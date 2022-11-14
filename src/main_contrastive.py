import os
import math
import time
import yaml
import uuid
import pickle
import logging
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from ast import literal_eval as make_tuple

import utils
from dataset import ContrastiveDataGeneratorPretraining, DataGenerator
from projector_plugin import ProjectorPlugin

from model import ContrastiveAE

# for visualizing latent space
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def run_experiment(cfg):
    
    if cfg['experiment']['name'] is None:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        ex_name = "{}_{}".format(str(timestr), uuid.uuid4().hex[:5])
    else:
        ex_name = cfg['experiment']['name']

    experiment_path = os.path.join(cfg['dir']['logging'], cfg['dataset']['name'], ex_name)
    Path(experiment_path).mkdir(parents=True)
    
    performance_logger = utils.get_logger(experiment_path, ex_name)

    print(f'cfg: {cfg}')
    performance_logger.info(f'cfg: {cfg}')
    performance_logger.info(f'num GPUs: {len(tf.config.list_physical_devices("GPU"))}')

    alpha = utils.get_alpha(cfg)

    x_train, x_test, y_train, y_test = utils.get_data(cfg)

    # PRETRAINING

    X = np.concatenate((x_train, x_test))
    Y = np.concatenate((y_train, y_test))

    pre_generator = ContrastiveDataGeneratorPretraining(X, batch_size=cfg['experiment']['pretrain_batch_size'])

    pretrain_generator = pre_generator.gen()

    feat_size = (32, 32, 3) # (32,32,3) for cifar10 or (96,96,3) for stl10
    pretrain_model = ContrastiveAE(cfg['model'], feat_size) 

    pretrain_optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['experiment']['lr_pretrain'])
    

    # train model
    pretrain_model.compile(pretrain_optimizer, loss={"output_1": utils.get_loss_fn(cfg, feat_size)})

    pretrain_model.fit(pretrain_generator, steps_per_epoch=int(len(Y)/(cfg['experiment']['pretrain_batch_size'])), epochs=cfg['experiment']['epochs_pretrain'], verbose=2)


    # visualize embeddings
    encoder = pretrain_model.encoder
    input = tfkl.Input(shape=feat_size)
    z, _ = encoder(input)
    z_model = tf.keras.models.Model(inputs=input, outputs=z)
    z = z_model.predict(X)
    z_test = z_model.predict(x_test)

    
    labels_transformed = np.squeeze(Y)
    labels_transformed_test = np.squeeze(y_test)
    label2class ={'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 
                    'frog':6, 'horse':7, 'ship':8, 'truck':9} # only valid for CIFAR-10, change for STL-10

    # only test embeddings (to avoid overcrowding)
    pca = PCA(n_components=2)
    z_transformed = pca.fit_transform(z_test)
    plt.figure(figsize=(10,10))
    scatter = plt.scatter(z_transformed[:,0], z_transformed[:,1], c=labels_transformed_test, cmap='tab10')
    plt.legend(handles=scatter.legend_elements()[0], labels=label2class)
    plt_save_path = os.path.join(experiment_path,'embedding_space_pca_onlytest.png')
    plt.savefig(plt_save_path)

    # only test embeddings (to avoid overcrowding)
    tsne = TSNE(n_components=2)
    z_transformed_tsne = tsne.fit_transform(z_test)
    plt.figure(figsize=(10,10))
    scatter = plt.scatter(z_transformed_tsne[:,0], z_transformed_tsne[:,1], c=labels_transformed_test, cmap='tab10')
    plt.legend(handles=scatter.legend_elements()[0], labels=label2class)
    plt_save_path = os.path.join(experiment_path,'embedding_space_tsne_onlytest.png')
    plt.savefig(plt_save_path)

    # measure AE performance
    estimator = GaussianMixture(n_components=cfg['model']['num_clusters'], covariance_type='diag', n_init=3)
    estimator.fit(z)
    yy = estimator.predict(z)
    pretrain_acc = utils.cluster_acc(yy, Y)
    performance_logger.info(f'pretrain accuracy: {pretrain_acc}')
    print(f'pretrain accuracy: {pretrain_acc}')
    performance_logger.info(f'yy: {yy}')
    performance_logger.info(f'Y: {Y}')
    
    # visualize for e.g. X[:100]
    temp_X = X[:900]

    proj = ProjectorPlugin(experiment_path)

    rec = pretrain_model.predict(temp_X)

    # convert bgr to rgb
    rec = rec[...,::-1]

    proj.save_image_sprites(rec, 32, 32, 3, False)

    performance_logger.info('finished pretraining')
    print('finished pretraining')

    
    # DC-GMM TRAINING
    generator = DataGenerator(x_train, y_train, num_constrains=cfg['training']['num_constrains'], alpha=alpha, q=cfg['training']['q'],
                        batch_size=cfg['training']['batch_size'], ml=cfg['training']['ml'])

    train_generator = generator.gen()

    test_generator = DataGenerator(x_test, y_test, batch_size=cfg['training']['batch_size']).gen()

    
    model = utils.get_model(cfg, feat_size)

    # initializing the dcgmm weights by calling it with an example batch for pretrained weights assignment
    b = next(train_generator)
    model(b[0])

    # assign weights to GMM mixtures of DCGMM
    mu_samples = estimator.means_
    sigma_samples = estimator.covariances_
    model.c_mu.assign(mu_samples)
    model.log_c_sigma.assign(np.log(sigma_samples))

    # assign pretrained encoder and decoder weights to DC-GMM
    model.get_layer('VGGEncoder').set_weights(pretrain_model.get_layer('VGGEncoder').get_weights())
    model.get_layer('VGGDecoder').set_weights(pretrain_model.get_layer('VGGDecoder').get_weights())

    performance_logger.info('assigned pretrained parameters to dcgmm model.')
    print('assigned pretrained parameters to dcgmm model.')


    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['training']['learning_rate'], 
                                         beta_1=cfg['training']['beta_1'],
                                         beta_2=cfg['training']['beta_2'])

    # handle callbacks
    callback_list = []

    if cfg['training']['lrs']:
        callback_list.append(tf.keras.callbacks.LearningRateScheduler(utils.get_learning_rate_scheduler(cfg)))

    
    # train model
    model.compile(optimizer, loss={"output_1": utils.get_loss_fn(cfg, feat_size)}, metrics={"output_4": utils.accuracy_metric})

    model.fit(train_generator, validation_data=test_generator, steps_per_epoch=int(len(y_train)/cfg['training']['batch_size']), 
              validation_steps=len(y_test)//cfg['training']['batch_size'], epochs=cfg['training']['epochs'], callbacks=callback_list, verbose=2)


    # measure training performance
    rec, z_sample, p_z_c, p_c_z = model.predict([x_train, np.zeros(len(x_train))])
    yy = np.argmax(p_c_z, axis=-1)
    performance_logger.info(f'after training dcgmm\ny_train: {y_train}\n yy: {yy}')
    acc = utils.cluster_acc(y_train, yy)
    nmi = normalized_mutual_info_score(y_train, yy)
    ari = adjusted_rand_score(y_train, yy)

    performance_logger.info("Train Accuracy: %f, NMI: %f, ARI: %f" % (acc, nmi, ari))


    # measure test performance
    
    rec, z_sample, p_z_c, p_c_z = model.predict([x_test, np.zeros(len(x_test))])
    yy = np.argmax(p_c_z, axis=-1)
    acc = utils.cluster_acc(y_test, yy)
    nmi = normalized_mutual_info_score(y_test, yy)
    ari = adjusted_rand_score(y_test, yy)

    performance_logger.info("Test Accuracy: %f, NMI: %f, ARI: %f.\n" % (acc, nmi, ari))

    return




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to a config file.")

    _args = parser.parse_args()

    cfg = utils.load_config(_args)

    run_experiment(cfg)
