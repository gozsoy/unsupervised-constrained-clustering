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
from dataset import ContrastiveDataGeneratorPretraining
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

    generator = ContrastiveDataGeneratorPretraining(X, batch_size=cfg['training']['batch_size'])

    train_generator = generator.gen()

    feat_size = (32,32,3) #(96,96,3) or 2048  # both are only valid for stl-10
    model = ContrastiveAE(cfg['model'], feat_size) 

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['training']['learning_rate'], 
                                         beta_1=cfg['training']['beta_1'],
                                         beta_2=cfg['training']['beta_2'])

    
    # handle callbacks
    callback_list = []

    if cfg['training']['lrs']:
        callback_list.append(tf.keras.callbacks.LearningRateScheduler(utils.get_learning_rate_scheduler(cfg)))

    if cfg['experiment']['save_model']:
        checkpoint_path = os.path.join(cfg['dir']['checkpoint'], cfg['dataset']['name'], ex_name)
        Path(checkpoint_path).mkdir(parents=True)
        callback_list.append(tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=0, monitor='cont_loss',
                                                              save_weights_only=True, save_best_only=True))


    # train model
    model.compile(optimizer)#, loss={"output_1": utils.get_loss_fn(cfg, feat_size)})

    model.fit(train_generator, steps_per_epoch=int(len(Y)/(cfg['training']['batch_size'])), epochs=cfg['training']['epochs'], callbacks=callback_list, verbose=2)
    
    # visualize embeddings
    encoder = model.encoder
    input = tfkl.Input(shape=feat_size)
    z = encoder(input)
    z_model = tf.keras.models.Model(inputs=input, outputs=z)
    z = z_model.predict(X)
    z_test = z_model.predict(x_test)

    labels_transformed = np.squeeze(Y)
    labels_transformed_test = np.squeeze(y_test)
    label2class ={'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 
                    'frog':6, 'horse':7, 'ship':8, 'truck':9}

    # both train and test embeddings
    pca = PCA(n_components=2)
    z_transformed = pca.fit_transform(z)
    plt.figure(figsize=(10,10))
    scatter = plt.scatter(z_transformed[:,0], z_transformed[:,1], c=labels_transformed, cmap='tab10')
    plt.legend(handles=scatter.legend_elements()[0], labels=label2class)
    plt_save_path = os.path.join(experiment_path,'embedding_space_pca.png')
    plt.savefig(plt_save_path)

    # both train and test embeddings
    tsne = TSNE(n_components=2)
    z_transformed_tsne = tsne.fit_transform(z)
    plt.figure(figsize=(10,10))
    scatter = plt.scatter(z_transformed_tsne[:,0], z_transformed_tsne[:,1], c=labels_transformed, cmap='tab10')
    plt.legend(handles=scatter.legend_elements()[0], labels=label2class)
    plt_save_path = os.path.join(experiment_path,'embedding_space_tsne.png')
    plt.savefig(plt_save_path)

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

    rec = model.predict(temp_X)

    # convert bgr to rgb
    rec = rec[...,::-1]

    proj.save_image_sprites(rec, 32, 32, 3, False)

    performance_logger.info('finished pretraining')
    print('finished pretraining')


    # DC-GMM TRAINING

    return




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to a config file.")

    _args = parser.parse_args()

    cfg = utils.load_config(_args)

    #utils.set_seeds(cfg)

    run_experiment(cfg)