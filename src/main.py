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
from dataset import DataGenerator, UnsupervisedDataGenerator
from projector_plugin import ProjectorPlugin

# for visualizing latent space
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def pretrain(cfg, model, performance_logger, feature_extractor, x_train, x_test, y_train, y_test, experiment_path, inp_shape):

    performance_logger.info('started pretraining')
    print('started pretraining')

    # use both train and test dataset
    X = np.concatenate((x_train, x_test))
    Y = np.concatenate((y_train, y_test))

    # extract features of all samples before pretraining if Task 1
    if cfg['model']['type'] == 'FC': # task 1
        print('extracting image features')
        dataloader = tf.data.Dataset.from_tensor_slices(X).batch(128)
        total_output = []
        for batch in dataloader:
            output = feature_extractor(batch, training=False)
            total_output.append(output)
        X = tf.concat(total_output,axis=0)
        print('extraction finished')
    

    # Get the AE from DCGMM
    input = tfkl.Input(shape=inp_shape)

    if cfg['model']['type'] == "FC":
        f = tfkl.Flatten()(input)
        e1 = model.encoder.dense1(f)
        e2 = model.encoder.dense2(e1)
        e3 = model.encoder.dense3(e2)
        z = model.encoder.mu(e3)
        d1 = model.decoder.dense1(z)
        d2 = model.decoder.dense2(d1)
        d3 = model.decoder.dense3(d2)
        dec = model.decoder.dense4(d3)

    elif cfg['model']['type'] == "VGG":
        enc = input
        for block in model.encoder.layers:
            enc = block(enc)
        f = tfkl.Flatten()(enc)
        z = model.encoder.mu(f)
        d_dense = model.decoder.dense(z)
        d_reshape = model.decoder.reshape(d_dense)
        dec = d_reshape
        for block in model.decoder.layers:
            dec = block(dec)
        dec = model.decoder.convT(dec)

        if cfg['model']['activation']=='sigmoid':
            dec = tf.sigmoid(dec)


    autoencoder = tfk.Model(inputs=input, outputs=dec)

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['experiment']['lr_pretrain'])

    if cfg['model']['type'] == 'FC': # task 1, inputs are features.
        autoencoder.compile(optimizer=optimizer, loss="mse")
    elif cfg['dataset']['name'] == 'CIFAR10' or cfg['dataset']['name'] == 'STL10': # task 2, inputs are images.
        autoencoder.compile(optimizer=optimizer, loss="binary_crossentropy")
    else:
        autoencoder.compile(optimizer=optimizer, loss="mse")

    # train AE
    autoencoder.fit(X, X, epochs=cfg['experiment']['epochs_pretrain'], batch_size=cfg['experiment']['pretrain_batch_size'], verbose=2)

    # get embeddings from trained model for initializing GMM mean and variances
    encoder = model.encoder
    input = tfkl.Input(shape=inp_shape)
    z, _ = encoder(input)
    z_model = tf.keras.models.Model(inputs=input, outputs=z)
    z = z_model.predict(X)

    estimator = GaussianMixture(n_components=cfg['model']['num_clusters'], covariance_type='diag', n_init=3)
    estimator.fit(z)

    # assign weights to GMM mixtures of DCGMM
    mu_samples = estimator.means_
    sigma_samples = estimator.covariances_
    model.c_mu.assign(mu_samples)
    model.log_c_sigma.assign(np.log(sigma_samples))


    # visualize embeddings
    pca = PCA(n_components=2)
    z_transformed = pca.fit_transform(z)
    labels_transformed = np.squeeze(Y)
    label2class ={'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 
                    'frog':6, 'horse':7, 'ship':8, 'truck':9} # only valid for CIFAR-10, change for STL-10
    plt.figure(figsize=(10,10))
    scatter = plt.scatter(z_transformed[:,0], z_transformed[:,1], c=labels_transformed, cmap='tab10')
    plt.legend(handles=scatter.legend_elements()[0], labels=label2class)
    plt_save_path = os.path.join(experiment_path,'embedding_space.png')
    plt.savefig(plt_save_path)

    # measure pretraining performance
    encoder = model.encoder
    input = tfkl.Input(shape=inp_shape)
    z, _ = encoder(input)
    z_model = tf.keras.models.Model(inputs=input, outputs=z)
    yy = estimator.predict(z_model.predict(X))
    pretrain_acc = utils.cluster_acc(yy, Y)
    performance_logger.info(f'pretrain accuracy: {pretrain_acc}')
    print(f'pretrain accuracy: {pretrain_acc}')
    performance_logger.info(f'yy: {yy}')
    performance_logger.info(f'Y: {Y}')
    
    
    # visualize reconstructions for Task 2
    if cfg['model']['type'] != 'FC': # task 2
        proj = ProjectorPlugin(experiment_path)

        if cfg['dataset']['name'] == 'CIFAR10':
            temp_X = X[:900]

            rec = autoencoder.predict(temp_X)

            # convert bgr to rgb
            rec = rec[...,::-1]

            proj.save_image_sprites(rec, 32, 32, 3, False)
        
        elif cfg['dataset']['name'] == 'STL10':
            temp_X = X[:100]

            rec = autoencoder.predict(temp_X)

            # convert bgr to rgb
            rec = rec[...,::-1]

            proj.save_image_sprites(rec, 96, 96, 3, False)

    performance_logger.info('finished pretraining')
    print('finished pretraining')

    return model



def run_experiment(cfg):
    
    # initialize logger and write info about training
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

    # load dataset (STL-10 or CIFAR-10)
    x_train, x_test, y_train, y_test = utils.get_data(cfg)

    # initialize resnet-50 feature extractor to be used online if Task 1
    if cfg['model']['type'] == 'FC': # task 1
        inp_shape = x_train.shape[1:]
        feature_extractor = utils.get_feature_extractor(inp_shape)
        feat_size = feature_extractor(x_train[:1]).shape[1] # 2048 for STL-10

    else: # task 2
        feature_extractor = None
        feat_size = x_train.shape[1:] # (32, 32, 3) for CIFAR-10 or (96, 96, 3) for STL-10
    

    model = utils.get_model(cfg, feat_size) 

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['training']['learning_rate'], 
                                         beta_1=cfg['training']['beta_1'],
                                         beta_2=cfg['training']['beta_2'])

    # handle callbacks
    callback_list = []

    if cfg['training']['lrs']:
        callback_list.append(tf.keras.callbacks.LearningRateScheduler(utils.get_learning_rate_scheduler(cfg)))

    # pretrain model
    model = pretrain(cfg, model, performance_logger, feature_extractor, x_train, x_test, y_train, y_test, experiment_path, feat_size)
    
    # train model
    model.compile(optimizer, loss={"output_1": utils.get_loss_fn(cfg, feat_size)}, metrics={"output_4": utils.accuracy_metric})

    # generate train and test datasets
    if cfg['model']['contrastive']: # our approach (unsupervised)
        generator = UnsupervisedDataGenerator(x_train, y_train, num_constrains=cfg['training']['num_constrains'], alpha=alpha, q=cfg['training']['q'],
                        batch_size=cfg['training']['batch_size'], ml=cfg['training']['ml'], feature_extractor=feature_extractor)
    else: # original DC-GMM (supervised)
        generator = DataGenerator(x_train, y_train, num_constrains=cfg['training']['num_constrains'], alpha=alpha, q=cfg['training']['q'],
                        batch_size=cfg['training']['batch_size'], ml=cfg['training']['ml'], feature_extractor=feature_extractor)
    train_generator = generator.gen()

    generator_2 = DataGenerator(x_test, y_test, batch_size=cfg['training']['batch_size'], feature_extractor=feature_extractor)
    test_generator = generator_2.gen()

    # train the DC-GMM model
    model.fit(train_generator, validation_data=test_generator, steps_per_epoch=int(len(y_train)/cfg['training']['batch_size']), 
              validation_steps=len(y_test)//cfg['training']['batch_size'], epochs=cfg['training']['epochs'], callbacks=callback_list, verbose=2)



    # measure training performance
    if cfg['model']['type'] == 'FC': # task 1
        # extract features of all training samples
        dataloader = tf.data.Dataset.from_tensor_slices(x_train).batch(128)
        total_output = []
        for batch in dataloader:
            output = feature_extractor(batch, training=False)
            total_output.append(output)
        x_train_extracted = tf.concat(total_output,axis=0)
        rec, z_sample, p_z_c, p_c_z = model.predict([x_train_extracted, np.zeros(len(x_train_extracted))])
    
    else: # task 2
        rec, z_sample, p_z_c, p_c_z = model.predict([x_train, np.zeros(len(x_train))])

    yy = np.argmax(p_c_z, axis=-1)
    acc = utils.cluster_acc(y_train, yy)
    nmi = normalized_mutual_info_score(y_train, yy)
    ari = adjusted_rand_score(y_train, yy)

    performance_logger.info("Train Accuracy: %f, NMI: %f, ARI: %f" % (acc, nmi, ari))


    # measure test performance
    if cfg['model']['type'] == 'FC': # task 1
        # extract features of all test samples
        dataloader = tf.data.Dataset.from_tensor_slices(x_test).batch(128)
        total_output = []
        for batch in dataloader:
            output = feature_extractor(batch, training=False)
            total_output.append(output)
        x_test_extracted = tf.concat(total_output,axis=0)
        rec, z_sample, p_z_c, p_c_z = model.predict([x_test_extracted, np.zeros(len(x_test_extracted))])
    
    else: # task 2
        rec, z_sample, p_z_c, p_c_z = model.predict([x_test, np.zeros(len(x_test))])

    yy = np.argmax(p_c_z, axis=-1)
    acc = utils.cluster_acc(y_test, yy)
    nmi = normalized_mutual_info_score(y_test, yy)
    ari = adjusted_rand_score(y_test, yy)

    performance_logger.info("Test Accuracy: %f, NMI: %f, ARI: %f.\n" % (acc, nmi, ari))

    # save test set confusion matrix
    conf_mat = utils.make_confusion_matrix(y_test, yy, cfg['model']['num_clusters'])
    np.save(os.path.join(experiment_path,'conf_mat.npy'), conf_mat)

    # save test set embeddings for images
    if cfg['experiment']['save_embedding'] and cfg['model']['type'] != 'FC':
        proj = ProjectorPlugin(experiment_path, z_sample)

        proj.save_labels(y_test)

        if cfg['dataset']['name'] == 'CIFAR10':
            proj.save_image_sprites(rec, 32, 32, 3, True)
        elif cfg['dataset']['name'] == 'STL10':
            proj.save_image_sprites(rec, 96, 96, 3, True)

        proj.finalize()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to a config file.")

    _args = parser.parse_args()

    cfg = utils.load_config(_args)

    utils.set_seeds(cfg)

    run_experiment(cfg)