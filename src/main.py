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
from dataset import ContrastiveDataGenerator, DataGenerator, MixedDataGenerator, UnsupervisedMixedDataGenerator, UnsupervisedDataGeneratorResNet
from projector_plugin import ProjectorPlugin

# for visualizing latent space
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def pretrain(cfg, model, performance_logger, feature_extractor, x_train, x_test, y_train, y_test, experiment_path):

    #x_train, x_test, y_train, y_test = utils.get_data(cfg, stl_pretrained=False)

    X = np.concatenate((x_train, x_test))
    Y = np.concatenate((y_train, y_test))


    # below not used for resnet vae
    '''print('extracting image features')
    dataloader = tf.data.Dataset.from_tensor_slices(X).batch(128)
    total_output = []
    for batch in dataloader:
        output = feature_extractor(batch, training=False)
        total_output.append(output)
    X = tf.concat(total_output,axis=0)
    print('extraction finished')
    inp_shape = X.shape[1]'''
    
    #inp_shape = (96,96,3) # stl10
    inp_shape = (32,32,3) # cifar10

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
    elif cfg['model']['type'] == "CNN":
        # my extension - not in use
        out = model.encoder.conv1(input)
        out = model.encoder.bn1(out)
        out = model.encoder.conv2(out)
        out = model.encoder.conv3(out)
        out = model.encoder.bn2(out)

        out = model.decoder.up1(out)
        out = model.decoder.conv1(out)
        out = model.decoder.bn1(out)
        dec = model.decoder.conv2(out)
        
        # original
        '''e1 = model.encoder.conv1(input)
        e2 = model.encoder.conv2(e1)
        f = tfkl.Flatten()(e2)
        z = model.encoder.mu(f)
        d1 = model.decoder.dense(z)
        d2 = model.decoder.reshape(d1)
        d3 = model.decoder.convT1(d2)
        d4 = model.decoder.convT2(d3)
        dec = model.decoder.convT3(d4)'''
        if cfg['model']['activation']=='sigmoid':
            dec = tf.sigmoid(dec)

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
        #dec = model.decoder.convT2(dec)

        if cfg['model']['activation']=='sigmoid':
            dec = tf.sigmoid(dec)

    elif cfg['model']['type'] == "ResNet":
        z_mu, _ = model.encoder(input)
        dec = model.decoder(z_mu)
    
    elif cfg['model']['type'] == "ResNetAE":
        dec = model.single_model(input)

    autoencoder = tfk.Model(inputs=input, outputs=dec)

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['experiment']['lr_pretrain'])
    if cfg['dataset']['name'] == 'MNIST' or cfg['dataset']['name'] == 'CIFAR10' or cfg['dataset']['name'] == 'STL10':
        autoencoder.compile(optimizer=optimizer, loss="binary_crossentropy")
    else: #elif cfg['dataset']['name'] == 'STL10': # use if pretrained resnet
        autoencoder.compile(optimizer=optimizer, loss="mse")

    pretrain_path = os.path.join(cfg['dir']['pretrain'],cfg['dataset']['name'],'autoencoder','cp.ckpt')
    gmm_save_path = os.path.join(cfg['dir']['pretrain'],cfg['dataset']['name'],'gmm_save.sav')

    # If the model should be run from scratch
    if cfg['experiment']['pretrain']:
        performance_logger.info('started pretraining')
        print('started pretraining')

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=pretrain_path,
                                                         save_weights_only=True, verbose=1)
        autoencoder.fit(X, X, epochs=cfg['experiment']['epochs_pretrain'], batch_size=256, callbacks=cp_callback, verbose=2)

        encoder = model.encoder
        input = tfkl.Input(shape=inp_shape)
        z, _ = encoder(input)
        z_model = tf.keras.models.Model(inputs=input, outputs=z)
        z = z_model.predict(X)

        estimator = GaussianMixture(n_components=cfg['model']['num_clusters'], covariance_type='diag', n_init=3)
        estimator.fit(z)
        pickle.dump(estimator, open(gmm_save_path, 'wb'))

        # visualize embeddings TRY TSNE AS WELL
        '''pca = PCA(n_components=2)
        z_transformed = pca.fit_transform(z)
        labels_transformed = np.squeeze(Y)
        label2class ={'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 
                      'frog':6, 'horse':7, 'ship':8, 'truck':9}
        plt.figure(figsize=(10,10))
        scatter = plt.scatter(z_transformed[:,0], z_transformed[:,1], c=labels_transformed, cmap='tab10')
        plt.legend(handles=scatter.legend_elements()[0], labels=label2class)
        plt_save_path = os.path.join(experiment_path,'embedding_space.png')
        plt.savefig(plt_save_path)'''


        performance_logger.info('finished pretraining')
        print('finished pretraining')

    else:
        autoencoder.load_weights(pretrain_path)
        estimator = pickle.load(open(gmm_save_path, 'rb'))

        performance_logger.info('loaded mnist pretrained weights')
        print('loaded mnist pretrained weights')

    
    # assign weights to GMM mixtures of DCGMM
    mu_samples = estimator.means_
    sigma_samples = estimator.covariances_
    model.c_mu.assign(mu_samples)
    model.log_c_sigma.assign(np.log(sigma_samples))

    # measure AE performance
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
    
    
    # visualize for e.g. X[:100]
    temp_X = X[:900]

    proj = ProjectorPlugin(experiment_path)

    rec = autoencoder.predict(temp_X)

    # convert bgr to rgb
    rec = rec[...,::-1]

    proj.save_image_sprites(rec, 32, 32, 3, False)


    return model



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

    # below 2 rows only valid when doing pre-feature extraction with resnet50
    inp_shape = x_train.shape[1:]
    feature_extractor = utils.get_feature_extractor(inp_shape) # WILL NOT BE USED FOR RESNET VAE
    
    #feat_size = feature_extractor(x_train[:1]).shape[1]
    feat_size = (32,32,3)
    #feat_size = (96, 96, 3) #(96,96,3) or 2048  # both are only valid for stl-10
    model = utils.get_model(cfg, feat_size)  # only works with "FC" 


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
        callback_list.append(tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1,
                                                              save_weights_only=True, period=100))

    # pretrain model
    model = pretrain(cfg, model, performance_logger, feature_extractor, x_train, x_test, y_train, y_test, experiment_path)
    
    # train model
    model.compile(optimizer, loss={"output_1": utils.get_loss_fn(cfg, feat_size)}, metrics={"output_4": utils.accuracy_metric})#, run_eagerly=True)

    # generate train and test datasets
    generator = UnsupervisedDataGeneratorResNet(x_train, y_train, num_constrains=cfg['training']['num_constrains'], alpha=alpha, q=cfg['training']['q'],
                        batch_size=cfg['training']['batch_size'], ml=cfg['training']['ml'], feature_extractor=feature_extractor)
    train_generator = generator.gen()

    generator_2 = DataGenerator(x_test, y_test, batch_size=cfg['training']['batch_size'], feature_extractor=feature_extractor)
    test_generator = generator_2.gen()

    model.fit(train_generator, validation_data=test_generator, steps_per_epoch=int(len(y_train)/cfg['training']['batch_size']), 
              validation_steps=len(y_test)//cfg['training']['batch_size'], epochs=cfg['training']['epochs'], callbacks=callback_list, verbose=2)



    # measure training performance
    # commented section is when using resnet for feature extraction
    '''dataloader = tf.data.Dataset.from_tensor_slices(x_train).batch(128)
    total_output = []
    for batch in dataloader:
        output = feature_extractor(batch, training=False)
        total_output.append(output)
    x_train_extracted = tf.concat(total_output,axis=0)
    rec, z_sample, p_z_c, p_c_z = model.predict([x_train_extracted, np.zeros(len(x_train_extracted))])'''

    rec, z_sample, p_z_c, p_c_z = model.predict([x_train, np.zeros(len(x_train))])
    yy = np.argmax(p_c_z, axis=-1)
    acc = utils.cluster_acc(y_train, yy)
    nmi = normalized_mutual_info_score(y_train, yy)
    ari = adjusted_rand_score(y_train, yy)
    # sc metric computation
    '''ml_ind1 = generator.ml_ind1
    ml_ind2 = generator.ml_ind2
    cl_ind1 = generator.cl_ind1
    cl_ind2 = generator.cl_ind2
    count = 0
    if cfg['training']['num_constrains'] == 0:
        sc = 0
    else:
        maxx = len(ml_ind1) + len(cl_ind1)
        for i in range(len(ml_ind1)):
            if yy[ml_ind1[i]] == yy[ml_ind2[i]]:
                count += 1
        for i in range(len(cl_ind1)):
            if yy[cl_ind1[i]] != yy[cl_ind2[i]]:
                count += 1
        sc = count / maxx'''

    '''performance_logger.info("Train Accuracy: %f, NMI: %f, ARI: %f, sc: %f." % (acc, nmi, ari, sc))'''
    performance_logger.info("Train Accuracy: %f, NMI: %f, ARI: %f" % (acc, nmi, ari))


    # measure test performance
    # commented section is when using resnet for feature extraction
    '''dataloader = tf.data.Dataset.from_tensor_slices(x_test).batch(128)
    total_output = []
    for batch in dataloader:
        output = feature_extractor(batch, training=False)
        total_output.append(output)
    x_test_extracted = tf.concat(total_output,axis=0)
    rec, z_sample, p_z_c, p_c_z = model.predict([x_test_extracted, np.zeros(len(x_test_extracted))])'''
    
    rec, z_sample, p_z_c, p_c_z = model.predict([x_test, np.zeros(len(x_test))])
    yy = np.argmax(p_c_z, axis=-1)
    acc = utils.cluster_acc(y_test, yy)
    nmi = normalized_mutual_info_score(y_test, yy)
    ari = adjusted_rand_score(y_test, yy)

    performance_logger.info("Test Accuracy: %f, NMI: %f, ARI: %f.\n" % (acc, nmi, ari))

    # save test set confusion matrix
    conf_mat = utils.make_confusion_matrix(y_test, yy, cfg['model']['num_clusters'])
    np.save(os.path.join(experiment_path,'conf_mat.npy'), conf_mat)

    # save test set embeddings
    if cfg['experiment']['save_embedding']:
        proj = ProjectorPlugin(experiment_path, z_sample)

        proj.save_labels(y_test)

        # Add images to projector (INSTEAD OF X_TEST IT SHOULD BE REC)
        if cfg['dataset']['name'] == 'MNIST':
            proj.save_image_sprites(x_test, 28, 28, 1, True)
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