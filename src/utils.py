from statistics import mode
import matplotlib
import yaml
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import socket
import logging
import random
import cv2
from PIL import Image
from matplotlib import cm
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
import math
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as scio

from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np

from model import DCGMM


import torchvision
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch


def set_seeds(cfg):
    seed = cfg["experiment"]["seed"]
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    
    return

def load_config(args):

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_loss_fn(cfg, inp_shape):
    # return function of the form loss = fn(y_true, y_pred)

    if cfg['dataset']['name'] == 'MNIST':
        
        pixel_count = tf.cast(inp_shape[0], dtype=tf.float32)

        def loss_fn(y_true, x_decoded_mean):
            loss = pixel_count * BinaryCrossentropy()(y_true, x_decoded_mean)
            return loss
        
        return loss_fn
    
    elif cfg['dataset']['name']== 'CIFAR10':

        pixel_count = tf.cast(tf.reduce_prod(inp_shape), dtype=tf.float32)

        def loss_fn(y_true, x_decoded_mean):
            
            loss = 1e-3 * pixel_count * BinaryCrossentropy()(y_true, x_decoded_mean)
            return loss
        
        return loss_fn
    
    elif cfg['dataset']['name'] == 'STL10':
        
        pixel_count = tf.cast(tf.reduce_prod(inp_shape), dtype=tf.float32)
        #pixel_count = inp_shape #only valid for pretrained resnet

        def loss_fn(y_true, x_decoded_mean):
            loss = pixel_count * MeanSquaredError()(y_true, x_decoded_mean)
            return loss
        
        return loss_fn
    
    else:
        raise NotImplementedError()


def get_data(cfg, stl_pretrained=False):
    if cfg['dataset']['name'] == 'MNIST':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train / 255.
        x_train = np.reshape(x_train, (-1, 28 * 28))
        x_test = x_test / 255.
        x_test = np.reshape(x_test, (-1, 28 * 28))

    elif cfg['dataset']['name'] == 'CIFAR10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        x_train = x_train / 255.
        x_test = x_test / 255.

        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)


    elif stl_pretrained:
        X = np.load("../dataset/stl10/stl_features.npy")
        X = X.astype('float32')
        y = np.load("../dataset/stl10/stl_label.npy")
        y = y.astype('int32')
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    
    elif cfg['dataset']['name'] == 'STL10':

        path = '/cluster/scratch/goezsoy/contrastive_dcgmm_datasets/stl10_matlab/'
        data=scio.loadmat(path+'train.mat')
        x_train = data['X']
        y_train = data['y'].squeeze()
        data=scio.loadmat(path+'test.mat')
        x_test = data['X']
        y_test = data['y'].squeeze()
        X = []
        Y = []
        X.append(x_train)
        X.append(x_test)
        Y.append(y_train)
        Y.append(y_test)
        X = np.concatenate(X,axis=0)
        Y = np.concatenate(Y,axis=0)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

        x_train = np.reshape(x_train,(-1,3,96,96))
        x_train = np.transpose(x_train,(0,1,3,2))
        x_train = np.transpose(x_train,(0,2,3,1)).astype('float32') # added astype newly
    

        x_test = np.reshape(x_test,(-1,3,96,96))
        x_test = np.transpose(x_test,(0,1,3,2))
        x_test = np.transpose(x_test,(0,2,3,1)).astype('float32') # added astype newly


        y_train = y_train - 1
        y_test = y_test - 1

        # [0,255] -> [0,1] ADDED FOR RESNET VAE. comment them for pretrained resnet
        #x_train = x_train / 255.
        #x_test = x_test / 255.

    else:
        raise NotImplementedError()

    
    return x_train, x_test, y_train, y_test


def get_model(cfg, inp_shape):
    return DCGMM(cfg['model'], inp_shape)


# NOT USED FOR RESNET VAE ACTUALLY
def get_feature_extractor(inp_shape):

    inputs = tf.keras.layers.Input(inp_shape, dtype = tf.uint8)
    x = tf.cast(inputs, tf.float32)
    x = tf.keras.applications.resnet.preprocess_input(x)
    outputs = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', pooling='avg')(x)
    feature_extractor = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    '''device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #print(f'resnet is on device: {device}')

    res50_model = torchvision.models.resnet50(pretrained=True)
    res50_conv = nn.Sequential(*list(res50_model.children())[:-2])
    res50_conv.eval()
    res50_conv = res50_conv#.to(device)'''
    return feature_extractor


def get_learning_rate_scheduler(cfg):

    def learning_rate_scheduler(epoch):
        initial_lrate = cfg['training']['learning_rate']
        drop = cfg['training']['decay_rate']
        epochs_drop = cfg['training']['epochs_lr']
        lrate = initial_lrate * math.pow(drop,
                                            math.floor((1 + epoch) / epochs_drop))
        return lrate
    
    return learning_rate_scheduler


def accuracy_metric(inp, p_c_z):
    y = inp
    y_pred = tf.math.argmax(p_c_z, axis=-1)
    return tf.numpy_function(cluster_acc, [y, y_pred], tf.float64)


def get_alpha(cfg):
    q = cfg['training']['q']
    if q > 0:
        alpha = 1000 * np.log((1 - q) / q)
    else:
        alpha = cfg['training']['alpha']
    return alpha


def get_logger(experiment_path, ex_name):

    performance_logger = logging.getLogger(ex_name + '_perf_logger')
    performance_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s  - %(message)s','%d-%m-%Y %H:%M')
    perf_file_handler = logging.FileHandler(os.path.join(experiment_path, ex_name + '_performance'))
    perf_file_handler.setLevel(logging.INFO)
    perf_file_handler.setFormatter(formatter)
    performance_logger.addHandler(perf_file_handler)

    return performance_logger


'''# below is not used anywhere
def setup_logger(results_path, create_stdlog):
    """Setup a general logger which saves all logs in the experiment folder"""

    f_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler = logging.FileHandler(str(results_path))
    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(f_format)

    root_logger = logging.getLogger(ROOT_LOGGER_STR)
    root_logger.handlers = []
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(f_handler)

    if create_stdlog:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        root_logger.addHandler(handler)'''


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((int(D), (D)), dtype=np.int64)
    for i in range(y_pred.size):
        w[int(y_pred[i]), int(y_true[i])] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def get_assigned_cluster_mapping(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((int(D), (D)), dtype=np.int64)
    for i in range(y_pred.size):
        w[int(y_pred[i]), int(y_true[i])] += 1
    ind = linear_assignment(w.max() - w)

    return ind[:, 1]


def make_confusion_matrix(y_true, y_pred, num_classes):
    assert len(y_pred) == len(y_true), "Lengths must match"
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int)

    cluster_mapping = list(get_assigned_cluster_mapping(y_true, y_pred))

    for i in range(len(y_pred)):
        conf_mat[y_true[i]][cluster_mapping[y_pred[i]]] += 1

    return conf_mat


def plot_image_rectangle(image_array, img_width, img_height, num_channels, path=None):
    num_width = image_array.shape[1]
    num_height = image_array.shape[0]

    sprite_image = np.ones((img_height * num_height, img_width * num_width, num_channels))

    for i in range(num_height):
        for j in range(num_width):
            this_img = image_array[i, j]
            sprite_image[i * img_height:(i + 1) * img_height, j * img_width:(j + 1) * img_width, 0:num_channels] \
                = this_img

    if path is not None:
        sprite_image *= 255.
        im = Image.fromarray(bgr_to_rgb(np.uint8(sprite_image)))
        im.save(path)
    else:
        cv2.imshow("image", sprite_image)
        cv2.waitKey(0)


def plot_image_square(image_array, img_width, img_height, num_channels, path=None, invert=False):
    num_images = image_array.shape[0]

    # Ensure shape
    if num_channels == 1:
        image_array = np.reshape(image_array, (-1, img_width, img_height))
    else:
        image_array = np.reshape(image_array, (-1, img_width, img_height, num_channels))

    # Invert pixel values
    if invert:
        image_array = 1 - image_array

    image_array = np.array(image_array)

    # Plot images in square
    n_plots = int(np.ceil(np.sqrt(num_images)))

    # Save image
    if num_channels == 1:
        sprite_image = np.ones((img_height * n_plots, img_width * n_plots))
    else:
        sprite_image = np.ones((img_height * n_plots, img_width * n_plots, num_channels))

    # fill the sprite templates
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < image_array.shape[0]:
                this_img = image_array[this_filter]
                if num_channels == 1:
                    sprite_image[i * img_height:(i + 1) * img_height, j * img_width:(j + 1) * img_width] = this_img
                else:
                    sprite_image[i * img_height:(i + 1) * img_height, j * img_width:(j + 1) * img_width,
                    0:num_channels] = this_img

    # save the sprite image
    if num_channels == 1:
        if path is not None:
            plt.imsave(path, sprite_image, cmap='gray')
        else:
            plt.axis('off')
            plt.imshow(sprite_image, cmap='gray')
            plt.show()
            plt.close()
    else:
        if path is not None:
            sprite_image *= 255.
            cv2.imwrite(path, sprite_image)
        else:
            cv2.imshow("image", sprite_image)
            cv2.waitKey(0)


def bgr_to_rgb(bgr):
    return bgr[:, :, ::-1]
