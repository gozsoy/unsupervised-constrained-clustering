import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
from ast import literal_eval as make_tuple
import math


tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras

from tensorflow.keras import layers


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), stride=(1, 1)):

        super(ResidualBlock, self).__init__()

        self.residual_block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                   strides=stride, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                   strides=stride, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
        ])

    def call(self, x, **kwargs):
        return x + self.residual_block(x)


class ResNetEncoder(tf.keras.models.Model):
    def __init__(self,
                 n_ResidualBlock=8,
                 n_levels=4,
                 z_dim=10,
                 bUseMultiResSkips=True):

        super(ResNetEncoder, self).__init__()

        self.max_filters = 2 ** (n_levels+3)
        self.n_levels = n_levels
        self.bUseMultiResSkips = bUseMultiResSkips

        self.conv_list = []
        self.res_blk_list = []
        self.multi_res_skip_list = []

        self.input_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3),
                                   strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
        ])

        for i in range(n_levels):
            n_filters_1 = 2 ** (i + 3)
            n_filters_2 = 2 ** (i + 4)
            ks = 2 ** (n_levels - i)

            self.res_blk_list.append(
                tf.keras.Sequential([ResidualBlock(n_filters_1)
                                     for _ in range(n_ResidualBlock)])
            )

            self.conv_list.append(
                tf.keras.Sequential([
                    tf.keras.layers.Conv2D(filters=n_filters_2, kernel_size=(2, 2),
                                           strides=(2, 2), padding='same'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(alpha=0.2),
                ])
            )

            if bUseMultiResSkips:
                self.multi_res_skip_list.append(
                    tf.keras.Sequential([
                        tf.keras.layers.Conv2D(filters=self.max_filters, kernel_size=(ks, ks),
                                               strides=(ks, ks), padding='same'),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(alpha=0.2),
                    ])
                )

        self.output_conv = tf.keras.layers.Conv2D(filters=z_dim, kernel_size=(3, 3),
                                                  strides=(1, 1), padding='same')

    def call(self, x, **kwargs):

        x = self.input_conv(x)

        skips = []
        for i in range(self.n_levels):
            x = self.res_blk_list[i](x)
            if self.bUseMultiResSkips:
                skips.append(self.multi_res_skip_list[i](x))
            x = self.conv_list[i](x)

        if self.bUseMultiResSkips:
            x = sum([x] + skips)

        x = self.output_conv(x)

        return x


class ResNetDecoder(tf.keras.models.Model):
    def __init__(self,
                 n_ResidualBlock=8,
                 n_levels=4,
                 output_channels=3,
                 bUseMultiResSkips=True):

        super(ResNetDecoder, self).__init__()

        self.max_filters = 2 ** (n_levels+3)
        self.n_levels = n_levels
        self.bUseMultiResSkips = bUseMultiResSkips

        self.conv_list = []
        self.res_blk_list = []
        self.multi_res_skip_list = []

        self.input_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=self.max_filters, kernel_size=(3, 3),
                                   strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
        ])

        for i in range(n_levels):
            n_filters = 2 ** (self.n_levels - i + 2)
            ks = 2 ** (i + 1)

            self.res_blk_list.append(
                tf.keras.Sequential([ResidualBlock(n_filters)
                                     for _ in range(n_ResidualBlock)])
            )

            self.conv_list.append(
                tf.keras.Sequential([
                    tf.keras.layers.Conv2DTranspose(filters=n_filters, kernel_size=(2, 2),
                                                    strides=(2, 2), padding='same'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(alpha=0.2),
                ])
            )

            if bUseMultiResSkips:
                self.multi_res_skip_list.append(
                    tf.keras.Sequential([
                        tf.keras.layers.Conv2DTranspose(filters=n_filters, kernel_size=(ks, ks),
                                                        strides=(ks, ks), padding='same'),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(alpha=0.2),
                    ])
                )

        self.output_conv = tf.keras.layers.Conv2D(filters=output_channels, kernel_size=(3, 3),
                                                  strides=(1, 1), padding='same')

    def call(self, z, **kwargs):

        z = z_top = self.input_conv(z)

        for i in range(self.n_levels):
            z = self.conv_list[i](z)
            z = self.res_blk_list[i](z)
            if self.bUseMultiResSkips:
                z += self.multi_res_skip_list[i](z_top)

        z = self.output_conv(z)

        return z


class ResNetAE(tf.keras.models.Model):
    def __init__(self,
                 input_shape=(256, 256, 3),
                 n_ResidualBlock=8,
                 n_levels=4,
                 z_dim=128,
                 bottleneck_dim=128,
                 bUseMultiResSkips=True):
        super(ResNetAE, self).__init__()

        assert input_shape[0] == input_shape[1]
        output_channels = input_shape[2]
        self.z_dim = z_dim
        self.img_latent_dim = input_shape[0] // (2 ** n_levels)

        self.encoder = ResNetEncoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     z_dim=z_dim, bUseMultiResSkips=bUseMultiResSkips)
        self.decoder = ResNetDecoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     output_channels=output_channels, bUseMultiResSkips=bUseMultiResSkips)

        self.fc1 = tf.keras.layers.Dense(bottleneck_dim)
        self.fc2 = tf.keras.layers.Dense(self.img_latent_dim * self.img_latent_dim * self.z_dim)

    def encode(self, x):
        h = self.encoder(x)
        h = tf.keras.backend.reshape(h, shape=(-1, self.img_latent_dim * self.img_latent_dim * self.z_dim))
        return self.fc1(h)

    def decode(self, z):
        z = self.fc2(z)
        z = tf.keras.backend.reshape(z, shape=(-1, self.img_latent_dim, self.img_latent_dim, self.z_dim))
        h = self.decoder(z)
        return tf.keras.backend.sigmoid(h)

    def call(self, x, **kwargs):
        return self.decode(self.encode(x))

'''# borrowed & modified from https://github.com/farrell236/ResNetAE/blob/master/ResNetAE.py
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), stride=(1, 1)):

        super(ResidualBlock, self).__init__()

        self.residual_block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                   strides=stride, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                   strides=stride, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
        ])

    def call(self, x):
        return x + self.residual_block(x)

# borrowed & modified from https://github.com/farrell236/ResNetAE/blob/master/ResNetAE.py
class ResNetEncoder(tf.keras.models.Model):
    def __init__(self,
                 input_shape=(32, 32, 3), # for stl10
                 n_ResidualBlock=2,
                 n_levels=4,
                 encoded_size=100):

        super(ResNetEncoder, self).__init__()

        self.n_levels = n_levels
        self.img_latent_dim = input_shape[0] // (2 ** n_levels)
        self.z_dim = encoded_size

        self.conv_list = []
        self.res_blk_list = []

        self.input_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3),
                                   strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
        ])

        for i in range(n_levels):
            n_filters_1 = 2 ** (i + 3)
            n_filters_2 = 2 ** (i + 4)
            ks = 2 ** (n_levels - i)

            self.res_blk_list.append(
                tf.keras.Sequential([ResidualBlock(n_filters_1)
                                     for _ in range(n_ResidualBlock)])
            )

            self.conv_list.append(
                tf.keras.Sequential([
                    tf.keras.layers.Conv2D(filters=n_filters_2, kernel_size=(2, 2),
                                           strides=(2, 2), padding='same'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(alpha=0.2),
                ])
            )

        self.output_conv = tf.keras.layers.Conv2D(filters=self.z_dim, kernel_size=(3, 3),
                                                  strides=(1, 1), padding='same')

        self.mu = tf.keras.layers.Dense(self.z_dim)
        self.sigma = tf.keras.layers.Dense(self.z_dim)

    def call(self, x):

        x = self.input_conv(x)
        
        for i in range(self.n_levels):
            x = self.res_blk_list[i](x)
            x = self.conv_list[i](x)

        x = self.output_conv(x)
        
        x = tf.keras.backend.reshape(x, shape=(-1, self.img_latent_dim * self.img_latent_dim * self.z_dim))

        return self.mu(x), self.sigma(x)

# borrowed & modified from https://github.com/farrell236/ResNetAE/blob/master/ResNetAE.py
class ResNetDecoder(tf.keras.models.Model):
    def __init__(self,
                 input_shape=(32, 32, 3), # for stl10
                 n_ResidualBlock=2,
                 n_levels=4, # should be same with encoder!
                 encoded_size=100, # should be same with encoder! same with config
                 activation=None): # should be same with config)

        super(ResNetDecoder, self).__init__()

        self.max_filters = 2 ** (n_levels+3)
        self.n_levels = n_levels
        
        self.output_channels = input_shape[2]
        self.img_latent_dim = input_shape[0] // (2 ** n_levels)
        self.z_dim = encoded_size
        self.activation = activation

        self.conv_list = []
        self.res_blk_list = []

        self.input_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=self.max_filters, kernel_size=(3, 3),
                                   strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
        ])

        for i in range(n_levels):
            n_filters = 2 ** (self.n_levels - i + 2)
            ks = 2 ** (i + 1)

            self.res_blk_list.append(
                tf.keras.Sequential([ResidualBlock(n_filters)
                                     for _ in range(n_ResidualBlock)])
            )

            self.conv_list.append(
                tf.keras.Sequential([
                    tf.keras.layers.Conv2DTranspose(filters=n_filters, kernel_size=(2, 2),
                                                    strides=(2, 2), padding='same'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(alpha=0.2),
                ])
            )

        self.output_conv = tf.keras.layers.Conv2D(filters=self.output_channels, kernel_size=(3, 3),
                                                  strides=(1, 1), padding='same')

        self.fc3 = tf.keras.layers.Dense(self.img_latent_dim * self.img_latent_dim * self.z_dim)

    def call(self, z):

        z = self.fc3(z)

        z = tf.keras.backend.reshape(z, shape=(-1, self.img_latent_dim, self.img_latent_dim, self.z_dim))

        z = self.input_conv(z)

        for i in range(self.n_levels):
            z = self.conv_list[i](z)
            z = self.res_blk_list[i](z)

        z = self.output_conv(z)

        if self.activation == "sigmoid":
            z = tf.sigmoid(z)

        return z'''


class VGGConvBlock(layers.Layer):
    def __init__(self, num_filters, block_id):
        super(VGGConvBlock, self).__init__(name="VGGConvBlock{}".format(block_id))
        self.conv1 = tfkl.Conv2D(filters=num_filters, kernel_size=(3, 3), activation='relu')
        self.conv2 = tfkl.Conv2D(filters=num_filters, kernel_size=(3, 3), activation='relu', strides=(2,2))
        #self.maxpool = tfkl.MaxPooling2D((2, 2))
        self.bn1 = tfkl.BatchNormalization()

    def call(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        #out = self.maxpool(out)
        out = self.bn1(out)

        return out


class VGGDeConvBlock(layers.Layer):
    def __init__(self, num_filters, block_id):
        super(VGGDeConvBlock, self).__init__(name="VGGDeConvBlock{}".format(block_id))
        self.upsample = tfkl.UpSampling2D((2, 2), interpolation='bilinear')
        self.convT1 = tfkl.Conv2DTranspose(filters=num_filters, kernel_size=(3, 3), padding='valid', activation='relu')
        self.convT2 = tfkl.Conv2DTranspose(filters=num_filters, kernel_size=(3, 3), padding='valid', activation='relu')
        self.bn1 = tfkl.BatchNormalization()

    def call(self, inputs):
        out = self.upsample(inputs)
        out = self.convT1(out)
        out = self.convT2(out)
        out = self.bn1(out)

        return out

# MY ADDITION 1
class VGGConvBlockAUX(layers.Layer):
    def __init__(self, num_filters, block_id):
        super(VGGConvBlockAUX, self).__init__(name="VGGConvBlockAUX{}".format(block_id))
        self.conv1 = tfkl.Conv2D(filters=num_filters, kernel_size=(3, 3), activation='relu', padding='same')
        self.conv2 = tfkl.Conv2D(filters=num_filters, kernel_size=(3, 3), activation='relu', padding='same')
        self.bn1 = tfkl.BatchNormalization()

    def call(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.bn1(out)

        return out

# MY ADDITION 2
class VGGDeConvBlockAUX(layers.Layer):
    def __init__(self, num_filters, block_id):
        super(VGGDeConvBlockAUX, self).__init__(name="VGGDeConvBlockAUX{}".format(block_id))
        self.convT1 = tfkl.Conv2DTranspose(filters=num_filters, kernel_size=(3, 3), padding='same', activation='relu')
        self.convT2 = tfkl.Conv2DTranspose(filters=num_filters, kernel_size=(3, 3), padding='same', activation='relu')
        self.bn1 = tfkl.BatchNormalization()

    def call(self, inputs):
        out = self.convT1(inputs)
        out = self.convT2(out)
        out = self.bn1(out)
        
        return out


class VGGEncoder(layers.Layer):
    def __init__(self, encoded_size):
        super(VGGEncoder, self).__init__(name='VGGEncoder')
        #self.layers = [VGGConvBlockAUX(32, 1), VGGConvBlockAUX(64, 2), VGGConvBlock(128, 3),VGGConvBlockAUX(128, 4),
        #              VGGConvBlock(256, 5),VGGConvBlockAUX(256, 6)]
        self.layers = [VGGConvBlock(128, 1),
                      VGGConvBlock(256, 2)]

        self.mu = tfkl.Dense(encoded_size, activation=None)
        self.sigma = tfkl.Dense(encoded_size, activation=None)

    def call(self, inputs):
        out = inputs

        # Iterate through blocks
        for block in self.layers:
            out = block(out)

        out_flat = tfkl.Flatten()(out)
        mu = self.mu(out_flat)
        sigma = self.sigma(out_flat)

        return mu, sigma



class VGGDecoder(layers.Layer):
    def __init__(self, input_shape, activation):
        super(VGGDecoder, self).__init__(name='VGGDecoder')

        #input_tuple = make_tuple(input_shape)
        input_tuple=input_shape
        if input_tuple == (64, 64, 1):
            target_shape = (13, 13, 64)
        elif input_tuple == (64, 64, 3):
            target_shape = (13, 13, 64)
        elif input_tuple == (32, 32, 3):  # compatible with cifar10
            target_shape = (5, 5, 256)
        elif input_tuple == (96, 96, 3): # compatible with stl10
            target_shape = (21, 21, 256)

        self.activation = activation
        self.dense = tfkl.Dense(target_shape[0] * target_shape[1] * target_shape[2])
        self.reshape = tfkl.Reshape(target_shape=target_shape)
        '''self.layers = [VGGDeConvBlock(256, 1),VGGDeConvBlockAUX(256, 2),
                       VGGDeConvBlock(128, 3),VGGDeConvBlockAUX(128, 4),
                       VGGDeConvBlockAUX(64, 5),VGGDeConvBlockAUX(32, 6)]s128, 3),VGGDeConvBlockAUX(128, 4),
                       VGGDeConvBlockAUX(64, 5),VGGDeConvBlockAUX(32, 6)]'''
        self.layers = [VGGDeConvBlock(256, 1),
                       VGGDeConvBlock(128, 2)]            
        self.convT = tfkl.Conv2DTranspose(filters=input_tuple[2], kernel_size=3, padding='same')
        #self.convT2 = tfkl.Conv2DTranspose(filters=input_tuple[2], kernel_size=3, padding='valid')

    def call(self, inputs):
        out = self.dense(inputs)
        out = self.reshape(out)

        # Iterate through blocks
        for block in self.layers:
            out = block(out)

        # Last convolution
        out = self.convT(out)
        #out = self.convT2(out)

        if self.activation == "sigmoid":
            out = tf.sigmoid(out)

        return out


'''class CNNEncoder(layers.Layer):
    def __init__(self, encoded_size):
        super(CNNEncoder, self).__init__(name='CNNEncoder')
        self.conv1 = tfkl.Conv2D(filters=32, kernel_size=4, strides=(2, 2), activation='relu')
        self.conv2 = tfkl.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')
        self.mu = tfkl.Dense(encoded_size, activation=None)
        self.sigma = tfkl.Dense(encoded_size, activation=None)

    def call(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out_flat = tfkl.Flatten()(out)  # Should be 15x15x64 for heart_echo, 7x7x64 for cifar10
        mu = self.mu(out_flat)
        sigma = self.sigma(out_flat)

        return mu, sigma


class CNNDecoder(layers.Layer):
    def __init__(self, input_shape, activation):
        super(CNNDecoder, self).__init__(name='CNNDecoder')
        self.activation = activation

        # TODO: Make this better
        input_tuple = make_tuple(input_shape)
        if input_tuple == (64, 64, 1):
            target_shape = (15, 15, 64)
        elif input_tuple == (64, 64, 3):
            target_shape = (15, 15, 64)
        elif input_tuple == (32, 32, 3):
            target_shape = (7, 7, 64)

        # self.dense = tfkl.Dense(15 * 15 * 64, activation='relu')
        # self.reshape = tfkl.Reshape(target_shape=(15, 15, 64))
        self.dense = tfkl.Dense(target_shape[0] * target_shape[1] * target_shape[2])
        self.reshape = tfkl.Reshape(target_shape=target_shape)
        self.convT1 = tfkl.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='valid',
                                           activation='relu')
        self.convT2 = tfkl.Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='valid',
                                           activation='relu')
        # self.convT3 = tfkl.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')
        self.convT3 = tfkl.Conv2DTranspose(filters=input_tuple[2], kernel_size=3, strides=1, padding='same')

    def call(self, inputs):
        out = self.dense(inputs)
        out = self.reshape(out)
        out = self.convT1(out)
        out = self.convT2(out)
        out = self.convT3(out)

        if self.activation == "sigmoid":
            out = tf.sigmoid(out)

        return out'''

class CNNEncoder(layers.Layer):
    def __init__(self, encoded_size):
        super(CNNEncoder, self).__init__(name='CNNEncoder')
        self.conv1 = tfkl.Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.bn1 = tfkl.BatchNormalization()
        self.conv2 = tfkl.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')
        self.conv3 = tfkl.Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.bn2 = tfkl.BatchNormalization()
        #self.mu = tfkl.Dense(encoded_size, activation=None)
        #self.sigma = tfkl.Dense(encoded_size, activation=None)

    def call(self, inputs):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn2(out)
    
        return out


class CNNDecoder(layers.Layer):
    def __init__(self, input_shape, activation):
        super(CNNDecoder, self).__init__(name='CNNDecoder')
        self.activation = activation

        self.up1 = tfkl.UpSampling2D()
        self.conv1 = tfkl.Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')  
        self.bn1 = tfkl.BatchNormalization()
        self.conv2 = tfkl.Conv2D(3,  kernel_size=1, strides=1, padding='same', activation='sigmoid')

    def call(self, inputs):
        out = self.up1(inputs)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.conv2(out)

        return out


class Encoder(layers.Layer):
    def __init__(self, encoded_size):
        super(Encoder, self).__init__(name='encoder')
        self.dense1 = tfkl.Dense(500, activation='relu')
        self.dense2 = tfkl.Dense(500, activation='relu')
        self.dense3 = tfkl.Dense(2000, activation='relu')
        self.mu = tfkl.Dense(encoded_size, activation=None)
        self.sigma = tfkl.Dense(encoded_size, activation=None)

    def call(self, inputs):
        x = tfkl.Flatten()(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma


class Decoder(layers.Layer):
    def __init__(self, input_shape, activation):
        super(Decoder, self).__init__(name='dec')
        # convert from tuple to int
        self.inp_shape = input_shape
        self.dense1 = tfkl.Dense(2000, activation='relu')
        self.dense2 = tfkl.Dense(500, activation='relu')
        self.dense3 = tfkl.Dense(500, activation='relu')
        if activation == "sigmoid":
            self.dense4 = tfkl.Dense(self.inp_shape, activation="sigmoid")
        else:
            self.dense4 = tfkl.Dense(self.inp_shape)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x


class DCGMM(tf.keras.Model):
    def __init__(self, cfg, inp_shape):
        super(DCGMM, self).__init__(name="DCGMM")
        self.encoded_size = cfg['latent_dim']
        self.num_clusters = cfg['num_clusters']
        self.inp_shape = inp_shape
        self.activation = cfg['activation']
        self.type = cfg['type']
        self.is_unsupervised = cfg['is_unsupervised']

        if self.type == "FC":
            self.encoder = Encoder(self.encoded_size)
            self.decoder = Decoder(self.inp_shape, self.activation)
        elif self.type == "CNN":
            self.encoder = CNNEncoder(self.encoded_size)
            self.decoder = CNNDecoder(self.inp_shape, self.activation)
        elif self.type == "VGG":
            self.encoder = VGGEncoder(self.encoded_size)
            self.decoder = VGGDecoder(self.inp_shape, self.activation)
        elif self.type == "ResNet":
            self.encoder = ResNetEncoder(encoded_size = self.encoded_size)
            self.decoder = ResNetDecoder(encoded_size = self.encoded_size, activation = self.activation)
        elif self.type == "ResNetAE":
            self.single_model = ResNetAE(input_shape = self.inp_shape, n_ResidualBlock=16, n_levels=5, bottleneck_dim=128, z_dim=128)
        else:
            raise NotImplemented("Invalid type {}".format(self.type))

        self.c_mu = tf.Variable(tf.ones([self.num_clusters, self.encoded_size]), name="mu")
        self.log_c_sigma = tf.Variable(tf.ones([self.num_clusters, self.encoded_size]), name="sigma")
        self.prior = tf.constant(tf.ones([self.num_clusters]) * (
                1 / self.num_clusters))  # tf.Variable(tf.ones([self.num_clusters]), name="prior")

    def call(self, inputs, training=True):
        inputs, W = inputs
        z_mu, log_z_sigma = self.encoder(inputs)

        # possible improvements for nan loss
        #log_z_sigma = tfk.activations.tanh(log_z_sigma) # clipping excessive variance values
        #log_z_sigma = tf.zeros_like(log_z_sigma, dtype=tf.float32) # unit variance
        
        print('---')
        #print(f'self.c_mu: {self.c_mu}')
        #print(f'self.log_c_sigma: {self.log_c_sigma}')
        print(f'log_z_sigma: {log_z_sigma}')

        
        z=tfd.MultivariateNormalDiag(loc=z_mu, scale_diag=tf.math.sqrt(tf.math.exp(log_z_sigma)))
        z_sample = z.sample()
        
        log_z_sigma_tile = tf.expand_dims(log_z_sigma, axis=-2)
        c = tf.constant([1, self.num_clusters, 1], tf.int32)
        log_z_sigma_tile = tf.tile(log_z_sigma_tile, c)

        z_mu_tile = tf.expand_dims(z_mu, axis=-2)
        c = tf.constant([1, self.num_clusters, 1], tf.int32)
        z_mu_tile = tf.tile(z_mu_tile, c)

        c_sigma = tf.math.exp(self.log_c_sigma)
        p_z_c = tf.stack([tf.math.log(
            tfd.MultivariateNormalDiag(loc=self.c_mu[i, :], scale_diag=tf.math.sqrt(c_sigma[i, :])).prob(
                z_sample) + 1e-30) for i in range(self.num_clusters)], axis=-1)

        prior = self.prior

        p_c_z = tf.math.log(prior + tf.keras.backend.epsilon()) + p_z_c

        norm_s = tf.math.log(1e-30 + tf.math.reduce_sum(tf.math.exp(p_c_z), axis=-1, keepdims=True))
        c = tf.constant([1, self.num_clusters], tf.int32)
        norm = tf.tile(norm_s, c)
        p_c_z = tf.math.exp(p_c_z - norm)

        loss_1a = tf.math.log(c_sigma + tf.keras.backend.epsilon())

        loss_1b = tf.math.exp(log_z_sigma_tile) / (c_sigma + tf.keras.backend.epsilon())

        loss_1c = tf.math.square(z_mu_tile - self.c_mu) / (c_sigma + tf.keras.backend.epsilon())

        loss_1d = self.encoded_size * tf.math.log(tf.keras.backend.constant(2 * np.pi))

        loss_1a = tf.multiply(p_c_z, tf.math.reduce_sum(loss_1a, axis=-1))
        loss_1b = tf.multiply(p_c_z, tf.math.reduce_sum(loss_1b, axis=-1))
        loss_1c = tf.multiply(p_c_z, tf.math.reduce_sum(loss_1c, axis=-1))
        loss_1d = tf.multiply(p_c_z, loss_1d)

        loss_1a = 1 / 2 * tf.reduce_sum(loss_1a, axis=-1)
        loss_1b = 1 / 2 * tf.reduce_sum(loss_1b, axis=-1)
        loss_1c = 1 / 2 * tf.reduce_sum(loss_1c, axis=-1)
        loss_1d = 1 / 2 * tf.reduce_sum(loss_1d, axis=-1)

        loss_2a = - tf.math.reduce_sum(tf.math.xlogy(p_c_z, prior), axis=-1)

        if training:
            ind1, ind2, data = W
            ind1 = tf.reshape(ind1, [-1])
            ind2 = tf.reshape(ind2, [-1])
            data = tf.reshape(data, [-1])
            ind = tf.stack([ind1, ind2], axis=0)
            ind = tf.transpose(ind)
            ind = tf.dtypes.cast(ind, tf.int64)
            W_sparse = tf.SparseTensor(indices=ind, values=data, dense_shape=[len(inputs), len(inputs)])
            W_sparse = tf.sparse.expand_dims(W_sparse, axis=-1)
            W_tile = tf.sparse.concat(-1, [W_sparse] * self.num_clusters)
            mul = W_tile.__mul__(p_c_z)
            sum_j = tf.sparse.reduce_sum(mul, axis=-2)
            loss_2a_constrain = - tf.math.reduce_sum(tf.multiply(p_c_z, sum_j), axis=-1)

            if not self.is_unsupervised:
                self.add_loss(tf.math.reduce_mean(loss_2a_constrain))
                self.add_metric(loss_2a_constrain, name='loss_2a_c', aggregation="mean")

        loss_2b = tf.math.reduce_sum(tf.math.xlogy(p_c_z, p_c_z), axis=-1)

        loss_3 = - 1 / 2 * tf.reduce_sum(log_z_sigma + 1, axis=-1)

        #threshold = 1e-2
        #loss_1b = loss_1b * threshold
        #loss_3 = loss_3 * threshold

        self.add_loss(tf.math.reduce_mean(loss_1a))
        self.add_loss(tf.math.reduce_mean(loss_1b))
        self.add_loss(tf.math.reduce_mean(loss_1c))
        self.add_loss(tf.math.reduce_mean(loss_1d))
        self.add_loss(tf.math.reduce_mean(loss_2a))
        self.add_loss(tf.math.reduce_mean(loss_2b))
        self.add_loss(tf.math.reduce_mean(loss_3))
        self.add_metric(loss_1a, name='loss_1a', aggregation="mean")
        self.add_metric(loss_1b, name='loss_1b', aggregation="mean")
        self.add_metric(loss_1c, name='loss_1c', aggregation="mean")
        self.add_metric(loss_1d, name='loss_1d', aggregation="mean")
        self.add_metric(loss_2a, name='loss_2a', aggregation="mean")
        self.add_metric(loss_2b, name='loss_2b', aggregation="mean")
        self.add_metric(loss_3, name='loss_3', aggregation="mean")

        # below are added for nan loss investigation
        #self.add_metric(log_z_sigma, name='log_z_sigma', aggregation="mean")
        
        dec = self.decoder(z_sample)

        # same as {'output_1':dec, 'output_2': z_sample, 'output_3': p_z_c, 'output_4': p_c_z}
        return dec, z_sample, p_z_c, p_c_z



# BELOW ARE ADDED FOR CONTRASTIVE PRETRAINING

# only used in pretraining - not final product
class VGGEncoderAE(layers.Layer):
    def __init__(self, encoded_size):
        super(VGGEncoderAE, self).__init__(name='VGGEncoderAE')
        self.layers = [VGGConvBlockAUX(32, 1), VGGConvBlockAUX(64, 2), VGGConvBlock(128, 3),VGGConvBlockAUX(128, 4),
                      VGGConvBlock(256, 5),VGGConvBlockAUX(256, 6),VGGConvBlockAUX(128, 7)]
        self.mu = tfkl.Dense(encoded_size, activation=None)

    def call(self, inputs):
        out = inputs

        # Iterate through blocks
        for block in self.layers:
            out = block(out)

        out_flat = tfkl.Flatten()(out)
        mu = self.mu(out_flat)

        return mu


# only used in pretraining - not final product
class VGGDecoderAE(layers.Layer):
    def __init__(self, input_shape, activation):
        super(VGGDecoderAE, self).__init__(name='VGGDecoderAE')

        #input_tuple = make_tuple(input_shape)
        input_tuple=input_shape
        if input_tuple == (64, 64, 1):
            target_shape = (13, 13, 64)
        elif input_tuple == (64, 64, 3):
            target_shape = (13, 13, 64)
        elif input_tuple == (32, 32, 3):  # compatible with cifar10
            target_shape = (5, 5, 128)
        elif input_tuple == (96, 96, 3): # compatible with stl10
            target_shape = (21, 21, 256)

        self.activation = activation
        self.dense = tfkl.Dense(target_shape[0] * target_shape[1] * target_shape[2])
        self.reshape = tfkl.Reshape(target_shape=target_shape)
        self.layers = [VGGDeConvBlockAUX(128, 1),
                       VGGDeConvBlockAUX(256, 2),VGGDeConvBlock(256, 3),
                       VGGDeConvBlockAUX(128, 4),VGGDeConvBlock(128, 5),
                       VGGDeConvBlockAUX(64, 6),VGGDeConvBlockAUX(32, 7)]
        self.convT = tfkl.Conv2DTranspose(filters=input_tuple[2], kernel_size=3, padding='same')

    def call(self, inputs):
        out = self.dense(inputs)
        out = self.reshape(out)

        # Iterate through blocks
        for block in self.layers:
            out = block(out)

        # Last convolution
        out = self.convT(out)

        if self.activation == "sigmoid":
            out = tf.sigmoid(out)

        return out

# place below function under utils
def compute_pairwise_sim(a):

    normalized_a = tf.math.l2_normalize(a,1)

    return tf.matmul(normalized_a, normalized_a , transpose_b=True)

# place below function under utils
def compute_pairwise_loss(row):
    temperature = 0.1

    sim_row , index = row

    exp_row = tf.math.exp(sim_row/temperature)

    total_sim = tf.math.reduce_sum(exp_row)
    corrected_total_sim = total_sim - tf.gather(exp_row,index)

    l = exp_row / corrected_total_sim

    return -tf.math.log(l)


class ContrastiveAE(tf.keras.Model):
    def __init__(self, cfg, inp_shape):
        super(ContrastiveAE, self).__init__(name="ContrastiveAE")
        self.encoded_size = cfg['latent_dim']
        self.num_clusters = cfg['num_clusters']
        self.inp_shape = inp_shape
        self.activation = cfg['activation']
        self.type = cfg['type']

        if self.type == "FC": # not implemented yet
            self.encoder = Encoder(self.encoded_size)
            self.decoder = Decoder(self.inp_shape, self.activation)
        elif self.type == "VGG":
            self.encoder = VGGEncoderAE(self.encoded_size)
            #self.decoder = VGGDecoderAE(self.inp_shape, self.activation)
        else:
            raise NotImplemented("Invalid type {}".format(self.type))

        self.dense1 = tfkl.Dense(128, activation='relu')
        self.dense2 = tfkl.Dense(128, activation=None)

    def call(self, inputs):
        
        batch_size = int(len(inputs)/2)

        z_mu = self.encoder(inputs)

        z_mu = self.dense2(self.dense1(z_mu))

        # compute contrastive batch loss
        sim_matrix = compute_pairwise_sim(z_mu)

        index_vector = tf.range(len(sim_matrix))
        loss_matrix = tf.vectorized_map(compute_pairwise_loss,(sim_matrix, index_vector))

        cont_loss = 0.0
        for i in range(batch_size):
            cont_loss += (loss_matrix[i,i+batch_size] + loss_matrix[i+batch_size,i])
        cont_loss /= float(2*batch_size)
        
        #cont_loss = cont_loss * 10.0
        self.add_loss(cont_loss)
        self.add_metric(cont_loss, name='cont_loss')

        #dec = self.decoder(z_mu)

        # same as {'output_1':dec}
        return