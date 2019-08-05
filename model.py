import os
from datetime import datetime
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, AveragePooling2D, Conv2DTranspose, Add, \
    Cropping2D, ZeroPadding2D, Activation, Concatenate, Deconv2D,BatchNormalization, LeakyReLU
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.initializers import RandomNormal, Constant
from keras import losses
from keras import metrics
from keras import callbacks
import keras as k
import glob
import math
import matplotlib.pyplot as plt
import cv2
# from tensorlayer.layers import *
from keras.layers import merge, Dropout, concatenate, add


def ofuse_pixel_error(y_true, y_pred):
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='predictions')
    error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(error, name='pixel_error')


def upsample_filt(shape):
    factor = (shape[0] + 1) // 2
    if shape[0] % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:shape[0], :shape[0]]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    return filt.reshape(shape)


class myUnet(object):
    def __init__(self, img_rows = 512, img_cols = 512, weight_filepath=None):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model = self.MSCN()
        self.current_epoch = 0
        self.weight_filepath = weight_filepath

    def CBRR_block(self, kn, ks, inputs):
        # conv_inputs =  Conv2D(kn, ks, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(kn, ks,  padding='same', kernel_initializer='he_normal')(inputs)
        # conv1_bn = BatchNormalization()(conv1)
        conv1_relu = LeakyReLU(alpha=0)(conv1)
        # merge = concatenate([inputs, conv_relu], axis=3)
        conv2 = Conv2D(kn, ks, padding='same', kernel_initializer='he_normal')(conv1_relu)
        # conv2_bn = BatchNormalization()(conv2)
        conv2_relu = LeakyReLU(alpha=0)(conv2)
        conv3 = Conv2D(kn, ks, padding='same', kernel_initializer='he_normal')(conv2_relu)
        # conv3_bn = BatchNormalization()(conv3)
        conv3_relu = LeakyReLU(alpha=0)(conv3)
        merge = Add()([inputs, conv3_relu])
        # merge = concatenate([inputs, conv3_relu], axis=3)
        return merge

    # def CBRR_block(self, kn, ks, inputs):
    #     # conv_inputs =  Conv2D(kn, ks, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    #     conv1 = Conv2D(kn, ks,  padding='same', kernel_initializer='he_normal')(inputs)
    #     conv1_bn = BatchNormalization()(conv1)
    #     conv1_relu = LeakyReLU(alpha=0)(conv1_bn)
    #     merge = concatenate([inputs, conv1_relu], axis=3)
    #     # conv2 = Conv2D(kn, ks, padding='same', kernel_initializer='he_normal')(conv1_relu)
    #     # conv2_bn = BatchNormalization()(conv2)
    #     # conv2_relu = LeakyReLU(alpha=0)(conv2_bn)
    #     # conv3 = Conv2D(kn, ks, padding='same', kernel_initializer='he_normal')(conv2_relu)
    #     # conv3_bn = BatchNormalization()(conv3)
    #     # conv3_relu = LeakyReLU(alpha=0)(conv3_bn)
    #     # merge = Add()([conv_inputs, conv1_relu])
    #     # merge = concatenate([inputs, conv3_relu], axis=3)
    #     return merge

    def sConv_block(self, kn, ks, inputs):
        conv = Conv2D(kn, ks, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)  # 256

        return pool

    def MSCN(self):
        inputs = Input((self.img_rows, self.img_cols, 3))

        conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=0)(conv1)
        conv2 = self.CBRR_block(32, 3, conv1)
        conv3 = self.sConv_block(32, 3, conv2)  # 256
        conv4 = self.CBRR_block(32, 3, conv3)
        conv5  = self.sConv_block(32, 3, conv4)  # 128
        conv6 = self.CBRR_block(32, 3, conv5)
        conv7 = self.sConv_block(32, 3, conv6)  # 64
        conv8 = self.CBRR_block(32, 3, conv7)
        conv9 = self.sConv_block(32, 3, conv8)  # 32
        conv10 = self.CBRR_block(32, 3, conv9)
        conv11 = self.sConv_block(32, 3, conv10)  # 16

        conv12 = Deconv2D(32, 2, strides=(2, 2), padding='same')(conv11) # 32
        conv13 = self.CBRR_block(32, 2, conv12)
        merge13 = concatenate([conv13, conv10], axis=3)

        conv14 = Deconv2D(32, 2, strides=(2, 2), padding='same')(merge13) # 64
        conv15 = self.CBRR_block(32, 2, conv14)
        merge15 = concatenate([conv15, conv8], axis=3)

        conv16 = Deconv2D(32, 2, strides=(2, 2), padding='same')(merge15)  # 128
        conv17 = self.CBRR_block(32, 2, conv16)
        merge17 = concatenate([conv17, conv6], axis=3)

        conv18 = Deconv2D(32, 2, strides=(2, 2), padding='same')(merge17)  # 256
        conv19 = self.CBRR_block(32, 2, conv18)
        merge19 = concatenate([conv19, conv4], axis=3)

        conv20 = Deconv2D(32, 2, strides=(2, 2), padding='same')(merge19)  # 512
        conv21 = self.CBRR_block(32, 2, conv20)
        merge21 = concatenate([conv21, conv2], axis=3)

        b1_conv = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(conv11)
        b1_bn = BatchNormalization()(b1_conv)
        b1_relu = LeakyReLU(alpha=0)(b1_bn)
        b1 = UpSampling2D(size=(32, 32), data_format=None)(b1_relu)

        b2_conv = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(merge13)
        b2_bn = BatchNormalization()(b2_conv)
        b2_relu = LeakyReLU(alpha=0)(b2_bn)
        b2 = UpSampling2D(size=(16, 16), data_format=None)(b2_relu)

        b3_conv = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(merge15)
        b3_bn = BatchNormalization()(b3_conv)
        b3_relu = LeakyReLU(alpha=0)(b3_bn)
        b3 = UpSampling2D(size=(8, 8), data_format=None)(b3_relu)

        b4_conv = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(merge17)
        b4_bn = BatchNormalization()(b4_conv)
        b4_relu = LeakyReLU(alpha=0)(b4_bn)
        b4 = UpSampling2D(size=(4, 4), data_format=None)(b4_relu)

        b5_conv = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(merge19)
        b5_bn = BatchNormalization()(b5_conv)
        b5_relu = LeakyReLU(alpha=0)(b5_bn)
        b5 = UpSampling2D(size=(2, 2), data_format=None)(b5_relu)

        fuse = concatenate([b1, b2, b3, b4, b5, merge21], axis=3)

        mask = Conv2D(3, 1, activation='softmax', padding='same')(fuse)

        model = Model(inputs=[inputs], outputs=[mask])

        model.compile(optimizer=Adam(lr=1e-5),
                      loss="categorical_crossentropy",
                      metrics=['accuracy'])
        return model


    def fit(self, generator, epochs=10, plot_callback=None, *args, **kwargs):
        # Loop over epochs
        for _ in range(epochs):

            self.model.fit_generator(
                generator,
                epochs=self.current_epoch + 1,
                initial_epoch=self.current_epoch,
                *args, **kwargs
            )

            # Update epoch
            self.current_epoch += 1

            # After each epoch predict on test images & show them
            if plot_callback:
                plot_callback(self.model)

            # Save logfile
            if self.weight_filepath:
                self.save()

    def predict(self, sample):
        return self.model.predict(sample)

    def summary(self):
        print(self.model.summary())

    def save(self):
        self.model.save_weights(self.current_weightfile())

    def load(self, filepath):
        self.model = self.MSCN()

        epoch = int(os.path.basename(filepath).split("_")[0])
        assert epoch > 0, "Could not parse weight file. Should start with 'X_', with X being the epoch"
        self.current_epoch = epoch
        self.model.load_weights(filepath)

    def current_weightfile(self):
        assert self.weight_filepath != None, 'Must specify location of logs'
        return self.weight_filepath + "{}_weights_{}.h5".format(self.current_epoch, self.current_timestamp())

    @staticmethod
    def current_timestamp():
        return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')



