

#-------------------------------------------------------------#
#   ResNet50/18
#-------------------------------------------------------------#
from __future__ import print_function

import keras.backend as K
import numpy as np
from keras import layers
from keras.applications.imagenet_utils import (decode_predictions,preprocess_input)
from keras.layers import (Activation,Reshape, AveragePooling2D, BatchNormalization,Concatenate,
                 Softmax, Conv2D, Conv2DTranspose, Dense, Dropout, Flatten,
                  Input, MaxPooling2D, ZeroPadding2D, Lambda, AveragePooling2D)
from keras.models import Model
from keras.preprocessing import image
from keras.regularizers import l2
from keras.utils.data_utils import get_file

def lidar_block(input_tensor, kernel_size, filters,N):

    filters1, filters2 = filters
    for i in range(N):
      x = Conv2D(filters1, kernel_size,padding='same', use_bias=False)(input_tensor)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)

      x = Conv2D(filters2, kernel_size,padding='same', use_bias=False)(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    return x

def identity_block(input_tensor, kernel_size, filters, stage, block):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(inputs):
    # 512x512x3
    x = ZeroPadding2D((3, 3))(inputs)
    # 256,256,64
    fs = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x[:,:,:,0:4])
    fs = BatchNormalization(name='bn_conv1')(fs)
    fs = Activation('relu')(fs)
    fd = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(tf.expand_dims(x[:,:,:,4],axis=-1))
    fd = BatchNormalization(name='bn_conv1')(fd)
    fd = Activation('relu')(fd)

    # 256,256,64 -> 128,128,64
    fs = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(fs)
    fd = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(fd)


    # 128,128,64 -> 128,128,256
    x = conv_block(fs, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    C2S = identity_block(x, 3, [64, 64, 256], stage=2, block='c')  


    # multi-modal feature fusion in stage2
    C2S = Conv2D(32, (1, 1), use_bias=False)(C2S)
    C2D = Conv2D(32, (1, 1), use_bias=False)(C2S)
    depare=Depthawaregate(C2S)
    C2S_=depare( [C2D, C2S] )
    specaware=Spectralawaregate( 3,1,C2D)
    C2D=specaware( [C2D, C2S])
    C2 = Conv2D(256, (1, 1), use_bias=False)(C2S_)
    C2D = Conv2D(128, (1, 1), use_bias=False)(C2S_)

    # 128,128,256 -> 64,64,512
    x = conv_block(C2, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    #x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    C3S = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    C3D=  lidar_block(C2D, 3, [128,128],2)

    # multi-modal feature fusion in stage3
    C3S = Conv2D(64, (1, 1), use_bias=False)(C3S)
    C3D = Conv2D(64, (1, 1), use_bias=False)(C3D)
    depare=Depthawaregate(C3S)
    C3S_=depare( [C3D, C3S] )
    specaware=Spectralawaregate( 3,1,C3D)
    C3D=specaware( [C3D, C3S])
    C3 = Conv2D(512, (1, 1), use_bias=False)(C3S_)

    # 64,64,512 -> 32,32,1024
    x = conv_block(C3, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    C4 = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # 32,32,1024 -> 16,16,2048
    x = conv_block(C4, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    C5 = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    return C5,C4,C3,C2
