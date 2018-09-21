import io
import sys
import numpy as np
import tensorflow as tf
keras=tf.contrib.keras
l2=keras.regularizers.l2

ATTENTIONX = 0
ATTENTIONI = 1
ATTENTIONO = 2

def res3d(inputs, weight_decay):
  # Res3D Block 1
  conv3d_1 = keras.layers.Conv3D(64, (3,7,7), strides=(1,2,2), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_1')(inputs)
  conv3d_1 = keras.layers.BatchNormalization(name='BatchNorm_1_0')(conv3d_1)
  conv3d_1 = keras.layers.Activation('relu', name='ReLU_1')(conv3d_1)
 
  # Res3D Block 2
  conv3d_2a_1 = keras.layers.Conv3D(64, (1,1,1), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_2a_1')(conv3d_1)
  conv3d_2a_1 = keras.layers.BatchNormalization(name='BatchNorm_2a_1')(conv3d_2a_1)
  conv3d_2a_a = keras.layers.Conv3D(64, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_2a_a')(conv3d_1)
  conv3d_2a_a = keras.layers.BatchNormalization(name='BatchNorm_2a_a')(conv3d_2a_a)
  conv3d_2a_a = keras.layers.Activation('relu', name='ReLU_2a_a')(conv3d_2a_a)
  conv3d_2a_b = keras.layers.Conv3D(64, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_2a_b')(conv3d_2a_a)
  conv3d_2a_b = keras.layers.BatchNormalization(name='BatchNorm_2a_b')(conv3d_2a_b)
  conv3d_2a = keras.layers.Add(name='Add_2a')([conv3d_2a_1, conv3d_2a_b])
  conv3d_2a = keras.layers.Activation('relu', name='ReLU_2a')(conv3d_2a)

  conv3d_2b_a = keras.layers.Conv3D(64, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_2b_a')(conv3d_2a)
  conv3d_2b_a = keras.layers.BatchNormalization(name='BatchNorm_2b_a')(conv3d_2b_a)
  conv3d_2b_a = keras.layers.Activation('relu', name='ReLU_2b_a')(conv3d_2b_a)
  conv3d_2b_b = keras.layers.Conv3D(64, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_2b_b')(conv3d_2b_a)
  conv3d_2b_b = keras.layers.BatchNormalization(name='BatchNorm_2b_b')(conv3d_2b_b)
  conv3d_2b = keras.layers.Add(name='Add_2b')([conv3d_2a, conv3d_2b_b])
  conv3d_2b = keras.layers.Activation('relu', name='ReLU_2b')(conv3d_2b)

  # Res3D Block 3
  conv3d_3a_1 = keras.layers.Conv3D(128, (1,1,1), strides=(2,2,2), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_3a_1')(conv3d_2b)
  conv3d_3a_1 = keras.layers.BatchNormalization(name='BatchNorm_3a_1')(conv3d_3a_1)
  conv3d_3a_a = keras.layers.Conv3D(128, (3,3,3), strides=(2,2,2), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_3a_a')(conv3d_2b)
  conv3d_3a_a = keras.layers.BatchNormalization(name='BatchNorm_3a_a')(conv3d_3a_a)
  conv3d_3a_a = keras.layers.Activation('relu', name='ReLU_3a_a')(conv3d_3a_a)
  conv3d_3a_b = keras.layers.Conv3D(128, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_3a_b')(conv3d_3a_a)
  conv3d_3a_b = keras.layers.BatchNormalization(name='BatchNorm_3a_b')(conv3d_3a_b)
  conv3d_3a = keras.layers.Add(name='Add_3a')([conv3d_3a_1, conv3d_3a_b])
  conv3d_3a = keras.layers.Activation('relu', name='ReLU_3a')(conv3d_3a)

  conv3d_3b_a = keras.layers.Conv3D(128, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_3b_a')(conv3d_3a)
  conv3d_3b_a = keras.layers.BatchNormalization(name='BatchNorm_3b_a')(conv3d_3b_a)
  conv3d_3b_a = keras.layers.Activation('relu', name='ReLU_3b_a')(conv3d_3b_a)
  conv3d_3b_b = keras.layers.Conv3D(128, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_3b_b')(conv3d_3b_a)
  conv3d_3b_b = keras.layers.BatchNormalization(name='BatchNorm_3b_b')(conv3d_3b_b)
  conv3d_3b = keras.layers.Add(name='Add_3b')([conv3d_3a, conv3d_3b_b])
  conv3d_3b = keras.layers.Activation('relu', name='ReLU_3b')(conv3d_3b)

  # Res3D Block 4
  conv3d_4a_1 = keras.layers.Conv3D(256, (1,1,1), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_4a_1')(conv3d_3b)
  conv3d_4a_1 = keras.layers.BatchNormalization(name='BatchNorm_4a_1')(conv3d_4a_1)
  conv3d_4a_a = keras.layers.Conv3D(256, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_4a_a')(conv3d_3b)
  conv3d_4a_a = keras.layers.BatchNormalization(name='BatchNorm_4a_a')(conv3d_4a_a)
  conv3d_4a_a = keras.layers.Activation('relu', name='ReLU_4a_a')(conv3d_4a_a)
  conv3d_4a_b = keras.layers.Conv3D(256, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_4a_b')(conv3d_4a_a)
  conv3d_4a_b = keras.layers.BatchNormalization(name='BatchNorm_4a_b')(conv3d_4a_b)
  conv3d_4a = keras.layers.Add(name='Add_4a')([conv3d_4a_1, conv3d_4a_b])
  conv3d_4a = keras.layers.Activation('relu', name='ReLU_4a')(conv3d_4a)

  conv3d_4b_a = keras.layers.Conv3D(256, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_4b_a')(conv3d_4a)
  conv3d_4b_a = keras.layers.BatchNormalization(name='BatchNorm_4b_a')(conv3d_4b_a)
  conv3d_4b_a = keras.layers.Activation('relu', name='ReLU_4b_a')(conv3d_4b_a)
  conv3d_4b_b = keras.layers.Conv3D(256, (3,3,3), strides=(1,1,1), padding='same', 
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False, 
                    name='Conv3D_4b_b')(conv3d_4b_a)
  conv3d_4b_b = keras.layers.BatchNormalization(name='BatchNorm_4b_b')(conv3d_4b_b)
  conv3d_4b = keras.layers.Add(name='Add_4b')([conv3d_4a, conv3d_4b_b])
  conv3d_4b = keras.layers.Activation('relu', name='ReLU_4b')(conv3d_4b)

  return conv3d_4b

def relu6(x):
  return keras.activations.relu(x,max_value=6)

def mobilenet(inputs, weight_decay):
  conv2d_1a = keras.layers.SeparableConv2D(256, (3,3), strides=(1,1), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_1a')(inputs)
  conv2d_1a = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_1a')(conv2d_1a)
  conv2d_1a = keras.layers.Activation(relu6, name='ReLU_Conv2d_1a')(conv2d_1a)

  conv2d_1b = keras.layers.SeparableConv2D(256, (3,3), strides=(2,2), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_1b')(conv2d_1a)
  conv2d_1b = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_1b')(conv2d_1b)
  conv2d_1b = keras.layers.Activation(relu6, name='ReLU_Conv2d_1b')(conv2d_1b)
  
  conv2d_2a = keras.layers.SeparableConv2D(512, (3,3), strides=(1,1), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_2a')(conv2d_1b)
  conv2d_2a = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_2a')(conv2d_2a)
  conv2d_2a = keras.layers.Activation(relu6, name='ReLU_Conv2d_2a')(conv2d_2a)

  conv2d_2b = keras.layers.SeparableConv2D(512, (3,3), strides=(1,1), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_2b')(conv2d_2a)
  conv2d_2b = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_2b')(conv2d_2b)
  conv2d_2b = keras.layers.Activation(relu6, name='ReLU_Conv2d_2b')(conv2d_2b)

  conv2d_2c = keras.layers.SeparableConv2D(512, (3,3), strides=(1,1), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_2c')(conv2d_2b)
  conv2d_2c = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_2c')(conv2d_2c)
  conv2d_2c = keras.layers.Activation(relu6, name='ReLU_Conv2d_2c')(conv2d_2c)

  conv2d_2d = keras.layers.SeparableConv2D(512, (3,3), strides=(1,1), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_2d')(conv2d_2c)
  conv2d_2d = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_2d')(conv2d_2d)
  conv2d_2d = keras.layers.Activation(relu6, name='ReLU_Conv2d_2d')(conv2d_2d)

  conv2d_2e = keras.layers.SeparableConv2D(512, (3,3), strides=(1,1), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_2e')(conv2d_2d)
  conv2d_2e = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_2e')(conv2d_2e)
  conv2d_2e = keras.layers.Activation(relu6, name='ReLU_Conv2d_2e')(conv2d_2e)

  conv2d_3a = keras.layers.SeparableConv2D(1024, (3,3), strides=(2,2), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_3a')(conv2d_2e)
  conv2d_3a = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_3a')(conv2d_3a)
  conv2d_3a = keras.layers.Activation(relu6, name='ReLU_Conv2d_3a')(conv2d_3a)

  conv2d_3b = keras.layers.SeparableConv2D(1024, (3,3), strides=(2,2), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_3b')(conv2d_3a)
  conv2d_3b = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_3b')(conv2d_3b)
  conv2d_3b = keras.layers.Activation(relu6, name='ReLU_Conv2d_3b')(conv2d_3b)

  return conv2d_3b

def res3d_aclstm_mobilenet(inputs, seq_len, weight_decay, atten_type):
  # Res3D Block
  res3d_featmap = res3d(inputs, weight_decay)

  # Attention ConvLSTM2D Block
  if atten_type==ATTENTIONX:
    clstm2d_1 = keras.layers.AttenXConvLSTM2D(256, (3,3), strides=(1,1), padding='same',
                      kernel_initializer='he_normal', recurrent_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
                      return_sequences=True, name='axclstm2d_1')(res3d_featmap)
  elif atten_type==ATTENTIONI:
    clstm2d_1 = keras.layers.AttenIConvLSTM2D(256, (3,3), strides=(1,1), padding='same',
                      kernel_initializer='he_normal', recurrent_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
                      return_sequences=True, name='aiclstm2d_1')(res3d_featmap)
  elif atten_type==ATTENTIONO:
    clstm2d_1 = keras.layers.AttenOConvLSTM2D(256, (3,3), strides=(1,1), padding='same',
                      kernel_initializer='he_normal', recurrent_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
                      return_sequences=True, name='aoclstm2d_1')(res3d_featmap)

  clstm2d_2 = keras.layers.GatedConvLSTM2D(256, (3,3), strides=(1,1), padding='same',
                    kernel_initializer='he_normal', recurrent_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
                    return_sequences=True, name='gclstm2d_2')(clstm2d_1)
  featmap_2d = keras.layers.Reshape((28,28,256), name='clstm_reshape')(clstm2d_2)

  # MobileNet
  features = mobilenet(featmap_2d, weight_decay)
  features = keras.layers.Reshape((seq_len/2,4,4,1024), name='feature_reshape')(features)
  gpooling = keras.layers.AveragePooling3D(pool_size=(seq_len/2,4,4), strides=(seq_len/2,4,4), 
                    padding='valid', name='Average_Pooling')(features)
  
  return gpooling
