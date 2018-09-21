import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import io
import sys
sys.path.append("./networks")
import numpy as np
import tensorflow as tf
keras=tf.contrib.keras
l2=keras.regularizers.l2
K=tf.contrib.keras.backend
import inputs as data
from res3d_aclstm_mobilenet import res3d_aclstm_mobilenet
from datagen import isoTestImageGenerator
from datetime import datetime

# Used ConvLSTM Type
ATTENTIONX = 0
ATTENTIONI = 1
ATTENTIONO = 2

# Modality
RGB = 0
#Depth = 1 # Not Trained
#Flow = 2  # Not Trained

cfg_type = ATTENTIONX
cfg_modality = RGB

seq_len = 32
batch_size = 8
num_classes = 249
testing_datalist = './dataset_splits/IsoGD/valid_rgb_list.txt'
weight_decay = 0.00005
model_prefix = './models/'
model_prefix = '/raid/gmzhu/tensorflow/ConvLSTMForGR/models/'
  
inputs = keras.layers.Input(shape=(seq_len, 112, 112, 3),
                            batch_shape=(batch_size, seq_len, 112, 112, 3))
feature = res3d_aclstm_mobilenet(inputs, seq_len, weight_decay, cfg_type)
flatten = keras.layers.Flatten(name='Flatten')(feature)
classes = keras.layers.Dense(num_classes, activation='linear', kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), name='Classes')(flatten)
outputs = keras.layers.Activation('softmax', name='Output')(classes)
model = keras.models.Model(inputs=inputs, outputs=outputs)
optimizer = keras.optimizers.SGD(lr=0.001, decay=0, momentum=0.9, nesterov=False)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

if cfg_type==ATTENTIONX:
  pretrained_model = '%s/isogr_rgb_attenxclstm_weights.h5'%model_prefix
elif cfg_type==ATTENTIONI:
  pretrained_model = '%s/isogr_rgb_atteniclstm_weights.h5'%model_prefix
elif cfg_type==ATTENTIONO:
  pretrained_model = '%s/isogr_rgb_attenoclstm_weights.h5'%model_prefix
print 'Loading pretrained model from %s' % pretrained_model
model.load_weights(pretrained_model, by_name=False)
for i in range(len(model.trainable_weights)):
  print model.trainable_weights[i]

_,test_labels = data.load_iso_video_list(testing_datalist)
test_steps = len(test_labels)/batch_size
print model.evaluate_generator(isoTestImageGenerator(testing_datalist, batch_size, seq_len, num_classes, cfg_modality),
                         steps=test_steps,
                         )
