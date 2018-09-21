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
from res3d_clstm_mobilenet import res3d_clstm_mobilenet
from callbacks import LearningRateScheduler 
from datagen import isoTrainImageGenerator, isoTestImageGenerator
from datagen import jesterTrainImageGenerator, jesterTestImageGenerator
from tensorflow.contrib.keras.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime

# Modality
RGB = 0
Depth = 1
Flow = 2

# Dataset
JESTER = 0
ISOGD = 1

cfg_modality = RGB
cfg_dataset = ISOGD

if cfg_modality==RGB:
  str_modality = 'rgb'
elif cfg_modality==Depth:
  str_modality = 'depth'
elif cfg_modality==Flow:
  str_modality = 'flow'

if cfg_dataset==JESTER:
  nb_epoch = 30
  init_epoch = 0
  seq_len = 16
  batch_size = 16
  num_classes = 27
  dataset_name = 'jester_%s'%str_modality
  training_datalist = './dataset_splits/Jester/train_%s_list.txt'%str_modality
  testing_datalist = './dataset_splits/Jester/valid_%s_list.txt'%str_modality
elif cfg_dataset==ISOGD:
  nb_epoch = 10
  init_epoch = 0
  seq_len = 32
  batch_size = 2
  num_classes = 249
  dataset_name = 'isogr_%s'%str_modality
  training_datalist = './dataset_splits/IsoGD/train_%s_list.txt'%str_modality
  testing_datalist = './dataset_splits/IsoGD/valid_%s_list.txt'%str_modality

weight_decay = 0.00005
model_prefix = './models/'
weights_file = '%s/%s_weights.{epoch:02d}-{val_loss:.2f}.h5'%(model_prefix,dataset_name)
  
_,train_labels = data.load_iso_video_list(training_datalist)
train_steps = len(train_labels)/batch_size
_,test_labels = data.load_iso_video_list(testing_datalist)
test_steps = len(test_labels)/batch_size
print 'nb_epoch: %d - seq_len: %d - batch_size: %d - weight_decay: %.6f' %(nb_epoch, seq_len, batch_size, weight_decay)

def lr_polynomial_decay(global_step):
  learning_rate = 0.001
  end_learning_rate=0.000001
  decay_steps=train_steps*nb_epoch
  power = 0.9
  p = float(global_step)/float(decay_steps)
  lr = (learning_rate - end_learning_rate)*np.power(1-p, power)+end_learning_rate
  return lr
  
inputs = keras.layers.Input(shape=(seq_len, 112, 112, 3), batch_shape=(batch_size, seq_len, 112, 112, 3))
feature = res3d_clstm_mobilenet(inputs, seq_len, weight_decay)
flatten = keras.layers.Flatten(name='Flatten')(feature)
classes = keras.layers.Dense(num_classes, activation='linear', kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), name='Classes')(flatten)
outputs = keras.layers.Activation('softmax', name='Output')(classes)
model = keras.models.Model(inputs=inputs, outputs=outputs)

#pretrained_model = '%s/isogr_%s_gatedclstm_weights.h5'%(model_prefix,str_modality)
#print 'Loading pretrained model from %s' % pretrained_model
#model.load_weights(pretrained_model, by_name=False)
for i in range(len(model.trainable_weights)):
  print model.trainable_weights[i]

optimizer = keras.optimizers.SGD(lr=0.001, decay=0, momentum=0.9, nesterov=False)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

lr_reducer = LearningRateScheduler(lr_polynomial_decay,train_steps) 
model_checkpoint = ModelCheckpoint(weights_file, monitor="val_acc", 
                   save_best_only=False,save_weights_only=True,mode='auto')
callbacks = [lr_reducer, model_checkpoint]

if cfg_dataset==JESTER:
  model.fit_generator(jesterTrainImageGenerator(training_datalist, batch_size, seq_len, num_classes, cfg_modality),
          steps_per_epoch=train_steps,
          epochs=nb_epoch,
          verbose=1,
          callbacks=callbacks,
          validation_data=jesterTestImageGenerator(testing_datalist, batch_size, seq_len, num_classes, cfg_modality),
          validation_steps=test_steps,
          initial_epoch=init_epoch,
          )
elif cfg_dataset==ISOGD:
  model.fit_generator(isoTrainImageGenerator(training_datalist, batch_size, seq_len, num_classes, cfg_modality),
          steps_per_epoch=train_steps,
          epochs=nb_epoch,
          verbose=1,
          callbacks=callbacks,
          validation_data=isoTestImageGenerator(testing_datalist, batch_size, seq_len, num_classes, cfg_modality),
          validation_steps=test_steps,
          initial_epoch=init_epoch,
          )
