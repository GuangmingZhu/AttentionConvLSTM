import numpy as np
import tensorflow as tf
K=tf.contrib.keras.backend
from tensorflow.contrib.keras.python.keras.callbacks import Callback



class LearningRateScheduler(Callback):
  """Learning rate scheduler.
  Arguments:
      schedule: a function that takes an batch index as input
          (integer, indexed from 0) and returns a new
          learning rate as output (float).
  """

  def __init__(self, schedule, steps):
    super(LearningRateScheduler, self).__init__()
    self.schedule = schedule
    self.epoch = 0
    self.steps_per_epoch = steps

  def on_batch_begin(self, batch, logs=None):
    if not hasattr(self.model.optimizer, 'lr'):
      raise ValueError('Optimizer must have a "lr" attribute.')
    global_step = self.epoch*self.steps_per_epoch+batch
    lr = self.schedule(global_step)
    if not isinstance(lr, (float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function '
                       'should be float.')
    K.set_value(self.model.optimizer.lr, lr)

  def on_epoch_begin(self, epoch, logs=None):
    self.epoch = epoch
