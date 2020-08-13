import tensorflow as tf
from lib.spec import Spec
from lib.cell import Cell
import lib.base_ops as ops
import numpy as np
import time
import os

class Stack(object):
  def __init__(self, spec: Spec, inputs, name=""):
    super(Stack, self).__init__()
    self.name = name
    self.spec = spec
    self.inputs = inputs
    self.input_channels = inputs.get_shape()[3]

  def build(self, downsample=True):
    tensors = self.inputs
    channels = tensors.get_shape()[3]
    if downsample:
      tensors = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(tensors)
      channels = channels * 2
    
    for c in range(3):
      tensors = Cell(self.spec, channels, tensors, name=("%s-%d" % (self.name, c))).build()
    
    return tensors

class NModel(object):
  def __init__(self, spec: Spec, name="", checkpoint=""):
    super(NModel, self).__init__()

    self.checkpoint = checkpoint

    if len(self.checkpoint) > 0:
      self.checkpoint_path = checkpoint+"/cp-{epoch:04d}.ckpt"
      self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

      self.cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=self.checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=5)


    self.name = name
    self.spec = spec
    self.inputs = tf.keras.Input(shape=(32, 32, 3))
    self.stem = tf.keras.layers.Conv2D(
      filters=128,
      kernel_size=(3,3),
      strides=(1,1),
      padding='same',
      use_bias=False,
      kernel_initializer='VarianceScaling'
    )
    self.dense = tf.keras.layers.Dense(10)

  def build(self):
    tensors = self.stem(self.inputs)
    for s in range(3):
      tensors = Stack(self.spec, tensors, name=("%s-%d" % (self.name, s))).build(downsample=(s > 0))
    
    tensors = tf.reduce_mean(tensors, [1,2])
    self.outputs = self.dense(tensors)

    self.model = tf.keras.Model(self.inputs, self.outputs, name=self.name)

    return self.model
  
  def load_checkpoint(self, ephocs):
    latest = tf.train.latest_checkpoint(self.checkpoint_dir)
    if not ephocs:
      latest = self.checkpoint_path.format(epoch=ephocs)

    self.model.load_weights(latest)
  
  def evaluate(self):
    (train_validate_images, train_validate_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    train_images, validate_images = np.split(train_validate_images, [int(.8 * len(train_validate_images))])
    train_labels, validate_labels = np.split(train_validate_labels, [int(.8 * len(train_validate_labels))])

    train_images, validate_images, test_images = train_images / 255.0, validate_images / 255.0,test_images / 255.0

    optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=tf.keras.experimental.CosineDecay(0.2, 256, alpha=0.0),
            momentum=0.9,
            epsilon=1.0)
    # optimizer_sgd = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    self.model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    evaluate = self.model.evaluate(test_images, test_labels)
    return evaluate

  def train_and_evaluate(self, batch_size=256, ephocs=12):
    optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=tf.keras.experimental.CosineDecay(0.2, batch_size, alpha=0.0),
            momentum=0.9,
            epsilon=1.0)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    self.model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    (train_validate_images, train_validate_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    train_images, validate_images = np.split(train_validate_images, [int(.8 * len(train_validate_images))])
    train_labels, validate_labels = np.split(train_validate_labels, [int(.8 * len(train_validate_labels))])

    train_images, validate_images, test_images = train_images / 255.0, validate_images / 255.0,test_images / 255.0
    
    time_his = TimeHistory()
    
    cb = [time_his]
    if len(self.checkpoint) > 0:
      if not os.path.isdir(self.checkpoint):
        self.model.save_weights(self.checkpoint_path.format(epoch=0))
      else:
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        self.model.load_weights(latest)

      cb.append(self.cp_callback)

    history = self.model.fit(train_images, train_labels, epochs=ephocs, batch_size=batch_size,
                  callbacks=cb,
                  validation_data=(validate_images, validate_labels))
    evaluate = self.model.evaluate(test_images, test_labels)

    data = {
      "trainable_parameters": np.sum([np.prod(v.get_shape().as_list()) for v in self.model.trainable_variables]),
      "params": history.params,
      "history": history.history,
      "training_time": time_his.training_time,
      "train_accuracy": history.history["accuracy"][-1],
      "validation_accuracy": history.history["val_accuracy"][-1],
      "test_accuracy": evaluate[1],
    }

    return data

class TimeHistory(tf.keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.times = []
    self.start_time = time.time()

  def on_epoch_begin(self, batch, logs={}):
    self.epoch_time_start = time.time()

  def on_epoch_end(self, batch, logs={}):
    self.times.append(time.time() - self.epoch_time_start)

  def on_train_end(self, logs={}):
    self.training_time = time.time() - self.start_time
