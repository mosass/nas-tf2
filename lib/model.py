import tensorflow as tf
from lib.spec import Spec
from lib.cell import Cell
import lib.base_ops as ops
import numpy as np


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
  def __init__(self, spec: Spec, inputs, name=""):
    super(NModel, self).__init__()
    self.name = name
    self.spec = spec
    self.inputs = inputs
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
    tensors = self.dense(tensors)
    return tensors
