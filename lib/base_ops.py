import tensorflow as tf

class Conv3x3BnRelu(object):
  def build(self, filters, inputs):
    conv = tf.keras.layers.Conv2D(
      filters=filters,
      kernel_size=(3,3),
      strides=(1,1),
      padding='same',
      use_bias=False,
      kernel_initializer='VarianceScaling'
    )
    bn = tf.keras.layers.BatchNormalization(
      axis=3,
      momentum=0.997,
      epsilon=1e-5
    )
    relu = tf.keras.layers.ReLU()

    x = conv(inputs)
    x1 = bn(x)
    return relu(x1)

class Conv1x1BnRelu(object):
  def build(self, filters, inputs):
    conv = tf.keras.layers.Conv2D(
      filters=filters,
      kernel_size=(1,1),
      strides=(1,1),
      padding='same',
      use_bias=False,
      kernel_initializer='VarianceScaling'
    )
    bn = tf.keras.layers.BatchNormalization(
      axis=3,
      momentum=0.997,
      epsilon=1e-5
    )
    relu = tf.keras.layers.ReLU()

    x = conv(inputs)
    x1 = bn(x)
    return relu(x1)

class MaxPool3x3(object):
  def build(self, filters, inputs):
    maxpool = tf.keras.layers.MaxPool2D(
      pool_size=(3, 3),
      strides=(1, 1),
      padding='same'
    )
    return maxpool(inputs)

class Projection(Conv1x1BnRelu):
  def __init__(self):
    super(Projection, self).__init__()

OP_MAP = {
  'conv3x3-bn-relu': Conv3x3BnRelu,
  'conv1x1-bn-relu': Conv1x1BnRelu,
  'maxpool3x3': MaxPool3x3,
  'projection': Projection
}