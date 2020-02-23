import tensorflow as tf

class Conv3x3BnRelu(tf.keras.Model):

  def __init__(self, filters):
    super(Conv3x3BnRelu, self).__init__()
    self.conv = tf.keras.layers.Conv2D(
      filters=filters,
      kernel_size=(3,3),
      strides=(1,1),
      padding='same',
      use_bias=False,
      kernel_initializer='variance_scaling'
    )
    self.bn = tf.keras.layers.BatchNormalization(
      axis=3,
      momentum=0.997,
      epsilon=1e-5
    )
    self.relu = tf.keras.layers.ReLU()

  def call(self, inputs):
    x = self.conv(inputs)
    x1 = self.bn(x)
    return self.relu(x1)

class Conv1x1BnRelu(tf.keras.Model):

  def __init__(self, filters):
    super(Conv1x1BnRelu, self).__init__()
    self.conv = tf.keras.layers.Conv2D(
      filters=filters,
      kernel_size=(1,1),
      strides=(1,1),
      padding='same',
      use_bias=False,
      kernel_initializer='variance_scaling'
    )
    self.bn = tf.keras.layers.BatchNormalization(
      axis=3,
      momentum=0.997,
      epsilon=1e-5
    )
    self.relu = tf.keras.layers.ReLU()

  def call(self, inputs):
    x = self.conv(inputs)
    x1 = self.bn(x)
    return self.relu(x1)

class MaxPool3x3(tf.keras.Model):

  def __init__(self, filters):
    super(MaxPool3x3, self).__init__()
    self.maxpool = tf.keras.layers.MaxPool2D(
      pool_size=(3, 3),
      strides=(1, 1),
      padding='same'
    )

  def call(self, inputs):
    return self.maxpool(inputs)


OP_MAP = {
    'conv3x3-bn-relu': Conv3x3BnRelu,
    'conv1x1-bn-relu': Conv1x1BnRelu,
    'maxpool3x3': MaxPool3x3,
}
