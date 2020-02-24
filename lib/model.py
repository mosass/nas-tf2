import tensorflow as tf
import lib.base_ops as ops
import lib.spec as spec
import lib.cell as cell
import numpy as np


class Stack(object):
  def __init__(self, spec: spec.Spec, channels, inputs):
    super().__init__()