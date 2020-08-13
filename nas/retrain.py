import argparse
import collections
import os
import random
from copy import deepcopy
import operator
import glob
import json
import sys

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)


import ConfigSpace
import numpy as np
from lib.spec import Spec
from lib.model import NModel
from nas.nasbase import NasBase
from nas.nasbase import Model
from datetime import datetime

cf = Spec.get_spec_from_json('{"edge_0": 0, "edge_1": 1, "edge_10": 1, "edge_11": 0, "edge_12": 1, "edge_13": 0, "edge_14": 1, "edge_15": 0, "edge_16": 0, "edge_17": 0, "edge_18": 0, "edge_19": 1, "edge_2": 0, "edge_20": 1, "edge_3": 1, "edge_4": 1, "edge_5": 0, "edge_6": 1, "edge_7": 0, "edge_8": 1, "edge_9": 0, "op_node_0": "conv1x1-bn-relu", "op_node_1": "conv1x1-bn-relu", "op_node_2": "maxpool3x3", "op_node_3": "maxpool3x3", "op_node_4": "maxpool3x3"}')
s = Spec(cf)
print(s.original_matrix)
print(s.original_ops)
# s.visualize()

nModel = NModel(spec=s, name="worst2", checkpoint='./chk/worst2')
model = nModel.build()
# model.summary()

data = nModel.train_and_evaluate(256, 8)


print(data)