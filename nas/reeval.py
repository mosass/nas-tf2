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

import matplotlib.pyplot as plt

# cf = Spec.get_spec_from_json('{"edge_0": 0, "edge_1": 0, "edge_10": 1, "edge_11": 1, "edge_12": 0, "edge_13": 1, "edge_14": 1, "edge_15": 0, "edge_16": 1, "edge_17": 0, "edge_18": 0, "edge_19": 1, "edge_2": 1, "edge_20": 1, "edge_3": 1, "edge_4": 1, "edge_5": 0, "edge_6": 0, "edge_7": 1, "edge_8": 0, "edge_9": 1, "op_node_0": "conv1x1-bn-relu", "op_node_1": "conv1x1-bn-relu", "op_node_2": "maxpool3x3", "op_node_3": "conv1x1-bn-relu", "op_node_4": "conv3x3-bn-relu"}')
# s = Spec(cf)
# print(s.original_matrix)
# print(s.original_ops)
# s.visualize()

# nModel = NModel(spec=s, name="best3", checkpoint='./chk/best3')
# nModel.build()

# for i in range(0, 105, 5):
#     nModel.load_checkpoint(i)
#     data = nModel.evaluate()
# print(data)

datafp = open('./retrain.json', 'r')
data = json.load(datafp)

plt.figure(figsize=(8, 5))

plt.plot(data[0], 'b-', label='SGD - best', alpha=0.8)
plt.plot(data[1], 'b--', label='RMSProp - best', alpha=0.8)
plt.plot(data[2], 'r-', label='SGD - worst', alpha=0.8)
plt.plot(data[3], 'r--', label='RMSProp - worst', alpha=0.8)

plt.ylabel('accuracy', fontsize=17)
plt.xlabel('epochs', fontsize=17)

plt.ylim(0.0, 0.9)
plt.grid()
plt.show()