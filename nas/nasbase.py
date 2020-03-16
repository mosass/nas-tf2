import numpy as np
import copy as copy
import tensorflow as tf
from lib.model import NModel
from lib.spec import Spec
from datetime import datetime
import json
import os

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

class Model(object):
    def __init__(self):
        self.arch = None
        self.data = None
        self.accuracy = None

    def __str__(self):
        """Prints a readable version of this bitstring."""
        return str(self.arch)
    
    def get_dict(self):
        return {
            "arch": self.arch.get_dictionary(),
            "accuracy": self.accuracy,
            "data": self.data
        }

class NasBase(object):

    def __init__(self):
        self.times = [0.0]
        self.best_model = []
        self.history = []

    def train_and_eval(self, model: Model):
        spec = Spec(model.arch)
        if spec.valid_spec == False:
            model.accuracy = -1
            return
        
        net = NModel(spec)
        net.build()
        data = net.train_and_evaluate()
        self.times.append(self.times[-1] + data["training_time"])
        model.data = data
        model.accuracy = self.calc_accuracy(data)

        self.history.append(model)
        if len(self.best_model) == 0 or model.accuracy > self.best_model[-1].accuracy :
            self.best_model.append(model)
        else:
            self.best_model.append(self.best_model[-1])
        
        return model

    def calc_accuracy(self, data):
        return -1 + data["validation_accuracy"]

    def save_state(self, output_path, state_id):
        now = datetime.now()
        date_time = now.strftime("%Y%m%d%H%M%S")
        self.save_file_his(output_path, state_id, date_time)
        self.save_file_state(output_path, state_id, date_time)

    def save_file_his(self, output_path, idx, timestamp):
        ls_his = [m.get_dict() for m in self.history]
        fh = open(os.path.join(output_path, 'his_%s_%s.json' % (idx, timestamp)), 'w')
        json.dump(ls_his, fh, cls=NpEncoder)
        fh.close()

    def save_file_state(self, output_path, idx, timestamp):
        ls_state = {
            "times": self.times,
            "best_model": [m.get_dict() for m in self.best_model]
        }
        fh = open(os.path.join(output_path, 'state_%s_%s.json' % (idx, timestamp)), 'w')
        json.dump(ls_state, fh, cls=NpEncoder)
        fh.close()
