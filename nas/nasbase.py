import numpy as np
import copy as copy
import tensorflow as tf
from lib.model import NModel
from lib.spec import Spec
from datetime import datetime
import json
import os
import random

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

    def __eq__(self, other):
        return self.accuracy == other.accuracy
    
    def __le__(self, other):
        return self.accuracy <= other.accuracy

    def __lt__(self, other):
        return self.accuracy < other.accuracy
    
    def __ge__(self, other):
        return self.accuracy >= other.accuracy

    def __gt__(self, other):
        return self.accuracy > other.accuracy

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
        self.pop_history = []

    def train_and_eval(self, model: Model, dry_run = False):
        spec = Spec(model.arch)
        if spec.valid_spec == False:
            model.accuracy = -1
            self.history.append(model)
            return model

        if dry_run:
            data = {
                "training_time": random.random() * 10000,
                "validation_accuracy": random.random()
            }
        else:
            try:
                net = NModel(spec)
                net.build()
                data = net.train_and_evaluate()
            except Exception as ex:
                print(ex)
                model.accuracy = -1
                self.history.append(model)
                return model

        self.times.append(self.times[-1] + data["training_time"])
        model.data = data
        model.accuracy = self.calc_accuracy(data)

        if len(self.best_model) == 0 or model > self.best_model[-1] :
            self.best_model.append(model)
        else:
            self.best_model.append(self.best_model[-1])
        
        self.history.append(model)
        return model

    def calc_accuracy(self, data):
        return -1 + data["validation_accuracy"]

    def save_state(self, output_path, state_id, pop):
        now = datetime.now()
        date_time = now.strftime("%Y%m%d%H%M%S")
        self.save_file_his(output_path, state_id, date_time)
        self.save_file_his_pop(output_path, state_id, date_time, pop)

    def save_file_his(self, output_path, idx, timestamp):
        ls_his = [m.get_dict() for m in self.history]
        fh = open(os.path.join(output_path, 'history.json'), 'w')
        json.dump(ls_his, fh, cls=NpEncoder)
        fh.close()

    def save_file_his_pop(self, output_path, idx, timestamp, pop):
        for p in pop:
            self.pop_history.append(p)
            
        ls_state = [m.get_dict() for m in self.pop_history]
        fh = open(os.path.join(output_path, 'state_history.json'), 'w')
        json.dump(ls_state, fh, cls=NpEncoder)
        fh.close()
