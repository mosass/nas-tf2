import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
import copy

MAX_EDGES=9
VERTICES=7

class Spec(object):

  @staticmethod
  def get_configuration_space():
    cs = CS.ConfigurationSpace()

    ops_choices = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
    cs.add_hyperparameter(CS.CategoricalHyperparameter("op_node_0", ops_choices))
    cs.add_hyperparameter(CS.CategoricalHyperparameter("op_node_1", ops_choices))
    cs.add_hyperparameter(CS.CategoricalHyperparameter("op_node_2", ops_choices))
    cs.add_hyperparameter(CS.CategoricalHyperparameter("op_node_3", ops_choices))
    cs.add_hyperparameter(CS.CategoricalHyperparameter("op_node_4", ops_choices))

    cs.add_hyperparameter(CS.UniformIntegerHyperparameter("num_edges", 0, MAX_EDGES))

    for i in range(VERTICES * (VERTICES - 1) // 2):
      cs.add_hyperparameter(CS.CategoricalHyperparameter("edge_%d" % i, [0,1]))

    return cs

  def __init__(self, config: CS.Configuration):
    matrix = np.zeros([VERTICES, VERTICES], dtype=np.int8)
    idx = np.triu_indices(np.shape(matrix)[0], k=1)
    for i in range(VERTICES * (VERTICES - 1) // 2):
      row = idx[0][i]
      col = idx[1][i]
      matrix[row, col] = config["edge_%d" % i]

    labeling = [config["op_node_%d" % i] for i in range(5)]
    labeling = ['input'] + list(labeling) + ['output']

    self.original_matrix = copy.deepcopy(matrix)
    self.original_ops = copy.deepcopy(labeling)

    self.matrix = copy.deepcopy(matrix)
    self.ops = copy.deepcopy(labeling)
    self.valid_spec = True
    self._prune()

  def _prune(self):
    """Prune the extraneous parts of the graph.

    General procedure:
      1) Remove parts of graph not connected to input.
      2) Remove parts of graph not connected to output.
      3) Reorder the vertices so that they are consecutive after steps 1 and 2.

    These 3 steps can be combined by deleting the rows and columns of the
    vertices that are not reachable from both the input and output (in reverse).
    """
    num_vertices = np.shape(self.original_matrix)[0]

    # DFS forward from input
    visited_from_input = set([0])
    frontier = [0]
    while frontier:
      top = frontier.pop()
      for v in range(top + 1, num_vertices):
        if self.original_matrix[top, v] and v not in visited_from_input:
          visited_from_input.add(v)
          frontier.append(v)

    # DFS backward from output
    visited_from_output = set([num_vertices - 1])
    frontier = [num_vertices - 1]
    while frontier:
      top = frontier.pop()
      for v in range(0, top):
        if self.original_matrix[v, top] and v not in visited_from_output:
          visited_from_output.add(v)
          frontier.append(v)

    # Any vertex that isn't connected to both input and output is extraneous to
    # the computation graph.
    extraneous = set(range(num_vertices)).difference(
        visited_from_input.intersection(visited_from_output))

    # If the non-extraneous graph is less than 2 vertices, the input is not
    # connected to the output and the spec is invalid.
    if len(extraneous) > num_vertices - 2:
      self.matrix = None
      self.ops = None
      self.valid_spec = False
      return

    self.matrix = np.delete(self.matrix, list(extraneous), axis=0)
    self.matrix = np.delete(self.matrix, list(extraneous), axis=1)
    for index in sorted(extraneous, reverse=True):
      del self.ops[index]

    if np.sum(self.matrix) > MAX_EDGES:
      self.valid_spec = False

if __name__ == "__main__":
    config = Spec.get_configuration_space().sample_configuration()

    spec = Spec(config)
    print(spec.matrix)
    print(spec.ops)