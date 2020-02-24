import tensorflow as tf
import lib.base_ops as ops
import lib.spec as spec
import numpy as np

class Cell(object):
  def __init__(self, spec: spec.Spec, channels, inputs, name=""):
    super(Cell, self).__init__()
    self.name = name
    self.spec = spec
    self.channels = channels
    self.inputs = inputs

    input_channels = inputs.get_shape()[3]
    self.node_channels = self.compute_vertex_channels(input_channels, self.channels, self.spec.matrix)
    self.number_of_node = len(self.node_channels)

  def build(self):
    tensors = [self.inputs]
    concat_out = []
    for t in range(1, self.number_of_node - 1):
      add_in = [self.truncate(tensors[s], self.node_channels[t]) for s in range(1, t) if self.spec.matrix[s,t]]
      
      if self.spec.matrix[0, t]:
        p_input = ops.Projection().build(self.node_channels[t], tensors[0])
        add_in.append(p_input)
      
      if len(add_in) == 1:
        added = add_in[0]
      else:
        added = tf.keras.layers.Add()(add_in)
      
      node = ops.OP_MAP[self.spec.ops[t]]().build(self.node_channels[t], added)
      tensors.append(node)
      
      if self.spec.matrix[t, -1]:
        concat_out.append(tensors[t])

    if not concat_out:
      assert self.spec.matrix[0, -1]
      output = ops.Projection().build(self.node_channels[-1], tensors[0])
    else:
      output = concat_out[0] if len(concat_out) == 1 else tf.keras.layers.Concatenate(axis=-1)(concat_out)

    return output

  def compute_vertex_channels(self, input_channels, output_channels, matrix):
    """Computes the number of channels at every vertex.

    Given the input channels and output channels, this calculates the number of
    channels at each interior vertex. Interior vertices have the same number of
    channels as the max of the channels of the vertices it feeds into. The output
    channels are divided amongst the vertices that are directly connected to it.
    When the division is not even, some vertices may receive an extra channel to
    compensate.

    Args:
      input_channels: input channel count.
      output_channels: output channel count.
      matrix: adjacency matrix for the module (pruned by model_spec).

    Returns:
      list of channel counts, in order of the vertices.
    """
    num_vertices = np.shape(matrix)[0]

    vertex_channels = [0] * num_vertices
    vertex_channels[0] = input_channels
    vertex_channels[num_vertices - 1] = output_channels

    if num_vertices == 2:
      # Edge case where module only has input and output vertices
      return vertex_channels

    # Compute the in-degree ignoring input, axis 0 is the src vertex and axis 1 is
    # the dst vertex. Summing over 0 gives the in-degree count of each vertex.
    in_degree = np.sum(matrix[1:], axis=0)
    interior_channels = output_channels // in_degree[num_vertices - 1]
    correction = output_channels % in_degree[num_vertices - 1]  # Remainder to add

    # Set channels of vertices that flow directly to output
    for v in range(1, num_vertices - 1):
      if matrix[v, num_vertices - 1]:
        vertex_channels[v] = interior_channels
        if correction:
          vertex_channels[v] += 1
          correction -= 1

    # Set channels for all other vertices to the max of the out edges, going
    # backwards. (num_vertices - 2) index skipped because it only connects to
    # output.
    for v in range(num_vertices - 3, 0, -1):
      if not matrix[v, num_vertices - 1]:
        for dst in range(v + 1, num_vertices - 1):
          if matrix[v, dst]:
            vertex_channels[v] = max(vertex_channels[v], vertex_channels[dst])
      assert vertex_channels[v] > 0

    # tf.logging.info('vertex_channels: %s', str(vertex_channels))

    # Sanity check, verify that channels never increase and final channels add up.
    final_fan_in = 0
    for v in range(1, num_vertices - 1):
      if matrix[v, num_vertices - 1]:
        final_fan_in += vertex_channels[v]
      for dst in range(v + 1, num_vertices - 1):
        if matrix[v, dst]:
          assert vertex_channels[v] >= vertex_channels[dst]
    assert final_fan_in == output_channels or num_vertices == 2
    # num_vertices == 2 means only input/output nodes, so 0 fan-in

    return vertex_channels

  def truncate(self, inputs, channels):
    """Slice the inputs to channels if necessary."""
    input_channels = inputs.get_shape()[3]

    if input_channels < channels:
      raise ValueError('input channel < output channels for truncate')
    elif input_channels == channels:
      return inputs   # No truncation necessary
    else:
      # Truncation should only be necessary when channel division leads to
      # vertices with +1 channels. The input vertex should always be projected to
      # the minimum channel count.
      assert input_channels - channels == 1
      return tf.slice(inputs, [0, 0, 0, 0], [-1, -1, -1, channels])