
# ga-re-a

- 20200717_183326
    - crossover rate = 0.9
    - mutaion tate = 0.5
    - population size = 10
    - (re) sample size = 5
    - (re) new indv per generation = 2
    - ga 4, re 2
    - migrate ga with ring topology every 3 generation
      migrate btw ga-re by neighbor + swap topology
    - (ga) non-strict to place new offspring at least one per generation

# ga-re-b

- 20200719_174625
    - crossover rate = 0.9
    - mutaion tate = 1.0
    - population size = 10
    - (re) sample size = 5
    - (re) new indv per generation = 2
    - ga 4, re 2
    - migrate ga with ring topology every 3 generation
      migrate btw ga-re by neighbor topology
    - (ga) strict to place new offspring at least one per generation

# re

- 20200719_032458
    - population size = 10
    - sample size = 5
    - new indv per generation = 1


# ga-c
- 




GA : 27
RE : 8

Single Cluster
 - 3-GA
 - 2-GA + 1-RE

Mutiple Cluster
- 3x2=6 GA
- 2x2=4 GA 1x2=2 RE




{"edge_0": 0, "edge_1": 0, "edge_10": 1, "edge_11": 0, "edge_12": 0, "edge_13": 1, "edge_14": 0, "edge_15": 0, "edge_16": 1, "edge_17": 1, "edge_18": 1, "edge_19": 1, "edge_2": 1, "edge_20": 0, "edge_3": 1, "edge_4": 1, "edge_5": 0, "edge_6": 0, "edge_7": 0, "edge_8": 1, "edge_9": 0, "op_node_0": "conv3x3-bn-relu", "op_node_1": "conv1x1-bn-relu", "op_node_2": "conv1x1-bn-relu", "op_node_3": "conv3x3-bn-relu", "op_node_4": "conv3x3-bn-relu"}