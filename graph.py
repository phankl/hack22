import time
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs

from node_independence import *

def unwrap_bool(n, b):
  result = np.zeros((n, n), dtype=int)
  row = 1
  column = 0

  for i, bit in enumerate(b):
    result[row][column] = bit
    column += 1
    if column == row:
      row += 1
      column = 0
    
  return result

def check_graph(n, b):
    n_edges = np.sum(b)
    if n_edges < n:
      return False
    
    # adjacency matrix
    a = unwrap_bool(n, b)
    
    # symmetrise
    a_sym = a + a.T

    # degree matrix
    deg = np.sum(a_sym, axis=0)    

    d = np.diag(deg)

    # laplace matrix
    l = d - a_sym

    # eigenvalues
    eigs = np.linalg.eigvalsh(l).astype(int)

    # second lowest eigenvalue
    conn = int(eigs[1])
    
    if conn == 0:
      return False
    else:
      g = nx.convert_matrix.from_numpy_matrix(a)
      nx.draw(g, arrows=True)
      #plt.show()
      plt.savefig('graph'+b+'.png')
      return True
 
def classify_path_nodes(path):
  result = [0] * (len(path) - 2)
  for i, node in enumerate(path[1:-1]):
    prev_node = path[i]
    next_node = path[i+2]

    if prev_node > node:
      prev_curr = 1
    else:
      prev_curr = 0
    if node > next_node:
      curr_next = 1
    else:
      curr_next = 0
    
    if prev_curr == curr_next:
      # chain
      result[i] = 0
    elif prev_curr == 0 and curr_next == 1:
      # fork
      result[i] = 1
    else:
      # collider
      result[i] = 2
  
  return result

def difference_combinations(pair, n):
  nodes = list(range(n))
  diff = [node for node in nodes if node not in pair]
  combs = [list(itertools.combinations(diff, i)) for i in range(n-1)]
  combs = [j for sub in combs for j in sub]

  return combs

def check_model(cache, a, node_map, paths, node_types, descendants, features):
  n = len(a)

  for pair, sub in paths.items():
    zs = difference_combinations(pair, n)
    for z in zs:
      all_paths_blocked = True
      for i, path in enumerate(sub):
        path_blocked = False
        for j, node in enumerate(path[1:-1]):
          node_type = node_types[pair][i][j]
          # check for chain or fork blocker
          if (node_type == 0 or node_type == 1) and node in z:
            path_blocked = True
            break
          # check for collider blocker
          if node_type == 2 and all([desc in z for desc in descendants[node]]):
            path_blocked = True
            break
        if not path_blocked:
          all_paths_blocked = False
          break
      
      x_node = node_map[pair[0]]
      y_node = node_map[pair[1]]
      z_nodes = [node_map[z_node] for z_node in z]
      
      x_label = features[x_node]
      y_label = features[y_node]
      z_labels = sorted([features[z_node] for z_node in z_nodes])
      
      input_tuple = tuple(sorted((x_label,) + (y_label,)))
      if len(z_labels) > 0:
        input_tuple += tuple(z_labels)

      pair_independent = cache[input_tuple]
      if all_paths_blocked != pair_independent:
        #print(input_tuple, cache[input_tuple])
        return False
        
  return True

def model_search(path, features):
    
  dat = pd.read_csv(path)[features] ## truncated set
  print(dat)
  #dat = tfl_preprocess(dat)

  n = len(dat.columns)

  graphs = np.loadtxt(f"graphs_{n}.gam", dtype=int)
  models = []

  # cache data dependencies
  cache = {}
  pairs = itertools.combinations(range(n), 2)
  for pair in pairs:
    zs = difference_combinations(pair, n)
    for z in zs:
      x_label = features[pair[0]]
      y_label = features[pair[1]]
      z_labels = sorted([features[z_node] for z_node in z])
      input_tuple = tuple(sorted((x_label,) + (y_label,)))
      if len(z_labels) == 0:
        z_labels = None
      else:
        input_tuple += tuple(z_labels)
      cache[input_tuple] = is_independent(dat, [x_label], [y_label], z_labels)[0]
      if not cache[input_tuple]:
        print(input_tuple)
    
  count = 0
  for i, graph in enumerate(graphs):
    print(i/len(graphs)*100)
    a = unwrap_bool(n, graph)
    g = nx.convert_matrix.from_numpy_matrix(a, create_using=nx.DiGraph)

    # all node mappings
    n_out_edges = np.sum(a, axis=1) 
    node_maps = itertools.permutations(range(n))
    
    # detect outgoing edges from output variable
    node_maps_red = (node_map for node_map in node_maps if n_out_edges[node_map.index(0)] == 0)  
    
    # get all simple paths between any pairs in the graph
    a_sym = a + a.T
    g_sym = nx.convert_matrix.from_numpy_matrix(a_sym)
    pairs = itertools.combinations(range(n), 2)
    paths = {pair: nx.algorithms.simple_paths.all_simple_paths(g_sym, *pair) for pair in pairs}

    # classify path nodes as chain, fork or collider
    node_types = {pair: [classify_path_nodes(path) for path in sub] for pair, sub in paths.items()}

    # get all descendant nodes for each node
    descendants = [nx.descendants(g, i) | {i} for i in range(n)]

    # iterate over all node_maps to get models
    for node_map in node_maps_red:

      # show model
      '''
      labels = {i: features[node_map[i]] for i in range(n)}
      nx.draw(g, arrows=True, with_labels=True, labels=labels)
      plt.show()
      '''

      if check_model(cache, a, node_map, paths, node_types, descendants, features):
        print("Model added")
        models += [(graph, node_map)]
      count += 1

  return models

def generate_dags(n, name):
  start = time.time()
  
  bool_arrays = itertools.product(range(2), repeat=(n*n - n)//2)
  graphs = [b for b in bool_arrays if check_graph(n, b)]
  graphs = np.array(graphs, dtype=int)
  np.savetxt(name, graphs, fmt='%i')
  
  end = time.time()
  #print(n, end-start)

filepath = "tests/simple_chain_testset.csv"
features = ['Z', 'Y', 'X']

#filepath = FILEPATH
#features = TEST_FEATURES_TFL

models = model_search(filepath, features)
print(models)
