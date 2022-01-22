import time
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs

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
      #g = nx.convert_matrix.from_numpy_matrix(a)
      #nx.draw(g, arrows=True)
      #plt.show()
      return True

def check_model(a, node_map, paths):
  
  return False
 
def classify_path_nodes(a, path):
  result = [0] * (len(path) - 2)
  for i, node in enumerate[path[1:-1]]:
    prev_node = path[i-1]
    next_node = path[i+1]
    
    prev_curr = a[prev_node][node]
    curr_next = a[node][next_node]

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

def model_search(n):
  graphs = np.loadtxt(f"graphs_{n}.gam", dtype=int)
  models = []

  count = 0
  for i, graph in enumerate(graphs):
    a = unwrap_bool(n, graph)
    
    # all node mappings
    n_out_edges = np.sum(a, axis=1) 
    node_maps = itertools.permutations(range(n))
    
    # detect outgoing edges from output variable
    node_maps_red = (node_map for node_map in node_maps if n_out_edges[node_map.index(0)] > 0)  
    
    # get all simple paths between any pairs in the graph
    a_sym = a + a.T
    g_sym = nx.convert_matrix.from_numpy_matrix(a_sym)
    pairs = itertools.combinations(range(n), 2)
    paths = [nx.algorithms.simple_paths.all_simple_paths(g, *pair) for pair in pairs]
    paths = [path for sub in paths for path in sub]
    classified_path_nodes = [classify_path_nodes(a, path) for path in paths]

    # iterate over all node_maps to get models
    for node_map in node_maps_red:
      if check_model(graph, node_map, paths):
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
  print(n, end-start)

print(model_search(7))

#for n in range(8, 9):
#  generate_dags(n, f'graphs_{n}.gam')
