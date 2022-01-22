import time
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

def generate_dags(n):
  bool_arrays = itertools.product(range(2), repeat=(n*n - n)//2)
  
  for b in bool_arrays:
    n_edges = np.sum(b)
    if n_edges < n:
      continue
    
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
    #eigs = np.unique(eigs)
    #print(eigs)

    # second lowest eigenvalue
    conn = int(eigs[1])
    
    if conn == 0:
      continue
    else:
      g = nx.convert_matrix.from_numpy_matrix(a)
      #nx.draw(g, arrows=True)
      #plt.show()

start = time.time()
generate_dags(7)
end = time.time()
print(end-start)
