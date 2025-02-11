import random
import numpy as np
from collections import defaultdict
from scipy.stats import binom
from src.gamma_sample import GammaSample

def find(x, i):
    if x[i] < 0:
        return i
    x[i] = find(x, x[i])
    return x[i]

def percolation_MC(edgelist, x, NO, m_max):
    """Perform a Monte Carlo percolation process on the edgelist
    using the Newman-Ziff algorithm."""
    # reset data structures
    x.fill(-1)
    # shuffle percolation order
    random.shuffle(edgelist)
    for idx, m in enumerate(range(1,m_max+1)):
        i, j = edgelist[idx]
        r_i = find(x, i)
        r_j = find(x, j)
        if r_i != r_j:
            if x[r_i] < x[r_j]:  # size of i is greater than j
                x[r_i] += x[r_j]  # size of i grows by the size of j
                x[r_j] = r_i  # assign all js to group of i
            else:  # size of i is less than j
                x[r_j] += x[r_i]  # size of j grows by the size of i
                x[r_i] = r_j  # assign all i's to group of j
        # check all qualities of interest
        calculate_observables(m, NO, x)


class NeighborhoodObservable():
    """Class to store the observable outcomes of interest for the 
    neighborhood percolation process. 

    We remove node $i$ from the list of nodes and edges, because
    we want to keep track of clusters of nodes that are not
    connected through $i$.
    """

    def __init__(self, i, neighbors_i, nodes, edgelist, infection_prob, v=None):
        
        self.i = i 
        #remove i from nodes
        nodes = [n for n in nodes if n != i]
        # assing indices to all nodes, i.e. node_map[idx] = node
        self.node_map = nodes

        self.edgelist = []
        for j,k in edgelist:
            if v is not None: # exclude vaccinated nodes
                if v[j] == 1 or v[k] == 1: 
                    #print(f"excluding ({j},{k}) because v[j]:{v[j]} or v[k]:{v[k]} is 1")
                    continue
            if i in (j,k):
                #print(f"excluding ({j},{k}) because i:{i} is in ({j},{k})")
                continue
            self.edgelist.append((nodes.index(j), nodes.index(k)))               
        # not re-indexed
        self.neighbors_i = neighbors_i
        
        self.samples = []
        self.infection_prob = infection_prob

    def __repr__(self):
        return f"""NeighborhoodObservable(
    i={self.i}, 
    neighbors_i={self.neighbors_i}, 
    nodes={self.node_map}, 
    edgelist={self.edgelist}, 
    infection_prob={self.infection_prob})"""



def is_reachable(i, j, x):
    """Determine if $i$ and $j$ are in the same cluster."""
    return find(x, i) == find(x, j)


def calculate_observables(m, NO, x):
    """Given a current percolation instance, calculate the observables
    of interest. In this case, we save the 
    """
    i = NO.i
    node_map = NO.node_map
    # calcualte probability of percolation outcome
    prob = binom.pmf(m, len(NO.edgelist), NO.infection_prob)
    if prob == 0:
        return
    # find all subsets of nodes at are connected in the percolation instance
    cluster2nodes = defaultdict(list)
    for idx in range(len(x)):
        cluster2nodes[find(x, idx)].append(node_map[idx])
    # for each cluster, find the number of neighbors(i) in the cluster    
    clusters = []
    num_neighbors = []
    neighbors_i = set(NO.neighbors_i)
    for nodes in cluster2nodes.values():
        # get the nodes that are neighbors of i and count them
        neighbors_subset = set(nodes).intersection(neighbors_i)
        num_reachable_neighbors = len(neighbors_subset)
        if num_reachable_neighbors == 0:
            continue
        # save the cluster and the number of neighbors in the cluster
        clusters.append(nodes)
        num_neighbors.append(num_reachable_neighbors)
    NO.samples.append(GammaSample(clusters, num_neighbors, prob))
