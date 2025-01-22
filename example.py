from src.neighborhood_message_passing import NeighborhoodMessagePassing

import networkx as nx
import numpy as np


g = nx.karate_club_graph()
r = 1
infection_rate = 0.2
t_max = 50
T = 10


def influence_maximization_example():
    """Example useage for influence maximization."""
    
    n = len(g)
    seeds = [0,10]

    # set s and v
    s = np.zeros(n)
    s[seeds] = 1
    # no vaccinated nodes
    v = np.zeros(n)

    NMP = NeighborhoodMessagePassing(g, r, infection_rate, t_max, 
                                     T=T, v = v, verbose=True, temporal = False)

    NMP.neighborhood_message_passing(s)
    marginals = NMP.marginals
    
    pi_t = np.sum(marginals, axis=0)
    print("Influence maximization",pi_t)


def vaccination_example():

    """Example useage for vaccination."""
        
    n = len(g)
    seeds = [0,10]

    # set s and v
    v = np.zeros(n)
    for i in seeds:
        v[i] = 1
    s = np.zeros(n)
    s += 1/(n-len(seeds))
    s[list(seeds)] = 0

    # run NMP
    NMP = NeighborhoodMessagePassing(g, r, infection_rate, t_max, T=T, v = v, temporal = False, verbose=False)
    NMP.neighborhood_message_passing(s)
    marginals = NMP.marginals
    
    pi_t = np.sum(marginals, axis=0)
    print("Vaccination",pi_t)


def sentinel_surveillance_example():
     
    """Example useage for sentinel surveillance."""

    n = len(g)
    seeds = [0,10]

    # set s and v
    v = np.zeros(n)
    for i in seeds:
        v[i] = 1
    s = np.zeros(n)
    s += 1/n

    NMP = NeighborhoodMessagePassing(g, r, infection_rate, t_max, T=T, v = v, verbose=False, temporal = True)
    NMP.neighborhood_message_passing(s,track_vaccinated=True)
    marginals = NMP.marginals

    marginals_S = marginals[seeds,:]
    pi_S_t = 1 - np.prod(1-marginals_S, axis=0)
    print("Sentinel surveillance",pi_S_t)


if __name__ == "__main__":
    influence_maximization_example()
    vaccination_example()
    sentinel_surveillance_example()