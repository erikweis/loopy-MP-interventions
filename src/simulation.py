import numpy as np
import random
from tqdm import tqdm


def simulate_discrete_SI_temporal(g, p, seeds, t_max = 100):

    """Simulate a discrete SIR process on a graph.

    Args:
        g: networkx.Graph
        p: float, infection probability
        seeds: list of int, initial infected nodes
        t_max: int, maximum number of time steps process, default = 100   
    """

    I = {i:-1 for i in g.nodes()}

    current_infected_nodes = list(seeds)
    next_infected_nodes = set()

    for t in range(0, t_max):   
        #infect nodes that are newly infected at time t
        for node in current_infected_nodes:
            I[node] = t
        
        # determine next infected nodes
        next_infected_nodes.clear()
        for i in current_infected_nodes:
            for j in g.neighbors(i):
                if I[j] < 0 and np.random.random() < p:
                    next_infected_nodes.add(j)
                    I[j] = t
        
        if not next_infected_nodes:
            break
        current_infected_nodes = list(next_infected_nodes) 
    return I


def simulate_discrete_SI(g, p, seeds, v, marginals, sentinels = None, return_t_final = True):
    
    """Simulate a discrete SIR process on a graph.
    Adds +1 to marginals[i,t] for each node i infected at time t.

    If sentinels are passed, the function returns the time that 
    the first sentinel is reached. If no sentinel is reached, the
    function returns the time at which the outbreak ends.

    Args:
        g: networkx.Graph
        p: float, infection probability
        seeds: list of int, initially infected nodes
        v: list of bool, vaccination status of nodes
        marginals: np.array, marginals data structure
        sentinels: list of int, sentinel nodes

    Returns:
        int, min(time of first sentinel reached, time of last infection)
    """


    # if seeds are not passed, each node is initially infected with probability 1/n',
    # where n' is the number of unvaccinated nodes
    if seeds is None:
        seeds = []
        for node in g.nodes():
            if not v[node] and np.random.random() < 1/len(g):
                seeds.append(node)
        #seeds = [random.choice(list(g.nodes()))]
    I = np.zeros(max(g.nodes())+1, dtype=bool)
    
    # initialize all seeds as infected
    for seed in seeds:
        I[seed] = True
    
    current_infected_nodes = list(seeds)
    next_infected_nodes = set()

    t_max = marginals.shape[1]
    for t in range(t_max):
        # infect nodes that are newly infected at time t
        for node in current_infected_nodes:
            marginals[node, t:] += 1
            # if sentinels are passed, return the time that the first one is reached
            if sentinels is not None and node in sentinels:
                return t
        
        # determine next infected nodes
        next_infected_nodes.clear()
        for i in current_infected_nodes:
            for j in g.neighbors(i):
                # node can only be infected if not vaccinated
                if v[j]:
                    continue
                # node cannot be already infected
                # infection happens with probability p
                elif not I[j] and np.random.random() < p:
                    next_infected_nodes.add(j)
                    I[j] = True
        
        if not next_infected_nodes:
            break
        current_infected_nodes = list(next_infected_nodes)
    
    if return_t_final:
        return t
    else:
        return None


def get_marginals_simulation(g, p, seeds = None, v=None, num_samples=10000, t_max=15, verbose = False):

    # establish marginals data structure
    marginals = np.zeros((max(g.nodes())+1, t_max), dtype=int)

    # if no v passed, no nodes are vaccinated
    if v is None:
        v = np.zeros(g.number_of_nodes(), dtype=bool)

    # track progress if verbose = True
    iter_ = range(num_samples)
    if verbose:
        iter_ = tqdm(iter_)

    # simulate SIR process for <num_samples> iterations
    for _ in iter_:
        simulate_discrete_SI(g, p, seeds, v, marginals)
    
    # normalize marginals and return
    marginals =  marginals.astype(float) / num_samples
    return marginals


def get_marginals_sentinel(g, p, sentinels, num_samples = 10000, t_max = 15, include_null = True):

    """Compute the vector of the probability a sentinel is reached at time t.
    If a sentinel is not found and `include_null=True`, the end of the outbreak is added to the vector

    Returns:
        np.array, sentinel marginals
    """

    # set up sentinel marginals
    sentinel_marginals = np.zeros(t_max)
    # unused data structures to be passed to simulate_discrete_SI
    marginals = np.zeros((max(g.nodes())+1, t_max), dtype=int)
    v = np.zeros(len(g), dtype=bool)
    seeds = None
    
    # simulate SIR process for <num_samples> iterations
    for _ in range(num_samples):
        t = simulate_discrete_SI(g, p, seeds, v, marginals, sentinels, return_t_final=include_null)
        if t is not None:
            sentinel_marginals[t:] += 1
    
    return sentinel_marginals / num_samples