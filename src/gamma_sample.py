import numpy as np


class GammaSample:
    """Class to store the outcome of a percolation process on a neighborhood.
    
    Attributes
    ----------
    reachable_nodes : list[list]
        A list of clusters of nodes (also a list) that are reachable from node i.
    num_neighbors : list
        The number of direct neighbors of i in each cluster.
    prob : float
        The probability weight of the percolation outcome in the sampling process.
    """

    def __init__(self, reachable_nodes, num_neighbors, prob):
        self.reachable_nodes = reachable_nodes
        self.num_neighbors = num_neighbors
        self.prob = prob

    def __repr__(self):
        return f"""GammaSample(
    clusters={self.reachable_nodes}, 
    num_neighbors={self.num_neighbors},
    prob={self.prob})"""

    def __eq__(self, other):
        return self.reachable_nodes == other.reachable_nodes and \
               self.num_neighbors == other.num_neighbors

    def __hash__(self):
        return hash((tuple(tuple(c) for c in self.reachable_nodes), tuple(self.num_neighbors)))


class TemporalGammaSample():
    """Class to store the outcome of a percolation process on a neighborhood,
    as well as the distance of each node from node i.
    
    Attributes
    ----------
    reachable_nodes : list
        A list of nodes reachable from i
    distances : list[list]
        The distance of each reachable node from node i.
    """

    def __init__(self, reachable_nodes, distances, prob):
        self.reachable_nodes = reachable_nodes
        self.distances = distances
        self.prob = prob

    def __repr__(self):
        return f"""TemporalGammaSample(reachable_nodes={self.reachable_nodes}, distances={self.distances})"""

    def __eq__(self, other):
        return self.reachable_nodes == other.reachable_nodes and \
               self.distances == other.distances

    def __hash__(self):
        return hash((tuple(self.reachable_nodes), tuple(self.distances)))


def _prob_i_infected_given_gamma(state, i, t, infection_prob, sample, s, temporal = False):
    """Calcualte the probability that node i is infected at time t,
    given a percolation sample.

    For each cluster, compute the probability each cluster doesn't
    infect i. The probability that i is infected is the 1 - the
    probability that no cluster infects i.

    Returns:
        prob_i_infected_given_gamma : float
            The probability that node i is infected at time t, given the percolation sample.
    """    
    if temporal:
        return 1 - np.prod([(1 - state[k][i][t-d]) for k, d in \
                                       zip(sample.reachable_nodes, sample.distances)])
    else:
        prob_no_cluster_infects_i = 1
        for c, nn in zip(sample.reachable_nodes, sample.num_neighbors):
            # prob cluster infected = (1 - prob no one in cluster is infected)
            prob_cluster_infected = 1 - np.prod([(1 - state[k][i][t-1]) for k in c])
            # prob cluster infects i, conditioned on it being infected
            prob_infected_cluster_infects_i = (1 - (1 - infection_prob)**nn)
            # prob cluster doesn't infect i
            prob_cluster_doesnt_infect_i = 1 - prob_infected_cluster_infects_i*prob_cluster_infected
            # add to aggregate
            prob_no_cluster_infects_i *= prob_cluster_doesnt_infect_i

        prob_i_infected_given_gamma = 1 - prob_no_cluster_infects_i
        return prob_i_infected_given_gamma
