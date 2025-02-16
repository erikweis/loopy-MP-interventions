import numpy as np
from scipy.stats import binom
import networkx as nx
from src.percolation import percolation_MC, NeighborhoodObservable
from src.simulation import simulate_discrete_SI_temporal
from src.gamma_sample import GammaSample, TemporalGammaSample


class Neighborhood:
    """
    A class to represent the neighborhood of a node i in a graph.

    Attributes
    ----------
    i : int
        The node index.
    nodes : list
        The nodes in the neighborhood.
    edges : list
        The edges in the neighborhood. This list may exclude 
        those that overlap with some other neighborhood.
        All edges that are incident to node i are excluded from 
        the list passed on initialization.
    neighbors_i : list
        The neighbors of node i in the neighborhood.
    Gamma_samples : list
        A list of GammaSample objects, obtained by sampling
        the neighborhood with a percolation process.
    """

    def __init__(self, edges, i, filter_r0_edges = True):
        self.i = i
        self.nodes, self.edges = self.get_reachable_nodes_edges(edges, i)
        self.Gamma_samples = []
        self.neighbors_i = []

        # calculate neighbors of i 
        # filter edges incident on i
        filtered_edges = []
        for edge in self.edges:
            if i in edge: # edge is incident on node i
                k = edge[1] if edge[0] == i else edge[0]
                self.neighbors_i.append(k)
                # add r=0 edges to the list
                if not filter_r0_edges:
                    filtered_edges.append(edge)
            else: # edge is not incident on node i
                filtered_edges.append(edge)

        self.edges = filtered_edges
        assert len(set(self.neighbors_i)) == len(self.neighbors_i)

    def __repr__(self):
        return f"""Neighborhood(
    i={self.i},
    nodes={self.nodes},
    edges={self.edges},
    neighbors_i={self.neighbors_i},
    Gamma_samples={self.Gamma_samples}
)"""

    @staticmethod
    def get_reachable_nodes_edges(edgelist, i):
        """From the edgelist passed, return the nodes and edges
        that are reachable from node i. Perform a DFS search.

        Returns:
            reachable_nodes : list
                The nodes reachable from node i.
            reachable_edges : list
                The edges reachable from node i.
        """
        reachable_edges = []
        reachable_nodes = set()
        for edge in edgelist:
            if i in edge:
                reachable_edges.append(edge)
                reachable_nodes.update(edge)
        finished = False
        while not finished:
            finished = True
            for edge in edgelist:
                if (edge[0] in reachable_nodes or edge[1] in reachable_nodes) and edge not in reachable_edges:
                    reachable_edges.append(edge)
                    reachable_nodes.update(edge)
                    finished = False
        return list(reachable_nodes), reachable_edges


def sample_gamma(nb, M, infection_prob, v = None, force=False, temporal = False):
    """Sample the neighborhood of a node i with a percolation process.
    
    Parameters
    ----------
    nb : Neighborhood
        The neighborhood object to sample.
    M : int
        The number of Monte Carlo steps to perform.
    infection_prob : float
        The probability an edge is kept in the percolation process.
    v : list, optional
        The vaccination status of each node in the graph.
    force : bool, optional
        If True, overwrite the Gamma samples if they already exist.
    """
    # check if Gamma samples already exist
    if len(nb.Gamma_samples) > 0 and not force:
        raise ValueError("Gamma samples already exist. Use force=True to overwrite.")
    # clear the Gamma samples and generate new samples
    nb.Gamma_samples = []
    if temporal:
        _sample_neighborhood_temporal(nb, M, infection_prob, v=v)
    else:
        if len(nb.edges) == 0:
            if infection_prob > 0:
                # if the neighborhood has no edges, then only one percolation outcome
                # is possible, so we add a null sample if it's probability is nonzero
                s = _get_null_sample(nb, infection_prob, temporal=temporal)
                nb.Gamma_samples.append(s)
        else:
            _sample_neighborhood_newman_ziff(nb, M, infection_prob, v=v)
            if infection_prob > 0:
                # add sample for the case where no edges appear in percolation process
                s = _get_null_sample(nb, infection_prob, temporal = temporal)
                nb.Gamma_samples.append(s)


def _sample_neighborhood_temporal(nb, M, infection_prob, v = None):
    """"
    Perform monte carlo sampling of the neighborhood percolation process
    using discrete-time SIR dynamics.
    """
    # create a graph of the neighborhood
    g = nx.Graph()
    g.add_edges_from(nb.edges)
    for k in nb.neighbors_i:
        g.add_edge(nb.i, k)
    if len(g) == 0:
        return
    # run the SIR process for M iterations
    seeds = [nb.i]
    for _ in range(M):
        I = simulate_discrete_SI_temporal(g, infection_prob, seeds, t_max = 20)
        reachable_nodes, distances = [],[]
        for node, distance in I.items():
            if distance >= 0 and node != nb.i:
                reachable_nodes.append(node)
                distances.append(distance)
        # save a sample which stores the nodes reachable from i, and their distance from i
        s = TemporalGammaSample(
            reachable_nodes,
            distances,
            prob = 1/M
        )
        nb.Gamma_samples.append(s)
    # add samples with appreciable probability to the sample list
    nb.Gamma_samples = consolidate_samples(nb.Gamma_samples)


def _sample_neighborhood_newman_ziff(nb, M, infection_prob, v = None):
    """"
    Perform monte carlo sampling of the neighborhood percolation process
    using the Newman-Ziff algorithm.
    """
    # create a neighborhood observable object to store the outcomes of the process
    NO = NeighborhoodObservable(nb.i, nb.neighbors_i, nb.nodes, nb.edges, infection_prob, v=v)
    # create a union-find structure for Newman-Ziff
    x = np.zeros(
        len(nb.nodes)-1, # exclude i in union-find structure
        dtype=int
    ) - 1
    # set the max number of edges to add
    m_max = len(NO.edgelist)
    # perform percolation sampling
    for _ in range(M):
        percolation_MC(NO.edgelist, x, NO, m_max)
    # add samples with appreciable probability to the sample list
    nb.Gamma_samples = consolidate_samples(NO.samples)
    # adjust probabilities to account for the number of samples, M
    for sample in nb.Gamma_samples:
        sample.prob = sample.prob / M


def _get_null_sample(nb, infection_prob, temporal = False):
    """"
    Return a GammaSample object, where no edges were retained
    in the percolation process.
    """
    assert temporal == False
    return GammaSample(
            [[k] for k in nb.neighbors_i],
            [1]*len(nb.neighbors_i),
            prob = binom.pmf(0,len(nb.edges),infection_prob)
        )


def consolidate_samples(samples, min_prob_threshold = 10**-10):
    """
    Consolidate neighborhood percolation samples with the same reachable nodes
    and combine their probabilities.
    """
    # use a hash function to consolidate samples
    out_samples = dict()
    for s in samples:
        if s in out_samples:
            out_samples[s].prob += s.prob
        else:
            out_samples[s] = s
    # filter samples with low probability / no reachable nodes
    out_samples = filter_samples(out_samples, min_prob_threshold)
    return out_samples


def filter_samples(samples, min_prob_threshold = 10**-5):
    """
    Filter samples with probability below a threshold.
    This is a computational efficiency measure to reduce the number
    of samples with negligible probability.
    """
    return [s for s in samples if s.prob > min_prob_threshold]
