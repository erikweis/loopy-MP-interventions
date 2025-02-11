import networkx as nx
import numpy as np
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
from src.gamma_sample import GammaSample, _prob_i_infected_given_gamma
from src.neighborhood import Neighborhood, sample_gamma
from src.percolation import percolation_MC, NeighborhoodObservable


def construct_neighborhoods_edgelists(g, r, v = None):
    """Construct neighborhoods for each node in the graph
    by calculating a list of primitive cycles. By default,
    networkx does not return cycles of length 2, so we
    include them manually.

    Each cycle is added to the neighborhood of each node
    in the cycle.

    Returns:
        A list of lists, where each list contains the edges
    """

    neighborhoods = [[] for _ in range(len(g.nodes))]
    limited_cycles = list(nx.simple_cycles(g, length_bound = r + 2))
    # add cycles of length 2
    limited_cycles.extend([[i,j] for i,j in g.edges])

    # construct neighborhoods by adding the cycle to each nodes neighborhood on the cycle
    for cycle in limited_cycles:

        # check if the cycle contains a node that is vaccinated
        if v is not None:
            if any([v[i] == 1 for i in cycle]):
                continue

        l = len(cycle)
        for idx in range(l):
            edge = tuple(sorted([cycle[idx], cycle[(idx + 1) % l]]))
            for i in cycle:
                if edge not in neighborhoods[i]:
                    neighborhoods[i].append(edge)
    
    return neighborhoods


class NeighborhoodMessagePassing:
    """Class to perform message passing on a graph using the neighborhood message passing algorithm.
    
    Args:
        g (nx.Graph): The graph on which to perform message passing.
        r (int): Hyperparameter specifying the size of the neighborhood.
        infection_prob (float): The probability of an edge being retained in the 
            independent cascade (IC) model, which is a global value $p = p_{ij}$ 
            for all edges.
        t_max (int): The maximum number of iterations to run the message passing algorithm.
        v (np.array): The vaccination status of each node, $v_i = 1$ if node i is vaccinated.
        M (int): The number of samples to draw from each neighborhood for the marginal calculations.
        verbose (bool): Whether to print progress information.
        temporal (bool): Whether to correct for time to infection within neighborhoods. Slower but needed for sentinel surveillance.
    """

    def __init__(self, g, r, infection_prob, t_max, 
                 v = None, M=10, verbose = False, temporal = False):
        
        # basic configs
        self.verbose = verbose
        self.temporal = temporal
        self.infection_prob = infection_prob
        self.t_max = t_max

        # save graph information
        self.N = len(g.nodes)
        self.adjlist = [list(g.neighbors(i)) for i in range(len(g))]
        
        # set vaccination for each node
        if v is None:
            self.v = np.zeros(self.N)
        else:
            assert len(v) == self.N
            self.v = v

        # get neighborhood edgelists
        neighborhood_edgelists = construct_neighborhoods_edgelists(g, r)
        self.neighborhood_edgelists = neighborhood_edgelists
        # construct neighborhoods
        self.neighborhoods = [Neighborhood(edges, i, filter_r0_edges=(not temporal)) \
                              for i, edges in enumerate(neighborhood_edgelists)]
        # construct conditional neighborhoods
        self.neighborhoods_i_except_j = [dict() for _ in range(self.N)]
        self.construct_neighborhoods_i_except_j()

        # sample neighborhoods
        self.sample_neighborhoods(M)
        self.sample_neighborhoods_i_except_j(M)

        # initialize state
        self.state = self.empty_state(100)
        self.state_size = 100
        
        # initialize marginals
        self.marginals = None
        
    def construct_neighborhoods_i_except_j(self):
        """Construct all neighborhoods for N_i excluding N_j."""
        if self.verbose:
            print("Constructing neighborhoods for N_i excluding N_j.")
            for i in tqdm(range(self.N)):
                for k in self.neighborhoods[i].nodes:
                    if i == k:
                        continue
                    self.neighborhoods_i_except_j[k][i] = Neighborhood(
                        [edge for edge in self.neighborhood_edgelists[k] if edge not in self.neighborhood_edgelists[i]], k,
                        filter_r0_edges=(not self.temporal)
                    )
        else:
            for i in range(self.N):
                for k in self.neighborhoods[i].nodes:
                    if i == k:
                        continue
                    self.neighborhoods_i_except_j[k][i] = Neighborhood(
                        [edge for edge in self.neighborhood_edgelists[k] if edge not in self.neighborhood_edgelists[i]], k,
                        filter_r0_edges=(not self.temporal)
                    )

    def sample_neighborhoods(self, M):
        """
        Generate neighborhood samples.

        Args:
            M (int): Number of samples to draw for each neighborhood
        """
        if self.verbose:
            print("Sampling neighborhoods for marginal calculations.")
            for i in tqdm(range(self.N)):
                sample_gamma(self.neighborhoods[i], M, self.infection_prob, v=self.v, temporal=self.temporal)
        else:
            for i in range(self.N):
                sample_gamma(self.neighborhoods[i], M, self.infection_prob, v=self.v, temporal=self.temporal)
                
    def sample_neighborhoods_i_except_j(self, M):
        """
        Sample neighborhoods for N_i excluding N_j.
        
        Args:
            M (int): Number of samples to draw for each neighborhood
        """
        if self.verbose:
            print("Sampling neighborhoods for N_i excluding N_j.")
            total_pairs = sum(len(self.neighborhoods_i_except_j[i].keys()) for i in range(self.N))
            with tqdm(total=total_pairs) as pbar:
                for i in range(self.N):
                    for k in self.neighborhoods_i_except_j[i].keys():
                        sample_gamma(self.neighborhoods_i_except_j[i][k], M, 
                                self.infection_prob, v=self.v, temporal=self.temporal)
                        pbar.update(1)
        else:
            for i in range(self.N):
                for k in self.neighborhoods_i_except_j[i].keys():
                    sample_gamma(self.neighborhoods_i_except_j[i][k], M,
                            self.infection_prob, v=self.v, temporal=self.temporal)

    def empty_state(self, size):
        """
        Create a dictionary of dictionaries to store
        the temporal state of the conditional marginals, pi_{i/j}(t).
        The initial size is arbitrary, but can be extended
        with the extend_state method.

        Returns:
            _type_: _description_
        """
        state = defaultdict(lambda: defaultdict(lambda: [0]*size))
        for i, nb in enumerate(self.neighborhoods):
            for j in nb.nodes:
                if i != j:
                    state[i][j] = [0]*size
            
            state[i] = dict(state[i])
        state=dict(state)
        return state
    
    def extend_state(self, size):
        """Add zeros to the end of all state vectors."""
        state = self.state
        for i, state_i in state.items():
            for j in state_i.keys():
                state[j][i].extend([0]*size)
        self.state_size += size


    def convergence_check(self, t, threshold=1e-4):

        """Check if message passing has converged, which happens if
        pi_{i/j}(t) - pi_{i/j}(t-1) < threshold for all i, j.
        """
        for _, state_i in self.state.items():
            for __, state_ij in state_i.items():
                if abs(state_ij[t] - state_ij[t-1]) > threshold:
                    return False
        return True

    def reset_state(self):
        """Reset the state, such that pi_{i/j}(t)=0 for all i,j,t."""
        for i, state_i in self.state.items():
            for j in state_i.keys():
                self.state[i][j] = [0]*100

    def compute_marginals(self, s, convergence_time, track_vaccinated = False):
        """Compute the marginals pi_i(t) for all nodes i at time t."""
        # initialize marginals
        marginals = np.zeros((self.N, convergence_time+1))
        marginals[:, 0] = s

        # get v
        v = self.v
        for t in range(1, convergence_time+1):
            for i, nb_i in enumerate(self.neighborhoods):
                # compute pi_i(t)
                for sample in nb_i.Gamma_samples:
                    marginals[i,t] += _prob_i_infected_given_gamma(
                        self.state, i, t, self.infection_prob, sample, s, temporal = self.temporal
                    ) * sample.prob
                marginals[i, t] = (s[i] + (1 - s[i]) * (marginals[i, t]))
                # if track_vaccinated, marginals for vaccinated nodes are not set to zero
                if not track_vaccinated:
                    marginals[i, t] *= (1 - v[i])

        self.marginals = marginals

    def neighborhood_message_passing(self, s, convergence_threshold=1e-6, track_vaccinated=False):
        """Compute the conditional marginals $pi_{i/j}(t)$ for all nodes i, j at time t.
        
        Args:
            s (numpy.ndarray): Initial state vector where s[i] is the probability node i 
                is initially infected.
            convergence_threshold (float, optional): Threshold for determining convergence. 
                Message passing stops when the change in conditional marginals between 
                consecutive time steps is below this value. Defaults to 1e-6.
            track_vaccinated (bool, optional): If True, continue tracking marginals for 
                vaccinated nodes. If False, set marginals to 0 for vaccinated nodes. 
                Defaults to False.
        
        Returns:
            int: The number of iterations until convergence.
                
        Raises:
            RuntimeError: If the algorithm does not converge within t_max iterations.
        """
        assert 0 <= self.infection_prob <= 1.0
        self.reset_state()
        for i, state_i in self.state.items():
            for j in state_i.keys():
                self.state[i][j][0] = s[i]
        v = self.v

        if self.verbose:
            pbar = tqdm(total=self.t_max-1, desc="Message passing iterations")
            
        for t in range(1, self.t_max):
            # extend state size if necessary
            if self.state_size <= t:
                self.extend_state(100)
                self.state_size += 100

            # compute conditional marginals
            for i, state_i in self.state.items():
                for j in state_i.keys():
                    nb_i_j = self.neighborhoods_i_except_j[i][j]
                    self.state[i][j][t] = _calculate_conditional_marginal(
                        self.state, i, j, nb_i_j, t, s, v, self.infection_prob, 
                        temporal=self.temporal, track_vaccinated=track_vaccinated
                    )

            if self.verbose:
                pbar.update(1)
                
            # check for convergence
            if self.convergence_check(t, threshold=convergence_threshold):
                if self.verbose:
                    pbar.close()
                    print(f"Converged early after {t} iterations")
                self.compute_marginals(s, t, track_vaccinated=track_vaccinated)
                return t

        if self.verbose:
            pbar.close()
            
        self.compute_marginals(s, self.t_max, track_vaccinated=track_vaccinated)
        raise RuntimeError(f"Message passing did not converge in {self.t_max} time steps.")


def _calculate_conditional_marginal(state, i, j, nb_i_j, t, s, v, infection_prob, temporal = False, track_vaccinated = False):
    """Compute the conditional marginal pi_{i/j}(t) for node i from precomputed neighborhood samples."""
    # compute \sum_{\gamma} p(i infected | \gamma) p(\gamma)
    prob_i_infected = 0
    for sample in nb_i_j.Gamma_samples:
        prob_i_infected += _prob_i_infected_given_gamma(
            state, i, t, infection_prob, sample, s, temporal = temporal
        ) * sample.prob
    # add seed status to the conditional marginal
    prob_i_infected = s[i] + (1 - s[i]) * prob_i_infected
    # set state to zero if node is vaccinated
    if not track_vaccinated:
        prob_i_infected *= (1 - v[i])
    
    return prob_i_infected
