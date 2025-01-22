
from pathlib import Path
import numpy as np
from collections import defaultdict
import pandas as pd
from ast import literal_eval
import sys
import os
import networkx as nx
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import load_graph

DIR = Path()
DATA = DIR / 'data'

#MARGINALS_DIR = DATA / 'marginals'
MARGINALS_DIR = DATA / 'marginals_T_sweep'
#DF_DIR = DATA / 'dataframes'
DF_DIR = DATA / 'dataframes/T_sweep'
with_idx = True

network_name = 'karate'

class ExpectationStat:

    def __init__(self):
        self.total = 0
        self.total2 = 0
        self.count = 0

    def add(self, x):
        assert isinstance(x, (int, float))
        self.total += x
        self.total2 += x**2
        self.count += 1

    def mean(self):
        return self.total / self.count
    
    def variance(self):
        return self.total2 / self.count - self.mean()**2
        
    
def parse(filepath, network_name=None):
    
    if filepath.suffix != ".npz":
        return dict()
    try:
        name = filepath.stem
        kvs = [x.split("=") for x in name.split('_')]
        
        d = {}
        #print(kvs)
        for k,v in kvs:
            if k in ['s','v']:
                d[k] = literal_eval(v)
                if isinstance(d[k], list):
                    d[k] = tuple(d[k])
            elif k == 'ip':
                d[k] = v
            elif k == "p":
                d[k] = float(v)
            else:
                d[k] = int(v)
            #d[k] = {k: float(v) if k=="p" else int(v) for k, v in kvs}
        d['filepath'] = filepath
        d['network'] = network_name
    except:
        print(f"Error parsing {filepath}")
        raise ValueError(f"Error parsing {filepath}")
    return d

def load_npz(filepath, full_marginals = False):
    d = np.load(filepath)
    if full_marginals:
        return d
    
    # save only marginals
    out = {}
    for k in d.keys():
        out[k] = d[k][:,-1]
    
    return out


def compute_sentinel_ts(marginals, sentinels, t_final_threshold = None):
    
    sentinel_ts = 1 - np.prod(1 - marginals[sentinels,:], axis = 0)

    if t_final_threshold is not None:
        margs = marginals.sum(axis = 0)
        prob_infections_at_t = np.diff(margs, prepend=0)
        # find the first element from the end that has a value > 0.01
        t_final = np.where(prob_infections_at_t > 0.05)[0][-1]
        sentinel_ts = sentinel_ts[:t_final+1]

    diff = np.diff(sentinel_ts, prepend=0)
    
    q = np.arange(len(diff)) # quality is the time the sentinel is found
    q = sum(diff*q) + len(diff)*(1 - sum(diff))

def main(ip):
    
    ####
    if ip == 'sentinel2':
        g = load_graph(DATA / f"edgelists/{network_name}.txt")
        sp = dict(nx.all_pairs_shortest_path_length(g))
        lsp = longest_shortest_path = max(sp[i][j] for i,j in g.edges())

    ##### read simulation data #####
    sim_outbreak_sizes = defaultdict(lambda :defaultdict(lambda :ExpectationStat()))
    sim_node_marginals = {k:defaultdict(lambda :defaultdict(lambda :ExpectationStat())) for k in [1,2]}
    for resultpath in (DATA / 'simulations' / network_name).iterdir():
        if resultpath.suffix != ".npz":
            continue
        
        params = parse(resultpath)
        if params['ip'] != ip:
            continue

        d = load_npz(resultpath,full_marginals=True if ip in ['sentinel','sentinel2'] else False)
        p = params['p']

        for s_str, marginals in d.items():
            
            s = s_str.split("=")[1]
            s_ = literal_eval(s)
            k = 1 if isinstance(s_,int) else len(s_)

            if ip in ['influencemax','vaccination']:
                total_outbreak_size = marginals.sum()
            elif ip == 'sentinel':
                sentinel_ts = marginals
                # diff[i] = <probability a sentinel is infected (or outbreak ended) at time t>
                diff = np.diff(sentinel_ts, prepend=0)
                q = np.arange(len(diff))
                q = sum(diff*q)
                total_outbreak_size = q
            elif ip == 'sentinel2':
                sentinel_ts = marginals
                diff = np.diff(sentinel_ts, prepend=0)
                q = np.arange(len(diff))
                q = sum(diff*q)
                total_outbreak_size = q + (1 - sum(diff))*lsp #np.sqrt(len(g))
            else:
                raise ValueError(f"Invalid ip: {ip}")

            sim_outbreak_sizes[p][s].add(total_outbreak_size)
            for i in range(marginals.shape[0]):
                sim_node_marginals[k][p][i].add(marginals[i])

    # print(sim_outbreak_sizes[0.05]['0'])
    # print(sim_node_marginals[0.05][0])

    ##### read message passing data #####
    #Ts = [10,20,200,1000]
    Ts = np.floor(np.linspace(1,10,25)**2).astype(int).tolist()
    Ts += np.floor(np.linspace(5,32,25)**2).astype(int).tolist()
    
    if with_idx:
        idx_list = np.arange(20)
        nmp_outbreak_sizes = {r: {T: {idx:defaultdict(lambda :defaultdict(lambda :ExpectationStat())) for idx in idx_list} for T in Ts} for r in [0,1,2]}
    else:
        nmp_outbreak_sizes = {r: {T:defaultdict(lambda :defaultdict(lambda :ExpectationStat())) for T in Ts} for r in [0,1,2]}
    nmp_node_marginals = {r: { T: {k:defaultdict(lambda :defaultdict(lambda :ExpectationStat())) for k in [1,2]} for T in Ts}  for r in [0,1,2]}

    for resultpath in (MARGINALS_DIR / network_name).iterdir():
        if resultpath.suffix != ".npz":
            continue
        
        params = parse(resultpath)
        if ip == 'sentinel2':
            if params['ip'] != 'sentinel':
                continue
        elif params['ip'] != ip:
            continue

        d = load_npz(resultpath,full_marginals=True if ip in ['sentinel','sentinel2'] else False)
        r = params['r']
        T = params['T']
        p = params['p']

        for s_str, marginals in d.items():
            total_outbreak_size = marginals.sum()
            s = s_str.split("=")[1]
            s_ = literal_eval(s)
            k = 1 if isinstance(s_,int) else len(s_)

            if ip in ['influence_max','vaccination']:
                total_outbreak_size = marginals.sum()
            elif ip == 'sentinel':
                
                # get sentinels
                if isinstance(s_,int):
                    sentinels = [s_]
                else:
                    sentinels = s_

                # compute probability sentinel found at time t
                sentinel_ts = 1 - np.prod(1 - marginals[sentinels,:], axis = 0)
                diff = np.diff(sentinel_ts, prepend=0)
                #print("diff",diff)

                # choose approximate final time for infection based on a threshold
                threshold = 0.01

                prob_infections_by_t = 1 - np.prod(1 - marginals,axis=0)
                #margs = marginals.sum(axis=0)
                prob_infections_at_t = np.diff(
                    prob_infections_by_t,
                    #margs, 
                    prepend=0)
                # find the first time from the end that has a probability of an infection > threshold
                t_final = np.where(prob_infections_at_t > threshold)[0][-1]
                
                #### option 1 #######
                # #truncate sentinel time series
                sentinel_ts = sentinel_ts[:t_final+1]
                
                #diff = np.diff(sentinel_ts, prepend=0)
                #q = np.arange(len(diff)) # quality is the time the sentinel is found
                # expected quality + t_final*<probability of not finding sentinel>
                #q = sum(diff*q) + len(diff)*(1 - sum(diff))
                
                ###### option 2 #######
                # prob_sentinels_not_infected = 1 - sum(diff)
                # #print("prob_sentinels_not_infected",prob_sentinels_not_infected)
                # q = t_final*prob_sentinels_not_infected
                # #print(q)
                # q += sum(diff*np.arange(len(diff)))
                # #print(q)

                ###### option 3 #######
                y = np.diff(marginals,axis=1,prepend=0)
                y2 = 1 - np.prod(1 - y, axis=0)
                #print("sum m2", sum(m2))
                avg_t_final = sum(np.arange(len(y2))*(y2/sum(y2)))

                q = np.arange(len(diff)) # quality is the time the sentinel is found    
                total_outbreak_size = sum(diff*q) + \
                    t_final*(1 - sum(diff))

            elif ip == 'sentinel2':
                # get sentinels
                if isinstance(s_,int):
                    sentinels = [s_]
                else:
                    sentinels = s_
                # compute infection time series
                sentinel_ts = 1 - np.prod(1 - marginals[sentinels,:], axis = 0)
                diff = np.diff(sentinel_ts, prepend=0)
                # get quality value
                q = np.arange(len(diff))
                q = sum(diff*q) + (1 - sum(diff))*lsp
                total_outbreak_size = q


            if with_idx:
                nmp_outbreak_sizes[r][T][params['idx']][p][s].add(total_outbreak_size)
            else:
                nmp_outbreak_sizes[r][T][p][s].add(total_outbreak_size)
            if ip not in ['sentinel','sentinel2']:
                for i in range(marginals.shape[0]):
                    nmp_node_marginals[r][T][k][p][i].add(marginals[i])

    def default2dict(d):
        if isinstance(d,(dict,defaultdict)):
            return {k:default2dict(v) for k,v in d.items()}
        return d

    #convert all data
    sim_outbreak_sizes = default2dict(sim_outbreak_sizes)
    sim_node_marginals = default2dict(sim_node_marginals)
    nmp_outbreak_sizes = default2dict(nmp_outbreak_sizes)
    nmp_node_marginals = default2dict(nmp_node_marginals)

    print(len(nmp_outbreak_sizes))

    ##### compare outbreak sizes #####
    if with_idx:
        data = []
        for r in [0,1,2]:
            for T in Ts:
                for idx in idx_list:
                    for p in nmp_outbreak_sizes[r][T][idx].keys():
                        if p not in sim_outbreak_sizes.keys():
                            print(f"ip={ip}, p={p} not in simulation data.")
                            continue
                        for s in nmp_outbreak_sizes[r][T][idx][p].keys():
                            nmp = nmp_outbreak_sizes[r][T][idx][p][s]
                            if s not in sim_outbreak_sizes[p].keys():
                                continue
                            sim = sim_outbreak_sizes[p][s]
                            
                            d = {}
                            d['r'] = r
                            d['T'] = T
                            d['idx'] = idx
                            d['p'] = p
                            d['s'] = s
                            d['pi_S_nmp'] = nmp.mean()
                            d['pi_S_sim'] = sim.mean()
                            d['e_S'] = nmp.mean() - sim.mean()
                            d['e_S^2'] = nmp.variance() + sim.variance()
                            data.append(d)
    else:
        data = []
        for r in [0,1,2]:
            for T in Ts:
                for p in nmp_outbreak_sizes[r][T].keys():
                    if p not in sim_outbreak_sizes.keys():
                        print(f"ip={ip}, p={p} not in simulation data.")
                        continue
                    for s in nmp_outbreak_sizes[r][T][p].keys():
                        nmp = nmp_outbreak_sizes[r][T][p][s]
                        if s not in sim_outbreak_sizes[p].keys():
                            continue
                        sim = sim_outbreak_sizes[p][s]
                        
                        d = {}
                        d['r'] = r
                        d['T'] = T
                        d['p'] = p
                        d['s'] = s
                        d['pi_S_nmp'] = nmp.mean()
                        d['pi_S_sim'] = sim.mean()
                        d['e_S'] = nmp.mean() - sim.mean()
                        #d['e_S^2'] = nmp.variance() + sim.variance()
                        data.append(d)
    
    df = pd.DataFrame(data)
    df.to_csv(DF_DIR /f'{network_name}_ip={ip}_e_S_data.csv', index=False)

    ##### compare node marginals #####
    if ip != 'sentinel':
        data = []
        for r in [0,1,2]:
            for T in Ts:
                for k in [1,2]:
                    for p in nmp_node_marginals[r][T][k].keys():
                        if p not in sim_node_marginals[k].keys():
                            print(f"ip = {ip}, p={p} not in simulation data.")
                            continue
                        for i in nmp_node_marginals[r][T][k][p].keys():
                            nmp = nmp_node_marginals[r][T][k][p][i]
                            sim = sim_node_marginals[k][p][i]
                            
                            d = {}
                            d['r'] = r
                            d['T'] = T
                            d['p'] = p
                            d['i'] = i
                            d['pi_i_nmp'] = nmp.mean()
                            d['pi_i_sim'] = sim.mean()
                            d['e_i'] = nmp.mean() - sim.mean()
                            d['e_i^2'] = nmp.variance() + sim.variance()
                            data.append(d)
        df = pd.DataFrame(data)
        df.to_csv(DF_DIR /f'{network_name}_ip={ip}_e_i_data.csv', index=False)

    ##### compute rank preservation ######

if __name__ == "__main__":
    main('influencemax')
    main('vaccination')
    main('sentinel2')
    #main('sentinel')
    