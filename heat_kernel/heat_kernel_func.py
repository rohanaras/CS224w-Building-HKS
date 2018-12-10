import networkx as nx
import pandas as pd
import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib
import matplotlib.pyplot as plt
import os
import time

def k_neighbors(points, weighted=True, k=3):
    '''
    weights are distance
    returns a weighted edgelist
    '''
    num_points = points.shape[0]
    
    # expanded out form of distance formula
    squared = (points ** 2).sum(axis=1, keepdims=True)
    mixed = points @ points.T
    dists = np.sqrt(squared + squared.T - (2 * mixed))

    closest_idx = np.argsort(dists, axis=1)[:, 1:(k + 1)]
    
    edge_list = []
    for i in range(num_points): # this can probably be parallelized
        cdists = dists[i, closest_idx[i]]
        if weighted:
            edge_list.extend([(i, j, cdists[k]) 
                              for k, j in enumerate(closest_idx[i])])
        else:
            edge_list.extend([(i, j, 1) 
                              for k, j in enumerate(closest_idx[i])])

    
    del dists
    del closest_idx
    
    return edge_list


def hks_s(G, ts, x, k=300, verbose=False, eigen_save_loc=None):
    """
    sparse version of hks
    
    G is a graph
    ts is a list of t (time)
    x is a list of node ids 
    k is the number of eigenvalue/vectors to use (starting from the first)
    eigen_save_loc is the place to save/load the eigen vectors/values from
    """
    if G.number_of_nodes() < k:
        k = G.number_of_nodes() - 1
        
    if type(ts) is not list:
        ts = [ts]
        
    try:  # check if eigen stuff is saved to avoid recomputing
        eigval_file_exists = os.path.isfile(eigen_save_loc + '_%d_values.npy' % k)
        eigvec_file_exists = os.path.isfile(eigen_save_loc + '_%d_vectors.npy' % k)
    except TypeError:
        eigval_file_exists, eigvec_file_exists, = False, False
        
    if eigval_file_exists and eigvec_file_exists:
        if verbose:
            t0 = time.time()
            print('Loading eigen values/vectors from file...')
        eig_val = np.load(eigen_save_loc + '_%d_values.npy' % k)
        eig_vec = np.load(eigen_save_loc + '_%d_vectors.npy' % k)
        if verbose:
            t1 = time.time()
            print('done! (%.2fs)' % (t1 - t0))
            
    else:
        if verbose:
            t0 = time.time()
            print('Calculating eigen values/vectors...')

        # L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes)).asfptype()
        L = nx.normalized_laplacian_matrix(G, nodelist=sorted(G.nodes)).asfptype()
        eig_val, eig_vec = eigsh(L, k)
        # eig_val, eig_vec = np.linalg.eig(L)

        if eigen_save_loc is not None:
            np.save(eigen_save_loc + '_%d_values' % k, eig_val)
            np.save(eigen_save_loc + '_%d_vectors' % k, eig_vec)

        if verbose:
            t1 = time.time()
            print('done! (%.2fs)' % (t1 - t0))
    
    hks_time_dict = {}
    for t in ts:  # compute hks for all times requested
        if type(x) is list:
            hks_list = {}
            for i in x:  # compute hks for all nodes requested
                phi_x = eig_vec[i, :]
                hks = np.sum(np.exp(-eig_val * t) * phi_x * phi_x)
                hks_list[i] = hks
                                    
            hks_time_dict[t] = hks_list
        else:  # compute hks for a single requested node
            phi_x = eig_vec[x, :]
            hks_time_dict[t] = np.sum(np.exp(-eig_val * t) * phi_x * phi_x)
            
        if verbose:
            print('Done with time %s...' % t)

            
    if verbose:
        t2 = time.time()
        print('Finished in %.2f seconds' % (t2 - t0))
    return hks_time_dict


def set_hks_attr(G, t):
    """
    Set the 'hks' attribute of Graph G with the hks calculated at time t
    DEPRECATED as is 
    """
    hks_dict = {node: hks(G, t, node) for node in G.nodes}
    nx.set_node_attributes(G, hks_dict, 'hks')
    
    
def compute_persistence(G, tau=np.inf, f_name='hks_s', f=None, verbose=False):
    """
    (Rohan's first attempt: incorrect)
    See "Persistence-based Segmentation of Deformable Shapes", Section 3
    """
    if f is None:
        f = nx.get_node_attributes(G, f_name)
    C = {node: {'parent': node, 'persistence': 0} for node in G.nodes()}
    roots = set(C.keys())
    
    PD_pairs = []
    
    def find(x, verbose=False):
        if verbose: print('find start', x, C[x])
        if C[x]['parent'] != x:
            C[x] = find(C[x]['parent'])
        if verbose: print('find end', x, C[x])
        return C[x] # return both the root, and persistence encoded at root
    
    num_iter = 0
    num_changes = -1
    while num_changes != 0:
        num_iter += 1        
        num_changes = 0
        
        f_subset = {k: f[k] for k in roots} # limit keys to local maxima
        sorted_keys = sorted(f_subset, key=lambda k: f[k], reverse=True)
        
        for x in sorted_keys:  # assign all nodes to a component
            # for nodes that have root x, union neighbors
            ys = set.union(*[set(G.neighbors(i)) for i in C.keys() if C[i]['parent']==x])
            ys = sorted(list(ys), key=lambda y: f[find(y)['parent']], reverse=True)
                        
            for y in ys:
                parent, persistence = find(y).values() 
                
                if x == parent:
                    break

                if f[x] > f[parent]: # max_fy:
                    if verbose: print('f(x) greater:', x, f[x], y, parent, f[parent])
                    break  # since x should be its own componenet

                if verbose: print('f(x) less/equal:', x, f[x], y, parent, f[parent])
                if verbose: print('C[%s]:'% y, C[y])

                if persistence < tau: # if > tau, segment into multiple clusters
                    num_changes += 1
                    roots.remove(x)
                    
                    PD_pairs.append((f[x], f[parent]))

                    if f[parent] - f[x] > persistence: 
                        # find new persistence of component
                        persistence = f[parent] - f[x]   
                    # set new parent, persistence of component it is in
                    C[x] = {'parent': parent, 'persistence': persistence}
                    # reset persistence of component at root
                    C[parent]['persistence'] = persistence

                    if verbose: print('assigned:', x, C[x])
                    break # done looking through y's
            if verbose: print()

        if verbose: print(len([k for k in C.keys() if k != C[k]['parent']]) / len(C.keys()))
        if verbose: print([k for k in C.keys() if k == C[k]['parent']])
    print('Finished with %s components in %s iterations' %
          (len([k for k in C.keys() if k == C[k]['parent']]), num_iter))
    return C, PD_pairs 

def persistence(G, tau=np.inf, f_name='hks_s', f=None, verbose=False):
    """
    Andrew's first attempt at implementation of the persistence-based 
    clustering algorithm.

    MOST LIKELY WRONG
    """
    if f is None:
        f = nx.get_node_attributes(G, f_name)
    C = {node: {'parent': node, 'persistence': 0} for node in G.nodes()}
    roots = set(C.keys())
    PD_pairs = []
    
    num_iter = 0
    num_changes = -1

        
    f_subset = {k: f[k] for k in roots}
    sorted_keys = sorted(f_subset, key=lambda k: f[k], reverse=True)

    for x in sorted_keys:
    	nbrs = G.neighbors(x)
    	
    	max_node = x
    	for nbr in nbrs:
    		if f[nbr] > f[x]:
    			max_node = nbr

    	if max_node is not x and C[x]['persistence'] < tau:
    		num_changes += 1
    		PD_pairs.append((f[x], f[max_node]))

    		roots.remove(x)
    		persistence = C[x]['persistence']
    		if f[max_node] - f[x] > persistence:
    			persistence = f[max_node] - f[x]
    		C[x] = {'parent': max_node, 'persistence': persistence}
    		C[max_node]['persistence'] = persistence
    return C, PD_pairs

def persistence_diagram(G, f_name='hks_s', f=None, tau=np.inf, verbose=False):
    """
    (correct version)
    """
    if f is None:
        f = nx.get_node_attributes(G, f_name)
    C = {}  # parent node for each node (defines clusters)
    PD_pairs = {}

    nodes = {k: f[k] for k in G.nodes()}  # hks values for each node

    # "We then process the vertices in decreasing value of f"
    sorted_keys = sorted(nodes, key=lambda k: f[k], reverse=True)  

    for x in sorted_keys:
        if verbose: print('=' * 10 + str(x) + '=' * 10)

        # "we first determine if it is a local maximum in the mesh by comparing
        # f(x) with f(y) for all y in a one-ring neighbor hood of x"
        nbrs = list(G.neighbors(x))
        
        # "If x is a local maximum, a new component is born and the vertex is
        # assigned to itself in the segmentation, C(x)=x."
        max_nbr = x
        for nbr in nbrs:
            # "If x is not a local maximum, we assign it to the neighbor with the
            # highest function value."
            if f[nbr] > f[x]: max_nbr = nbr
        C[x] = max_nbr  # Set parent

        if verbose: print('parent of %s: %s' % (x, C[x]))
        if verbose: print('own parents: %s' % [node for node, parent in C.items() if node==parent])

        # roots of neighbors that are part of components
        local_comps = set(get_root(node, C) for node in set(nbrs).intersection(set(C.keys())))

        if verbose: 
            print('nbrs: %s' % {nbr: (get_root(nbr, C) if nbr in C else None) for nbr in nbrs })
            print('all nodes in component: %s' % list(C.keys()))
            print('local components: %s' % local_comps)

        # "If the vertex is adjacent to two or more existing components"
        if len(local_comps) > 1: # should this part be done even if C(x)=x?
            # sort by (birth) hks value of root (descending)
            sorted_comps = sorted(list(local_comps), key=lambda k: f[get_root(k, C)], reverse=True)  

            length = len(sorted_comps)
            biggest_nbrcomp = get_root(sorted_comps[0], C)
            for i in range(1, length):  # loop through all neighbor components except
                                        # the one with the largest hks (everything is
                                        # compared to it)
                                        # TODO: it's possible that once tau is exceeded, 
                                        # the following components should be compared to 
                                        # the one that exceeded tau
                curr_nbrcomp = get_root(sorted_comps[i], C)

                # "we check the persistence of the components and merge them only if
                # they are not τ-persistent"
                if verbose: print(sorted_comps[0], biggest_nbrcomp, '|', sorted_comps[i], curr_nbrcomp)
                if  f[biggest_nbrcomp] - f[curr_nbrcomp] < tau:
                    # "To merge two segments with maxima x1 and x2 such that
                    # f(x1)<f(x2), we set C(x1)=x2"
                    if verbose: print(True)
                    C[curr_nbrcomp] = biggest_nbrcomp  

                    # "When computing the PD, every time we merge two components,
                    # we output the pair (f(x1),f(x)), where x1 is the maximum with 
                    # the smaller value of f and x is the point currently being processed."
                    # The implementation below is reverse of what they have in the paper
                    # so that the points appear above the diagonal line 
                    PD_pairs[(f[x], f[curr_nbrcomp])] = curr_nbrcomp
                # else: # This is an attempt at the above todo
                #     biggest_nbrcomp = get_root(sorted_comps[i], C)

    if verbose: print('=+=' * 10)
    return C, PD_pairs


def plot_PD(PD_points, tau=None):
    def abline(slope, intercept):
        """Plot a line from slope and intercept"""
        axes = plt.gca()
        xlim = axes.get_xlim()
        ylim = axes.get_ylim()
        x_vals = np.array((min(xlim[0], ylim[0]), max(xlim[1], ylim[1])))
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, '--')

    plt.scatter(*zip(*PD_points), alpha=0.2)
    plt.axis('equal')
    abline(1, 0)

    if tau is not None:
        abline(1, tau)
       
def get_root(node, C):
    parent = C[node]
    while node != parent:
        node = parent
        parent = C[node]
    root = parent
    return root


def orig_plot_segments(G, C, pos=None, labels=False, **kwargs):
    all_roots = set(get_root(node, C) for node in C)
    cmap = {c: plt.cm.get_cmap('tab20', 20)(i % 20) 
            for i, c in enumerate(all_roots)}
    colors = [cmap[get_root(node, C)] for node in G.nodes()]
    nx.draw(G, pos=pos, node_color=colors, with_labels=labels, **kwargs)
    plt.show()


def plot_segments(G, C, cmap=None, pos=None, labels=False, cutoff=3, **kwargs):
    all_roots = set(get_root(node, C) for node in C)
    
    if cmap is None:
        cmap = {c: -1 for c in all_roots}
        possible_colors = {c: set(range(20)) for c in all_roots}

        clusters = {c: set() for c in all_roots}
        for node in C:
            clusters[get_root(node, C)].add(node)

        for root, cluster in clusters.items():
            neighbors = set()
            for node in cluster:
                neighbors.update(nx.single_source_shortest_path_length(G, node, cutoff))
            neighbors.difference_update(cluster)
            for neighbor in neighbors:
                neighbor_root = get_root(neighbor, C)
                if cmap[neighbor_root] >= 0 and cmap[neighbor_root] in possible_colors[root]:
                    possible_colors[root].remove(cmap[neighbor_root])
                if len(possible_colors[root]) == 0:
                    raise Exception("Can't Naively find Color Solution")
            cmap[root] = next(iter(possible_colors[root]))
            # print(cmap)

    colors = [plt.cm.get_cmap('tab20', 20)(cmap[get_root(node, C)]) for node in G.nodes()]
    nx.draw(G, pos=pos, node_color=colors, with_labels=labels, **kwargs)
    plt.show()
    return cmap