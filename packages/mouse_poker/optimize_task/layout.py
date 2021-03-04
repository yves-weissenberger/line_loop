import numpy as np
import itertools
import networkx as nx
import matplotlib.pyplot as plt

poke_pos = np.array([    [149,0],
                     [68,19],[231,19],
                   [0,62],[149,62],[298,62],
                     [68,105],[231,105],
                          [149,124]])

def get_all_valid_physical_layouts(rew_nodes,nNodes):
    
    """ Takes props of graph and returns all possible permutations of node locations
        respecting that rewards should be at the edges
        
        Checked by making sure rewards stay at edges and outputs are unique
    """
    
    Rlocs = np.array([0,3,5,8])
    Rmaps = list(itertools.permutations(rew_nodes))
    
    
    nonRlocs = np.array([i for i in range(9) if i not in Rlocs])
    nonR = [i for i in range(nNodes) if i not in rew_nodes]
    nonR_maps = list(itertools.permutations(nonR))
    
    orders = []
    
    for rMap in Rmaps:
        
        for nrMap in nonR_maps:
            
            order = np.zeros(nNodes)
            order[np.array(rMap)] = Rlocs

            order[np.array(nrMap)] = nonRlocs
            
            orders.append(order.copy().astype('int'))
    
    return orders



def get_transition_set(links,nStates=9):
    """ Takes as input the list of edges and edges and returns an array 
        that gives the transitions available as a function of state
    """
    edge_set = [[] for _ in range(nStates)]#np.zeros([self.nNodes,self.nActions],dtype='int')
    cntArr = [0]*nStates#np.zeros(self.nNodes,dtype='int')
    for fst,snd in links:
        edge_set[fst].append(snd)
        cntArr[fst] += 1

    return edge_set

def find_physical_edge_lengths(edges,pos):
    """ This function takes in the edges of the graph as well as the physical position of
        the nodes and returns the distribution of physical distances between nodes
    """
    pos = np.array(pos).astype('float')
    
    dists = []
    for edge in edges:
        dists.append(np.sum(np.abs(pos[edge[0]],pos[edge[1]])))
    return np.array(dists)

def angle_between(e1,c,e2):

    ba = e1 - c
    bc = e2 - c

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def get_all_edge_angles_and_dists(edge_set,layouts,poke_pos=poke_pos):
    all_angle = []
    all_dists = []

    for lyt in layouts:
        tmpA = []
        tmpD = []

        for kk,e in enumerate(edge_set):
            if len(e)==2:
                tmp1 = angle_between(np.array(poke_pos[lyt[e[0]]]),
                                                np.array(poke_pos[lyt[kk]]),
                                                np.array(poke_pos[lyt[e[1]]]))
                tmpA.append(np.abs(tmp1%180))

                d_ = np.abs(np.array(poke_pos[lyt[e[0]]]) - np.array(poke_pos[lyt[e[1]]]))
                tmpD.append(d_)
        all_angle.append(tmpA.copy())
        all_dists.append(tmpD.copy())
    return all_angle, all_dists


def graph_same(layout1,layout2):
    """ Check if two graphs have the same layout"""
    
    flp = lambda x,y: flip(x,y)
    
    layout1 = np.array(layout1)
    layout1_rev = np.flipud(layout1)
    layout2_ = np.array(layout2).copy()
    
    for rot in range(4):
        
        if np.logical_or.reduce([np.all(layout1==layout2_) or 
                                 np.all(layout1==flp(layout2_,'V')) or 
                                 np.all(layout1==flp(layout2_,'H')) or
                                 np.all(layout1==flp(flp(layout2_,'H'),'V')),
                                 np.all(layout1_rev==layout2_) or 
                                 np.all(layout1_rev==flp(layout2_,'V')) or 
                                 np.all(layout1_rev==flp(layout2_,'H')) or
                                 np.all(layout1_rev==flp(flp(layout2_,'H'),'V'))]):
            return True
        
        
        layout2_ = np.array(rotate(layout2_))
        


    if np.logical_or.reduce([np.all(layout1==layout2_) or 
                             np.all(layout1==flp(layout2_,'V')) or 
                             np.all(layout1==flp(layout2_,'H')) or
                             np.all(layout1==flp(flp(layout2_,'H'),'V')),
                             np.all(layout1_rev==layout2_) or 
                             np.all(layout1_rev==flp(layout2_,'V')) or 
                             np.all(layout1_rev==flp(layout2_,'H')) or
                             np.all(layout1_rev==flp(flp(layout2_,'H'),'V'))]):
        return True
    else:
        return False


def check_set_of_graphs_uniq(initial_set,layouts):
    """ Check if a set of graphs is the same """
    final_set = []
    for g in initial_set:
        uniq = True
        for g2 in final_set:
            if graph_same(layouts[g],layouts[g2]):
                uniq = False
                #break
        if uniq:
            final_set.append(g)
    return final_set, len(final_set)==len(initial_set)

def flip(layout,axis='V'):
    if axis=='V':
        #defines new position of nodes after flip, 
        #node 0 should go to pos 8; node 1 to pos 6 etc
        flip = [8,6,7,3,4,5,1,2,0]
    elif axis=='H':
        flip = [0,2,1,5,4,3,7,6,8]
    else:
        raise ValueError('can only flip vertically ("V")or horizontally ("H")')
        
    layout2 = [flip[i] for i in layout]
    return layout2


def rotate(layout,direction='cw'):
    if direction=='cw':
        #defines new position of nodes after rotation, 
        #node 0 should go to pos 5; node 1 to pos 2 etc
        rot = [5,2,7,0,4,8,1,6,3]  
    elif direction=='ccw':
        rot = [3,6,1,8,4,0,7,2,5]
    else:
        raise ValueError('Can only rotate the graph clockwise ("cw") or counterclockwise ("ccw")')
    
    layout2 = [rot[i] for i in layout]
    return layout2


# Fairly fast for many datapoints, less fast for many costs, somewhat readable
def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    from: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient



def select_graphs_by_percentile(costs,percentile=60):
    
    """ Returns indices of graphs which are in the top 'percentile'
        percentile according to all costs
    """
    pct = 100-percentile
    mus = np.array([np.percentile(i,pct,) for i in costs])

    goodIxs = np.where(np.logical_and.reduce([i<=j for i,j in zip(costs,mus)]))[0]
    badIxs = np.array([i for i in range(len(costs[0])) if i not in goodIxs])
    return goodIxs,badIxs,mus


##################################################################################################
# Plots
##################################################################################################

def plot_bunch_o_graphs(plot_ixs,layouts,edges,mu_angle=None,var_angle=None,var_dist=None,nX=4,nY=6,check_uniq=True):
    """ Plot the graphs in """
    
    if mu_angle==None:  mu_angle = np.zeros(len(layouts))
    if var_angle==None: var_angle = np.zeros(len(layouts))
    if var_dist==None:  var_dist = np.zeros(len(layouts))

    G = nx.DiGraph()
    G.add_edges_from(edges)

    plt.figure(figsize=(17,22))
    ijk = 1
    seen_gfs = []
    for ix in plot_ixs:
        seen = False
        for i in seen_gfs:
            seen = seen or graph_same(i,layouts[ix])
        if not seen:
            if check_uniq:
                seen_gfs.append(np.array(layouts[ix]))
            plt.subplot(nY,nX,ijk)
            plt.gca().annotate('ix:{:.0f} | minA:{:.1f} | stdA:{:.1f} | stdD:{:.1f}'.format(ix,mu_angle[ix],var_angle[ix],var_dist[ix]),
                               [-.1,1.1],xycoords='axes fraction',fontsize=12)
            nx.draw(G,
                pos=np.array(poke_pos)[np.array(layouts[ix])],#[np.random.permutation(range(9))],
                connectionstyle='arc3, rad=0.0',
                with_labels=True)

            ijk += 1


def convert_to_taskFile_graph(ly):
    """ This function converts from the layout indices
        by how the graph should be plotted for the
        networkx library to correct layout for pokes in 
        the pycontrol task file
    """
    return [flip(ly,'V').index(i) for i in range(9)]


def plot_param_stats(mu_angle,var_angle,var_dist):
    plt.figure(figsize=(18,10))
    plt.subplot(2,3,1)
    plt.title("min(A)")
    seaborn.distplot(mu_angle,kde=0)
    plt.ylabel("# of graphs")
    plt.xlabel("Min angle between choices")

    plt.subplot(2,3,2)
    plt.title("var(A)")
    seaborn.distplot(var_angle,kde=0)
    plt.xlabel("var angle between choices")

    plt.subplot(2,3,3)
    plt.title("var(D)")
    seaborn.distplot(var_dist,kde=0)
    plt.xlabel("var dist between choices")



    plt.subplot(2,3,4)
    plt.scatter(mu_angle,var_angle)
    plt.ylabel("Var angle between choices")
    plt.xlabel("Mean angle between choices")

    plt.subplot(2,3,5)
    plt.scatter(mu_angle,var_dist)
    plt.ylabel("Var distance between choices")
    plt.xlabel("Mean angle between choices")

    plt.subplot(2,3,6)
    plt.scatter(var_dist,var_angle)
    plt.ylabel("Var angle between choices")
    plt.xlabel("Var distance between choices")

    plt.tight_layout()

    seaborn.despine()
    return None
