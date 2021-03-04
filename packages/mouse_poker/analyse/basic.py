import numpy as np
import itertools
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import seaborn

poke_pos = np.array([1,-1])* np.array( [  [149,0],
                                     [68,19],[231,19],
                                   [0,62],[149,62],[298,62],
                                     [68,105],[231,105],
                                          [149,124]])


def get_binned_pokes(events,event_times,binsize=50):
    
    """ Function that returns binned poke times in an 
        nPokes x nTimepoints array. binsize argument is
        in milliseconds
    """
    inPoke_events=  ['poke_'+str(i) for i in range(1,10)]
    tot_ms = int(np.max(event_times)*1000)
    y = np.zeros([9,tot_ms])
    for kk,iPke in enumerate(inPoke_events):
        ixs = np.where(events==iPke)[0]
        #print(ixs)
        y[kk,(1000*event_times[ixs]).astype("int")-1] = 1
    
    y_clip = y[:,:int(binsize*np.floor(tot_ms/binsize))]
    Y = y_clip.reshape(9,-1,binsize).sum(axis=2) #checked axis of reshaping


    poke_seq = [np.where(Y[:,i])[0][0] for i in np.where(Y.sum(axis=0))[0]]
    return Y,poke_seq



def get_distribution_of_poke_times(events,event_times):
    inPoke_events=  ['poke_'+str(i) for i in range(1,10)]
    ts = []
    for kk,event in enumerate(events):
        if event in inPoke_events:
            #print(event)
            tmp = np.where(np.array(events)[kk:]==events[kk] + '_out')
            #print(tmp)
            if tmp:
                outIx = np.argmin(tmp[0]) + kk + 1
                #print(kk,outIx)
                deltaT = event_times[outIx] - event_times[kk]
                ts.append(deltaT)
    return ts



def get_transition_overview(y,dat_dict,valid_transitions_only=True):

    """ Visualise"""
    
    all_edges = list(itertools.product(*[range(9),range(9)]))
    
    if valid_transitions_only:
        poke_seq = []
        for i in dat_dict['port']:
            poke_seq.append(i[0])
    else:
        poke_seq = [np.where(y[:,i])[0][0] for i in np.where(y.sum(axis=0))[0]]
    
    seq_counter = [0]*len(all_edges)
    for p1,p2 in zip(poke_seq[:-1],poke_seq[1:]):

        for kk,nP in enumerate(all_edges):
            if (p1,p2)==nP:
                seq_counter[kk] += 1
    seq_counter2 = np.array(seq_counter).reshape(-1,9)
    edge_weights = (seq_counter2/np.sum(seq_counter2,axis=0)).flatten(order='F')
    edge_weights[np.isnan(edge_weights)] = 0
    #print(edge_weights)
    
    
    tmp1 = [i[0] for i in dat_dict['port']]
    poke_count = [tmp1.count(i)/float(len(tmp1)) for i in range(9)]

    return all_edges,edge_weights, poke_count, seq_counter

def plot_transition_overview(y,dat_dict,valid_transitions_only=True,cbar='edges'):

    all_edges,edge_weights, poke_count,_ = get_transition_overview(y,dat_dict,
                                                                 valid_transitions_only=valid_transitions_only)
    
    
    G = nx.DiGraph()
    for i in range(9):
        G.add_node(i,pos=(poke_pos[i][0],poke_pos[i][1]))

    for e,w in zip(all_edges,edge_weights):
        G.add_edge(e[0],e[1],weight=10*w)
        

    nx.draw_networkx_nodes(G,pos=poke_pos,node_color=poke_count,cmap=matplotlib.cm.Reds,vmin=0,vmax=1.)
    


    nx.draw_networkx_edges(G,poke_pos,
                           edge_color=edge_weights,
                           edge_vmin=0,edge_vmax=1.,
                           connectionstyle='arc3, rad=0.1',
                           edge_cmap=matplotlib.cm.Blues)
    
    if cbar=='edges':
        cmap = matplotlib.cm.Blues
    else:
        cmap = matplotlib.cm.Reds
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = 0, vmax=1))
    sm._A = []
    plt.colorbar(sm)

    seaborn.despine(bottom=True,left=True)
    return edge_weights