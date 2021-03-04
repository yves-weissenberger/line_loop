import numpy as np
import networkx as nx
import itertools
#This file contains functions that are useful to optimizing the 
#design of the behavioral tasktransition structure



#######################################################################################
# ------ Functions for optimizing the graph layout
#######################################################################################





def find_shortest_reward_loop(gw):
    """ Find the shortest loop back to reward """
    if not hasattr(gw,'G'):
        gw.build_nx_graph()
    rewIx = int(np.where(gw.rewards)[0])
    t1,t2 = gw.edge_transitions[rewIx]
    shortest_path = np.min([nx.shortest_path_length(gw.G,source=t1,target=rewIx),
                            nx.shortest_path_length(gw.G,source=t2,target=rewIx)])
    return shortest_path





#######################################################################################
# ------ Functions for optimizing the physical layout
#######################################################################################

poke_pos = [  [265,456],
         [184,475],[347,475],
    [116,518],[265,518],[414,518],
         [184,561],[347,561],
              [265,580]]



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


def find_physical_edge_lengths(edges,pos):
    """ This function takes in the edges of the graph as well as the physical position of
        the nodes and returns the distribution of physical distances between nodes
    """
    pos = np.array(pos).astype('float')
    
    dists = []
    for edge in edges:
        dists.append(np.sum(np.abs(pos[edge[0]],pos[edge[1]])))
    return np.array(dists)


def find_number_of_turnarounds(edge_transitions,pos):
    """ This function takes in the edges of the graph as well as the physical position of
        the nodes and find out how often mice need to change physical direction
    """
    
    nNodes = len(pos)
    
    paths = []
    for node in range(nNodes):
        
        outputs = edge_transitions[node]             #these are nodes that node proejcts to
        inputs = np.where(gw.edge_transitions==node)[0] #these are nodes that project to node 
        
        paths = paths + [[i,node,j] for i in inputs for j in outputs]

    all_angles = []
    for p in paths:
        e1,c,e2 = p
        all_angles.append(angle_between(pos[e1],pos[c],pos[e2]))

    
    return np.array(all_angles),np.array(paths)


def angle_between(e1,c,e2):

    ba = e1 - c
    bc = e2 - c

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


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