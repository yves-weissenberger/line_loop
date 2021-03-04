import numpy as np
import re
import networkx as nx


class graphworld(object):
    """ To do list:
            incorporate the random transitions
            make the final state absorbing
    """
    def __init__(self,graph_spec,nNodes=9,nActions=2,teleport_probability=.0,build_nx_graph=False):
        
        self.graph_spec = graph_spec
        self.edges, self.edge_transitions = self.read_gunnar_graph(self.graph_spec)

        self.terminal_states = []
        self.rewards = np.zeros(self.nNodes)
        self.teleport_probability = teleport_probability
        self.build_trans_mat()
        
        if build_nx_graph:
            self.G = nx.DiGraph()
            self.G.add_edges_from(self.edges)

        
    def set_reward_state(self,nodeNr,terminal=False):
        "right now handles setting one state as the rewarded one"
        assert (nodeNr<= self.nNodes)
        
        #reinitialise
        self.rewards = np.zeros(self.nNodes)
        self.terminal_states = []
        
        #set values
        self.rewards[nodeNr] = 1.
        if terminal:
            self.terminal_states.append(nodeNr)
        self.build_trans_mat()

    def read_gunnar_graph(self,g_spec):
        """convert output of gunnar algorithm to list of list of edges
            as well as a matrix defining the transitions as a function
            of actions
        """


        nNodes,nEdges = g_spec.split(' ')[:2]
        self.nNodes = self.n_states = int(nNodes)
        self.nEdges = int(nEdges)
        self.nActions = int(float(nEdges)/float(nNodes))
        #print(self.nActions,self.nNodes)

        links = [int(i) for i in re.findall('[0-9]',g_spec[5:])]

        edges = []
        edge_set = np.zeros([self.nNodes,self.nActions],dtype='int')
        cntArr = np.zeros(self.nNodes,dtype='int')
        for fst,snd in zip(links[:-1][::2],links[1:][::2]):
            edges.append([fst,snd])
            edge_set[fst,cntArr[fst]] = snd
            cntArr[fst] += 1
        return edges,edge_set

    def build_trans_mat(self):
        self.trans_mat = np.zeros([self.nNodes,self.nActions,self.nNodes]) + self.teleport_probability/(self.nNodes-1.)
        for s in range(self.nNodes):
            if not s in self.terminal_states:
                for a in range(self.nActions):
                    self.trans_mat[s,a,self.edge_transitions[s,a]] = 1 - self.teleport_probability
            else:
                self.trans_mat[s] = 0
                
    def plot_policy_onto_graph(self,P,pos=None):
        
        self.G = nx.DiGraph()
        self.G.add_edges_from(self.edges)

        colors = self._get_edge_colors(P)
        nx.draw(self.G,pos=pos,with_labels=True,edge_color=colors)
    
    def build_nx_graph(self):
        self.G = nx.DiGraph()
        self.G.add_edges_from(self.edges)

        
    def _get_edge_colors(self,P):
        #What we want to do here is to 
        colors = ['k']*len(self.edges)
        for stateIx in range(self.n_states):
            p_ = P[stateIx]
            if p_[0]!=p_[1]:
                goodMove = np.argmax(p_)
                badMove = np.argmin(p_)

                good_edge = [stateIx,self.edge_transitions[stateIx,goodMove]]
                bad_edge = [stateIx,self.edge_transitions[stateIx,badMove]]

                good_edge_ix = [kk for kk,i in enumerate(self.G.edges) if list(i)==good_edge][0]
                bad_edge_ix = [kk  for kk,i in enumerate(self.G.edges) if list(i)==bad_edge][0]
                #print(good_edge,good_edge_ix)

                colors[good_edge_ix] = 'g'
                colors[bad_edge_ix] = 'r'
        return colors