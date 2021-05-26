import numpy as np

def get_st_dist(state_seq,pk_ctr,rew_loc):
    """ get distances between current and next state and reward during navigation on the line """
    
    d0 = np.abs(state_seq[pk_ctr]-rew_loc)
    d1 = np.abs(state_seq[pk_ctr+1]-rew_loc)
    st_dist = state_seq[pk_ctr]-state_seq[pk_ctr+1]
    
    return d0,d1,st_dist
    