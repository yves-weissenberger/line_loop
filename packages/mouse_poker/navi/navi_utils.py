import numpy as np

def get_st_dist(state_seq,pk_ctr,rew_loc):
    """ get distances between current and next state and reward during navigation on the line """
    
    d0 = np.abs(state_seq[pk_ctr]-rew_loc)
    d1 = np.abs(state_seq[pk_ctr+1]-rew_loc)
    st_dist = state_seq[pk_ctr]-state_seq[pk_ctr+1]
    
    return d0,d1,st_dist
    

def policy_changed_with_rew_loc(pk_ctr,state_seq,rew_loc,prev_diff_rew_loc,if_is_rew_loc=False):
    """ is the policy different between rew_loc and prev_diff_rew_loc?
        An important question is the below, what happens when you are in
        the previous reward location?
    """
    if prev_diff_rew_loc is not None:
        same_as_prev_pol = (((state_seq[pk_ctr]-rew_loc)>0)==           #direction to reward with location
                            ((state_seq[pk_ctr]-prev_diff_rew_loc)>0))  #direction to reward with prev location
    else:
        same_as_prev_pol = False
    
    if state_seq[pk_ctr]==prev_diff_rew_loc:
        same_as_prev_pol = if_is_rew_loc
    return same_as_prev_pol