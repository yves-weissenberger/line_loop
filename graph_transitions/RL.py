import numpy as np


#This contains function for doing RL on this stuff

def value_iteration(world, g=.9):
    '''Evaluate state values under the optimal policy for specified world and 
    discount factor g.'''
    d = 1.                       # Value change
    V = np.zeros(world.n_states) # Value estimate.
    while d > 1e-12:
        prev_V = V[:].copy()
        #this this line of V
        V = np.max(np.sum(world.trans_mat * (world.rewards + g*V)[None,None,:],2),1)
        d = np.max(np.abs(prev_V-V))
    Q = world.trans_mat @ (world.rewards + g*V)  # Action values.
    policy = (Q == np.max(Q,1)[:,None]).astype(float)
    policy = policy/np.sum(policy,1)[:,None]
    return V, Q, policy