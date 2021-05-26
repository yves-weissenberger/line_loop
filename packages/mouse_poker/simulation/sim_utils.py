import numpy as np


def get_modulo_distance(target_state,state,n_states=6):
    """ Get distance between two states on a circle"""
    return np.min(np.abs([target_state-state,target_state-(state+n_states),target_state-(state-n_states)]))
