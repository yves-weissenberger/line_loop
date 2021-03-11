import re
import os
import matplotlib.pyplot as plt
import numpy as np


def get_poke_to_state_map(lines):
    """ """
    tmp = []
    for i,j in zip([int(re.findall('POKEDPORT_([0-9])',i)[0]) for i in lines if '_POKEDPORT' in i],
                         [int(re.findall('POKEDSTATE_([0-9])',i)[0]) for i in lines if '_POKEDSTATE' in i]):

        if [i,j] not in tmp:
            tmp.append([i,j])
    poke_to_state_map = [i[1] for i in sorted(tmp)]
    return poke_to_state_map,tmp


def map_poke_to_state_fun(pkst_map,poke):
    "i[0] are ports i[1] are states"
    tmp1 = [i[0] for i in pkst_map]
    tmp2 = [i[1] for i in pkst_map]
    return tmp2[tmp1.index(poke)]


def map_state_to_poke_fun(pkst_map,poke):
    "i[0] are ports i[1] are states"
    tmp1 = [i[0] for i in pkst_map]
    tmp2 = [i[1] for i in pkst_map]
    return tmp1[tmp2.index(poke)]


def extract_navi_dat(lines):
    """ Separate data loading function for navigation data"""
    rew_list = [0]
    state_seq = []
    port_seq = []
    forced_seq = [0]

    poke_to_state_map,full_pkst_map = get_poke_to_state_map(lines)
    used_states = sorted([i[1] for i in full_pkst_map])
    for lix,l in enumerate(lines):
        if 'POKEDSTATE' in l:
            poked_state = int(re.findall('POKEDSTATE_([0-9])_',l)[0]) 
            prev = int(re.findall('PREVSTATE_([0-9])_',l)[0]) 
            #now = int(re.findall('NOWSTATE_([0-9])_',l)[0]) 
            port = int(re.findall('POKEDPORT_([0-9])_',lines[lix+1])[0])
            rew = 'REW_True' in lines[lix+1]
            forced = len(eval(re.findall('AVAILSTATES_(\[.*\])_',l)[0]))==1

            if rew:
                if (poked_state==used_states[0]) or (poked_state!=used_states[-1]):
                    forced = False
                if (forced_seq[-1] and ((state_seq[-1]!=used_states[0]) and (state_seq[-1]!=used_states[-1]))):
                    forced = True




            if rew_list[-1]:
                rew_list.append(False)
                port_seq.append(None)
                forced_seq.append(False)
                state_seq.append(prev)
                
            rew_list.append(rew)
            state_seq.append(poked_state)
            port_seq.append(port)
            forced_seq.append(forced)
            
    del rew_list[0]
    del forced_seq[0]
    return state_seq, rew_list, port_seq,forced_seq


def get_transitions(state_seq,rew_list,port_seq,forced_seq,rew_indices,map_poke_to_state,minNrew=5,set_rew_indices=None,firstOnly=False):
    """ This function obtains empirical counts for transitions from a given state to another
        as a function of """
    used_states = sorted([i[1] for i in map_poke_to_state.args[0]])
    if set_rew_indices is None: set_rew_indices=used_states
    perf = np.zeros([9,9,9])
    perf_ctr = np.zeros([9,9,9])
    rew_hist = []
    transition_mtx = np.zeros([9,9,9])
    state_ctr = np.zeros([9,9])
    all_rew_loc = []
    for rew_ctr,(st,nd) in enumerate(zip(np.where(rew_list)[0][:-2],np.where(rew_list)[0][1:-1])):
        rew_loc = state_seq[nd]
        if (rew_loc in set_rew_indices):
            c_rew_index = rew_indices.index(port_seq[nd])
            all_rew_loc.append(rew_loc)
            if not rew_hist:
                rew_hist.append(rew_loc)
            elif rew_loc==rew_hist[-1]:
                rew_hist.append(rew_loc)
            else:
                rew_hist = []

            has_visited= []
            if len(rew_hist)>minNrew:

                for pk_ctr in range(st+1,nd+1):
                    if not forced_seq[pk_ctr]:
                        state = state_seq[pk_ctr]
                        if state not in has_visited:

                            next_state = state_seq[pk_ctr+1]

                            transition_mtx[state,next_state,rew_loc] += 1
                            state_ctr[state,rew_loc] += 1
                            if firstOnly: has_visited.append(state)

    
    return transition_mtx, state_ctr, np.unique(all_rew_loc)



def get_trajectories(state_seq,rew_list,port_seq,forced_seq,used_states):
    
    """ Get empirical behavioural trajectories to be able to model behaviour in RL terms. """

    forced = []
    decision_seq = []
    for rew_ctr,(st,nd) in enumerate(zip(np.where(rew_list)[0][:-1],np.where(rew_list)[0][1:])):
        rew_loc = state_seq[nd]
        #print(state_seq[nd],port_seq[nd])
        seq_ = []
        fcd_ = []
        for pk_ctr in range(st+1,nd+1):
            
            state = state_seq[pk_ctr]
            #print(pk_ctr,state)
            seq_.append(state-used_states[0])
            if forced_seq[pk_ctr]: fcd_.append(1);     
            else: fcd_.append(0)
                    
        forced.append(fcd_.copy())
        decision_seq.append(seq_.copy())
    return decision_seq, forced