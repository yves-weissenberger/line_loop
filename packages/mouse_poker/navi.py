import re
import os
import matplotlib.pyplot as plt
import numpy as np


def get_performance(state_seq,rew_list,port_seq,forced_seq,rew_indices,map_poke_to_state,minNrew=5,set_rew_indices=None,firstOnly=False,maxNrew=100):
    
    """ 
    Calculate the fraction correct in each state
    
    Arguments:
    ============================
    
    state_seq (list):             The sequence of states visited by the subject
    
    rew_list (list):              A boolean list that specifies whether a reward was received at this location or not
    
    port_seq (list):              The sequence of ports visited by the subject
    
    forced_seq (list):            A boolean list that specifies whether the decision the animal made was forced
    
    map_poke_to_state (function): Takes as input a port and returns the state this port corresponds to
    
    rew_indices (list):           The set of states (not ports!) where in this session rewards were delivered
    
    minNrew (int):                The required minimum number of sequential rewards at a particular rewarded state
                                  that are required before they are included in the perf and and perf_ctr arrays.
                                  For calculating the fraction correct, for example, you may want to ignore the first
                                  trial, when the animal doesn't know where the reward is
                                  
    minNrew (int):                Analogous to minNrew but specifies an upper limit
    
    set_rew_indices (list):       If you only want to consider behaviour when the reward is in one of 
                                  a particular set of states, rather than all states that had reward in
                                  a particular session, you can specify this parameter.
                                  
    first_only (boolean):         If True, for each trial (i.e. from a given starting state to receiving the reward)
                                  then it only adds the decision you make during the first visit to each state on that
                                  trial. For example, if reward is at 5 and the state sequence is 3-4-3-4-5. It will
                                  say that you made 1 correct decision in state 3 and 1 incorrect decision in state 4.
                                  Otherwise, you would have 2 correcy decisions in state 3, 2 incorrect in state 4 and
                                  1 correct in state 4.
                                    
    """
    
    used_states = sorted([i[1] for i in map_poke_to_state.args[0]])  #
    if set_rew_indices is None: set_rew_indices=used_states
    perf = np.zeros([9,9])  #this is an array that keeps track of the number of correct choices made in each state at each possible reward location
    perf_ctr = np.zeros([9,9]) #this is an array that keeps track of the TOTAL number choices made in each state at each possible reward location
    rew_hist = []
    all_rew_loc = []
    
    #the main logic of this code is to use the rew_list to separate the state sequence into trials.
    #and then calculate things on a trial by trial basis.
    for rew_ctr,(st,nd) in enumerate(zip(np.where(rew_list)[0][:-2],np.where(rew_list)[0][1:-1])):
        
        
        rew_loc = state_seq[nd]   #the reward location on this trial
        
        #include this trial only if it is at one of the reward locations you are considering
        if (rew_loc in set_rew_indices):
            
            #This block of code is used to calculate the number of
            #sequential rewards at the same location
            all_rew_loc.append(rew_loc)
            if not rew_hist:
                rew_hist.append(rew_loc)
            elif rew_loc==rew_hist[-1]:
                rew_hist.append(rew_loc)
            else:
                rew_hist = []

            has_visited= []  #this is a list of states that you have visited in a given 'trial', relevant if first_only=True
            
            #If you are in the range of number of rewards of interest calculate stuff in the trial
            if np.logical_and(len(rew_hist)>=minNrew,len(rew_hist)<=maxNrew): 
                
                #for each poke in the trial
                for pk_ctr in range(st+1,nd+1):
                    rewarded = rew_list[pk_ctr]  #if it is rewarded. Offset by 1 index (i.e. if the sequence is 3-4-5).
                    

                    #only proceed if it is NOT a forced trial
                    if not forced_seq[pk_ctr]:
                        
                        d0 = np.abs(state_seq[pk_ctr]-rew_loc)     #calculate the distance, in state space, from the current state to the reward 
                        d1 = np.abs(state_seq[pk_ctr+1]-rew_loc)   #calculate the distance, in state space, from the NEXt state to the reward

                        state = state_seq[pk_ctr]
                        rewarded = rew_list[pk_ctr]  #if it is rewarded. Offset by 1 index (i.e. if the sequence is 3-4-5).

                        
                        if state not in has_visited: #if this is the first visit to the state (note if first_only=False, you don't store which states you have visited)
                            
                            if d1<d0:   #if the distance of the next state from reward is smaller than the current state you made the correct choice
                                perf[state,rew_loc] += 1
                                perf_ctr[state,rew_loc] += 1
                            else:        #you made the wrong choice
                                perf[state,rew_loc] += 0
                                perf_ctr[state,rew_loc] += 1
                            if firstOnly: has_visited.append(state)
    
    for i in np.unique(rew_hist):
        perf[i,i] = np.nan
        perf_ctr[i,i] = np.nan

    return perf, perf_ctr

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