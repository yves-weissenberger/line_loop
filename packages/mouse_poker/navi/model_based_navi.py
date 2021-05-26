import numpy as np

from .navi import extract_navi_dat, get_transition_matrix
from .navi_utils import get_st_dist


def first_block_model_based_batch(all_fs,ignore_distance=False,
                                  ignore_first_visit=False,ignore_has_updated=True,
                                  verbose=False):

    perf = 0
    trial_ctr = 0

    for fpath in all_fs:

        with open(fpath,'r') as f:
            lines = f.readlines()
        state_seq, rew_list, _,forced_seq = extract_navi_dat(lines)
        perf_tmp,trial_ctr_tmp = run_first_block_model_based_analysis(state_seq,rew_list,forced_seq,
                                                                      ignore_distance,ignore_first_visit,
                                                                      ignore_has_updated,verbose)

        perf += perf_tmp
        trial_ctr += trial_ctr_tmp
    return perf, trial_ctr


def run_first_block_model_based_analysis(state_seq,rew_list,forced_seq,ignore_distance=False,ignore_first_visit=False,ignore_has_updated=True,verbose=False):

    """ Check whether behaviour is model based.
    
    Arguments:
    ==================================
    state_seq (list):   list of states that have been visited
    """
    perf = 0
    trial_ctr = 0


    #organise data into trials
    rewarded_pokes = np.where(rew_list)[0]
    trial_starts = np.concatenate([[0],rewarded_pokes[:-1]])
    trial_ends = rewarded_pokes


    #initialise variables
    prev_rew_loc = None  #this is the reward location that was last updated
    direction = None
    has_updated = False
    exp_dirs = []
    prev_rew_loc = None

    #this is the start of a
    for st,nd in zip(trial_starts,trial_ends):

        rew_loc = state_seq[nd]  #this is state that is rewarded
        #print(state_seq[nd])  #this is the first state the animals enter into
        prev_direction = direction

        if rew_loc!=prev_rew_loc:
            #print("HERE")
            direction = None
            has_updated = False
            prev_direction = None
            exp_dirs = [] #experienced directions
            prev_diff_rew_loc = prev_rew_loc

        visited_states = []
        direction = (rew_loc - state_seq[st+1])>0 #which side are you approaching the reward from


        for pk_ctr in range(st+1,nd):  #for each poke between two rewards

            d0,d1,_ = get_st_dist(state_seq,pk_ctr,rew_loc)
            free_choice_trial = forced_seq[pk_ctr] is False
            state = state_seq[pk_ctr]
            next_state = state_seq[pk_ctr+1]


            update_condition_list = [free_choice_trial,              #NOT a forced trial
                                    prev_diff_rew_loc is None,      #Only look at first block
                                    prev_direction is not None,     #make sure NOT looking at first run-to-rew (when rew_loc is unknown)
                                    direction not in exp_dirs,      #hasn't experienced this direction in this block yet
                                    (ignore_has_updated or (not has_updated)),           #toggle if only look at first relevant POKE in block
                                    (ignore_first_visit or(state not in visited_states)),    #look only at first visits to each state
                                    (ignore_distance or (d0>1)),
                                    #np.abs(start_state_trial-rew_loc)>1,
                                    state !=rew_loc
                                    ]

            if all(update_condition_list):
                if verbose:
                    print(("!"*80 + "\n")*3)
                    print_list = (state,next_state,rew_loc,int(d1<d0))
                    print('state:{},nextstate:{},rew_loc:{},correct:{}'.format(*print_list))
                    print(state_seq[st+1:nd+1])
                    print('\n')

                perf += int(d1<d0)
                trial_ctr += 1

                has_updated = True  #if decisions from this block have led to updated

            visited_states.append(state)

        if direction not in exp_dirs:
            exp_dirs.append(direction)
        prev_rew_loc = rew_loc

    return perf,trial_ctr
