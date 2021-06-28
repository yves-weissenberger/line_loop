from os import stat
import numpy as np

from .navi import extract_navi_dat, get_transition_matrix
from .navi_utils import get_st_dist, policy_changed_with_rew_loc


def first_block_model_based_batch(all_fs,ignore_distance=False,
                                  ignore_first_visit=False,ignore_has_updated=True,
                                  verbose=False):

    perf = 0
    trial_ctr = 0

    for fpath in all_fs:

        with open(fpath,'r') as f:
            lines = f.readlines()
        state_seq, rew_list, _,forced_seq = extract_navi_dat(lines)
        perf_tmp,trial_ctr_tmp = first_block_model_based_analysis(state_seq,rew_list,forced_seq,
                                                                      ignore_distance,ignore_first_visit,
                                                                      ignore_has_updated,verbose)

        perf += perf_tmp
        trial_ctr += trial_ctr_tmp
    return perf, trial_ctr


def first_block_model_based_analysis(state_seq,rew_list,forced_seq,ignore_distance=False,ignore_first_visit=False,ignore_has_updated=True,verbose=False):

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

        if prev_diff_rew_loc!=state_seq[st]:
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




def model_based_batch(all_fs,ignore_distance=False,
                    ignore_first_visit=False,ignore_has_updated=True,
                    use_block_transitions=False,if_is_rew_loc=True,
                    ignore_prev_rew_loc_in_seq=False,
                    verbose=False,minNrew=15,min_rew_in_session=20):
    """ Run the main model based analysis on a dataset
    
    
    Arguments:
    ==============================

    ignore_first_visit (bool):           if True, then will include every visit to a given state (in one run)
                                         not just the first visit
    
    ignore_distance (bool):              if False, will only include decisions where the distance to reward 
                                         was greater than 1.
    
    if_is_rew_loc (bool):                When calculating if the required policy with the current reward location
                                         is different to the previous_rew_location, there is ambiguity about what
                                         to do when you are in the previously rewarded state. This determines what
                                         happens in this case.

    ignore_has_updated (bool):           if True, only look at the very first decision (rather than the first run to reward)
                                         after a rew location transitions that meets all criteria

    minNrew (int):                       when calculating what the transition would have been,

    use_block_transitions (bool):        if True, will calculat the transition matrix for each block
    """
    perf = 0
    trial_ctr = 0
    prob_array_for_pois_binom = []
    choices = []

    for fpath in all_fs:

        with open(fpath,'r') as f:
            lines = f.readlines()
        state_seq, rew_list, _,forced_seq = extract_navi_dat(lines)
        if np.sum(rew_list)>min_rew_in_session:
            if verbose:
                print(fpath)
            res = model_based_analysis_single_session(state_seq,rew_list,forced_seq,
                                                    ignore_distance=ignore_distance,
                                                    ignore_first_visit=ignore_first_visit,
                                                    ignore_has_updated=ignore_has_updated,
                                                    use_block_transitions=use_block_transitions,
                                                    ignore_prev_rew_loc_in_seq=ignore_prev_rew_loc_in_seq,
                                                    verbose=verbose,
                                                    if_is_rew_loc=if_is_rew_loc,
                                                    minNrew=minNrew)
            perf_tmp, trial_ctr_tmp, prob_array_for_pois_binom_tmp, choices_tmp = res

            perf += perf_tmp
            trial_ctr += trial_ctr_tmp
            prob_array_for_pois_binom.extend(prob_array_for_pois_binom_tmp)
            choices.extend(choices_tmp)
    return perf, trial_ctr, prob_array_for_pois_binom, choices


def model_based_analysis_single_session(state_seq,rew_list,forced_seq,
                                        ignore_distance=False,ignore_first_visit=False,
                                        ignore_has_updated=True,
                                        ignore_prev_rew_loc_in_seq=False,
                                        minNrew=15,use_block_transitions=False,
                                        if_is_rew_loc=True,verbose=False,):
    """ Check for model based behaviour on trials"""


    #define variables that will be output
    prob_array_for_pois_binom = []
    choices = []
    perf = 0
    trial_ctr = 0

    transition_mtx,_,_,_= get_transition_matrix(state_seq,
                                                rew_list,
                                                forced_seq,
                                                minNrew=minNrew)



    #organise data into trials
    rewarded_pokes = np.where(rew_list)[0]
    trial_starts = np.concatenate([[0],rewarded_pokes[:-2]])
    trial_ends = rewarded_pokes


    #initialise variables
    prev_rew_loc = None  #this is the reward location that was last updated
    direction = None
    has_updated = False
    same_as_prev_pol = True
    exp_dirs = []
    prev_block_start = 0
    block_start = 0
    check1 = []
    for st,nd in zip(trial_starts,trial_ends):

        rew_loc = state_seq[nd]  #this is state that is rewarded

        prev_direction = direction

        if rew_loc!=prev_rew_loc:
            #print("HERE")
            direction = None
            has_updated = False
            prev_direction = None
            exp_dirs = [] #experienced directions
            prev_diff_rew_loc = prev_rew_loc
            visited_states = []
            prev_block_start = block_start
            block_start = st

        if state_seq[st+1]==(rew_loc):
            exp_dirs = [True,False]
        direction = (rew_loc - state_seq[st+1])>0 #which side are you approaching the reward from
        visited_states = []
        prev_rew_loc_in_seq = False

        if rew_loc!=state_seq[st+1] and rew_loc:
            for pk_ctr in range(st+1,nd):  #for each poke between two rewards  

                d0,d1,_ = get_st_dist(state_seq,pk_ctr,rew_loc)
                free_choice_trial = forced_seq[pk_ctr]==False
                state = state_seq[pk_ctr]
                next_state = state_seq[pk_ctr+1]
                
                #if ((not prev_rew_loc_in_seq) and (state==prev_diff_rew_loc)):
                #    prev_rew_loc_in_seq = True
                #else:
                #    prev_rew_loc_in_seq = False

                if state==prev_diff_rew_loc:
                    prev_rew_loc_in_seq = True
                else:
                    prev_rew_loc_in_seq = False


                same_as_prev_pol = policy_changed_with_rew_loc(pk_ctr,state_seq,rew_loc,prev_diff_rew_loc,if_is_rew_loc=if_is_rew_loc)
                if np.abs(state-next_state)!=1:
                    print(state,next_state,rew_list[pk_ctr])
                #check1.append(np.abs(state-next_state))
                inclusion_condition_list = [free_choice_trial,          #NOT a forced trial
                                        prev_diff_rew_loc is not None,  #ignore first block in session
                                        not same_as_prev_pol,           #ensure that this a policy change is required to make correct decision
                                        prev_direction is not None,     #make sure NOT looking at first run-to-rew after a block transition (when rew_loc is unknown)
                                        direction not in exp_dirs,      #hasn't experienced this direction in this block yet
                                        (ignore_has_updated or (not has_updated)),               #toggle if only look at first relevant POKE in block
                                        (ignore_first_visit or (state not in visited_states)),    #look only at first visits to each state
                                        (ignore_distance or (d0>1)),
                                        state!=rew_loc,  #this is necessary due to a bug in the code
                                        ((ignore_prev_rew_loc_in_seq) or (not prev_rew_loc_in_seq)),
                                        ]

                #print(inclusion_condition_list)
                if all(inclusion_condition_list):
                    
                    if use_block_transitions:
                        
                        transition_mtx,_,_,_= get_transition_matrix(state_seq[prev_block_start:block_start],
                                                                    rew_list[prev_block_start:block_start],
                                                                    forced_seq[prev_block_start:block_start],
                                                                    minNrew=minNrew)

                        
                    choice_correct = int(d1<d0)
                    perf += choice_correct
                    trial_ctr += 1
                    t_p = transition_mtx[prev_diff_rew_loc,state,next_state]
                    
                    #because we are counting the sum of correct choices, need to look at probability of correct choice
                    #under previous model, not of empirical transition
                    if not choice_correct:
                        t_p = 1-t_p
                    
                    #only include
                    try:
                        if not np.isnan(t_p):
                            prob_array_for_pois_binom.append(t_p)
                            choices.append(int(d1<d0))
                    except:
                        pass
                    #These should be largely nans except because of bug in behaviour code, you can in some sessions
                    #start in the rewarded state

                    #if prev_diff_rew_loc==state:
                    #    print(t_p)
                    #this is print command to make sure everything is working correctyl.
                    if verbose:
                        print(("!"*80 + "\n")*3)
                        print_list = (state,state_seq[pk_ctr+1],rew_loc,prev_diff_rew_loc,choice_correct,np.round(t_p,decimals=2))
                        print('direction:{},exp_dirs:'.format(direction) + str(exp_dirs))
                        print('sapp:{}'.format(same_as_prev_pol))
                        print('prev_rew_loc_in_seq:{}'.format(prev_rew_loc_in_seq))
                        print('state:{},nextstate:{},rew_loc:{},prev_rew_loc:{},correct:{},prev_tp:{}'.format(*print_list))
                        #print(pk_ctr,os.path.split(fpath)[-1])
                        print(state_seq[st-1:nd+1],st,pk_ctr)
                        #print(forced_seq[st-1:nd+1],st,pk_ctr)
                        print(rew_list[st-1:nd+1])

                        print('\n')

                    has_updated = True  #if decisions from this block have led to updated

                visited_states.append(state)

        if direction not in exp_dirs:
            exp_dirs.append(direction)
        prev_rew_loc = rew_loc
        assert all(np.asarray(check1)==1)
    return perf, trial_ctr, prob_array_for_pois_binom, choices