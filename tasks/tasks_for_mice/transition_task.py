from pyControl.utility import *
import hardware_definition as hw
import random

""" Task code for tasks with optimized graphs
    
    This is a general task


    Physical layout of pokes is:

            1
        2       3
    4       5       6
        7       8
            9
"""

states = ['handle_poke',
          'reward_consumption',
          'deliver_reward']

events = ['poke_4', 'poke_4_out',
          'poke_6', 'poke_6_out',
          'poke_5', 'poke_5_out',
          'poke_1', 'poke_1_out',
          'poke_9', 'poke_9_out',
          'end_consumption','session_timer']

nStates = 9
events = ['poke_'+str(i) for i in range(nStates)]
events.extend(['poke_'+str(i)+'_out' for i in range(nStates)])
events.append('session_timer')
events.append('end_consumption')


#list of solenoids to do stuff with
hw.sol_list = [hw.SOL_1,hw.SOL_2,hw.SOL_3,
               hw.SOL_4,hw.SOL_5,hw.SOL_6,
               hw.SOL_7,hw.SOL_8,hw.SOL_9]

#list of pokes to do stuff with
hw.poke_list = [hw.poke_1,hw.poke_2,hw.poke_3,
                hw.poke_4,hw.poke_5,hw.poke_6,
                hw.poke_7,hw.poke_8,hw.poke_9]

#should only use one of these
#v.T = []
v.edge_list = [[0, 1], [0, 3], [1, 0], [1, 3], [2, 0],
               [2, 1], [3, 4], [3, 7], [4, 2], [4, 6],
               [5, 2], [5, 4], [6, 7], [6, 8], [7, 5],
               [7, 8], [8, 6], [8, 5]]


#-------------------------------------------------------------------------        
# Parameters of the task
#-------------------------------------------------------------------------


v.reward_locations = [0,1,2,3]
v.nStates = nStates
v.random_transition_P = 0.
v.switch_rewarded_states = False  #determines whether reward location swaps
v.reward_durations = [.1,.1]


#by default map pokes to states 1:1
v.poke_to_state_map = range(9)  #entry 0 tells you which state poke 0 corresponds to
v.stop_reward_consumption_dur = 250*ms



#-------------------------------------------------------------------------        
# Variables
#-------------------------------------------------------------------------
v.n_rewards = 0
v.poked_port = random.randint(0,9)
v.lights_set = False
v.current_state = None
v.visitation_num = [0]*nStates
v.state_str = [str(i) for i in range(9)]
v.available_transitions = get_transition_set(v.edge_list)


#-------------------------------------------------------------------------        
# Non-state machine code.
#-------------------------------------------------------------------------

def map_poke_to_state(poke):
    return v.poke_to_state_map[poke]


def map_state_to_poke(state):
    return v.poke_to_state_map.index(state)

def get_transition_set(links):
    """ Takes as input the list of edges and edges and returns an array 
        that gives the transitions available as a function of state
    """
        edge_set = [[None,None] for _ in range(v.nStates)]#np.zeros([self.nNodes,self.nActions],dtype='int')
        cntArr = [0]*v.nStates#np.zeros(self.nNodes,dtype='int')
        for fst,snd in links:
            edge_set[fst][cntArr[fst]] = snd
            cntArr[fst] += 1
    return edge_set


def check_graph():
    """ Function that checks that the graph input is valid Not done"""
    return None

def update_lights(state):
    """ This function lights up ports that are available from the current state """

    for s_ in range(v.nStates):
        if s_ in v.available_transitions[v.current_state]:
            hw.poke_list[map_state_to_poke(s_)].LED.on()
        else:
            hw.poke_list[map_state_to_poke(s_)].LED.on()
    return None

#-------------------------------------------------------------------------        
# State machine code.
#-------------------------------------------------------------------------


def run_start(): 
    # Set session timer and turn on houslight.
    set_timer('session_timer', v.session_duration)  
    hw.houselight.on()                             
    
def run_end():
    # Turn off all hardware outputs.  
    hw.off()



def handle_poke(event):
    """ Basically run all of the task """

    if event=='entry':
        v.current_state = map_poke_to_state(v.poked_port)  #NB this is declared in code
        v.poked_port = None
        update_lights()

    if event[-1] in v.state_str:  #check that event is an in-poke

        poked_port = int(event[-1])  #which poke was poked

        #if poked port is one of available transitions do the transition update
        if poked_port in v.available_transitions[v.current_state]:

            #track the poke that was just poked as this is going to be the next state
            v.poked_port = poked_port


            if poked_port in v.reward_locations:
                goto_state('deliver_reward')
            else:
                #goto_state('wait_for_poke')
                v.current_state = map_poke_to_state(v.poked_port)
                v.poked_port = None
                update_lights()


def reward_consumption(event):
    # Wait until subject has stayed out of the poke for v.stop_reward_consumption_dur ms.
    if event == 'entry':
        if not (hw.poke_1.value() or hw.poke_4.value() or hw.poke_6.value() or hw.poke_9.value()): # subject already left poke.

            set_timer('end_consumption', v.stop_reward_consumption_dur)

    elif event in ('poke_1','poke_4','poke_6', 'poke_9'):
        disarm_timer('end_consumption')

    elif event in ('poke_1_out','poke_4_out','poke_6_out','poke_9_out'):
        set_timer('end_consumption', v.stop_reward_consumption_dur)

    elif event == 'end_consumption':
        goto_state('handle_poke')


def deliver_reward(event):
    # Deliver reward to appropriate poke
    if event == 'entry':
        timed_goto_state('reward_consumption', v.reward_durations[1])
        hw.solenoid_list[v.poked_port].on()
    elif event == 'exit':
        hw.solenoid_list[v.poked_port].on()
