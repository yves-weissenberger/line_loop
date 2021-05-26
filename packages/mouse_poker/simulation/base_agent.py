import numpy as np


class base_agent(object):
    """
    This is a generic class that other agents (with different policies)
    inherit from. This class implements the basic task_logic
    """
    def __init__(self,learning_params=None,task_params=None):
        """ 
        For simplicity, in the first instance, will assume that
        
        Arguments:
        =======================
        
        task_params (dict):    dictionary containing parameters that are required for running of the task
        
        learning_params (dict): parameters necessary for specifying and updating the policy
        
        """
        
        if task_params is None:
            task_params = {'len_graph':9,
                           'graph_type':'line',
                           'rew_locs_session':list(range(9)),
                           'rewards_pre_switch': 20,
                           'reward_switch_p': 0.2,
                           'choices_pre_forced': 20
                            }

        
        self.learning_params = learning_params
        self.task_params = task_params
        
        
        #helper params
        self._all_states = list(range(self.task_params['len_graph']))
        

        self._init_task_variables()
        self._init_stores()
        

        
    def _init_task_variables(self):
        #initialise variables
        self.forced = False
        self.reward_location = None
        self.current_state = None
        self.current_port = None
        self.current_reward_location = None
        self.available_states = []
        self.choices_since_reward = 0
        self.n_rewards_at_loc = 0
        self.n_reward_total = 0
        
    def _init_stores(self):
        #this is the information to extract at the end to feed into
        #the standard analysis pipeline
        self.state_seq = []
        self.rew_list = []
        self.port_seq = []
        self.forced_seq = []

    def set_available_states(self):
        """ updates available_states variables"""
        
        #if bottom edge
        if self.current_state==0:
            if self.task_params['graph_type']=='line':
                self.available_states = [1]
            else:
                self.available_states = [self.task_params['len_graph']-1,1]

        #elif top edge
        elif self.current_state==(self.task_params['len_graph']-1):

            if self.task_params['graph_type']=='line':
                self.available_states = [self.task_params['len_graph']-2]
            else:
                self.available_states = [self.task_params['len_graph']-2,0]

        #with states either side
        else:
            self.available_states = [self.current_state-1,self.current_state+1]

        #if its a forced trial
        if self.choices_since_reward>self.task_params['choices_pre_forced']:
            self.forced_seq.append(True)
            self.forced = True
            #if this is a forced trial
            if len(self.available_states)>1:
                
                if self.task_params['graph_type']=='line':
                    best_ix = np.argmin([np.abs(self.available_states[0]-self.reward_location),
                                          np.abs(self.available_states[1]-self.reward_location)])
                    self.available_states = [self.available_states[best_ix]]
                else:
                    best_ix = np.argmin([get_modulo_distance(self.reward_location,self.available_states[1]),
                                          get_modulo_distance(self.reward_location,self.available_states[1])])
                    self.available_states = [self.available_states[best_ix]]

        else:
            self.forced_seq.append(False)

    
    def init_trial(self):
        """ Initialise the task"""
        if ((self.reward_location is None) or 
            (self.n_rewards_at_loc>self.task_params['rewards_pre_switch'] and
                np.random.uniform(0,1)>self.task_params['reward_switch_p'])):
            self.update_reward_location()
        
            

        self.current_state = np.random.choice(self.poss_starting_states)
        self.set_available_states()
        #print(self.curr)
        #print(self.available_states)
        self.state_seq.append(self.current_state)
        self.port_seq.append(self.current_port)
        self.rew_list.append(False)

        
    def run(self,n_trials=3e3):
        """ Run behaviour """
        self.init_trial()
        #if self.task_params
        #print(n_trials)
        rew = False
        while self.n_reward_total<n_trials:
            if rew>0:
                self.update_policy(rew=rew,end_of_trial=True)
                self.init_trial()
                

            #print(self.available_states)
            rew = self.update_state()
            self.update_policy(rew=rew,end_of_trial=False)
            self.state_seq.append(self.current_state)
            self.port_seq.append(self.current_port)
            self.rew_list.append(rew)
                    
        
    
    def update_state(self):
        """ Received choice of 0 or 1"""
        self.do_policy()
        self.set_available_states()

        rew = False
        if self.current_state==self.reward_location:
            self.choices_since_reward = 0
            self.n_rewards_at_loc += 1
            self.choices_since_reward =0 
            self.n_reward_total += 1
            #self.rew_list.append(True)
            rew = True
        else:
            self.choices_since_reward += 1

        return rew
            
    
    def update_reward_location(self):
        self.n_rewards_at_loc = 0
        self.reward_location = np.random.choice(self.task_params['rew_locs_session'])
        self.poss_starting_states = [i for i in self.task_params['rew_locs_session'] if i!=self.reward_location]

    
    def do_policy(self):
        pass #return None
    def update_policy(self,rew,end_of_trial=False):
        """ 
        Use this function when the behavioural policy is actually updated 
        (e.g. Q-learning),not if the behavioural policy involves inference.
        use end_of_trial signal for episodic updates
        """
        pass