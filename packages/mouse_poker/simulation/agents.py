import numpy as np
from .base_agent import base_agent

class diffusion_agent(base_agent):
    """ This agent simply makes random choices at each time point."""

    def do_policy(self):
        self.current_state = np.random.choice(self.available_states)


class towards_middle_agent(base_agent):
    """ This is an agent where you have two policies. It selects between these
        two policies based on how many states are in each direction. Right now
        this is the simplest possible implementation that samples a direction and 
        sticks with it. You could imagine one where you essentially calculate values based on 
        number of states available and then make a direction decision at each point. 
        Additionally, you can just add stochasticity at all points 
        (i.e. go in direction and change direction)
    """
   
    def do_policy(self):
        if self.task_params['graph_type']=='line':
            
            if self.choices_since_reward==0:
                self.direction = int(np.abs(self.current_state-9)>np.abs(self.current_state-0)) #
                if len(self.available_states)==2:
                    self.current_state = self.available_states[self.direction]
                else:
                    self.current_state = self.available_states[0]
                #    self.direction = 1
                #else:
                #    self.direction = -1
            else:
                if len(self.available_states)==2:
                    self.current_state = self.available_states[self.direction]
                else:
                    if not self.forced:
                        if self.current_state==0: self.direction = 1
                        elif self.current_state==(self.task_params['len_graph']-1): self.direction = 0
                            
                    self.current_state = self.available_states[0]
        else:
            raise Exception('No idea what could happen on the loop here')
            #self.current_state = np.random.choice(self.available_states)


class Qlearner(base_agent):
    """ Single-step Q-learning agent. This doesn't work, something is buggy"""
    def __init__(self,learning_params=None,task_params=None):
        super().__init__(learning_params,task_params)
        self.sigmoid = lambda x: 1/(1+np.exp(-(x[1]-x[0])))
        self.decs = []
        self.Q_values = np.zeros([self.task_params['len_graph'],2])
        self.G = 0.99
        self.alpha = .7
        self.cost = .1
        self.rVal = 10
        self.prev_state = None
        self.up = None
        
    def do_policy(self):
        self.prev_state = self.current_state + 0
        if len(self.available_states)==2:
            p = self.sigmoid(self.Q_values[self.current_state])
            choice = np.random.choice([0,1],p=[1-p,p])
            self.current_state = self.available_states[choice]
        else:
            choice = self.current_state==0
            self.current_state = self.available_states[0]
        self.up = choice
        #self.decs.append([self.prev_state,choice,(self.current_state==self.reward_location)-self.cost])
        #print(prev_state,self.current_state)
    def update_policy(self,rew=False,end_of_trial=False):
        """ update Q-values """
        #if end_of_trial:
            #gammas = [self.G**i for i in reversed(range(len(self.decs)))]
        Q_new = self.Q_values.copy()

        if end_of_trial:
            nextS_term = 0
            nextS_term = np.max(self.Q_values[self.current_state])
        else:
            nextS_term = np.max(self.Q_values[self.current_state])
        Q_new[self.prev_state,self.up] = self.Q_values[self.prev_state,self.up] + self.alpha*((rew)*self.rVal -self.cost + self.G*nextS_term - self.Q_values[self.prev_state,self.up])
        self.decs = []
        self.Q_values = Q_new.copy()


class Model_based_agent(base_agent):
    """ This is a perfect model based agent. Exploration policy right now is just
        random walk. Another thing to consider here is what happens on the loop when
        both directions are equal. Right now it just chooses direction 0, could make
        this stochastic or do something else."""
    def __init__(self,learning_params=None,task_params=None):
        super().__init__()
        self.known_reward_location = None

    def exploration_policy(self):
        """ explore the environment randomly"""
        self.current_state = np.random.choice(self.available_states)

    def do_policy(self):
        """ Run the policy """
        if self.known_reward_location is None:
            self.exploration_policy()
        else:
        
            if len(self.available_states)==2:

                if self.task_params['graph_type']=='line':

                    d0 = np.abs(self.known_reward_location-self.available_states[0])
                    d1 = np.abs(self.known_reward_location-self.available_states[1])
                else:
                    d0 = get_modulo_distance(self.known_reward_location,
                                            self.available_states[0],
                                            nStates=self.task_params['len_graph'])

                    d1 = get_modulo_distance(self.known_reward_location,
                                            self.available_states[1],
                                            nStates=self.task_params['len_graph'])
                    
                self.current_state = self.available_states[np.argmin([d0,d1])]

            else:
                self.current_state = self.available_states[0]
                
    def update_policy(self,end_of_trial,rew):
        #print(1)
        if end_of_trial:
            if self.known_reward_location is None:
                self.known_reward_location = self.current_state
        #else:
            elif (self.current_state!=self.known_reward_location):
                self.known_reward_location = self.current_state
        else:
            if (self.current_state==self.known_reward_location):
                self.known_reward_location = None

