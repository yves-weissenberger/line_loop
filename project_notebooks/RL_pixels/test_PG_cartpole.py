import gym

import sys
import os
import numpy as np
import time


#Imports for neural network
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim


class Policy(nn.Module):
    def __init__(self,window_size,num_filter=2,size=2,pad=0,stride=2,lstm_hidden=0):
        super(Policy, self).__init__()


        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('leaky_relu'))

        self.num_filter = num_filter
        self.stride = stride
        self.size = size
        self.pad =pad

        #self.conv1 = nn.Conv2d(1, self.num_filter, self.size,self.stride)
        #self.conv2 = nn.Conv2d(self.num_filter, self.num_filter, self.size,self.stride)
        #self.conv3 = nn.Conv2d(self.num_filter, self.num_filter, self.size,self.stride)
        #self.conv4 = init_(nn.Conv2d(self.num_filter, self.num_filter, self.size, self.stride, self.pad))

        #self.affine1 = nn.Linear(np.prod(window_size),28,bias=True)
        #self.dropout = nn.Dropout(p=0.1)


        self.lstm_hidden = lstm_hidden
        if self.lstm_hidden:
            self.lstm = nn.LSTMCell(input_size=2, hidden_size=self.lstm_hidden)
            self.h_1 = self.c_1 = torch.zeros([1,self.lstm_hidden])

            self.affine2 = nn.Linear(self.lstm_hidden, 2,bias=True)
        else:
            self.affine2 = nn.Linear(2, 2,bias=True)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        #x = self.affine1(x)
        #x = self.dropout(x)
        #x = F.relu(x)
        #action_scores = self.affine2(x)
        #return F.softmax(action_scores, dim=1)

        #x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))

        if self.lstm_hidden:
            self.h_1, self.c_1 = self.lstm(x.view([1,-1]), (self.h_1, self.c_1))
            action_scores = self.affine2(self.h_1)

        else:
            action_scores = self.affine2(x.view([1,-1]))



        return F.softmax(action_scores, dim=1)



    def reset_lstm(self, buf_size=None, reset_indices=None):
        """
        Resets the inner state of the LSTMCell
        """
        if self.lstm_hidden:
            with torch.no_grad():
                #if reset_indices is None:
                    # set device to that of the underlying network
                    # (it does not matter, the device of which layer is queried)
                self.h_1 = self.c_1 = torch.zeros([1,self.lstm_hidden])
                #else:
                    # set device to that of the underlying network
                    # (it does not matter, the device of which layer is queried)
                    #resetTensor = torch.as_tensor(reset_indices.astype(np.uint8))

                    #if resetTensor.sum():
                    #    self.h_t1 = (1 - resetTensor.view(-1, 1)).float() * self.h_t1
                    #    self.c_t1 = (1 - resetTensor.view(-1, 1)).float() * self.c_t1


def select_action(state):
    #state = torch.from_numpy(state).float().unsqueeze(0)
    #probs = policy(state)
    probs = policy(torch.Tensor(state).view([1,-1]))
    #print(probs)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode(gamma=.99):
    #gamma = 
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward(retain_graph=True)
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]




if __name__=='__main__':
    window_size = (86,86)
    #env = gym.make('CartPole-v0')
    #env = gym.make("MountainCar-v0")
    
    nFrames_run = 0

    policy = Policy(window_size,lstm_hidden=10)
    policy.reset_lstm()
    optimizer = optim.Adam(policy.parameters(), lr=5e-3)
    eps = np.finfo(np.float32).eps.item()


    st = time.time()

    allR = []
    action = 0
    #for t_ in range(1,3000):
    t_ = 1
    allA_tmp = []
    showA = 0

    reset_every = 201
    alpha_= 0.99
    obs = env.reset()
    done = False
    epLen = 0.
    avgR = 200#22 for cartpole

    episode_nr = 0
    while True:

        #dat,rew = env.draw()
        if avgR<120:
            env.render()
        

        action = select_action(obs)
        obs,rew,done,_ = env.step(action)
        #print(rew)
        #allA_tmp.append(action)
        #env.handle_keys(action)

        #all_actions.append(action)
        #handle_keys(action)
        allR.append(rew)
        
        policy.rewards.append(rew)



        epLen += 1
        if done:
            finish_episode()
            policy.h_1 = policy.h_1.detach()
            policy.c_1 = policy.c_1.detach()
            policy.reset_lstm()


            obs = env.reset()
            sys.stdout.write('\r episode_number:{}   |episode length:{:.3f} running average:{:.3f}'.format(episode_nr,epLen,avgR))
            avgR = alpha_*avgR + (1-alpha_)*epLen
            episode_nr += 1
            epLen = 0
