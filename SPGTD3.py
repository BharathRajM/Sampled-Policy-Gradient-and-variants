#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import numpy as np
from math import sqrt

# In[2]:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim,action_dim, max_action):
        super(Actor,self).__init__()
        
        self.l1 = nn.Linear(state_dim,256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,action_dim)
        
        self.max_action = max_action
        
    def forward(self,state):
        
        action = self.l1(state)
        action = F.relu(action)
        action = self.l2(action)
        action = F.relu(action)
        action = self.l3(action)
        action = torch.tanh(action)
        action = self.max_action * action
        
        return action

# In[4]:


class Critic(nn.Module):
    def __init__(self, state_dim,action_dim):
        super(Critic,self).__init__()
        
        #Q1 architecture
        
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,1)
        
        #Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256,256)
        self.l6 = nn.Linear(256,1)
        
    def forward(self,state,action):
        
        state_action = torch.cat([state,action],1)
        
        q1 = self.l1(state_action)
        q1 = F.relu(q1)
        q1 = self.l2(q1)
        q1 = F.relu(q1)
        q1 = self.l3(q1)
        
        q2 = self.l4(state_action)
        q2 = F.relu(q2)
        q2 = self.l5(q2)
        q2 = F.relu(q2)
        q2 = self.l6(q2)
        
        
        return q1,q2
        
    def Q1(self,state,action):
        state_action = torch.cat([state,action],1)
        
        q1 = self.l1(state_action)
        q1 = F.relu(q1)
        q1 = self.l2(q1)
        q1 = F.relu(q1)
        q1 = self.l3(q1)
        
        return q1


# In[6]:


class SPGTD3(object):
    def __init__(self,state_dim,action_dim,max_action,
                 discount=0.99,tau=0.005,policy_noise=0.2,
                 noise_clip=0.5,policy_freq=2):
        
        self.actor = Actor(state_dim,action_dim,max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        
        self.critic = Critic(state_dim,action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        self.total_it = 0
        
    def select_action(self,state):
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self,replay_buffer,sigma_noise,search,batch_size = 256):
        self.total_it +=1
        
        state,action,next_state,reward,not_done = replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip,self.noise_clip)
            
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action,self.max_action)
            
            #target Q values
            target_Q1,target_Q2 = self.critic_target(next_state,next_action)
            target_Q = torch.min(target_Q1,target_Q2)
            target_Q = reward + not_done * self.discount * target_Q
            
        current_Q1,current_Q2 = self.critic(state,action)
        
        critic_loss = F.mse_loss(current_Q1,target_Q) + F.mse_loss(current_Q2,target_Q)
        
        
        
        actor_loss = None
        
        # delayed policy updates
        if(self.total_it % self.policy_freq == 0):
            pi_s = self.actor(state)
            #policy_Q1,policy_Q2 = self.critic(state,pi_s)
            #policy_Q = torch.min(policy_Q1,policy_Q2)
            policy_Q = self.critic.Q1(state,pi_s)
            best = pi_s

            #get action from replaybuffer for state s
            #current_Q1_buffer,current_Q2_buffer = self.critic(state,action)
            #current_Q_buffer = torch.min(current_Q1_buffer,current_Q2_buffer)
            current_Q_buffer = self.critic.Q1(state,action)

            cond_1 = current_Q_buffer > policy_Q
            indices_1 = cond_1.nonzero(as_tuple=False)
            best[indices_1] = action[indices_1]

            for _ in range(search):

                sampled_action = best + (torch.randn(best.size())*sqrt(sigma_noise)).to(device) #add gaussian noise to best
                #Q1_1,Q1_2 = self.critic(state,sampled_action)
                #Q1 = torch.min(Q1_1,Q1_2)
                Q1 = self.critic.Q1(state,sampled_action)

                #Q2_1,Q2_2 = self.critic(state,best)
                #Q2 = torch.min(Q2_1,Q2_2)
                Q2 = self.critic.Q1(state,best)
                cond_2 = Q1 > Q2
                indices_2 = cond_2.nonzero(as_tuple=False)
                best[indices_2] = sampled_action[indices_2]
            best = torch.clip(best, -1, 1).to(device)
            
            current_action = self.actor(state)
            actor_loss = F.mse_loss(current_action,best)
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            #update the target models
            for param,target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)
                
            for param,target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)
                
        #optimise the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
         
        #return actor_loss,critic_loss,Q1_loss,Q2_loss,,target_Q
                
    def save(self,filename):
        torch.save(self.critic.state_dict(),filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(),filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),filename + "_actor_optimizer")
        
    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


