#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[19]:


# implementation of DDPG based on TD3 paper


# In[12]:


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor,self).__init__()
        
        self.l1 = nn.Linear(state_dim,256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,action_dim)
        
        self.max_action = max_action #this is used to scale back the tanh output to the env_action[high]
        
    def forward(self, state):
        
        action = self.l1(state)
        action = F.relu(action)
        
        action = self.l2(action)
        action = F.relu(action)
        
        action = self.l3(action)
        action = torch.tanh(action)
        
        action = self.max_action * action
        
        return action


# In[17]:


class Critic(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Critic,self).__init__()
        
        self.l1 = nn.Linear(state_dim+action_dim,256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,1)
        
    def forward(self,state,action):
        
        Q_value = self.l1(torch.cat([state,action],1) )
        Q_value = F.relu(Q_value)
        
        Q_value = self.l2(Q_value)
        Q_value = F.relu(Q_value)
        
        Q_value = self.l3(Q_value)
        
        return Q_value


# In[20]:


class DDPG(object):
    
    def __init__(self,state_dim,action_dim,max_action,discount=0.99,tau=0.001):
        
        self.actor = Actor(state_dim,action_dim,max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=3e-4)
        
        self.critic = Critic(state_dim,action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=3e-4)
        
        self.discount = discount # =gamma = for discounting future rewards
        self.tau = tau #used for updating target networks
        
    def select_action(self,state):
        
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
        
    def train(self, replay_buffer, batch_size = 256):
        
        state,action,next_state,reward,not_done = replay_buffer.sample(batch_size)
        
        
        # CRITIC LOSS
        #Q(s) = R + gamma*Q(s+1,a+1)
        
        # Compute the target Q estimate
        target_Q = self.critic_target(next_state,self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()
        
        # current Q
        current_Q = self.critic(state,action)
        
        # compute loss
        critic_loss = F.mse_loss(current_Q,target_Q)
        
        #optimise the critic and take steps in this direction
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        
        # ACTOR LOSS
        actor_loss = - self.critic(state,self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        
        #Update the parameters to target network
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1-self.tau)*target_param.data)
            
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1-self.tau)*target_param.data)
            
    
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
        
        


# In[ ]:





# In[ ]:




