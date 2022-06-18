#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Normal
from math import sqrt


# In[3]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[1]:


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
        
        pre_tanh_action = self.l3(action)
        
        action = torch.tanh(pre_tanh_action)
        
        action = self.max_action * action
        
        return pre_tanh_action,action


# In[4]:


class Critic(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Critic,self).__init__()
        
        self.l1 = nn.Linear(state_dim+action_dim,256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,1)
        
        self.l4 = nn.Linear(state_dim+action_dim,256)
        self.l5 = nn.Linear(256,256)
        self.l6 = nn.Linear(256,1)
        
    def forward(self,state,action):
        
        Q_value1 = self.l1(torch.cat([state,action],1) )
        Q_value1 = F.relu(Q_value1)
        
        Q_value1 = self.l2(Q_value1)
        Q_value1 = F.relu(Q_value1)
        
        Q_value1 = self.l3(Q_value1)
        
        
        Q_value2 = self.l4(torch.cat([state,action],1) )
        Q_value2 = F.relu(Q_value2)
        
        Q_value2 = self.l5(Q_value2)
        Q_value2 = F.relu(Q_value2)
        
        Q_value2 = self.l6(Q_value2)
        
        
        return Q_value1,Q_value2
    


# In[6]:


class SPGOAC(object):
    
    def __init__(self, state_dim,action_dim,max_action,discount=0.99,tau=0.001):
        
        self.actor = Actor(state_dim,action_dim,max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=3e-4)
        
        self.critic = Critic(state_dim,action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=3e-4)
        
        self.discount = discount 
        self.tau = tau

        
    def select_action(self,state):
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        return self.actor(state)[1].cpu().data.numpy().flatten()
    
    def get_optimistic_action(self,state,beta_UB=4.66,delta=23.53):
         
        LOG_SIG_MAX = 2
        LOG_SIG_MIN = -20
        
        # obtain the pre_tanh activatio actions
        pre_tanh_action,action = self.actor(state)
        std = torch.std(pre_tanh_action)
        pre_tanh_action.requires_grad_()
        
        # get upper bound of Q estimates
        Q1,Q2 = self.critic(state,action)
        
        #mean
        mu_Q = (Q1+Q2)/2.0
        
        sigma_Q = torch.abs(Q1-Q2)/2.0
        
        Q_UB = mu_Q + beta_UB * sigma_Q
        
        #obtain the gradient of Q_UB w.r.t pre_tanh_action        
        grad_Q_UB = torch.autograd.grad(Q_UB,pre_tanh_action)
        grad = grad_Q_UB[0]
        
        sigma_T = torch.pow(std,2)
        
        denom = torch.sqrt(torch.sum(
                                    torch.mul(torch.pow(grad,2),sigma_T)
                                    )
                          ) + 10e-6
        
        #get change in mu
        mu_C = sqrt(2.0 * delta) * torch.mul(sigma_T,grad) / denom
        
        mu_E = pre_tanh_action + mu_C
        

        
        dist = Normal(mu_E, std)
        
        ac = dist.sample()[0]
        
        return ac
        
        
    def train(self,replay_buffer,sigma_noise,search,batch_size=256):
        
        #print("sigma:",sigma_noise)
        
        #sample from buffer
        state,action,next_state,reward,not_done = replay_buffer.sample(batch_size)
        
        #if Q(s,a)>Q(s,pi(s)) 
        #    then action = a else action = pi(s)

        pre_tanh_action,pi_s = self.actor(state)
        policy_Q = self.critic(state,pi_s)
        best = pi_s
        
        #get action from replaybuffer for state s
        current_Q_buffer = self.critic(state,action)

        #if(current_Q_buffer>policy_Q):
        #    best = action
        #else:
        #    best = mu_s

        #best = best.cpu().data.numpy().flatten() #numpy array
        
        
        #cond_1 = current_Q_buffer > policy_Q
        #indices_1 = cond_1.nonzero(as_tuple=False)
        #best[indices_1] = action[indices_1]
        
        #get the optimistic action from OAC and compare with current best action
        
        optimistic_action = self.get_optimistic_action(state,beta_UB=4.66,delta=23.53)
        best = optimistic_action

        #SPG: Now apply gaussian noise w.r.t this best action by taking sigma as the std deviation (No searches involved)

        #SPG-OffGE: Do this "S=Search" times with the gaussian noise to find the best action with Q>Q(best)
        for _ in range(search):
            
            #sampled_action = best + np.random.normal(0,sigma_noise,size=best.shape) #add gaussian noise to best
            #sampled_action = torch.FloatTensor(sampled_action).to(device) #convert back to tensor
            
            sampled_action = best + (torch.randn(best.size())*sqrt(sigma_noise)).to(device) #add gaussian noise to best

            Q1,_ = self.critic(state,sampled_action)
            Q2,_ = self.critic(state,best)

            #if(Q1>Q2):
            #    best = sampled_action
            #best = best.cpu().data.numpy().flatten() # convert back to numpy
            #convert numpy array back to tensor
            #best = torch.FloatTensor(best).to(device)
            
            cond_2 = Q1 > Q2
            indices_2 = cond_2.nonzero(as_tuple=False)
            best[indices_2] = sampled_action[indices_2]
        best = torch.clip(best, -1, 1).to(device)

#        spg_Q = self.critic(state,best)
#        if(spg_Q>policy_Q):
#            target = best
#        else:
#            target = pi_s

        #Now calculate critic loss and actor loss
        #CRITIC LOSS
        #Q(st,at) = r + gamma*Q(st+1,at+1)

        target_Q = self.critic_target(next_state,self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # compute loss
        current_Q_buffer = self.critic(state,action)
        critic_loss = F.mse_loss(current_Q_buffer,target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #ACTOR LOSS : mean squared error loss between the action from buffer and target
        pre_tanh_action,current_action = self.actor(state)
        actor_loss = F.mse_loss(current_action,best)
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