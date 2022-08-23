#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Normal
from math import sqrt


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
LOG_STD_MIN_MAX = (-20,2)

# In[ ]:


class TanhNormal(Distribution):
    def __init__(self,normal_mean,normal_std):
        super(TanhNormal,self).__init__()
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        
        self.standard_normal = Normal(torch.zeros_like(self.normal_mean,device = device),
                                      torch.ones_like(self.normal_std,device = device))
        
        self.normal = Normal(normal_mean,normal_std)
        self.epsilon = 1e-6
        
    def log_prob(self,pre_tanh):
        log_det = 2 * np.log(2) + F.logsigmoid( 2 * pre_tanh) + F.logsigmoid(-2 * pre_tanh)
        result = self.normal.log_prob(pre_tanh) - log_det
        return result
    
    def rsample(self):
        pretanh = self.normal_mean + self.normal_std * self.standard_normal.sample()
        return torch.tanh(pretanh),pretanh
    
    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)
        
    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)


# In[8]:


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor,self).__init__()
        
        self.l1 = nn.Linear(state_dim,256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,2 * action_dim)
        
        self.max_action = max_action #this is used to scale back the tanh output to the env_action[high]
        self.action_dim = action_dim
        
    def forward(self, state):
        
        action = self.l1(state)
        action = F.relu(action)
        
        action = self.l2(action)
        action = F.relu(action)
        
        action = self.l3(action)
        
        mean,log_std = action.split([self.action_dim,self.action_dim],dim=1)
        log_std = log_std.clamp(*LOG_STD_MIN_MAX)
        
        #if self.training:
        std = torch.exp(log_std)
        tanh_normal = TanhNormal(mean,std)
        action,pre_tanh = tanh_normal.rsample()
        action = self.max_action * action
        return pre_tanh,action, std
        


# In[9]:





# In[10]:


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
    
    def Q1(self,state,action):
        state_action = torch.cat([state,action],1)
        
        q1 = self.l1(state_action)
        q1 = F.relu(q1)
        q1 = self.l2(q1)
        q1 = F.relu(q1)
        q1 = self.l3(q1)
        
        return q1

# In[ ]:





# In[ ]:


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
        _,action,_ = self.actor(state)
        return action.cpu().data.numpy().flatten()
    
    
    def get_optimistic_exploration_action(self,state,beta_UB=4.66,delta=23.53):
        
        
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        
        # obtain the pre_tanh activatio actions
        pre_tanh_mu_T,tanh_mu_T, std = self.actor(state)
        pre_tanh_mu_T.requires_grad_()
        
        #tanh_mu_T = torch.tanh(pre_tanh_mu_T)
        
        # get upper bound of Q estimates
        Q1,Q2 = self.critic(state,tanh_mu_T)
        
        #mean
        mu_Q = (Q1+Q2)/2.0
        sigma_Q = torch.abs(Q1-Q2)/2.0
        Q_UB = mu_Q + beta_UB * sigma_Q
        
        
        #obtain the gradient of Q_UB w.r.t "action" evaluated at mu_T        
        grad_Q_UB = torch.autograd.grad(Q_UB,pre_tanh_mu_T)
        grad = grad_Q_UB[0]
        sigma_T = torch.pow(std,2)
        denom = torch.sqrt(torch.sum(
                                    torch.mul(torch.pow(grad,2),sigma_T)
                                    )
                          ) + 10e-6

        #get change in mu
        mu_C = sqrt(2.0 * delta) * torch.mul(sigma_T,grad) / denom
        mu_E = pre_tanh_mu_T + mu_C
        dist = TanhNormal(mu_E, std)
        ac = dist.sample()
        ac_np = ac.cpu().data.numpy().flatten()
        return ac_np
    
    def train(self, replay_buffer,sigma_noise,search,batch_size=256):
        
        #sample from buffer
        state,action,next_state,reward,not_done = replay_buffer.sample(batch_size)
        
        # ACTOR LOSS
        actor_loss = None
        _,pi_s,_ = self.actor(state)
        policy_Q = self.critic.Q1(state,pi_s)
        best = pi_s
        
        current_Q_buffer = self.critic.Q1(state,action)
        cond_1 = current_Q_buffer > policy_Q
        indices_1 = cond_1.nonzero(as_tuple=False)
        best[indices_1] = action[indices_1]
        
        for _ in range(search):
            sampled_action = best + (torch.randn(best.size())*sqrt(sigma_noise)).to(device)
            Q1 = self.critic.Q1(state,sampled_action)
            Q2 = self.critic.Q1(state,best)
            cond_2 = Q1>Q2
            indices_2 = cond_2.nonzero(as_tuple=False)
            best[indices_2] = sampled_action[indices_2]
            
        best = torch.clip(best,-1,1).to(device)
        
        _,current_action,_ = self.actor(state)
        actor_loss = F.mse_loss(current_action,best)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # CRITIC LOSS
        #Q(s) = R + gamma*Q(s+1,a+1)
        
        # Compute the target Q estimate
        with torch.no_grad():
            #target Q values
            pre_tanh_next_action,_,_ = self.actor_target(next_state)
            next_action = torch.tanh(pre_tanh_next_action)
            target_Q1,target_Q2 = self.critic_target(next_state,next_action)
            target_Q = torch.min(target_Q1,target_Q2)
            target_Q = reward + not_done * self.discount * target_Q
        
        current_Q1,current_Q2 = self.critic(state,action)
        critic_loss = F.mse_loss(current_Q1,target_Q) + F.mse_loss(current_Q2,target_Q)
        
        #optimise the critic and take steps in this direction
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        
        
        #update the target models
        for param,target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)

        for param,target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)
            
            
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





# In[1]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




