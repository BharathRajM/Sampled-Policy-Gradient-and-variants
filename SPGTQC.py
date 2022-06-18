#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.distributions import Distribution,Normal


# In[2]:


LOG_STD_MIN_MAX = (-20,2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

# In[5]:

#def quantile_huber_loss_f(quantiles,samples):
#    pairwise_delta = samples[:None,None,:] - quantiles[:, :, :, None]
#    abs_pairwise_delta = torch.abs(pairwise_delta)
#    huber_loss = torch.where(abs_pairwise_delta>1,
#                             abs_pairwise_delta - 0.5,
#                             pairwise_delta ** 2 * 0.5)
    
#    n_quantiles = quantiles.shape[2]
#    tau = torch.arange(n_quantiles, device = DEVICE).float() / n_quantiles + 1 / 2 / n_quantiles
#    loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
#    return loss

def quantile_huber_loss_f(quantiles,samples):
    samples = samples.reshape(quantiles.shape)
    pairwise_delta = samples[:,:] - quantiles[:, :]
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta>1,
                             abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)
    
    n_quantiles = quantiles.shape[2]
    tau = torch.arange(n_quantiles, device = device).float() / n_quantiles + 1 / 2 / n_quantiles
    
    loss = (torch.abs(tau[None, None, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
    return loss
                                   

class TanhNormal(Distribution):
    def __init__(self,normal_mean,normal_std):
        super(TanhNormal,self).__init__()
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        
        self.standard_normal = Normal(torch.zeros_like(self.normal_mean,device = device),
                                      torch.ones_like(self.normal_std,device = device))
        
        self.normal = Normal(normal_mean,normal_std)
        
    def log_prob(self,pre_tanh):
        log_det = 2 * np.log(2) + F.logsigmoid( 2 * pre_tanh) + F.logsigmoid(-2 * pre_tanh)
        result = self.normal.log_prob(pre_tanh) - log_det
        return result
    
    def rsample(self):
        pretanh = self.normal_mean + self.normal_std * self.standard_normal.sample()
        return torch.tanh(pretanh),pretanh


# In[8]:


class Mlp(nn.Module):
    def __init__(self, input_size, hidden_sizes,output_size):
        #hidden_sizes = [256,256] list
        super(Mlp,self).__init__()
        self.fcs = []
        in_size = input_size
        for i,next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size,next_size)
            self.add_module(f'fc{i}',fc)
            self.fcs.append(fc)
            in_size = next_size
        self.last_fc = nn.Linear(in_size,output_size)
        
    def forward(self,inp):
        h = inp
        for fc in self.fcs:
            h = F.relu(fc(h),inplace=False)
        output = self.last_fc(h)
        return output


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
        action = F.relu(action,inplace=False)
        
        action = self.l2(action)
        action = F.relu(action,inplace=False)
        
        action = self.l3(action)
        
        mean,log_std = action.split([self.action_dim,self.action_dim],dim=1)
        log_std = log_std.clamp(*LOG_STD_MIN_MAX)
        
        if self.training:
            std = torch.exp(log_std)
            tanh_normal = TanhNormal(mean,std)
            action,pre_tanh = tanh_normal.rsample()
            log_prob = tanh_normal.log_prob(pre_tanh)
            log_prob = log_prob.sum(dim=1,keepdim=True)
        
        else:
            action = torch.tanh(mean)
            log_prob = None
        
        return action,log_prob
    


# In[ ]:


class Critic(nn.Module):
    def __init__(self,state_dim,action_dim,n_quantiles,n_nets):
        super(Critic,self).__init__()
        
        self.nets = []
        self.n_quantiles = n_quantiles
        self.n_nets = n_nets
        
        for i in range(n_nets):
            net = Mlp(state_dim+action_dim,[256,256,256],n_quantiles)
            self.add_module(f'qf{i}',net)
            self.nets.append(net)
        
    def forward(self,state,action):
        
        sa_distribution = torch.cat([state,action],1)
        quantiles = torch.stack(tuple(net(sa_distribution) for net in self.nets ),dim=1)
        return quantiles



class SPGTQC(object):
    
    def __init__(self, state_dim, action_dim, n_quantiles, n_nets, top_quantiles_to_drop, max_action, target_entropy, discount=0.99, tau=0.001):
        
        
        
        self.actor = Actor(state_dim,action_dim,max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=3e-4)
        
        self.critic = Critic(state_dim,action_dim,n_quantiles,n_nets).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=3e-4)
        
        self.log_alpha = torch.zeros((1,),requires_grad=True,device = device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha],lr=3e-4)
        
        self.discount = discount 
        self.tau = tau
        self.top_quantiles_to_drop = top_quantiles_to_drop*n_nets
        self.quantiles_total = self.critic.n_quantiles * self.critic.n_nets
        
        self.target_entropy = target_entropy
        
        
        
    def select_action(self,state):
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        action,_ = self.actor(state)
        #print("ACTION:--------\n",action)
        return action.cpu().data.numpy().flatten()
        
    def train_tqc(self,replay_buffer,sigma_noise,search,batch_size=256):
        
        #sample from buffer
        state,action,next_state,reward,not_done = replay_buffer.sample(batch_size)
        
        alpha = torch.exp(self.log_alpha)
        
        with torch.no_grad():
            
            new_next_action, next_log_pi = self.actor(next_state)
            
            next_z = self.critic_target(next_state,new_next_action)
            sorted_z,_ = torch.sort(next_z.reshape(batch_size,-1))
            sorted_z_part = sorted_z[:, :self.quantiles_total-self.top_quantiles_to_drop]
            
            #compute target
            target = reward + not_done*self.discount*(sorted_z - alpha*next_log_pi)
        
        cur_z = self.critic(state,action)
        critic_loss = quantile_huber_loss_f(cur_z,target)
              
        new_action,log_pi = self.actor(state)
        alpha_loss = -self.log_alpha*(log_pi + self.target_entropy).detach().mean()
        actor_loss = (alpha*log_pi - self.critic(state,new_action).mean(2).mean(1,keepdim=True)).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1-self.tau)*target_param.data)
            
    def train(self,replay_buffer,sigma_noise,search,batch_size=256):
        
        alpha = torch.exp(self.log_alpha)
        
        #sample from buffer
        state,action,next_state,reward,not_done = replay_buffer.sample(batch_size)
        
        pi_s,_ = self.actor(state)
        policyQ_d = self.critic(state,pi_s).mean(2).mean(1,keepdim=True)
        best = pi_s
        
        #current Qd for (s,a)
        current_Q_d = self.critic(state,action).mean(2).mean(1,keepdim=True)
        
        cond_1 = current_Q_d > policyQ_d
        indices_1 = cond_1.nonzero(as_tuple=False)
        best[indices_1] = action[indices_1]
        
        #SPG now apply gaussian noise w.r.t this current best action by taking sigma as the std deviation
        
        with torch.no_grad():
            
            for s in range(search):
                
                sampled_action = best + (torch.randn(best.size())*sqrt(sigma_noise)).to(device)
                Q_d_1 = self.critic(state,sampled_action).mean(2).mean(1,keepdim=True)
                Q_d_2 = self.critic(state,best).mean(2).mean(1,keepdim=True)
                
                cond_2 = Q_d_1>Q_d_2
                indices_2 = cond_2.nonzero(as_tuple=False)
                best[indices_2] = sampled_action[indices_2]
            best = torch.clip(best,-1,1).to(device)

        
            # critic loss
            new_next_action, next_log_pi = self.actor(next_state)

            next_z = self.critic_target(next_state,new_next_action)
            sorted_z,_ = torch.sort(next_z.reshape(batch_size,-1))
            sorted_z_part = sorted_z[:, :self.quantiles_total-self.top_quantiles_to_drop]

            #compute target
            target = reward + not_done*self.discount*(sorted_z - alpha*next_log_pi)
        
        cur_z = self.critic(state,action)
        critic_loss = quantile_huber_loss_f(cur_z,target)
              
        new_action,log_pi = self.actor(state)
        alpha_loss = -self.log_alpha*(log_pi + self.target_entropy).detach().mean()
        #actor_loss = (alpha*log_pi - self.critic(state,new_action).mean(2).mean(1,keepdim=True)).mean()
        
        current_action,_ = self.actor(state)
        actor_loss = F.mse_loss(current_action,best)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1-self.tau)*target_param.data)

            
    def save(self,filename):
        torch.save(self.critic.state_dict(),filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(),filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),filename + "_actor_optimizer")
        
        torch.save(self.log_alpha,filename+"_log_alpha")
        torch.save(self.alpha_optimizer.state_dict(),filename + "_alpha_optimizer")
        
        
        
    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
        
        self.log_alpha = torch.load(filename+"_log_alpha")
        self.alpha_optimizer.load_state_dict(torch.load(filename + "_log_alpha_optimizer"))
        




# In[ ]:





