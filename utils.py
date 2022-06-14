import torch
import numpy as np
class  ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size = int(1e6)):
        
        self.max_size = max_size
        self.memory_counter = 0
        self.size = 0
        
        self.state = np.zeros((self.max_size,state_dim))
        self.action = np.zeros((self.max_size,action_dim))
        self.next_state = np.zeros((self.max_size,state_dim))
        self.reward = np.zeros((self.max_size,1)) #[[0.],[0.]]
        self.not_done = np.zeros((max_size,1))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def add(self, state, action, next_state, reward, done):
        
        self.memory_counter = (self.memory_counter) % self.max_size
        
        self.state[self.memory_counter] = state
        self.action[self.memory_counter] = action
        self.next_state[self.memory_counter] = next_state
        self.reward[self.memory_counter] = reward
        self.not_done[self.memory_counter] = 1. - done
        
        self.memory_counter += 1
        
    def sample(self,batch_size):
        batch  = np.random.randint(0,self.memory_counter,size = batch_size)
        
        states = self.state[batch]
        actions = self.action[batch]
        next_states = self.next_state[batch]
        rewards = self.reward[batch]
        not_dones = self.not_done[batch]
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        not_dones = torch.FloatTensor(not_dones).to(self.device)

        
        return (states,actions,next_states,rewards,not_dones)