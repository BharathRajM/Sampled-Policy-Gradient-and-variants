#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import gym
import argparse
import os
import utils

import OurDDPG
import SPG
import TD3
import SPGTQC

# In[17]:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_policy(policy, env_name,seed,eval_episodes=10):
    
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    avg_reward = 0
    
    for _ in range(eval_episodes):
        
        state,done = eval_env.reset(), False
        
        while not done:
            action = policy.select_action(np.array(state))
            state,reward,done,_ = eval_env.step(action)
            avg_reward+=reward
            
    avg_reward/=eval_episodes
    
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward
    
    

def get_estimation_bias(policy_name,policy,env_name,discount,seed,max_steps,max_action):
    
    
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    
    # get the starting state
    state,done = eval_env.reset(),False
    episode_reward = 0

    if(policy_name=="SPG" or policy_name=="OurDDPG" or policy_name=="DDPG"):
        #estimated Q(s,a)
        starting_state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        starting_action = policy.actor(starting_state)
        estimated_Q = policy.critic(starting_state,starting_action)

        starting_action = starting_action.cpu().data.numpy().flatten()
        estimated_Q = estimated_Q.cpu().data.numpy().flatten()

        true_Q = 0

        for t in range(int(max_steps)):

            if(done):
                break

            action = (policy.select_action(np.array(state))).clip(-max_action,max_action)
            next_state,reward,done,_ = eval_env.step(action)
            state = next_state
            episode_reward+=reward

            true_Q=true_Q + (discount**t)*reward 
    
    if(policy_name=="TD3"):
        #estimated Q(s,a)
        starting_state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        starting_action = policy.actor(starting_state)
        estimated_Q1,estimated_Q2 = policy.critic(starting_state,starting_action)
        estimated_Q = torch.min(estimated_Q1,estimated_Q2)

        starting_action = starting_action.cpu().data.numpy().flatten()
        estimated_Q = estimated_Q.cpu().data.numpy().flatten()

        true_Q = 0

        for t in range(int(max_steps)):

            if(done):
                break

            action = (policy.select_action(np.array(state))).clip(-max_action,max_action)
            next_state,reward,done,_ = eval_env.step(action)
            state = next_state
            episode_reward+=reward

            true_Q=true_Q + (discount**t)*reward
            
    return estimated_Q,true_Q

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",default="OurDDPG")
    parser.add_argument("--env",default="HalfCheetah-v4")
    parser.add_argument("--seed",default=0,type=int)
    parser.add_argument("--start_timesteps",default=25e3,type=int) #timesteps to start training from
    parser.add_argument("--eval_freq",default=5e3,type=int)
    parser.add_argument("--max_timesteps",default=1e6,type=int)
    
    parser.add_argument("--expl_noise",default=0.1) #standard deviation of gaussian exploration noise
    
    parser.add_argument("--batch_size",default=256,type=int)
    parser.add_argument("--discount",default=0.99)   #
    parser.add_argument("--tau",default = 0.005)
    
    #td3 parameters
    parser.add_argument("--policy_noise",default = 0.2)
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    
    #SPG parameters
    parser.add_argument("--sigma_noise_start",default = 0.7)
    parser.add_argument("--search", default=8,type=int)                # Range to clip target policy noise
    
    
    
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--get_estimates", default=1)                 # Model load file name, "" doesn't load, "default" uses file_name
    
    args = parser.parse_args()
    
    #print("args max_timesteps",args.max_timesteps)
    
    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("------------------------------------------")
    print(f"Policy : {args.policy} , Env : {args.env} , Seed: {args.seed}")
    print("------------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if True and not os.path.exists("./models"):
        os.makedirs("./models")
        
    if not os.path.exists("./results/"+args.env):
        os.makedirs("./results/"+args.env)
        
    if not os.path.exists("./results/"+args.env+"/"+args.policy):
        os.makedirs("./results/"+args.env+"/"+args.policy)


    env = gym.make(args.env)

    # set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    sigma_decay = 1025000
    sigma_noise_start=args.sigma_noise_start
    sigma_noise_final=0.01
    sigma_noise_array = np.logspace(np.log(sigma_noise_start), np.log(sigma_noise_final),int(args.max_timesteps), base=np.exp(1))
    search = args.search
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim" : state_dim,
        "action_dim" : action_dim,
        "max_action" : max_action,
        "discount" : args.discount,
        "tau" : args.tau
    }

    #initialise the policy
    if args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)

    elif args.policy == "TD3" :
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)

    elif args.policy == "DDPG" :
        policy = DDPG.DDPG(**kwargs)

    elif args.policy == "SPG":
        policy = SPG.SPG(**kwargs)

    elif args.policy == "SPGR":
        policy = SPGR.SPGR(**kwargs)

    elif args.policy == "SPGTQC":
        kwargs["n_quantiles"] = 25
        kwargs["n_nets"] = 5
        kwargs["top_quantiles_to_drop"] = 2
        kwargs["target_entropy"] = -np.prod(env.action_space.shape).item()
        policy = SPGTQC.SPGTQC(**kwargs)


    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim,action_dim)

    #evaluate untrained policy    
    evaluations = [eval_policy(policy, args.env, args.seed)]
    #true Q value and estimated Q values
    #true Q value and estimated Q values
    true_Q = []
    estimated_Q = []

    state,done = env.reset(),False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    
    #keep track of rewards accumulated per timestep
    training_rewards_list=[]

    for t in range(int(args.max_timesteps)):

        episode_timesteps+=1

        if t<args.start_timesteps:
            action = env.action_space.sample()

        else:
            action = (policy.select_action(np.array(state)) + np.random.normal(0,max_action * args.expl_noise,size = action_dim)
                     ).clip(-max_action,max_action)


        next_state,reward,done,_ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        #store in replay buffer
        replay_buffer.add(state,action,next_state,reward,done_bool)

        state = next_state
        episode_reward+=reward
        
        
        
        if t>args.start_timesteps:
            
            if(args.policy == "SPG" or args.policy == "SPGR"):
                sigma_noise = max(sigma_noise_final, sigma_noise_start - episode_timesteps/sigma_decay)
                policy.train(replay_buffer,sigma_noise,search,args.batch_size)
            
            else:
                policy.train(replay_buffer,args.batch_size)
            
            

        if done:
            
            print(f"Total Timesteps: {t+1} Episode Num: {episode_num+1} Episode Timestep:{episode_timesteps} Reward:{episode_reward:.3f}")
            training_rewards_list.append(episode_reward)
            state,done = env.reset(),False
            episode_reward = 0
            episode_timesteps = 0
            episode_num+=1
            


        #Evaluate
        if (t+1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy,args.env,args.seed))
            np.save("./results/"+args.env+"/"+args.policy+"/"+file_name+"_eval_rewards",evaluations)
            if(True):
                # save the model, rewards accumulated during training, rewards accumulated testing,
                policy.save(f"./models/{file_name}")
                np.save("./results/"+args.env+"/"+args.policy+"/"+file_name+"_training_rewards",training_rewards_list)
                #np.save(f"./results/{file_name}_training_rewards_list",training_rewards_list)
                
            if(args.get_estimates):
                #Get Q value estimates
                e_Q,t_Q = get_estimation_bias(args.policy,policy,args.env,args.discount,args.seed,args.max_timesteps,max_action)
                estimated_Q.append(e_Q)
                true_Q.append(t_Q)
                np.save("./results/"+args.env+"/"+args.policy+"/"+file_name+"_true_Q",true_Q)
                np.save("./results/"+args.env+"/"+args.policy+"/"+file_name+"_estimated_Q",estimated_Q)    



