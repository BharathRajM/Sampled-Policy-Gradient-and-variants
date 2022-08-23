#!/bin/bash

# Scripts 

for ((i=0;i<10;i+=1))
do
	python main.py \
	--policy "TD3" \
	--env "HalfCheetah-v4" \
	--seed $i
	
	python main.py \
	--policy "TD3" \
	--env "Hopper-v4" \
	--seed $i
	
	python main.py \
	--policy "TD3" \
	--env "Walker2d-v4" \
	--seed $i
	
	python main.py \
	--policy "TD3" \
	--env "Ant-v4" \
	--seed $i
	
	python main.py \
	--policy "TD3" \
	--env "Humanoid-v4" \
	--seed $i
	
	python main.py \
	--policy "TD3" \
	--env "InvertedPendulum-v4" \
	--seed $i
	
	python main.py \
	--policy "TD3" \
	--env "InvertedDoublePendulum-v4" \
	--seed $i
	
	python main.py \
	--policy "TD3" \
	--env "Reacher-v4" \
	--seed $i

done
