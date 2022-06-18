#!/bin/bash

# Scripts 

for ((i=0;i<10;i+=1))
do
	python main.py \
	--policy "SPG" \
	--env "HalfCheetah-v4" \
	--seed $i
	
	python main.py \
	--policy "SPG" \run
	--env "Hopper-v4" \
	--seed $i
	
	python main.py \
	--policy "SPG" \
	--env "Walker2d-v4" \
	--seed $i
	
	python main.py \
	--policy "SPG" \
	--env "Ant-v4" \
	--seed $i
	
	python main.py \
	--policy "SPG" \
	--env "Humanoid-v4" \
	--seed $i
	
	python main.py \
	--policy "SPG" \
	--env "InvertedPendulum-v4" \
	--seed $i
	
	python main.py \
	--policy "SPG" \
	--env "InvertedDoublePendulum-v4" \
	--seed $i
	
	python main.py \
	--policy "SPG" \
	--env "Reacher-v4" \
	--seed $i

done
