#!/bin/bash

# Scripts 

for ((i=0;i<10;i+=1))
do
	python main.py \
	--policy "OurDDPG" \
	--env "HalfCheetah-v4" \
	--seed $i
	
	python main.py \
	--policy "OurDDPG" \
	--env "Hopper-v4" \
	--seed $i
	
	python main.py \
	--policy "OurDDPG" \
	--env "Walker2d-v4" \
	--seed $i
	
	python main.py \
	--policy "OurDDPG" \
	--env "Ant-v4" \
	--seed $i
	
	python main.py \
	--policy "OurDDPG" \
	--env "Humanoid-v4" \
	--seed $i
	
	python main.py \
	--policy "OurDDPG" \
	--env "InvertedPendulum-v4" \
	--seed $i
	
	python main.py \
	--policy "OurDDPG" \
	--env "InvertedDoublePendulum-v4" \
	--seed $i
	
	python main.py \
	--policy "OurDDPG" \
	--env "Reacher-v4" \
	--seed $i

done
