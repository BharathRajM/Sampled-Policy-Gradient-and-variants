# Sampled-Policy-Gradient-and-variants
 
# A suite of deterministic algorithms along with proposed extensions of SPG

## Introduction

This repository consists of all the codebase and the supporting files for the MSc thesis entitled "Extensions of Sampled Policy Gradient".
With good coding habits and I present a clean and easy implementation of all the proposed algorithms, well maintained and documented in this repository. We follow the same file pattern as done in Fujimoto's TD3 repository.

## Setup
To begin setting up the environment, create a `pip` virtual environment as follows:
```
python3 -mvenv env
source env/bin/activate
pip install -r requirements.txt
```

## Train the agents

For a quickstart, to train the agents you need to choose the `policy`: DDPG, TD3, SPG, SPGTD3, SPGTQC, SPGOAC. Then you mention the environment `env`. For an access to all the hyper parameters, you can refer the `main.py` script for possible arguments. To train an `SPG` agent in the `Humanoid-v4` environment you do as follows:
```
python main.py --policy SPG --env Humanoid-v4 
```

When the model is undergoing training, the results of the model will be stored in the `results/` directory and the trained model will be stored in the `models/` directory. 

## Results and visualizations

To see the model performance curves and their overestimation bias curves, you can run the ipython notebooks `Graphs.ipnyb` and `Overestimation bias.ipnyb` respectively. To visualize the models performance in the simulation please run the `Visualize_results.ipnyb` notebook. 

## Environments
For our study we used the [MuJoCo library](https://github.com/openai/mujoco-py). Kindly look here for more information
