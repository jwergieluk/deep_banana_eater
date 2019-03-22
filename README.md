# Deep banana eater

A deep reinforcement learning agent collecting bananas.

This is a Python implementation of deep-q-network based reinforcement learning agent 
learning to collect yellow bananas and avoid blue bananas in a simulated 3D environment. 

![Environment screenshot](env-screenshot.png)

# Description of the environment

State and action space, rewards; when is the env solved?

For this project, you will train an agent to navigate (and collect bananas!) in
a large, square world.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1
is provided for collecting a blue banana. Thus, the goal of your agent is to
collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with
ray-based perception of objects around the agent's forward direction. Given
this information, the agent has to learn how to best select actions. Four
discrete actions are available, corresponding to:

    0 - move forward.
    1 - move backward.
    2 - turn left.
    3 - turn right.

The task is episodic, and in order to solve the environment, your agent must
get an average score of +13 over 100 consecutive episodes.

# Installation instructions

The scripts in this repository require Python 3.6 and the following packages to run properly: 

* pytorch 1.0
* numpy
* pandas
* matplotlib
* requests

The installation instructions are as follows (tested on a Linux system): 

0. Clone this repository using
```commandline
git clone https://github.com/jwergieluk/deep_banana_eater.git
```

1. Install Anaconda Python distribution: https://www.anaconda.com/distribution/#download-section
2. Create a virtual environment with all the necessary packages and activate it:

```commandline
conda create -n deep_banana_eater -c pytorch python=3.6 pytorch torchvision numpy pandas matplotlib requests
conda activate deep_banana_eater
```

3. Clone Udacity's `deep-reinforcement-learning` repository and install the necessary Python package
into the environment:
```commandline
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python 
pip install .
```

4. Download the environment files using the provided script:
```commandline
python download_external_dependencies.py
```

5. Clone and install the `ml-agents` package provided by Unity: 
```commandline
git clone https://github.com/Unity-Technologies/ml-agents.git
cd ml-agents/ml-agents
pip install .
```

All development and testing of this code was performed on an Arch Linux system in Mar 2019. 

# Usage

## Watch a trained agent

Use the following command to load a pretrained agent and watch the agent's interactions with the environment: 
```commandline
python deep_banana_eater.py test --load-weights-from dqn-weights.bin
```

## Training the agent

The `train` command of the `deep_banana_eater.py` script can be used to train an agent 
and save the learned parameters to disk: 
```commandline
python deep_banana_eater.py train --max-episodes 1800
```

The above command runs 1800 training episodes and saves the results to the `runs` directory.

# License

deep_banana_eater is released under the MIT License. See LICENSE file for details.

Copyright (c) 2019 Julian Wergieluk

# Instructions from Udacity (delete this)

a README that describes how someone not familiar with this project should use your 
repository. The README should be designed for a general audience that may not be 
familiar with the Nanodegree program; you should describe the environment that 
you solved, along with how to install the requirements before running the code in your repository.


