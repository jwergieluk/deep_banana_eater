# deep_banana_eater

Deep reinforcement learning agent collecting bananas

# Requirements

* The README describes the the project environment details (i.e., the state and
  action spaces, and when the environment is considered solved).

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.

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

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

* The README has instructions for installing dependencies or downloading needed files.
* The README describes how to run the code in the repository, to train the agent.

# Installation instructions

* Unity's mlagents package requires Python 3.6



Clone this git repository including the submodules: 

    git clone --recurse-submodules https://github.com/jwergieluk/deep_banana_eater.git

Required Python modules (see also requirements.txt)

* pytorch
* mlagents (Unity's ML Agents)
* tensorboardX
* click

Installation instructions for mlagents module can be found under 
https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md
Alternatively, the mlagents is a submodule of this repository. 

Clone the repository

    https://github.com/udacity/deep-reinforcement-learning.git
    
navigate to the `deep-reinforcement-learning/python` directory and install the package with 
`pip install .`.



# Questions

* How to control the stand-alone environment?


# Instructions from Udacity (delete this)

a README that describes how someone not familiar with this project should use your 
repository. The README should be designed for a general audience that may not be 
familiar with the Nanodegree program; you should describe the environment that 
you solved, along with how to install the requirements before running the code in your repository.


