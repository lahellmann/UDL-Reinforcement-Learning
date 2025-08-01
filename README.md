# Reinforcement Learning (based on the book Understanding Deep Learning by Simon J.D. Prince[[1]](#references))
This repository contains a project for the seminar 'Understanding Deep Learning' by Lukas Niehaus in the summer term 2025 at the University of Osnabrück. 
## **Table of Contents**
- [Overview](#overview)
- [QuickStart](#quickstart)
- [Structure of this repository](#structure-of-this-repository)
- [The two Agents](#the-two-agents)
- [References](#references)
---
## Overview
Reinforcement Learning is a type of Machine Learning where an agent learns to make decisions by interacting with an environment and recieves rewards for good behaviour and penalties for bad ones.

Chess therefore is a great option for reinforcement learning, because 
- it has clear rules and goals
- every move affects future possibilities
- it allows the agent to explore, learn from mistakes, and improve over time.
  
The aim of this project is to develop reinforcement learning agents for bullet chess, where each player has only 60 seconds total to complete the game. Unlike traditional chess agents that focus solely on move quality, our agents are designed to also manage time pressure, learning to make decisions that are not only strong but also fast when required.

We are going to use DQN and policy-value with alpha-zero style training for our agents. 
You can play against any of these, if you please.

## QuickStart
To start of quickly and be able to execute the code properly, follow this guide.

Fist we need to install [Git](#git) to be able to clone this repository.
Then decide, whether you want to set up your virtual environment with [venv](#venv) (built into Python) or [Conda](#conda) (a package and environment manager from Anaconda/Miniconda).

### Install Git
<a name="git"></a>
Download and install Git:

- Visit the [official Git website](https://git-scm.com/) to download the latest version of Git.
- Follow the installation instructions for your operating system.

### Clone the Git Repository

- Open a terminal or command prompt.
- Go to the directory where you want to store everything regarding the course:
```bash
cd <directory_name>
```
- Clone the Git repository:
```bash
git clone https://github.com/lahellmann/UDL-Reinforcement-Learning
```
- Change into the cloned repository:
```bash
cd UDL-Reinforcement-Learning
```

### Set Up a Virtual Environment (venv)
<a name="venv"></a>

Download and install Python:
- Visit the [official Python website](https://www.python.org/) to download the latest version of Python.
- During installation, make sure to check the option that adds Python to your system's PATH.

- Create a virtual environment:
```bash 
python -m venv venv
```
- Activate the virtual environment:
--> On Windows:
```bash
.\venv\Scripts\activate
```
--> On Unix or MacOS:
```bash
source venv/bin/activate
```
- Install required packages
```bash
pip install -r requirements.txt
```

### Set Up a Virtual Environment (conda)
<a name="conda"></a>
- Create a virtual environment:
1. Open your terminal (Command Prompt on Windows, Terminal on macOS/Linux).
2. Navigate to the directory where you saved the environment.yml file. (This should be YOUR_PATH/OED_Game_Theoretic_Framework_MBRL/ )
3. Execute the following command to create the environment:

```bash 
conda create -m venv -f environment.yml
```
- Activate the virtual environment:
--> On Windows, Unix and MacOS:
```bash
conda activate venv
```
- Install required packages
```bash
pip install -r requirements.txt
```

## Structure of this repository
'agent_dqn.py' - This file contains all relevant functions and definitions for our agent that uses Deep Q-networks to learn. ([open](agent_dqn.py))

'agent_policy_value.py' - This file contains all relevant functions and definitions for our agents policy. ([open](agent_policy_value.py))

'chess_dqn_model.pth' - This file contains the trained DQN-agent. ([open](chess_dqn_model.pth))

'environment.py' - This file contains all relevant functions and definitions for the environment our agent acts in. ([open](environment.py))

'main.ipynb' - This notebook contains the script for initializing, training and finally saving the DQN model. We will add the agent-critic model here as well. ([open](main.ipynb))

'play.ipynb' - This notebooks let's you play against our agent in Bullet Chess. You'll enter moves in UCI-format. Have fun! ([open](play.ipynb))

'utils.py' - This file contains useful functions that might be used in the future. ([open](utils.py))

## The two Agents
### DDQN(Double Deep Q-Network)
DDQN is part of the family of value-based reinforcement learning algorithms. It extends the classical Deep Q-Network (DQN) by addressing the problem of overestimation in action-value estimates. In standard DQN, the same network is used to both select and evaluate the best action, which can lead to optimistic value estimates. DDQN mitigates this by decoupling action selection from action evaluation.
Core components:

*    Online Q-network: predicts Q-values for the current state and is updated every training step.
*    Target Q-network: a delayed copy of the online network used to evaluate the Q-value of the next state, updated less frequently (e.g. every few hundred steps) to stabilize learning.
*    Replay buffer: stores experiences of the form (state, action, reward, next state, done) to break temporal correlations and allow for more sample-efficient learning.
*    Epsilon-greedy policy: balances exploration and exploitation during training by sometimes selecting random actions.

Basic idea:
```bash
1. Start an episode at initial state s
2. Choose an action a using an ε-greedy policy derived from the online Q-network
3. Take action a in the environment
4. Observe reward r and next state s′
5. Choose next action a′ using the online network, but evaluate it using the target Q-network
6. Compute target value:
    y=r+γQtarget(s′,arg⁡max⁡aQonline(s′,a))y=r+γQtarget​(s′,argmaxa​Qonline​(s′,a))
7. Update Q-network by minimizing the loss between current Q-value and target y
8. Set s ← s′, a ← a′, and repeat until the episode ends
9. Periodically, copy the online network weights into the target network

Continue training until a convergence threshold is reached or a set number of episodes complete
The policy is derived from the trained Q-values:
π(s)=arg⁡max⁡aQ(s,a)π(s)=argmaxa​Q(s,a)
```

### Policy-Value with AlphaZero-style training
This agent is based on AlphaZero, a powerful reinforcement learning framework that combines policy and value learning with Monte Carlo Tree Search (MCTS). Unlike DDQN, which estimates action-values directly, AlphaZero learns a policy (which actions to prefer) and a value function (how good a position is) using a shared neural network.

Instead of learning purely from rewards after taking actions, the agent improves itself through self-play guided by MCTS, which explores and refines the policy using simulations from the neural network[[2]](#references).

Core components:

*    Policy-Value Neural Network: Given a state (e.g. a chess board), it outputs:
      *    A policy (π) — a probability distribution over legal actions
      *    A value (v) — the expected game outcome from the current position
*    Monte Carlo Tree Search:
      *    Uses the policy and value predictions to guide exploration of possible future states.
      *    Outputs improved target policy (based on visit counts) and simulated value.
*    Replay buffer: stores (state, π, value) triplets collected during self-play games.
*    Training loop: alternates between self-play and training using stored experiences.

Basic idea
```bash
Start an episode (a self-play game) at initial state s

At each move:
a. Use Monte Carlo Tree Search (MCTS) guided by the current policy-value network
b. MCTS returns a probability distribution π over legal actions
c. Choose action a by sampling from π (adds exploration)
d. Record training tuple: (state s, policy π, current player)
e. Take action a → observe next state s′
f. Set s ← s′, and continue until the game ends

After game ends, assign game result z (+1 win, -1 loss, 0 draw)

For each stored tuple (s, π, player), compute value = z * player_sign
Add (s, π, value) to the replay buffer

After collecting enough games:
a. Sample mini-batches from the buffer
b. Perform gradient descent to minimize loss:
    Policy loss: cross-entropy between predicted and target π
    Value loss: mean squared error between predicted and true value
Repeat the self-play → training cycle

Continue until the policy converges or a preset training limit is reached
The learned policy is directly given by the network’s output distribution over actions
```

## References
<a name="references"></a>
[1] S. J. D. Prince, Understanding Deep Learning. The MIT Press, 2023. [website to book](https://udlbook.github.io/udlbook/)
[2] Silver, D., Schrittwieser, J., Simonyan, K. et al. Mastering the game of Go without human knowledge. Nature 550, 354–359 (2017). [https://doi.org/10.1038/nature2427]
