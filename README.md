# Reinforcement Learning (based on the book Understanding Deep Learning by Simon J. D. Prince[[1]](#references))
This repository contains a project for the seminar 'Understanding Deep Learning' by Lukas Niehaus in the summer term 2025 at the University of Osnabrück. 

## Overview
Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment, recieving rewards for good behaviour and penalties for bad behaviour.

Chess is therefore a great option for reinforcement learning because 
- it has clear rules and goals,
- every move affects future possibilities,
- it allows the agent to explore, learn from mistakes, and improve over time.
  
The aim of this project is to develop reinforcement learning agents for bullet chess, in which each player has only 60 seconds in total to complete the game. Unlike traditional chess agents, which focus solely on move quality, our agents are designed to also manage time pressure, learning to make decisions that are not only strong but also fast, when required.

We are going to use Deep Q-Networks (DQN) and agent-critic policies for our agents. Since this repository is still being worked on, there is only a trained DQN agent so far, that you could play against. We are currently improving the game and training the agent-critic policy. 

## QuickStart
To get started quickly and be able to execute the code properly, follow this guide.
For fastest usage, go to this [jupyter notebook](https://colab.research.google.com/drive/1ae7qFCyGvhH7TT2yE0qTrL9GH5_HAyJ9?usp=sharing) in google collab.


First we need to install [Git](#git) to be able to clone this repository.
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

### Set Up a Virtual Environment (pip)
<a name="venv"></a>

Download and install Python:
- Visit the [official Python website](https://www.python.org/) to download the latest version of Python.
- During installation, make sure to check the option that adds Python to your system's PATH.

- Create a virtual environment:
```bash 
python -m venv venv
```
- Activate the virtual environment:
```bash
.\venv\Scripts\activate # On Windows
source venv/bin/activate # On Unix or MacOS
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
```bash
conda activate venv
```
- Install required packages
```bash
pip install -r requirements.txt
```

## How to use this repository
### Structure of this repository
`agent_dqn.py` - This file contains all relevant functions and definitions for our agent that uses Deep Q-Networks to learn. ([open](agent_dqn.py))

`agent_policy_value.py` - This file contains all relevant functions and definitions for our agents policy. ([open](agent_policy_value.py))

`models/chess_dqn_model.pth` - This file contains the trained DQN-agent. ([open](models/chess_dqn_model.pth))

`environment.py` - This file contains all relevant functions and definitions for the environment our agent acts in. ([open](environment.py))

`main.ipynb` - This notebook contains the script for initializing, training and finally saving the DQN model. We will add the agent-critic model here as well. ([open](main.ipynb))

`play.ipynb` - This notebooks let's you play against our agent in Bullet Chess. You'll enter moves in UCI-format. Have fun! ([open](play.ipynb))

`utils.py` - This file contains useful functions that might be used in the future. ([open](utils.py))

## References
<a name="references"></a>
[1] S. J. D. Prince, Understanding Deep Learning. The MIT Press, 2023. [website to book](https://udlbook.github.io/udlbook/)
