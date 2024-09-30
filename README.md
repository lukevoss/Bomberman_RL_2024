<div align="center">

# Reinforcement Learning for Bomberman SS2024

</div>
<div align="center">
  <img src="images/bomberman.png" width="500" alt="Bomberman">
</div>

**Authors:**
- Max Tiedl
- Bastian Müller
- Luke Voß


A project for the course:  
*Machine Learning Essentials*

---

**Institution:**  
Ruprecht-Karls-University Heidelberg  
Faculty of Mathematics and Computer Science  
Department of Computer Science  

**Last Update:** 30.09.2024


## Project Overview

Due to the confidentiality requirements of this project, the full report cannot be made publicly available. Below is a brief overview of our work and a presentation of the various agents we have developed.

### Best Performing Agents

#### 1. Echo 

**Framework:** Proximal Policy Optimization (PPO)  
**Average Score:** 5.21  
**Location:** `agent_code/echo`

The Echo Agent, our top performer, utilized imitation learning to acquire skills. This learning approach was primarily based on behaviors observed from our second-best agent, Atom.

#### 2. Atom

**Framework:** Q-Learning  
**Average Score:** 5.04  
**Location:** `agent_code/atom`


Atom has demonstrated robust performance and serves as the foundational model for the Echo Agent's training through imitation.

### Data Generation for Imitation Learning

**Agents:** Rule-Based Agent, Atom  
**Function:** Generates datasets detailing actions taken and rewards received during gameplay.  
**Location:** `agent_code/data_generator_*`  

The datasets generated by these agents are crucial for training other agents via imitation learning. The data includes each action taken by the agent and the corresponding reward, facilitating detailed analysis and model training.

### Imitation Learning Process

**Tools Used:** PyTorch  
**Location:** `imitation_learning`  

1. **Dataset Creation:** Convert raw data into a structured PyTorch dataset using `create_dataset` files found in the `imitation_learning` folder.
2. **Training:** The logic to mimic specified agent behaviors is contained within the imitation learning scripts in this folder. Execution of these scripts requires a pre-collected dataset.

### Dataset Availability

Due to size constraints on GitHub, we are unable to host the full dataset directly. If you require access to this dataset, please contact one of the authors.

---
For more details or access to the datasets, please reach out to the corresponding authors via the GitHub repository.




