import logging
import numpy as np

from tqdm import tqdm

from agent_code.q_learning import QLearningAgent
from agent_code.utils import ACTIONS


NUM_EPOCHS = 1


def train(agent, train_dataset):

    for _, (old_state, action_idx, rewards, new_state) in tqdm(enumerate(train_dataset),desc="Training Q-Table"):
        if (new_state == -1).any():
            new_state = None
        else:
            new_state = new_state.tolist()
        action = ACTIONS[int(action_idx)]
        old_state = old_state.tolist()
        agent.training_step(old_state, action, rewards, new_state)
        


def validation(agent, validation_dataset):

    correct_actions = 0
    
    for (state, action_idx, rewards, new_states) in validation_dataset:
        action = ACTIONS[int(action_idx)]
        agent_action = agent.act(state.tolist(),n_round=0,train=False)
        if agent_action == action:
            correct_actions += 1
    table = agent.q_table 
    print(f"Validation Accuracy: {(correct_actions/len(validation_dataset))*100}%")
    print(f"Length Q-Table: {len(agent.q_table)}")


def main():

    file_path = "./expert_data.npz"
    expert_dataset = np.load(file_path)
    old_states = expert_dataset['old_states']
    new_states = expert_dataset['new_states']
    actions_idx = expert_dataset['actions']
    rewards = expert_dataset['rewards']
    len_dataset = len(actions_idx)
    train_size = int(len_dataset*0.8)

    agent = QLearningAgent(pretrained_model="q_table.pkl", learning_rate=0.00001)
    train_dataset = list(zip(old_states[:train_size], actions_idx[:train_size], rewards[:train_size], new_states[:train_size]))
    val_dataset = list(zip(old_states[train_size:], actions_idx[train_size:], rewards[train_size:], new_states[train_size:]))


    for i in range(NUM_EPOCHS):
        print("Epoch: ", i)
        train(agent, train_dataset)
        validation(agent, val_dataset)

    
    # Store the model
    agent.save(model_name="imitation_q_table.pkl")


if __name__ == "__main__":
    main()
