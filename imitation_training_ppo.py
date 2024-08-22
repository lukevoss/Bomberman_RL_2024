"""
File to train the Actor Critic backbone of the PPO algorithm to act like the rule based agent
Efficient training and testing of different Network architectures 
and give a good starting point for reinforcement learning
"""


import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.data import Dataset, random_split
from agent_code.actor_critic import ActorCriticMLP
import agent_code.ppo as ppo

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
USING_PRETRAINED = False
MODEL_NAME = 'imitation_model.pt'
FIELD_SIZE = 15
FEATURE_SIZE = 7
NUM_INPUT = 30
NUM_OUTPUT = 6
HIDDEN_SIZE = 512
NUM_EPOCHS = 5
MAX_LR = 1e-3
START_LR = 2e-3
BATCH_SIZE = 64


class ExpertDataset(Dataset):
    def __init__(self):

        file_path = "./expert_data_ppo.npz"

        data = np.load(file_path)


        self.states = torch.from_numpy(data['states']).float()
        self.actions = torch.from_numpy(data['actions']).float()
        self.masks = torch.from_numpy(data['masks']).float()
        self.rewards = torch.from_numpy(data['rewards']).float()


    def __getitem__(self, index):
        states = self.states[index]
        actions = self.actions[index]
        masks = self.masks[index]
        rewards = self.rewards[index]

        return states, actions, masks, rewards

    def __len__(self):
        return len(self.actions)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


def get_data_loader(expert_dataset, percent_for_training=0.8):
    train_size = int(percent_for_training*len(expert_dataset))
    validation_size = len(expert_dataset)-train_size

    train_dataset, validation_dataset = random_split(
        expert_dataset, [train_size, validation_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        validation_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)

    return train_loader, val_loader


def train(model, device, train_loader, criterion, optimizer):
    model.train()
    for _, (state, action, masks, rewards) in enumerate(train_loader):
        state =  state.to(device)
        action = action.to(device)
        masks = masks.to(device)
        rewards = rewards.to(device)
        optimizer.zero_grad()

        if masks[-1] == 0:
            next_value = torch.tensor(0).to(device).unsqueeze(0)  # Next value doesn't exist
        else:
            _, next_value = model(state[-1])
            
        dist, estimated_returns = model(state)

        returns = ppo.compute_gae_tensors(next_value, rewards,
                                    masks, estimated_returns).to(device)

        action_prediction = dist.logits
        action = action.long()

        
        loss_actor = criterion(action_prediction, action)
        loss_critic = (returns - estimated_returns).pow(2).mean()
        loss = loss_actor + loss_critic
        loss.backward()
        optimizer.step()


def validation(model, device, validation_loader, criterion):
    model.eval()
    val_loss_actor = 0
    val_loss_critic = 0
    
    with torch.no_grad():
        for (state, action, masks, rewards) in validation_loader:
            state =  state.to(device)
            action = action.to(device)
            masks = masks.to(device)
            rewards = rewards.to(device)

            if masks[-1] == 0:
                next_value = torch.tensor(0).to(device).unsqueeze(0)  # Next value doesn't exist
            else:
                _, next_value = model(state[-1])
                
            dist, estimated_returns = model(state)

            returns = ppo.compute_gae_tensors(next_value, rewards,
                                        masks, estimated_returns).to(device)

            action_prediction = dist.logits
            action = action.long()

            
            val_loss_actor += criterion(action_prediction, action)
            val_loss_critic += (returns - estimated_returns).pow(2).mean()
 
    val_loss_actor /= len(validation_loader)
    val_loss_critic /= len(validation_loader)
    print(f"Validation Loss: \n Actor: {val_loss_actor}\n Critic: {val_loss_critic}")


def main():
    expert_dataset = ExpertDataset()
    train_loader, val_loader = get_data_loader(expert_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCriticMLP(NUM_INPUT, HIDDEN_SIZE, NUM_OUTPUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=START_LR)

    for i in range(NUM_EPOCHS):
        print("Epoch: ", i)
        train(model, device, train_loader, criterion, optimizer)
        validation(model, device, val_loader, criterion)

    # Store the model
    model_path = "./imitation_model.pt"
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()
