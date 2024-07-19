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

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
USING_PRETRAINED = False
MODEL_NAME = 'imitation_model.pt'
FIELD_SIZE = 15
FEATURE_SIZE = 7
NUM_INPUT = FIELD_SIZE * FIELD_SIZE * FEATURE_SIZE
NUM_OUTPUT = 6
HIDDEN_SIZE = 256
NUM_EPOCHS = 8
MAX_LR = 1e-3
START_LR = 1e-3
BATCH_SIZE = 128


class ExpertDataset(Dataset):
    def __init__(self):

        file_path = "./expert_data.npz"

        data = np.load(file_path)
        input = torch.from_numpy(data['states']).float()
        output = torch.from_numpy(data['actions']).float()

        self.X = input
        self.y = output

    def __getitem__(self, index):
        local_x = self.X[index]
        local_y = self.y[index]
        return local_x, local_y

    def __len__(self):
        return len(self.y)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class ActorCriticLSTM(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCriticLSTM, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.lstm = nn.LSTMCell(self.num_inputs, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)

        self.critic_linear = nn.Linear(hidden_size, 1)
        self.actor_linear = nn.Sequential(
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=-1)
        )

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

        self.apply(init_weights)

    def forward(self, x):
        x, _ = self.lstm(x)
        x, _ = self.lstm2(x)
        action_probs = self.actor_linear(x)
        dist = Categorical(action_probs)
        value = self.critic_linear(x)

        return dist, value


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
    for batch_idx, (state, action) in enumerate(train_loader):
        state, action = state.to(device), action.to(device)
        optimizer.zero_grad()

        dist, _ = model(state)
        action_prediction = dist.logits
        action = action.long()
        loss = criterion(action_prediction, action)
        loss.backward()
        optimizer.step()


def validation(model, device, validation_loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for state, action in validation_loader:
            state, action = state.to(device), action.to(device)
            dist, _ = model(state)
            action_prediction = dist.logits
            action = action.long()
            val_loss += criterion(action_prediction, action)
    val_loss /= len(validation_loader)
    print("Validation Loss: ", val_loss)


def main():
    expert_dataset = ExpertDataset()
    train_loader, val_loader = get_data_loader(expert_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCriticLSTM(NUM_INPUT, NUM_OUTPUT, HIDDEN_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=START_LR)

    for i in range(NUM_EPOCHS):
        print("Epoch: ", i)
        train(model, device, train_loader, optimizer, criterion)
        validation(model, device, val_loader, criterion)

    # Store the model
    model_path = "./imitation_model.pt"
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()
