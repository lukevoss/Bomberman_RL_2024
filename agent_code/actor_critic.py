import torch
import torch.nn as nn
from torch.distributions import Categorical


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class ActorCriticMLP(nn.Module):
    """
    Actor Critic with MLP backbone
    """

    def __init__(self, num_inputs, hidden_size, num_outputs, std=0.0):
        super(ActorCriticMLP, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, 1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, 1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=-1)
        )

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

        self.apply(init_weights)

    def forward(self, x):

        action_probs = self.actor(x)
        dist = Categorical(action_probs)
        value = self.critic(x)

        return dist, value


class ActorCriticLSTM(nn.Module):
    """
    Actor Critic with LSTM backbone, shown more capable in behavioral cloning
    """

    def __init__(self, num_inputs, hidden_size, num_outputs, std=0.0):
        super(ActorCriticLSTM, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.lstm = nn.LSTMCell(self.num_inputs, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        # self.lstm3 = nn.LSTMCell(750, 500)
        # self.lstm4 = nn.LSTMCell(500, 250)
        # self.lstm5 = nn.LSTMCell(250, hidden_size)

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
        # x, _ = self.lstm3(x)
        # x, _ = self.lstm4(x)
        # x, _ = self.lstm5(x)

        action_probs = self.actor_linear(x)
        dist = Categorical(action_probs)
        value = self.critic_linear(x)

        return dist, value
