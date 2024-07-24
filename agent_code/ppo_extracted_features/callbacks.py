""" 
This File is called by the environment and manages the agents movements
Implementation of a PPO algorithm with LSTM and MLP networks as Actor Critic

Deep learning approach without feature engineering:
Board is representet in 15x15x7 vector
Symmetry of board is leveraged

Current status:
Agent learn, but gets stuck on bad local maxima. Behavioral cloning to solve issue, but results are still bad
Ideas:
Network not deep enough, reward system not dense enough, feature engeneering maybe nececarry


Author: Luke Voss
"""

from collections import deque

import torch

from ppo import PPOAgent
from extracted_features import state_to_features


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    """
    # Hyperparameter
    self.MAX_COORD_HISTORY = 6
    FEATURE_SIZE = 20
    HIDDEN_SIZE = 256
    NETWORK_TYPE = 'LSTM'

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.current_round = 0
    # Will get changed to true if setup_trainig is called
    self.train = False

    # Agent Position history before normalization
    self.agent_coord_history = deque([], self.MAX_COORD_HISTORY)

    self.agent = PPOAgent(pretrained_model=None,
                          input_feature_size=FEATURE_SIZE,
                          hidden_size=HIDDEN_SIZE,
                          network_type=NETWORK_TYPE,
                          device=self.device)


def reset_self(self, game_state):
    self.agent_coord_history = deque([], self.MAX_COORD_HISTORY)
    self.current_round = game_state["round"]


def is_new_round(self, game_state):
    return game_state["round"] != self.current_round


def act(self, game_state: dict) -> str:
    """
    Agent parses the input, thinks, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    Author: Luke Voss
    """
    if self.is_new_round(game_state):  # TODO: Is "self" correct?
        self.reset_self(game_state)

    # Board History before agent position is normalized
    self.agent_coord_history.append(game_state['self'][3])

    feature_vector = state_to_features(game_state).to(self.device)
    next_action = self.agent.act(feature_vector, train=self.train)

    return next_action
