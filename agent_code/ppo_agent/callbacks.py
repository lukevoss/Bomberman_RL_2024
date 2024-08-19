""" 
This File is called by the environment and manages the agents movements
Implementation of a PPO algorithm with LSTM and MLP networks as Actor Critic

Deep learning approach with feature engineering:

Current status:
Agent learn, but gets stuck on bad local maxima. Behavioral cloning to solve issue, but results are still bad
Ideas:
Network not deep enough, reward system not dense enough, feature engeneering maybe nececarry


Author: Luke Voss
"""

from collections import deque

import torch

from agent_code.ppo import PPOAgent
from agent_code.feature_extraction import state_to_small_features_ppo, FEATURE_VECTOR_SIZE
from agent_code.utils import print_large_feature_vector


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
    self.MAX_COORD_HISTORY = 7
    HIDDEN_SIZE = 512
    NETWORK_TYPE = 'MLP'
    PRETRAINED_MODEL = "ppo_model.pt"
    self.MODEL_NAME = "ppo_model"

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model is run on: {self.device}")
    self.current_round = 0
    self.max_opponents_score = 0
    self.sum_points_per_game = 0
    self.last_points = 0

    self.all_coins_game = []

    # Agent Position history before normalization
    self.agent_coord_history = deque([], self.MAX_COORD_HISTORY)

    self.agent = PPOAgent(pretrained_model=PRETRAINED_MODEL,
                          input_feature_size=FEATURE_VECTOR_SIZE,
                          hidden_size=HIDDEN_SIZE,
                          network_type=NETWORK_TYPE,
                          device=self.device)


def reset_self(self, game_state: dict):
    self.agent_coord_history = deque([], self.MAX_COORD_HISTORY)
    self.current_round = game_state["round"]
    self.max_opponents_score = 0
    self.all_coins_game = []


def is_new_round(self, game_state: dict) -> bool:
    return game_state["round"] != self.current_round


def act(self, game_state: dict) -> str:
    """
    Agent parses the input, thinks, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    Author: Luke Voss
    """

    if is_new_round(self, game_state):  # TODO Correct?
        reset_self(self, game_state)
        self.sum_points_per_game += self.last_points
        round = game_state['round']
        if ((round-1) % 10) == 0:
            print(f"\nAverage points per game: {self.sum_points_per_game/10}")
            self.sum_points_per_game = 0

    self.last_points = game_state['self'][1]

    self.agent_coord_history.append(game_state['self'][3])
    living_opponent_scores = [opponent[1] for opponent in game_state['others']]
    if living_opponent_scores:
        max_living_opponent_score = max(living_opponent_scores)
        self.max_opponents_score = max(
            self.max_opponents_score, max_living_opponent_score)

    coins = game_state['coins']
    if coins != []:
        for coin in coins:
            if coin not in self.all_coins_game:
                self.all_coins_game.append(coin)

    num_coins_already_discovered = len(self.all_coins_game)

    feature_vector = state_to_small_features_ppo(
        game_state, num_coins_already_discovered).to(self.device)
    
    
    next_action = self.agent.act(feature_vector)

    # print_feature_vector(feature_vector)
    # print(f"Action took: {next_action}")

    return next_action
