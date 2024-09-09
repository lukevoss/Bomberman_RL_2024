""" 
This File is called by the environment and manages the agents movements
Implementation of a Q-learning algorithm

Current Status:
Agent "atom" continiously achieves about 5.1 points per game, 
in the classic setting with 3 rule based agents as opponents


Author: Luke Voss
"""
from collections import deque

from agent_code.feature_extraction import state_to_small_features
from agent_code.q_learning import QLearningAgent



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
    self.MODEL_NAME = "who_is_gonna_be_the_one.pkl"

    self.current_round = 0
    self.all_coins_game = []
    self.agent_coord_history = deque([], self.MAX_COORD_HISTORY)

    # Pretraining
    # 1) (200 rounds, Max rounds = 100)
    self.agent = QLearningAgent(logger=self.logger, learning_rate=0.9, gamma = 0, max_epsilon = 0.8, min_epsilon = 0.05, decay_rate = 0.001)
    # 2) - 6) (200 rounds)
    # self.agent = QLearningAgent(pretrained_model=self.MODEL_NAME, logger=self.logger, learning_rate=0.9, gamma = 0, max_epsilon = 0.8, min_epsilon = 0.05, decay_rate = 0.001)
    # -- ENABLE BOMBS --
    # 7) - 12) (200 rounds)
    # self.agent = QLearningAgent(pretrained_model=self.MODEL_NAME, logger=self.logger, learning_rate=0.9, gamma = 0, max_epsilon = 0.8, min_epsilon = 0.05, decay_rate = 0.001)
    # Main Training (10000 rounds, Max rounds = 400)
    # self.agent = QLearningAgent(pretrained_model=self.MODEL_NAME, logger=self.logger, learning_rate=0.1, gamma = 0.8, max_epsilon = 0.8, min_epsilon = 0.05, decay_rate = 0.001)


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
    if is_new_round(self, game_state): 
        reset_self(self, game_state)

    self.agent_coord_history.append(game_state['self'][3])

    coins = game_state['coins']
    if coins != []:
        for coin in coins:
            if coin not in self.all_coins_game:
                self.all_coins_game.append(coin)

    num_coins_already_discovered = len(self.all_coins_game)
    feature_vector = state_to_small_features(
        game_state, num_coins_already_discovered)
    
    return self.agent.act(feature_vector, 
                          n_round = game_state["round"], 
                          train = self.train)
    
    
