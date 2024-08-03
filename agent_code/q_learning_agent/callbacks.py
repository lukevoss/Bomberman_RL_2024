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
import os
from collections import deque
import pickle

import torch

from agent_code.q_learning_agent.feature_extraction import state_to_features
from agent_code.q_learning import *

# path to the QTable models
cwd = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = f"{cwd}/model.pkl"

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

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Model is run on: {self.device}")
    self.current_round = 0
    self.all_coins_game = []

    # Agent Position history before normalization
    self.agent_coord_history = deque([], self.MAX_COORD_HISTORY)

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as file:
            self.model = pickle.load(file)
        print("Using pretained model")
    else:
        self.model = QTable.initialize_q_table(self)
        print("Using new model")


def reset_self(self, game_state: dict):
    self.agent_coord_history = deque([], self.MAX_COORD_HISTORY)
    self.current_round = game_state["round"]
    self.max_opponents_score = 0


def is_new_round(self, game_state: dict) -> bool:
    return game_state["round"] != self.current_round


def act(self, game_state: dict) -> str:
    """
    Agent parses the input, thinks, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    Author: Luke Voss
    """
    print(len(self.model))
    if is_new_round(self, game_state):  # TODO Correct?
        reset_self(self, game_state)

    self.agent_coord_history.append(game_state['self'][3])

    coins = game_state['coins']
    if coins != []:
        for coin in coins:
            if coin not in self.all_coins_game:
                self.all_coins_game.append(coin)

    num_coins_already_discovered = len(self.all_coins_game)
    feature_vector = state_to_features(
        game_state, num_coins_already_discovered)#.to(self.device)
    

    state = tuple(feature_vector)
    
    if  self.model.get(state): # If the state has been already ecountered
        model_result = self.model[state]
        if model_result.values() is not None: # If the encountered state has values in it for the actions
             
            random_int = random.uniform(0,1)
            if self.train:# If in training MODE
                epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * game_state['round'])
                self.logger.debug(f"Randomness is at {epsilon}")

                if random_int <= epsilon: # Go for random action for a decreasing percentage of randomness
                    action = random.choice(ACTIONS)# without bomb
                    self.logger.debug(f"***************Choosing  {action}  purely at random in {state} ")
                    return action
                # Choose from the Q_Table the best action that is stored, if more than one best action, choose one randomly
                max_value = max(model_result.values())
                possible_actions = [key for key, val in model_result.items() if abs(max_value - val) < 1e-4]
                action = random.choice(possible_actions)
                self.logger.info(f"Picking {action} from state {state}.")
                self.logger.info(f"Number of encountered states: {len(self.model)}  in round: {game_state['round']}.")
                return action
            # Having some randomness even if there is best action 
            # epsilon =  MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * game_state['step']) #
            epsilon = .01
            if random_int <= epsilon:
                action = random.choice(ACTIONS)
                self.logger.debug(f"*************Choosing {action} purely at random in {state}")
                return action
            # Choosing from Q_Table
            action = np.argmax(list(model_result.values()))
            action = ACTIONS[action]
            self.logger.info(f"Picking {action} from state {state}.")
            return action
    else:
        # Add the new state to the q_table
        action = random.choice(ACTIONS)
        self.model[state]= dict.fromkeys(ACTIONS, ZERO)
        # print(state)
        return action
