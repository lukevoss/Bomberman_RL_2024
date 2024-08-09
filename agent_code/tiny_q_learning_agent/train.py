"""
This File is called by the environment and manages the agents training
Implementation of a PPO algorithm with LSTM and MLP networks as Actor Critic

Deep learning approach with strong feature engineering
"""
import os
from typing import List
import time
import pickle

import numpy as np
import torch

import events as e
import own_events as own_e
from agent_code.feature_extraction import state_to_tiny_features
from agent_code.add_own_events import tiny_add_own_events, GAME_REWARDS
from agent_code.q_learning import *

# path to the QTable models
cwd = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = f"{cwd}/model.pkl"

# Hyper parameters:
SAVE_EVERY_N_EPOCHS = 100
loop = 0
LR = 1e-4


def setup_training(self):
    """
    Initialise self for training purpose.
    This function is called after `setup` in callbacks.py.
    """
    pass


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.
    Keeps track of transitions and updates the Agent network

        Parameter:
            self: This object is passed to all callbacks and you can set arbitrary values.
            old_game_state (dict): The state that was passed to the last call of `act`.
            self_action (str): The action that you took.
            new_game_state (dict): The state the agent is in now.
            events (list): The events that occurred when going from  `old_game_state` to `new_game_state`

    Author: Luke Voss
    """
   

    # Log Events
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    num_coins_already_discovered = len(self.all_coins_game)


    start_time = time.time()
    old_feature_state = state_to_tiny_features(old_game_state, num_coins_already_discovered)#.to(self.device)
    new_feature_state = state_to_tiny_features(new_game_state, num_coins_already_discovered)#.to(self.device)
    time_feature_extraction = (time.time() - start_time)

    # Hand out self shaped events
    start_time = time.time()
    events = tiny_add_own_events(old_game_state, 
                            old_feature_state,
                            self_action,
                            events,
                            end_of_round=False,
                            agent_coord_history=self.agent_coord_history,
                            max_opponents_score=self.max_opponents_score)
    time_own_events = (time.time() - start_time)

    reward = reward_from_events(self, events)

    self.agent.training_step(old_feature_state, self_action, reward, new_feature_state)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

        Parameter:
            self: This object is passed to all callbacks and you can set arbitrary values.
            last_game_state (dict): The last state that was passed to the last call of `act`.
            last_action (str): The last action that you took.
            events (list): The last events that occurred in the last step

    Author: Luke Voss
    """
    

    # Log Events
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    reward = reward_from_events(self, events)
    num_coins_already_discovered = len(self.all_coins_game)

    start_time = time.time()
    old_feature_state = state_to_tiny_features(last_game_state, num_coins_already_discovered)#.to(self.device)
    time_feature_extraction = (time.time() - start_time)

    # Hand out self shaped events
    start_time = time.time()
    events = tiny_add_own_events(last_game_state,
                            old_feature_state, 
                            last_action,
                            events,
                            end_of_round=False,
                            agent_coord_history=self.agent_coord_history,
                            max_opponents_score=self.max_opponents_score)
    time_own_events = (time.time() - start_time)

    self.agent.training_step(old_feature_state, last_action, reward, None)
    
    # Store the model
    n_round = last_game_state['round']
    if (n_round % SAVE_EVERY_N_EPOCHS) == 0:
        print(f"Saving table of length {len(self.agent.q_table)}")
        self.agent.save()


def reward_from_events(self, events: List[str]) -> int:
    """
    Calculate the Rewards sum from all current events in the game

        Parameter:
            events (list[str]) = List of occured events

        Return:
            reward_sum [float] = Sum of all reward for occured events

    Author: Luke Voss
    """

    

    reward_sum = 0
    for event in events:
        if event in GAME_REWARDS:
            reward_sum += GAME_REWARDS[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum