"""
This File is called by the environment and manages the agents training
Implementation of a PPO algorithm with LSTM and MLP networks as Actor Critic

Deep learning approach with strong feature engineering
"""
import numpy as np
import torch

from typing import List

import events as e
import own_events as own_e
from extracted_features import state_to_features
from ppo import ACTIONS
from add_own_events import add_own_events


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
    # Hand out self shaped events
    events = add_own_events(self, old_game_state, self_action, events,
                            end_of_round=False, agent_coord_history=self.agent_coord_history)

    # Log Events
    self.logger.debug(f'Encountered game event(s) {", ".join(
        map(repr, events))} in step {new_game_state["step"]}')

    old_feature_state = state_to_features(old_game_state)
    new_feature_state = state_to_features(new_game_state)
    reward = reward_from_events(self, events)
    is_not_terminal = True

    self.agent.training_step(old_feature_state, new_feature_state,
                             self_action, reward, is_not_terminal)


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

    # Hand out self shaped events
    events = add_own_events(self, last_game_state, last_action, events,
                            end_of_round=False, agent_coord_history=self.agent_coord_history)

    # Log Events
    self.logger.debug(f'Encountered event(s) 
                      {", ".join(map(repr, events))} in final step')

    reward = reward_from_events(self, events)
    old_feature_state = state_to_features(last_game_state)
    is_not_terminal = False

    self.agent.training_step(old_feature_state, None, last_action, reward,  is_not_terminal)

    # Store the model
    n_round = last_game_state['round']
    if (n_round % SAVE_EVERY_N_EPOCHS) == 0:
        self.agent.save_model()


def reward_from_events(self, events: List[str]) -> int:
    """
    Calculate the Rewards sum from all current events in the game

        Parameter:
            events (list[str]) = List of occured events

        Return:
            reward_sum [float] = Sum of all reward for occured events

    Author: Luke Voss
    """
    # Base rewards:
    aggressive_action = 0.3
    coin_action = 0.2
    escape = 0.6
    waiting = 0.5

    game_rewards = {
        # SPECIAL EVENTS
        own_e.ESCAPE: escape,
        own_e.NOT_ESCAPE: -escape,
        own_e.WAITED_NECESSARILY: waiting,
        own_e.WAITED_UNNECESSARILY: -waiting,
        own_e.CLOSER_TO_PLAYERS: aggressive_action,
        own_e.AWAY_FROM_PLAYERS: -aggressive_action,
        own_e.CLOSER_TO_COIN: coin_action,
        own_e.AWAY_FROM_COIN: -coin_action,
        own_e.CONSTANT_PENALTY: -0.001,
        own_e.WON_ROUND: 10,
        own_e.GET_IN_LOOP: -0.025 * self.loop_count,

        # DEFAULT EVENTS
        e.INVALID_ACTION: -1,
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,
        e.CRATE_DESTROYED: 0.4,
        e.COIN_FOUND: 0,
        e.COIN_COLLECTED: 2,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -8,
        e.GOT_KILLED: -10,
        e.OPPONENT_ELIMINATED: 0,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
