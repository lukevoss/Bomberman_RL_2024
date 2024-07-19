""" 
This Agent acts as data generator. The rule_based_agent acts as expert predictor. 
This file save the state action pair file .npz file in each step of the game

Author: Luke Voss
"""


import numpy as np

from typing import List

import events as e
from .callbacks import state_to_features, normalize_state, ACTIONS


def setup_training(self):
    """
    Initialise self for training purpose. 
    This function is called after `setup` in callbacks.py.

        Parameter:
            self: This object is passed to all callbacks and you can set arbitrary values.

    Author: Luke Voss
    """
    self.data_count = 1




def game_events_occurred(self, old_game_state: dict, expert_action: str, new_game_state: dict, events: List[str]):
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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # events.append(CONSTANT_PENALTY)

    # Handle None actions
    if expert_action is None:
        expert_action = 'WAIT'

    # Normalize game states
    self.action_map, self.reverse_action_map = normalize_state(old_game_state)
    _,_ = normalize_state(new_game_state)
    normalized_expert_action = self.reverse_action_map(expert_action)
    idx_normalized_expert_action = ACTIONS.index(normalized_expert_action)
    feature_state = state_to_features(old_game_state)


    np.savez_compressed("./data/expert_data_{}.npz".format(self.data_count), state=feature_state, action=idx_normalized_expert_action)
    self.data_count += 1
    
        
        



def end_of_round(self, last_game_state: dict, last_expert_action: str, events: List[str]):
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

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Handle None Actions
    if last_expert_action is None:
        last_expert_action = 'WAIT'

    # Normalize states and actions
    self.action_map, self.reverse_action_map = normalize_state(last_game_state)
    normalized_expert_action = self.reverse_action_map(last_expert_action)
    idx_normalized_expert_action = ACTIONS.index(normalized_expert_action)
    feature_state = state_to_features(last_game_state)

    np.savez_compressed("./data/expert_data_{}.npz".format(self.data_count), state=feature_state, action=idx_normalized_expert_action)
    self.data_count += 1


def reward_from_events(self, events: List[str]) -> int:
    """
    Calculate the Rewards sum from all current events in the game

        Parameter:
            events (list[str]) = List of occured events

        Return:
            reward_sum [float] = Sum of all reward for occured events

    Author: Luke Voss
    """

    game_rewards = {
        e.COIN_COLLECTED: 0.5,
        e.INVALID_ACTION: -0.01,
        e.KILLED_SELF: 0.1, # It is better to kill himself than to get killed
        e.GOT_KILLED: -1,
        e.CRATE_DESTROYED: 0.05,
        e.KILLED_OPPONENT: 1,
    }


    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
