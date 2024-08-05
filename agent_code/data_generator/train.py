""" 
This Agent acts as data generator. The rule_based_agent acts as expert predictor. 
This file save the state action pair file .npz file in each step of the game

Author: Luke Voss
"""


import numpy as np

from typing import List

import events as e
from agent_code.utils import ACTIONS
from agent_code.feature_extraction import state_to_features_large
from agent_code.add_own_events import add_own_events


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


    action_idx = ACTIONS.index(expert_action)
    feature_state = state_to_features_large(old_game_state, self.max_opponents_score)
    events = add_own_events(old_game_state, expert_action, events, True, self.agent_coord_history, self.max_opponents_score)

    np.savez_compressed("./data/expert_data_{}.npz".format(self.data_count),
                        state=feature_state, action=action_idx, events = events, is_terminal = False)
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
    
    action_idx = ACTIONS.index(last_expert_action)
    feature_state = state_to_features_large(last_game_state, self.max_opponents_score)
    events = add_own_events(last_game_state, last_expert_action, events, True, self.agent_coord_history, self.max_opponents_score)

    np.savez_compressed("./data/expert_data_{}.npz".format(self.data_count),
                        state=feature_state, action=action_idx, events = events, is_terminal = True)
    self.data_count += 1
