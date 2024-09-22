""" 
This Agent is based on the q_learning_agent and acts as data generator.
The generated data samples consist of the normalized game state and the following action. 

Author: Max Tiedl
"""

from typing import List

from agent_code.feature_extraction import state_to_small_features
from agent_code.add_own_events import add_own_events_q_learning, GAME_REWARDS
from agent_code.q_learning import *

# Hyper parameters:
SAVE_EVERY_N_EPOCHS = 100


def setup_training(self):
    """
    Initialise self for training purpose.
    This function is called after `setup` in callbacks.py.
    """
    directory='data'
    file_count = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
    self.data_count = file_count


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


    old_feature_state = state_to_small_features(old_game_state, num_coins_already_discovered)#.to(self.device)
    new_feature_state = state_to_small_features(new_game_state, num_coins_already_discovered)#.to(self.device)

    # Hand out self shaped events
    events = add_own_events_q_learning(old_game_state, 
                            old_feature_state,
                            self_action,
                            events,
                            end_of_round=False,
                            agent_coord_history=self.agent_coord_history,
                            max_opponents_score=self.max_opponents_score)

    reward = reward_from_events(self, events)

    self.agent.training_step(old_feature_state, self_action, reward, new_feature_state)

    # Handle None actions
    if self_action is None:
        self_action = 'WAIT'

    num_coins_already_discovered = len(self.all_coins_game)

    action_idx = ACTIONS.index(self_action)

    np.savez_compressed("./data/expert_data_{}.npz".format(self.data_count),
                        state=old_feature_state, action=action_idx, events = events, is_terminal = False)
    self.data_count += 1

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

    old_feature_state = state_to_small_features(last_game_state, num_coins_already_discovered)#.to(self.device)

    # Hand out self shaped events
    events = add_own_events_q_learning(last_game_state,
                            old_feature_state, 
                            last_action,
                            events,
                            end_of_round=False,
                            agent_coord_history=self.agent_coord_history,
                            max_opponents_score=self.max_opponents_score)

    self.agent.training_step(old_feature_state, last_action, reward, None)
    
    # Handle None Actions
    if last_action is None:
        last_action = 'WAIT'
    
    action_idx = ACTIONS.index(last_action)

    np.savez_compressed("./data/expert_data_{}.npz".format(self.data_count),
                        state=old_feature_state, action=action_idx, events = events, is_terminal = True)
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

    

    reward_sum = 0
    for event in events:
        if event in GAME_REWARDS:
            reward_sum += GAME_REWARDS[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
