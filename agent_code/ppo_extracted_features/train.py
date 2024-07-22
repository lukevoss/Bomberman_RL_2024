""" 
This File is called by the environment and manages the agents training
Implementation of a PPO algorithm with LSTM and MLP networks as Actor Critic

Deep learning approach with strong feature engineering:
Board is abstracted as a boolean vector of size 20 with each feature as following: 

    Direction to closest Coin
        [0]:Up
        [1]:Right
        [2]:Down
        [3]:Left
        [4]:Wait

    Direction to closest Crate
        [5]: Up
        [6]:Right
        [7]:Down
        [8]:Left
        [9]:Wait

    Direction in which placing a bomb would kill another player
        [10]: Up
        [11]:Right
        [12]:Down
        [13]:Left
        [14]:Place now

    If in Danger, Direction to safety:
        [15]: Up
        [16]:Right
        [17]:Down
        [18]:Left
        [19]:Wait

    [20] Could we survive a placed Bomb

Author: Luke Voss
"""
import numpy as np
import torch

from typing import List

import events as e
from .callbacks import state_to_features, ACTIONS


# Hyper parameters:
SAVE_EVERY_N_EPOCHS = 100
loop = 0
LR = 1e-4
NUM_STEPS = 100
MINI_BATCH_SIZE = 25
PPO_EPOCHS = 8


def setup_training(self):
    """
    Initialise self for training purpose. 
    This function is called after `setup` in callbacks.py.

        Parameter:
            self: This object is passed to all callbacks and you can set arbitrary values.

    Author: Luke Voss
    """
    self.states = []
    self.actions = []
    self.rewards = []
    self.values = []
    self.masks = []
    self.log_probs = []
    self.loss_sum = 0
    self.n_updates = 0

    self.round_rewards = 0

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)


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
    self.logger.debug(f'Encountered game event(s) {", ".join(
        map(repr, events))} in step {new_game_state["step"]}')

    # Hand out self shaped events
    events = get_events(self, old_game_state, self_action, events)

    normalized_action = self.reverse_action_map(self_action)
    idx_normalized_action = torch.tensor(
        [ACTIONS.index(normalized_action)], device=self.device)
    done = 0
    reward = reward_from_events(self, events)
    self.round_rewards += reward

    # Save Tansistions in Lists
    self.states.append(state_to_features(old_game_state).to(self.device))
    self.actions.append(idx_normalized_action)
    self.rewards.append(reward)
    self.values.append(self.value)
    self.masks.append(1 - done)
    self.log_probs.append(self.action_logprob.unsqueeze(0))

    if (len(self.actions) % NUM_STEPS) == 0:
        next_state = state_to_features(new_game_state).to(self.device)
        _, next_value = self.model(next_state)
        returns = compute_gae(next_value, self.rewards,
                              self.masks, self.values)

        returns = torch.stack(returns).detach()
        log_probs = torch.stack(self.log_probs).detach()
        values = torch.stack(self.values).detach()
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        advantages = returns - values

        # Update step of PPO algorithm
        loss = ppo_update(self, PPO_EPOCHS, MINI_BATCH_SIZE,
                          states, actions, log_probs, returns, advantages)
        self.loss_sum += loss
        self.n_updates += 1

        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.masks = []
        self.log_probs = []


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
    events = get_events(self, last_game_state, last_action, events)

    self.logger.debug(f'Encountered event(s) {
                      ", ".join(map(repr, events))} in final step')
    n_round = last_game_state['round']

    normalized_action = self.reverse_action_map(last_action)
    idx_normalized_action = torch.tensor(
        [ACTIONS.index(normalized_action)], device=self.device)
    reward = reward_from_events(self, events)
    self.round_rewards += reward
    done = 1

    # Save Tansistions in Lists
    self.states.append(state_to_features(last_game_state).to(self.device))
    self.actions.append(idx_normalized_action)
    self.rewards.append(reward)
    self.values.append(self.value)
    self.masks.append(1-done)
    self.log_probs.append(self.action_logprob.unsqueeze(0))

    if (last_game_state['step'] % NUM_STEPS) == 0:
        next_value = 0  # Next value doesn't exist
        returns = compute_gae(next_value, self.rewards,
                              self.masks, self.values)

        returns = torch.stack(returns).detach()
        log_probs = torch.stack(self.log_probs).detach()
        values = torch.stack(self.values).detach()
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        advantages = returns - values

        # Update step of PPO algorithm
        if states.size(0) > 0:
            loss = ppo_update(self, PPO_EPOCHS, MINI_BATCH_SIZE,
                              states, actions, log_probs, returns, advantages)
            self.loss_sum += loss
            self.n_updates += 1

        print(' Total rewards of {}, Loss: {}'.format(
            self.round_rewards, self.loss_sum/self.n_updates))

        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.masks = []
        self.log_probs = []

    self.round_rewards = 0

    # Store the model
    if (n_round % SAVE_EVERY_N_EPOCHS) == 0:
        model_path = "./models/ppo_model.pt"
        torch.save(self.model.state_dict(), model_path)


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
        ESCAPE: escape,
        NOT_ESCAPE: -escape,
        WAITED_NECESSARILY: waiting,
        WAITED_UNNECESSARILY: -waiting,
        CLOSER_TO_PLAYERS: aggressive_action,
        AWAY_FROM_PLAYERS: -aggressive_action,
        CLOSER_TO_COIN: coin_action,
        AWAY_FROM_COIN: -coin_action,
        CONSTANT_PENALTY: -0.001,
        WON_GAME: 10,
        GET_IN_LOOP: -0.025 * self.loop_count,

        # DEFAULT EVENTS
        e.INVALID_ACTION: -1,

        # bombing
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,

        # crates, coins
        e.CRATE_DESTROYED: 0.4,
        e.COIN_FOUND: 0,
        e.COIN_COLLECTED: 2,

        # kills
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
