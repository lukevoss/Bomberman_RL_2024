import os
import random
import pickle

import events as e
import own_events as own_e
from agent_code.utils import *


GAME_REWARDS = {
        # SPECIAL EVENTS
        own_e.CONSTANT_PENALTY: -0.001,
        own_e.WON_ROUND: 10,
        own_e.BOMBED_1_TO_2_CRATES: 0,
        own_e.BOMBED_3_TO_5_CRATES: 0.5,
        own_e.BOMBED_5_PLUS_CRATES: 0.5,
        own_e.GOT_IN_LOOP: -0.3,
        own_e.ESCAPING: 0.03,
        own_e.OUT_OF_DANGER: 0.05,
        own_e.NOT_ESCAPING: -0.01,
        own_e.CLOSER_TO_COIN: 0.05,
        own_e.AWAY_FROM_COIN: -0.02,
        own_e.CLOSER_TO_CRATE: 0.01,
        own_e.AWAY_FROM_CRATE: -0.05,
        own_e.SURVIVED_STEP: 0,
        own_e.DESTROY_TARGET: 0.03,
        own_e.MISSED_TARGET: -0.01,
        own_e.WAITED_NECESSARILY: 0.05,
        own_e.WAITED_UNNECESSARILY: -2,
        own_e.CLOSER_TO_PLAYERS: 0.02,
        own_e.AWAY_FROM_PLAYERS: -0.01,
        own_e.SMART_BOMB_DROPPED: 0.7,
        own_e.DUMB_BOMB_DROPPED: -0.5,

        # DEFAULT EVENTS
        e.INVALID_ACTION: -1,
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,
        e.CRATE_DESTROYED: 0.01,
        e.COIN_FOUND: 0,
        e.COIN_COLLECTED: 3,
        e.KILLED_OPPONENT: 6,
        e.KILLED_SELF: -8,
        e.GOT_KILLED: -10,
        e.OPPONENT_ELIMINATED: 0,
    }

class QLearningAgent:
    def __init__(self, pretrained_model=None, logger = None, learning_rate=0.1, gamma = 0.99, max_epsilon = 1, min_epsilon = 0.08, decay_rate = 1):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.q_table = {}
        self.logger = logger
        if pretrained_model:
            self.load_pretrained_model(pretrained_model)

    def load_pretrained_model(self, model_name):
        model_path = os.path.join('./models', model_name)
        if os.path.isfile(model_path):
            with open(model_path, 'rb') as file:
                self.q_table = pickle.load(file)
            if self.logger:
                self.logger.info("Using pretrained model")
        else:
            raise FileNotFoundError(f"Pretrained model at {model_path} not found.")

    def act(self, feature_vector, n_round, train=True):
        state = tuple(feature_vector)
        epsilon = self._compute_epsilon(n_round) if train else 0
        
        if state not in self.q_table or random.uniform(0, 1) <= epsilon:
            action = random.choice(ACTIONS)
            if self.logger:
                self.logger.debug(f"Choosing {action} randomly")
            return action
        
        best_action_idx = self._greedy_policy(state)
        action = ACTIONS[best_action_idx]
        if self.logger:
            self.logger.debug(f"Choosing {action} from Q-Table")
        return action
    
    def _compute_epsilon(self, n_round):
        """Compute epsilon for the epsilon-greedy policy based on the round number."""
        return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * n_round)
    
    def _greedy_policy(self, state):
        """Select the action with the highest value from the Q-table. If more than one choose randomly"""
        action_values = self.q_table[state]
        max_value = max(action_values)
        best_actions_idx = [index for index, value in enumerate(action_values) if value == max_value]
        return random.choice(best_actions_idx)

    def update_q_value(self, state, action_idx, reward, new_state):
        current_q = self.q_table[state][action_idx]
        if new_state:
            future_q = max(self.q_table[new_state], default=0)
        else:
            future_q = 0 # Last action in a game where the next state doesnt exist
        self.q_table[state][action_idx] = current_q + self.learning_rate * (reward + self.gamma * future_q - current_q)

    def training_step(self, state:List[int], action, reward, new_state: List[int]):
        state = tuple(state)
        if state not in self.q_table:
            self.q_table[state] = [0] * len(ACTIONS)

        if new_state != None:
            new_state = tuple(new_state)
            if new_state not in self.q_table:
                self.q_table[new_state] = [0] * len(ACTIONS)

        action_idx = ACTIONS.index(action)
        if state and new_state:
            self.update_q_value(state, action_idx, reward, new_state)

    def save(self, model_name="q_table.pkl"):
        model_path = os.path.join('./models', model_name)
        with open(model_path, 'wb') as file:
            pickle.dump(self.q_table, file)
        if self.logger:
            self.logger.info("Q-table saved to {}".format(model_path))