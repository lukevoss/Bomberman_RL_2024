import random

import events as e
import own_events as own_e
from agent_code.utils import *

# Training parameters
LEARNING_RATE = 0.9  # 0.7
# Environment parameters
GAMMA = 0.99  # 0.95
# Exploration parameters
MAX_EPSILON = 1  # 1
MIN_EPSILON = 0.1  # 0.1
DECAY_RATE = 0.0001  # 0.001

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

class QTable():
    """
        the structure of the q_table is a linked dictionary to reduce the domain of search
        {state:{actions: Q_value}} --> e.g: {feature:{'LEFT':-0.95}},
        Features are binary
        one example of one row from q_table at time of initializing:
        { 0, 0, 0, ... , 0, 0, 0): {'UP': 0,'RIGHT': 0,'DOWN': 0,'LEFT': 0,'WAIT': 0,'BOMB': 0}

    """

    def __init__(self, game_state: GameState):
        super(QTable, self).__init__()
        self.game_state = game_state

    def initialize_q_table(self) -> dict:
        """initializing an empty qtable
        Returns:
            dict: dictionary of {state:{actions: Q_value}}
        """
        features_dict = {}

        return features_dict
    
def _epsilon_greedy_policy(model: dict, state: list,  epsilon: float) -> str:
    """
    With a Probability of 1 - ɛ, we do exploitation, and with the probability ɛ,
    we do exploration. 
    In the epsilon_greedy_policy we will:
    1-Generate the random number between 0 to 1.
    2-If the random number is greater than epsilon, we will do exploitation.
        It means that the agent will take the action with the highest value given
        a state.
    3-Else, we will do exploration (Taking random action). 

    """
    random_int = random.uniform(0, 1)
    if state and random_int > epsilon:
        action = _greedy_policy(model, state)
    else:
        action = random.choice(ACTIONS)
    return action


def _greedy_policy(model: dict, state: list) -> str:
    """
    Q-learning is an off-policy algorithm which means that the policy of 
    taking action and updating function is different.
    In this example, the Epsilon Greedy policy is acting policy, and 
    the Greedy policy is updating policy.
    The Greedy policy will also be the final policy when the agent is trained.
    It is used to select the highest state and action value from the Q-Table.
    """
    state = tuple(state)
    action = np.argmax(model[state])
    action = ACTIONS[action]
    return action


def update_model(self, game_state: GameState, state: list | None, new_state: list | None, action: str | None, reward: float) -> dict:
    """Updating the Q_Value ragarding the state and the action the agent choose

    Returns:
        dict of state and actions
    """
    epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * \
        np.exp(-DECAY_RATE * game_state['step'])  # Updating must be per step
    # end of the game
    if game_state is None:
        pass

    if state:  # if state is not None
        state = tuple(state)
    if not action:  # if action is None go for random action
        action = _epsilon_greedy_policy(self.model, state, epsilon)

    if new_state:  # if New_state is not None, which means we haven't invalid action
        new_state = tuple(new_state)

    # invalid action or not such state in the model then penalize the agent with invalid action
    if new_state is None or self.model.get(new_state) is None:
        self.model[state][action] = self.model[state][action] + (
            LEARNING_RATE*(reward + GAMMA * (GAME_REWARDS[e.INVALID_ACTION]) - self.model[state][action]))

    # if action is valid, update the Q_value of that state_action
    elif (self.model[state][action] is not None) and new_state:

        model_new_result = self.model[new_state]
        max_result = max(model_new_result.values())

        self.model[state][action] = self.model[state][action] + LEARNING_RATE*(
            reward + GAMMA * max_result - self.model[state][action])

    return self.model