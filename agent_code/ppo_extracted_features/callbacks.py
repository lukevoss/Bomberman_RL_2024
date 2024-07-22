""" 
This File is called by the environment and manages the agents movements
Implementation of a PPO algorithm with LSTM and MLP networks as Actor Critic

Deep learning approach without feature engineering:
Board is representet in 15x15x7 vector
Symmetry of board is leveraged

Current status:
Agent learn, but gets stuck on bad local maxima. Behavioral cloning to solve issue, but results are still bad
Ideas:
Network not deep enough, reward system not dense enough, feature engeneering maybe nececarry


Author: Luke Voss
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
USING_PRETRAINED = True
MODEL_NAME = 'imitation_model.pt'
FIELD_SIZE = 15
FEATURE_SIZE = 7

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class ActorCritic(nn.Module):
    """
    Actor Critic with MLP backbone
    """
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, 1000),
            nn.ReLU(),
            nn.Linear(1000, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, 1000),
            nn.ReLU(),
            nn.Linear(1000, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=-1)
        )

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)

    def forward(self, x):
    
        action_probs = self.actor(x)
        dist  = Categorical(action_probs)
        value = self.critic(x)

        return dist, value
    
class ActorCriticLSTM(nn.Module):
    """
    Actor Critic with LSTM backbone, shown more capable in behavioral cloning
    """
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCriticLSTM, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.lstm = nn.LSTMCell(self.num_inputs, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        # self.lstm3 = nn.LSTMCell(750, 500)
        # self.lstm4 = nn.LSTMCell(500, 250)
        # self.lstm5 = nn.LSTMCell(250, hidden_size)

        self.critic_linear = nn.Linear(hidden_size, 1)
        self.actor_linear = nn.Sequential(
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=-1)
        )

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)

    def forward(self, x):
    
        x, _ = self.lstm(x)
        x, _ = self.lstm2(x)
        # x, _ = self.lstm3(x)
        # x, _ = self.lstm4(x)
        # x, _ = self.lstm5(x)

        action_probs = self.actor_linear(x)
        dist  = Categorical(action_probs)
        value = self.critic_linear(x)

        return dist, value

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    num_inputs  = FIELD_SIZE * FIELD_SIZE * FEATURE_SIZE # 2023
    hidden_size = 256
    num_outputs = len(ACTIONS) # 6
    
    model_file = os.path.join('./models', MODEL_NAME)
    self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.isfile(model_file) & USING_PRETRAINED:
        print("Using pretrained model")
        self.model = ActorCriticLSTM(num_inputs, num_outputs, hidden_size).to(self.device)
        self.model.load_state_dict(torch.load(model_file))
    else:
        print("Using new model")
        self.logger.info("Setting up model from scratch.")
        self.model = ActorCriticLSTM(num_inputs, num_outputs, hidden_size).to(self.device)


def normalize_state(game_state):
    """ 
    Function that normalizes the game state to leverage the ocurring symmetries. 
    Game state is normalized in-place to avoid computitional expensive copying. 
    Thus the game_state parameter is already normalized in the functions game_event_ocurred and end_of_round
    since it is normalized in the act() function!

        Parameters:
            game_state (dict): environment state to normalize (in-place!!).

        Returns: 
            action_map (func): function to map action in normalized state to action in input_state
            reverse_action_map (func): function to map action in input_state to action in normalized state.
    
    Author: Luke Voss
    """
    if game_state is None:
        return lambda a: a, lambda a: a

    x_coord_agent, y_coord_agent = game_state['self'][3]

    transposed = False
    x_flipped = False
    y_flipped = False


    def flip_tuple(t, flip_x, flip_y):
        """
        Make use of left/right and up/down symmetry by fliping tuple from state

            Parameter:
                flip_x (bool): flip tuple L/R 
                flip_y (bool): flip tuple UP/DOWN

            Return:
                x(int),y(int): new coords of flipped tuple
        """
        x, y = t
        if flip_x:
            x = 16 - x
        if flip_y:
            y = 16 - y
        return x, y
    
    # Make use of diagonal symmetry
    def transpose_tuple(t):
        return t[1], t[0]
    
    # Define mapping for actions
    action_mapping = {
        'RIGHT': 'UP',
        'UP': 'RIGHT',
        'LEFT': 'DOWN',
        'DOWN': 'LEFT',
        'WAIT': 'WAIT',
        'BOMB': 'BOMB'
    }
    
    def map_action(a, flip_x, flip_y, transpose):
        if flip_x:
            a = 'RIGHT' if a == 'LEFT' else ('LEFT' if a == 'RIGHT' else a)
        if flip_y:
            a = 'UP' if a == 'DOWN' else ('DOWN' if a == 'UP' else a)
        if transpose:
            a = action_mapping.get(a, a)
        return a
    
    def map_reverse_action(a, flip_x, flip_y, transpose):
        if transpose:
            a = action_mapping.get(a, a)
        if flip_x:
            a = 'RIGHT' if a == 'LEFT' else ('LEFT' if a == 'RIGHT' else a)
        if flip_y:
            a = 'UP' if a == 'DOWN' else ('DOWN' if a == 'UP' else a)
        return a
    
    if x_coord_agent > 8:
        game_state['field'] = np.flipud(game_state['field'])
        game_state['bombs'] = [(flip_tuple(pos, True, False), time) for pos, time in game_state['bombs']]
        game_state['explosion_map'] = np.flipud(game_state['explosion_map'])
        game_state['coins'] = [flip_tuple(coin, True, False) for coin in game_state['coins']]
        name, score, can_place_bomb, pos = game_state['self']
        game_state['self'] = (name, score, can_place_bomb, flip_tuple(pos, True, False))
        game_state['others'] = [(name, score, can_place_bomb, flip_tuple(pos, True, False)) for name, score, can_place_bomb, pos in game_state['others']]
        x_flipped = True

    if y_coord_agent > 8:
        game_state['field'] = np.fliplr(game_state['field'])
        game_state['bombs'] = [(flip_tuple(pos, False, True), time) for pos, time in game_state['bombs']]
        game_state['explosion_map'] = np.fliplr(game_state['explosion_map'])
        game_state['coins'] = [flip_tuple(coin, False, True) for coin in game_state['coins']]
        name, score, can_place_bomb, pos = game_state['self']
        game_state['self'] = (name, score, can_place_bomb, flip_tuple(pos, False, True))
        game_state['others'] = [(name, score, can_place_bomb, flip_tuple(pos, False, True)) for name, score, can_place_bomb, pos in game_state['others']]
        y_flipped = True

    agent_x_update, agent_y_update = game_state['self'][3]

    if agent_y_update > agent_x_update:
        game_state['field'] = np.transpose(game_state['field'])
        game_state['bombs'] = [(transpose_tuple(pos), time) for pos, time in game_state['bombs']]
        game_state['explosion_map'] = np.transpose(game_state['explosion_map'])
        game_state['coins'] = [transpose_tuple(coin) for coin in game_state['coins']]
        name, score, can_place_bomb, pos = game_state['self']
        game_state['self'] = (name, score, can_place_bomb, transpose_tuple(pos))
        game_state['others'] = [(name, score, can_place_bomb, transpose_tuple(pos)) for name, score, can_place_bomb, pos
                                in game_state['others']]
        transposed = True

    action_map = lambda a: map_action(a, x_flipped, y_flipped, transposed)
    reverse_action_map = lambda a: map_reverse_action(a, x_flipped, y_flipped, transposed)

    return action_map, reverse_action_map


def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to a feature vector of size 15x15x7
    Walls of board are not represented to reduce dimensionality
    7 feature bits (boolean) are representet in the following logic:
        [0] = wall
        [1] = crate
        [2] = agent
        [3] = opponent
        [4] = bomb
        [5] = coin
        [6] = explosion

        Parameter
            game_state:  A dictionary describing the current game board.
        
        Return: 
            np.array of size 15x15x7

    Author: Luke Voss
    """
    # Handle the case when the game_state is None
    if game_state is None:
        return None
    
    # Initialize the feature vector
    feature_vector = np.zeros((17,17,7), dtype=bool) # TODO type is ok?

    # Extract relevant information from the game_state
    field = game_state['field']
    x_agent, y_agent = game_state['self'][3]
    opponents = game_state['others']
    bombs = game_state['bombs']
    coins = game_state['coins']
    explosion_map = game_state['explosion_map']

    # Set feature bits based on extracted information
    feature_vector[:,:,0] = (field == -1) # Walls
    feature_vector[:,:,1] = (field == 1) # Creates
    feature_vector[x_agent,y_agent,2] = 1  # Agent #TODO test if x and y axis are correct
    
    for _, opponent in enumerate(opponents): 
        x_coord, y_coord = opponent[3]
        feature_vector[x_coord,y_coord,3] = 1 # Opponents
    
    for _, bomb in enumerate(bombs):
        x_coord, y_coord = bomb[0]
        feature_vector[x_coord,y_coord,4] = 1 # Bombs
    
    for _, coin in enumerate(coins):
        x_coord, y_coord = coin
        feature_vector[x_coord,y_coord,5] = 1 # Coins

    feature_vector[:,:,6] = (explosion_map != 0) # Explosions

    # TODO: do we have to include the walls on border for training? or does this work to reduce dimension?
    feature_vector = feature_vector[1:16,1:16,:]

    return torch.tensor(feature_vector.flatten(), dtype=torch.float32)

    
def act(self, game_state: dict) -> str:
    """
    Agent parses the input, thinks, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.
    The game_state is normalized here

        Parameter:
            self: The same object that is passed to all of your callbacks.
            game_state (dict): The dictionary that describes everything on the board, in-place normalization.
        
        Return:
            next_action (str): The action to take as a string.
    
    Author: Luke Voss
    """
    # IMPORTANT: This normalized the state also in game_events_ocurred
    self.action_map, self.reverse_action_map = normalize_state(game_state) 
    feature_vector = state_to_features(game_state).to(self.device)
    self.dist, self.value = self.model(feature_vector)

    if self.train:
        # Exploration: Sample from Action Distribution
        idx_action = self.dist.sample() 
        self.action_logprob = self.dist.log_prob(idx_action) 
    else:
        # Exploitation: Get Action with higest probability
        idx_action = self.dist.probs.argmax() # TODO this correct?
    
    
    next_action = ACTIONS[idx_action]

    return self.action_map(next_action)
    


    


    



   
