""" This File is called by the environment and manages the agents movements

Author: Luke Voss
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
USING_PRETRAINED = True
MODEL_NAME = 'ppo_model.pt'
FIELD_SIZE = 17
FEATURE_SIZE = 7

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class ActorCritic(nn.Module):
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
        )

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)

    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        # print(self.actor[0].weight)
        # print("Max", torch.max(self.actor[0].weight))
        # print("Min", torch.min(self.actor[0].weight))
        # if torch.isnan(mu).any():
        #     print(self.actor[0].weight)
        #     print(self.actor[0].bias)
        #     print(self.critic[0].weight)
        #     print(self.critic[0].bias)
        std   = self.log_std.exp()# .expand_as(mu) TODO Batch size problem?
        dist  = Normal(mu, std)
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
        self.model = ActorCritic(num_inputs, num_outputs, hidden_size).to(self.device)
        self.model.load_state_dict(torch.load(model_file))
    else:
        print("Using new model")
        self.logger.info("Setting up model from scratch.")
        self.model = ActorCritic(num_inputs, num_outputs, hidden_size).to(self.device)


def normalize_state(game_state):
    """ 
    Function that normalizes the game state to leverage the ocurring symmetries

        Parameters:
            game_state (dict): environment state to normalize (in-place).

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
    Converts the game state to a feature vector of size 17x17x7
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
            np.array of size 17x17x7

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
    # feature_vector = feature_vector[1:16,1:16,:]

    return torch.tensor(feature_vector.flatten(), dtype=torch.float32)

    
def act(self, game_state: dict) -> str:
    """
    Agent parses the input, thinks, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

        Parameter:
            self: The same object that is passed to all of your callbacks.
            game_state (dict): The dictionary that describes everything on the board.
        
        Return:
            next_action (str): The action to take as a string.
    
    Author: Luke Voss
    """
    action_map, reverse_action_map = normalize_state(game_state) #TODO Copy of Dict nececarry?
    feature_vector = state_to_features(game_state).to(self.device)
    self.dist, self.value = self.model(feature_vector)

    if self.train:
        # Exploration: Sample from Action Distribution
        idx_action = torch.argmax(self.dist.sample()).item()  
    else:
        # Exploitation: Get Action with higest probability
        idx_action = self.dist.mean.argmax().item() 
    
    next_action = ACTIONS[idx_action]
    # print("Took action: ", action_map(next_action))
    return action_map(next_action)
    


    


    



   
