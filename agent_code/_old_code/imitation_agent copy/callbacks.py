""" This File is called by the environment and manages the agents movements

Author: Luke Voss
"""

import os
from random import shuffle
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
USING_PRETRAINED = True
MODEL_NAME = 'imitation_agent.pt'
FIELD_SIZE = 15
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
            nn.Softmax(dim=-1)
        )

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)

    def forward(self, x):

        action_probs = self.actor(x)
        dist  = Categorical(action_probs)
        value = self.critic(x)

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
    num_inputs  = FIELD_SIZE * FIELD_SIZE * FEATURE_SIZE # 1575
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

    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0


def normalize_state(game_state):
    """ 
    Function that normalizes the game state to leverage the ocurring symmetries. 
    Game state is normalized in-place to avoid computitional expensive copying. 
    Thus the game_state parameter is already normalized in the functions game_event_ocurred and end_of_round

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
    feature_vector = feature_vector[1:16,1:16,:]

    return torch.tensor(feature_vector.flatten(), dtype=torch.float32)

def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]
    

def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0


def act(self, game_state):
    """
    Agent parses the input, thinks, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.
    

        Parameter:
            self: The same object that is passed to all of your callbacks.
            game_state (dict): The dictionary that describes everything on the board, in-place normalization.
        
        Return:
            next_action (str): The action to take as a string.
    
    Author: Luke Voss
    """
    

    self.logger.info('Picking action according to rule set')

    if self.train:
        # Check if we are in a different round
        if game_state["round"] != self.current_round:
            reset_self(self)
            self.current_round = game_state["round"]
        # Gather information about the game state
        arena = game_state['field']
        _, score, bombs_left, (x, y) = game_state['self']
        bombs = game_state['bombs']
        bomb_xys = [xy for (xy, t) in bombs]
        others = [xy for (n, s, b, xy) in game_state['others']]
        coins = game_state['coins']
        bomb_map = np.ones(arena.shape) * 5
        for (xb, yb), t in bombs:
            for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
                if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                    bomb_map[i, j] = min(bomb_map[i, j], t)

        # If agent has been in the same location three times recently, it's a loop
        if self.coordinate_history.count((x, y)) > 2:
            self.ignore_others_timer = 5
        else:
            self.ignore_others_timer -= 1
        self.coordinate_history.append((x, y))

        # Check which moves make sense at all
        directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        valid_tiles, valid_actions = [], []
        for d in directions:
            if ((arena[d] == 0) and
                    (game_state['explosion_map'][d] < 1) and
                    (bomb_map[d] > 0) and
                    (not d in others) and
                    (not d in bomb_xys)):
                valid_tiles.append(d)
        if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
        if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
        if (x, y - 1) in valid_tiles: valid_actions.append('UP')
        if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
        if (x, y) in valid_tiles: valid_actions.append('WAIT')
        # Disallow the BOMB action if agent dropped a bomb in the same spot recently
        if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
        self.logger.debug(f'Valid actions: {valid_actions}')

        # Collect basic action proposals in a queue
        # Later on, the last added action that is also valid will be chosen
        action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        shuffle(action_ideas)

        # Compile a list of 'targets' the agent should head towards
        cols = range(1, arena.shape[0] - 1)
        rows = range(1, arena.shape[0] - 1)
        dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                    and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
        crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
        targets = coins + dead_ends + crates
        # Add other agents as targets if in hunting mode or no crates/coins left
        if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
            targets.extend(others)

        # Exclude targets that are currently occupied by a bomb
        targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

        # Take a step towards the most immediately interesting target
        free_space = arena == 0
        if self.ignore_others_timer > 0:
            for o in others:
                free_space[o] = False
        d = look_for_targets(free_space, (x, y), targets, self.logger)
        if d == (x, y - 1): action_ideas.append('UP')
        if d == (x, y + 1): action_ideas.append('DOWN')
        if d == (x - 1, y): action_ideas.append('LEFT')
        if d == (x + 1, y): action_ideas.append('RIGHT')
        if d is None:
            self.logger.debug('All targets gone, nothing to do anymore')
            action_ideas.append('WAIT')

        # Add proposal to drop a bomb if at dead end
        if (x, y) in dead_ends:
            action_ideas.append('BOMB')
        # Add proposal to drop a bomb if touching an opponent
        if len(others) > 0:
            if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
                action_ideas.append('BOMB')
        # Add proposal to drop a bomb if arrived at target and touching crate
        if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
            action_ideas.append('BOMB')

        # Add proposal to run away from any nearby bomb about to blow
        for (xb, yb), t in bombs:
            if (xb == x) and (abs(yb - y) < 4):
                # Run away
                if (yb > y): action_ideas.append('UP')
                if (yb < y): action_ideas.append('DOWN')
                # If possible, turn a corner
                action_ideas.append('LEFT')
                action_ideas.append('RIGHT')
            if (yb == y) and (abs(xb - x) < 4):
                # Run away
                if (xb > x): action_ideas.append('LEFT')
                if (xb < x): action_ideas.append('RIGHT')
                # If possible, turn a corner
                action_ideas.append('UP')
                action_ideas.append('DOWN')
        # Try random direction if directly on top of a bomb
        for (xb, yb), t in bombs:
            if xb == x and yb == y:
                action_ideas.extend(action_ideas[:4])

        
        # Pick last action added to the proposals list that is also valid
        while len(action_ideas) > 0:
            a = action_ideas.pop()
            if a in valid_actions:
                # Keep track of chosen action for cycle detection
                if a == 'BOMB':
                    self.bomb_history.append((x, y))
                return a
    else:
        # IMPORTANT: This normalized the state also in game_events_ocurred
        action_map, _ = normalize_state(game_state) 
        feature_vector = state_to_features(game_state).to(self.device)
        dist, _ = self.model(feature_vector)

        # Exploitation: Get Action with higest probability
        idx_action = dist.probs.argmax()
        
        next_action = ACTIONS[idx_action]

        return action_map(next_action)

    


    


    



   
