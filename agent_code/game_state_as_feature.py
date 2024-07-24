from numpy import np
import torch


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
    feature_vector = np.zeros((17, 17, 7), dtype=bool)  # TODO type is ok?

    # Extract relevant information from the game_state
    field = game_state['field']
    x_agent, y_agent = game_state['self'][3]
    opponents = game_state['others']
    bombs = game_state['bombs']
    coins = game_state['coins']
    explosion_map = game_state['explosion_map']

    # Set feature bits based on extracted information
    feature_vector[:, :, 0] = (field == -1)  # Walls
    feature_vector[:, :, 1] = (field == 1)  # Creates
    # Agent #TODO test if x and y axis are correct
    feature_vector[x_agent, y_agent, 2] = 1

    for _, opponent in enumerate(opponents):
        x_coord, y_coord = opponent[3]
        feature_vector[x_coord, y_coord, 3] = 1  # Opponents

    for _, bomb in enumerate(bombs):
        x_coord, y_coord = bomb[0]
        feature_vector[x_coord, y_coord, 4] = 1  # Bombs

    for _, coin in enumerate(coins):
        x_coord, y_coord = coin
        feature_vector[x_coord, y_coord, 5] = 1  # Coins

    feature_vector[:, :, 6] = (explosion_map != 0)  # Explosions

    return torch.tensor(feature_vector.flatten(), dtype=torch.float32)


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
        game_state['bombs'] = [(flip_tuple(pos, True, False), time)
                               for pos, time in game_state['bombs']]
        game_state['explosion_map'] = np.flipud(game_state['explosion_map'])
        game_state['coins'] = [flip_tuple(
            coin, True, False) for coin in game_state['coins']]
        name, score, can_place_bomb, pos = game_state['self']
        game_state['self'] = (name, score, can_place_bomb,
                              flip_tuple(pos, True, False))
        game_state['others'] = [(name, score, can_place_bomb, flip_tuple(
            pos, True, False)) for name, score, can_place_bomb, pos in game_state['others']]
        x_flipped = True

    if y_coord_agent > 8:
        game_state['field'] = np.fliplr(game_state['field'])
        game_state['bombs'] = [(flip_tuple(pos, False, True), time)
                               for pos, time in game_state['bombs']]
        game_state['explosion_map'] = np.fliplr(game_state['explosion_map'])
        game_state['coins'] = [flip_tuple(
            coin, False, True) for coin in game_state['coins']]
        name, score, can_place_bomb, pos = game_state['self']
        game_state['self'] = (name, score, can_place_bomb,
                              flip_tuple(pos, False, True))
        game_state['others'] = [(name, score, can_place_bomb, flip_tuple(
            pos, False, True)) for name, score, can_place_bomb, pos in game_state['others']]
        y_flipped = True

    agent_x_update, agent_y_update = game_state['self'][3]

    if agent_y_update > agent_x_update:
        game_state['field'] = np.transpose(game_state['field'])
        game_state['bombs'] = [(transpose_tuple(pos), time)
                               for pos, time in game_state['bombs']]
        game_state['explosion_map'] = np.transpose(game_state['explosion_map'])
        game_state['coins'] = [transpose_tuple(
            coin) for coin in game_state['coins']]
        name, score, can_place_bomb, pos = game_state['self']
        game_state['self'] = (name, score, can_place_bomb,
                              transpose_tuple(pos))
        game_state['others'] = [(name, score, can_place_bomb, transpose_tuple(pos)) for name, score, can_place_bomb, pos
                                in game_state['others']]
        transposed = True

    def action_map(a): return map_action(a, x_flipped, y_flipped, transposed)

    def reverse_action_map(a): return map_reverse_action(
        a, x_flipped, y_flipped, transposed)

    return action_map, reverse_action_map
