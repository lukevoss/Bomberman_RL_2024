"""
Board is abstracted as a boolean vector of size 20 with each feature as following:

    Direction to closest Coin
        [0]:Up
        [1]:Right
        [2]:Down
        [3]:Left
        [4]:Wait

    Direction to closest Crate
        [5]:Up
        [6]:Right
        [7]:Down
        [8]:Left
        [9]:Wait

    Direction in which placing a bomb would kill another player
        [10]:Up
        [11]:Right
        [12]:Down
        [13]:Left
        [14]:Place now

    If in Danger, Direction to safety:
        [15]:Up
        [16]:Right
        [17]:Down
        [18]:Left
        [19]:Wait

    Amount of danger in next moves from 0 (Safe)  to 1 (Immideatly dead)
        [20]:Up
        [21]:Right
        [22]:Down
        [23]:Left
        [24]:Wait

    [25] Could we survive a placed Bomb

    [26] Can we place a bomb

    #and we can place one?
    [27] Very smart bomb position: Would destroy 4 or more crates or opponent in trap

    [28] Only one opponent left

    [29] Currently in the lead


# TODO make crossing of bomb danger zone possible
# TODO sometimes placing a bomb before running out of danger can trap a opponent between two bombs
# TODO Should we do masking?
# TODO are any coins left in game? If not start hunting and dont destroy more crates

Author: Luke Voss
"""
import numpy as np


from utils import *

FEATURE_VECTOR_SIZE = 30
EXTREME_DANGER = 1
HIGH_DANGER = 0.75
MEDIUM_DANGER = 0.5
LOW_DANGER = 0.25
NO_DANGER = 0


def is_coin(coords, game_state: GameState):
    return coords in game_state.coins


def is_near_crate(coords, game_state: GameState) -> bool:
    """Return True if the given coordinate is near a crate."""
    for direction in MOVEMENT_DIRECTIONS:
        new_coords = coords[0]+direction[0], coords[1] + direction[1]
        if game_state.field[new_coords] == 1:
            return True
    return False


def is_opponent_in_blast_range(coords, game_state: GameState) -> bool:
    """Return True if the player is within blast range of the enemy."""
    for opponent in game_state.others:
        if opponent[3] in get_blast_effected_coords(coords, game_state.field):
            return True
    return False


def is_save_step_criterion(coords, game_state: GameState):
    sorted_dangerous_bombs = sort_and_filter_out_dangerous_bombs(
        coords, game_state)
    return (is_valid_action(coords, game_state) and
            not is_dangerous(coords, explosion_map, sorted_dangerous_bombs))


def find_shortest_path(start_coords, game_state: GameState, stop_criterion: function):
    """
    Returns the shortest path to one of the given goal coordinates, currently bombs and opponents block movements
    with next game state estimation
    # TODO put waiting into exploring?
    """
    queue = deque([start_coords, game_state])
    visited = set([start_coords])
    parent = {start_coords: None}

    while queue:
        current_coords, current_game_state = queue.popleft()
        if stop_criterion(current_coords, current_game_state):
            step = current_coords
            path = []
            while step != start_coords:
                path.append(step)
                step = parent[step]
            return path[::-1]

        for action in MOVEMENT_ACTIONS:
            next_game_state = get_next_game_state(action, current_game_state)
            if next_game_state != None:
                new_coords = next_game_state['self'][3]
                if new_coords not in visited:
                    queue.append(new_coords)
                    visited.add(new_coords)
                    parent[new_coords] = new_coords
    return []


def get_action_idx_from_coords(agent_coords, new_coords):
    direction = (new_coords[0]-agent_coords[0], new_coords[1]-agent_coords[1])
    return MOVEMENT_DIRECTIONS.index(direction)


def get_action_idx_to_closest_thing(game_state: GameState, stop_criterion: function):
    agent_coords = game_state.self[3]
    shortest_path = find_shortest_path(
        agent_coords, game_state, stop_criterion)

    if shortest_path:
        first_step_coords = shortest_path[0]
        return get_action_idx_from_coords(first_step_coords)
    else:
        return ACTIONS.index('WAIT')


def get_danger_in_each_direction(coords, game_state: GameState):
    danger_per_action = np.zeros(len(DIRECTIONS_AND_WAIT))
    for idx_action, direction in enumerate(DIRECTIONS_AND_WAIT):
        new_coords = coords[0] + direction[0], coords[1] + direction[1]
        sorted_dangerous_bombs = sort_and_filter_out_dangerous_bombs(
            new_coords, game_state)
        if game_state.explosion_map[new_coords] == 1:
            danger_per_action[idx_action] = EXTREME_DANGER
        for bomb_coords, timer in sorted_dangerous_bombs:
            blast_coords = get_blast_effected_coords(
                bomb_coords, game_state.field)
            if new_coords in blast_coords:
                match timer:
                    case 0:
                        danger_per_action[idx_action] = max(
                            danger_per_action[idx_action], EXTREME_DANGER)
                    case 1:
                        danger_per_action[idx_action] = max(
                            danger_per_action[idx_action], HIGH_DANGER)
                    case 2:
                        danger_per_action[idx_action] = max(
                            danger_per_action[idx_action], MEDIUM_DANGER)
                    case 3:
                        danger_per_action[idx_action] = max(
                            danger_per_action[idx_action], LOW_DANGER)

    return danger_per_action


def is_deadend(coords, game_state: GameState):
    count_free_tiles = 0
    for direction in MOVEMENT_DIRECTIONS:
        new_coords = move_in_direction(coords, direction)
        if game_state.field[new_coords] == FREE:
            count_free_tiles += 1
    return count_free_tiles <= 1


def get_possible_directions(agent_coords, game_state: GameState):
    possible_directions = []
    for direction in MOVEMENT_DIRECTIONS:
        new_coords = march_forward(agent_coords, direction)
        if is_valid_action(new_coords, game_state):
            possible_directions.append(direction)
    return possible_directions


def are_opposite_directions(directions):
    if directions == 2:
        direction_1 = directions[0]
        direction_2 = directions[1]
        return (direction_1[0] + direction_2[0] == 0 and
                direction_1[1] + direction_2[1] == 0)
    return False


def opponent_in_deadend(opponent, game_state):
    """
    Returns if opponent is in deadend (e.g only has two availabe opposite directions that 
    he can walk and one of them leads into a deadend)
    """
    opponent_coord = opponent[3]
    possible_directions = get_possible_directions(opponent_coord, game_state)
    if len(possible_directions) <= 2 and are_opposite_directions(possible_directions):
        for direction in possible_directions:
            new_coords = opponent_coord
            for i in range(0, s.BOMB_POWER):
                new_coords = move_in_direction(new_coords, direction)
                if is_deadend(new_coords, game_state):
                    return True
    return False


def would_surely_kill_opponent(bomb_blast_coords, game_state: GameState):
    for opponent in game_state.others:
        if opponent_in_deadend(opponent, game_state) and potentially_destroying_opponent(bomb_blast_coords, [opponent]):
            return True
    return False


def state_to_features(game_state: dict, scores) -> np.array:
    feature_vector = np.zeros(FEATURE_VECTOR_SIZE)

    game_state = GameState(**game_state)
    agent_coords = game_state.self[3]

    # How to get to closes coin
    action_idx_to_coin = get_action_idx_to_closest_thing(game_state, is_coin)
    feature_vector[action_idx_to_coin] = 1

    # How to get to closest crate
    action_idx_to_crate = get_action_idx_to_closest_thing(
        game_state, is_near_crate)
    feature_vector[action_idx_to_crate + 5] = 1

    # How to get in the reach of opponents
    action_idx_to_opponents = get_action_idx_to_closest_thing(
        game_state, is_opponent_in_blast_range)
    feature_vector[action_idx_to_opponents + 10] = 1

    # How to get to safety
    action_idx_to_safety = get_action_idx_to_closest_thing(
        game_state, is_save_step_criterion)
    feature_vector[action_idx_to_safety + 15] = 1

    # How much danger is estimated in each direction
    feature_vector[20:25] = get_danger_in_each_direction(
        agent_coords, game_state)

    # Could we survive a bomb?
    can_reach_safety, _ = simulate_bomb(agent_coords, game_state)
    feature_vector[25] = can_reach_safety

    # Can we place a bomb?
    feature_vector[26] = game_state.self[2]

    # Is it a perfect spot for a bomb?
    bomb_blast_coords = get_blast_effected_coords(
        agent_coords, game_state.field)
    n_destroying_crates = get_number_of_destroying_crates(
        bomb_blast_coords, game_state.field)
    would_kill_opponent = would_surely_kill_opponent(
        bomb_blast_coords, game_state)

    feature_vector[27] = n_destroying_crates >= 4 or would_kill_opponent

    # Is currently only one opponent left?
    living_opponent = game_state.others
    feature_vector[28] = len(living_opponent) <= 1

    # Are we currently in the lead?
    own_score = game_state.self[1]
    feature_vector[29] = own_score > max(scores)
