
import torch

from agent_code.utils import *


def state_to_features_large(game_state: dict, max_opponents_score: int, num_coins_already_discovered: int) -> torch.tensor:
    """
    Board is abstracted as a boolean vector of size 29 with each feature as following:

        If in Danger, Direction to safety:
            [0]:Up
            [1]:Down
            [2]:Left
            [3]:Right
            [4]:Wait

        Amount of danger in next moves from 0 (Safe)  to 1 (Immideatly dead)
            [5]:Up
            [6]:Down
            [7]:Left
            [8]:Right
            [9]:Wait

        Direction to closest Crate (only if coins are left in game)
            [10]:Up
            [11]:Down
            [12]:Left
            [13]:Right
            [14]:Wait

        Direction in which placing a bomb would kill another player
            [15]:Up
            [16]:Down
            [17]:Left
            [18]:Right
            [19]:Place now

        Direction to closest Coin
            [20]:Up
            [21]:Down
            [22]:Left
            [23]:Right

        [24] Could we survive a placed Bomb

        [25] Can we place a bomb

        [26] Very smart bomb position: Would destroy 4 or more crates or opponent in trap

        [27] Normalized number of opponents left (1 = all still alive, 0 = no one left)

        [28] Currently in the lead

        Additional Ideas:
            - Number of coins left in Game (9 total)
            - sometimes placing a bomb before running out of danger can trap a opponent between two bombs
            - search for many possible paths and give back more than one action idx (especially for safety!)
                -> keep searching as long as distance is the same (Safety can search for more)

    Author: Luke Voss
    """
    feature_vector = torch.zeros(27)

    state = GameState(**game_state)
    agent_coords = state.self[3]

    # How to get to safety
    if state.is_dangerous(agent_coords):
        action_idx_to_safety = state.get_action_idx_to_closest_thing('safety')
        if action_idx_to_safety != None:
            feature_vector[action_idx_to_safety] = 1
    elif state.is_danger_all_around(agent_coords):
        feature_vector[4] = 1

    # How much danger is estimated in each direction
    feature_vector[5:10] = torch.from_numpy(
        state.get_danger_in_each_direction(agent_coords)).type_as(feature_vector)

    # How to get to closest crate
    if num_coins_already_discovered < NUM_COINS_IN_GAME:
        action_idx_to_crate = state.get_action_idx_to_closest_thing('crate')
        if action_idx_to_crate != None:
            feature_vector[action_idx_to_crate + 10] = 1
    elif num_coins_already_discovered > NUM_COINS_IN_GAME:
        raise ValueError("Number of discovered coins cant be that high")

    # How to get in the reach of opponents
    action_idx_to_opponents = state.get_action_idx_to_closest_thing('opponent')
    if action_idx_to_opponents != None:
        feature_vector[action_idx_to_opponents + 15] = 1


    # How to get to closest coin
    action_idx_to_coin = state.get_action_idx_to_closest_thing('coin')
    if (action_idx_to_coin != None) and (action_idx_to_coin != 4):
        feature_vector[action_idx_to_coin+20] = 1

    # Can we place a Bomb and survive it?
    can_reach_safety, _ = state.simulate_own_bomb()
    feature_vector[24] = can_reach_safety and state.self[2]

    # Is it a perfect spot for a bomb?
    feature_vector[25] = state.is_perfect_bomb_spot(agent_coords)

    # Could we survive a placed Bomb
    can_reach_safety, _ = state.simulate_own_bomb()
    feature_vector[25] = can_reach_safety

    # Can we place a bomb?
    feature_vector[26] = state.self[2]

    # Is it a perfect spot for a bomb?
    feature_vector[27] = state.is_perfect_bomb_spot(agent_coords)

    # Normalized Number of living opponent
    feature_vector[28] = len(state.others) / 3

    # Are we currently in the lead?
    own_score = state.self[1]
    feature_vector[29] = own_score > max_opponents_score

    return feature_vector


def state_to_small_features(game_state: dict, num_coins_already_discovered: int) -> List:
    """
    Board is abstracted as a boolean vector of size 21 with each feature as following:

    If in Danger, Direction to safety:
        [0]:Up
        [1]:Down
        [2]:Left
        [3]:Right
        [4]:Wait

    Direction to closest Crate (only if coins are left in the game)
        [5]:Up
        [6]:Down
        [7]:Left
        [8]:Right
        [9]:Wait

    Direction in which placing a bomb would kill another player
        [10]:Up
        [11]:Down
        [12]:Left
        [13]:Right
        [14]:Place now

    Direction to closest Coin
        [15]:Up
        [16]:Down
        [17]:Left
        [18]:Right

    [19] Can we place a bomb and survive it?

    [20] Can we place a smart bomb?

    """
    feature_vector = [0] * 21

    state = GameState(**game_state)
    agent_coords = state.self[3]

    # How to get to safety
    if state.is_dangerous(agent_coords):
        action_idx_to_safety = state.get_action_idx_to_closest_thing('safety')
        if action_idx_to_safety != None:
            feature_vector[action_idx_to_safety] = 1
    elif state.is_danger_all_around(agent_coords):
        feature_vector[4] = 1


    # How to get to closest crate
    if num_coins_already_discovered < NUM_COINS_IN_GAME:
        action_idx_to_crate = state.get_action_idx_to_closest_thing('crate')
        if action_idx_to_crate != None:
            feature_vector[action_idx_to_crate + 5] = 1
    elif num_coins_already_discovered > NUM_COINS_IN_GAME:
        raise ValueError("Number of discovered coins cant be that high")

    # How to get in the reach of opponents
    action_idx_to_opponents = state.get_action_idx_to_closest_thing('opponent')
    if action_idx_to_opponents != None:
        feature_vector[action_idx_to_opponents + 10] = 1

    # How to get to closest coin
    action_idx_to_coin = state.get_action_idx_to_closest_thing('coin')
    if (action_idx_to_coin != None) and (action_idx_to_coin != 4):
        feature_vector[action_idx_to_coin + 15] = 1

    # Can we place a Bomb and survive it?
    can_reach_safety, _ = state.simulate_own_bomb()
    feature_vector[20] = int(can_reach_safety and state.self[2])

    # Is it a perfect spot for a bomb?
    feature_vector[21] = int(state.is_perfect_bomb_spot(agent_coords))

    return feature_vector

def state_to_very_small_features_imitation(game_state: dict, num_coins_already_discovered: int) -> List:
    """
    Version of state_to_very_small_features for imitation learning, this version keeps showing the path to the closest crate
    even if no coins exist anymore in the game, so the states can be correctly imitated from the rule based player
    Also the shortest path search stops when next to opponent not already when in bomb reach
    Board is abstracted as a boolean vector of size 20 with each feature as following:


    If in Danger, Direction to safety:
        [0]:Up
        [1]:Down
        [2]:Left
        [3]:Right
        [4]:Wait

    Direction to closest Crate (if not imitation learning than only if coins are left in the game)
        [5]:Up
        [6]:Down
        [7]:Left
        [8]:Right
        [9]:Wait

    Direction in which placing a bomb would kill another player
        [10]:Up
        [11]:Down
        [12]:Left
        [13]:Right
        [14]:Place now

    Direction to closest Coin
        [15]:Up
        [16]:Down
        [17]:Left
        [18]:Right

    [19] Can we place a bomb and survive it?

    """
    feature_vector = [0] * 20

    state = GameState(**game_state)
    agent_coords = state.self[3]

    # How to get to safety
    if state.is_dangerous(agent_coords):
        action_idx_to_safety = state.get_action_idx_to_closest_thing('safety')
        if action_idx_to_safety != None:
            feature_vector[action_idx_to_safety] = 1
    elif state.is_danger_all_around(agent_coords):
        feature_vector[4] = 1

    # How to get to closest crate
    #if num_coins_already_discovered < NUM_COINS_IN_GAME:
    action_idx_to_crate = state.get_action_idx_to_closest_thing('crate')
    if action_idx_to_crate != None:
        feature_vector[action_idx_to_crate + 5] = 1

    # How to get in the reach of opponents
    action_idx_to_opponents = state.get_action_idx_to_closest_thing('next_to_opponent')
    if action_idx_to_opponents != None:
        feature_vector[action_idx_to_opponents + 10] = 1

    # How to get to closest coin
    action_idx_to_coin = state.get_action_idx_to_closest_thing('coin')
    if (action_idx_to_coin != None) and (action_idx_to_coin != 4):
        feature_vector[action_idx_to_coin + 15] = 1

    # Can we place a Bomb and survive it?
    can_reach_safety, _ = state.simulate_own_bomb()
    feature_vector[19] = int(can_reach_safety and state.self[2])


    return feature_vector

def state_to_very_small_features(game_state: dict, num_coins_already_discovered: int) -> List:
    """
    Board is abstracted as a boolean vector of size 20 with each feature as following:

    If in Danger, Direction to safety:
        [0]:Up
        [1]:Down
        [2]:Left
        [3]:Right
        [4]:Wait

    Direction to closest Crate (only if coins are left in the game)
        [5]:Up
        [6]:Down
        [7]:Left
        [8]:Right
        [9]:Wait

    Direction in which placing a bomb would kill another player
        [10]:Up
        [11]:Down
        [12]:Left
        [13]:Right
        [14]:Place now

    Direction to closest Coin
        [15]:Up
        [16]:Down
        [17]:Left
        [18]:Right

    [19] Can we place a bomb and survive it?

    """
    feature_vector = [0] * 20

    state = GameState(**game_state)
    agent_coords = state.self[3]

    # How to get to safety
    if state.is_dangerous(agent_coords):
        action_idx_to_safety = state.get_action_idx_to_closest_thing('safety')
        if action_idx_to_safety != None:
            feature_vector[action_idx_to_safety] = 1
    elif state.is_danger_all_around(agent_coords):
        feature_vector[4] = 1

    # How to get to closest crate
    if num_coins_already_discovered < NUM_COINS_IN_GAME:
        action_idx_to_crate = state.get_action_idx_to_closest_thing('crate')
        if action_idx_to_crate != None:
            feature_vector[action_idx_to_crate + 5] = 1
    elif num_coins_already_discovered > NUM_COINS_IN_GAME:
        raise ValueError("Number of discovered coins can't be that high")

    # How to get in the reach of opponents
    action_idx_to_opponents = state.get_action_idx_to_closest_thing('opponent')
    if action_idx_to_opponents != None:
        feature_vector[action_idx_to_opponents + 10] = 1

    # How to get to closest coin
    action_idx_to_coin = state.get_action_idx_to_closest_thing('coin')
    if (action_idx_to_coin != None) and (action_idx_to_coin != 4):
        feature_vector[action_idx_to_coin + 15] = 1

    # Can we place a Bomb and survive it?
    can_reach_safety, _ = state.simulate_own_bomb()
    feature_vector[19] = int(can_reach_safety and state.self[2])

    # if feature_vector == [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]:
    #     pass


    return feature_vector


def state_to_tiny_features(game_state: dict, num_coins_already_discovered: int) -> List:
    """
    Board is abstracted as a boolean vector of size 16 with each feature as following:

    Direction to Coin or Safety (If there is no coin reachable without dying then show direction to safety)
        [0]:Up
        [1]:Down
        [2]:Left
        [3]:Right
        [4]:Wait

    Direction to closest Crate (only if coins are left in the game)
        [5]:Up
        [6]:Down
        [7]:Left
        [8]:Right
        [9]:Wait

    Direction in which placing a bomb would kill another player
        [10]:Up
        [11]:Down
        [12]:Left
        [13]:Right
        [14]:Place now

    [15] Can we place a bomb and survive it?

    """
    feature_vector = [0] * 16

    state = GameState(**game_state)
    agent_coords = state.self[3]

    # How to get to closest coin or safety
    if exists_coin(state):
        action_idx_to_coin = state.get_action_idx_to_closest_thing('coin')
        if (action_idx_to_coin != None) and (action_idx_to_coin != 4):
            feature_vector[action_idx_to_coin] = 1
        elif state.is_dangerous(agent_coords):
            action_idx_to_safety = state.get_action_idx_to_closest_thing('safety')
            if (action_idx_to_safety != None):
                feature_vector[action_idx_to_safety] = 1
    elif state.is_dangerous(agent_coords):
        action_idx_to_safety = state.get_action_idx_to_closest_thing('safety')
        if (action_idx_to_safety != None):
            feature_vector[action_idx_to_safety] = 1


    # How to get to closest crate
    if num_coins_already_discovered < NUM_COINS_IN_GAME:
        action_idx_to_crate = state.get_action_idx_to_closest_thing('crate')
        if action_idx_to_crate != None:
            feature_vector[action_idx_to_crate + 5] = 1
    elif num_coins_already_discovered > NUM_COINS_IN_GAME:
        raise ValueError("Number of discovered coins can't be that high")

    # How to get in the reach of opponents
    action_idx_to_opponents = state.get_action_idx_to_closest_thing('opponent')
    if action_idx_to_opponents != None:
        feature_vector[action_idx_to_opponents + 10] = 1

    # Can we place a Bomb and survive it?
    can_reach_safety, _ = state.simulate_own_bomb()
    feature_vector[15] = int(can_reach_safety and state.self[2])

    # if feature_vector == [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]:
    #     pass


    return feature_vector
