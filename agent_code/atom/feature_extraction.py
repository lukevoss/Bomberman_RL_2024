
import torch

from agent_code.atom.utils import *

FEATURE_VECTOR_SIZE=20


def state_to_small_features(game_state: dict, num_coins_already_discovered: int) -> List:
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
        # If we are in danger and a path exist, dont calculate other directions
        if action_idx_to_safety != None:
            feature_vector[action_idx_to_safety] = 1
            return feature_vector
    # If we are sourrounded by danger then wait and don't calculate other directions
    elif state.is_danger_all_around(agent_coords):
        feature_vector[4] = 1
        return feature_vector

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


    # If we can't place a bomb and survive it, don't show bombing opponent as an option
    feature_vector[14] &= feature_vector[19] 


    return feature_vector

