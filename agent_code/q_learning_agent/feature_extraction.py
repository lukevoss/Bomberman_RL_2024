"""
Board is abstracted as a boolean vector of size 20 with each feature as following:

    Direction to closest Coin
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

    If in Danger, Direction to safety:
        [15]:Up
        [16]:Down
        [17]:Left
        [18]:Right
        [19]:Wait


    [20] Can we place a bomb and survive it?

    [21] Very smart bomb position: Would destroy 4 or more crates or opponent in trap

    Additional Ideas:
        - sometimes placing a bomb before running out of danger can trap a opponent between two bombs
        - search for many possible paths and give back more than one action idx (especially for safety!)
            -> keep searching as long as distance is the same (Safety can search for more)

Author: Luke Voss
"""
import torch

from agent_code.utils import *

FEATURE_VECTOR_SIZE = 22


def state_to_features(game_state: dict, num_coins_already_discovered: int) -> torch.tensor:
    feature_vector = [0] * FEATURE_VECTOR_SIZE

    state = GameState(**game_state)
    agent_coords = state.self[3]

    # How to get to closest coin
    action_idx_to_coin = state.get_action_idx_to_closest_thing('coin')
    if action_idx_to_coin != None:
        feature_vector[action_idx_to_coin] = 1

    # How to get to closest crate
    if num_coins_already_discovered < NUM_COINS_IN_GAME:
        action_idx_to_crate = state.get_action_idx_to_closest_thing('crate')
        if action_idx_to_crate != None:
            feature_vector[action_idx_to_crate + 5] = 1

    # How to get in the reach of opponents
    action_idx_to_opponents = state.get_action_idx_to_closest_thing('opponent')
    if action_idx_to_opponents != None:
        feature_vector[action_idx_to_opponents + 10] = 1

    # How to get to safety
    if state.is_dangerous(agent_coords):
        action_idx_to_safety = state.get_action_idx_to_closest_thing('safety')
        if action_idx_to_safety != None:
            feature_vector[action_idx_to_safety + 15] = 1
    elif state.is_danger_all_around(agent_coords):
        feature_vector[19] = 1

    # Can we place a Bomb and survive it?
    can_reach_safety, _ = state.simulate_own_bomb()
    feature_vector[20] = can_reach_safety and state.self[2]

    # Is it a perfect spot for a bomb?
    feature_vector[21] = state.is_perfect_bomb_spot(agent_coords) and feature_vector[20]

    return feature_vector
