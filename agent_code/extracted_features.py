"""
Board is abstracted as a boolean vector of size 20 with each feature as following:

    Direction to closest Coin
        [0]:Up
        [1]:Down
        [2]:Left
        [3]:Right
        [4]:Wait

    Direction to closest Crate
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

    Amount of danger in next moves from 0 (Safe)  to 1 (Immideatly dead)
        [20]:Up
        [21]:Down
        [22]:Left
        [23]:Right
        [24]:Wait

    [25] Could we survive a placed Bomb

    [26] Can we place a bomb

    [27] Very smart bomb position: Would destroy 4 or more crates or opponent in trap

    [28] Normalized number of opponents left (1 = all still alive, 0 = no one left)

    [29] Currently in the lead

    ??[30] Number of coins left in Game ??


# TODO make crossing of bomb danger zone possible
# TODO sometimes placing a bomb before running out of danger can trap a opponent between two bombs
# TODO Should we do masking?
# TODO are any coins left in game? If not start hunting and dont destroy more crates
# TODO Controll correct action_idx
Author: Luke Voss
"""
import numpy as np


from utils import *

FEATURE_VECTOR_SIZE = 30


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
        game_state, is_save_step)
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
    feature_vector[27] = is_perfect_bomb_spot(agent_coords, game_state)

    # Normalized Number of living opponent
    feature_vector[28] = len(game_state.others) / 3

    # Are we currently in the lead?
    own_score = game_state.self[1]
    feature_vector[29] = own_score > max(scores)
