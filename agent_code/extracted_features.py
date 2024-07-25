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

    Amount of danger in next moves from -1 (Safe)  to 1 (Immideatly dead)
        [20]:Up
        [21]:Right
        [22]:Down
        [23]:Left
        [24]:Wait
        # TODO Add Bomb?

    [25] Could we survive a placed Bomb

    [26] Can we place a bomb

    [27] Very smart bomb position: Would destroy 4 or more crates or opponent in trap #and we can place one?

    [28] Only one opponent left

    [29] Currently in the lead



# TODO make crossing of bomb danger zone possible
# TODO sometimes placing a bomb before running out of danger can trap a opponent between two bombs
# TODO Kill self if no crates or coin left and only one other opponent (make sure opponent not in reach)
# TODO Trap agents with bombs in dead end -> rule_based_agents run towards dead end and dont hunt for 5 turns. One Block before dead end is optimal position, since rule_based_agent hasn't dropped bomb yet

Author: Luke Voss
"""

import numpy as np


def state_to_features(game_state: dict) -> np.array:
    pass
