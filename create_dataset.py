import os
from typing import List

import numpy as np

import events as e
import own_events as own_e

def main():

    input_dir = "./agent_code/data_generator/data"
    npz_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]
    N_DATASET = 40000

    all_states = np.zeros((N_DATASET,27))
    all_action_idx = np.zeros(N_DATASET)
    all_masks = np.zeros(N_DATASET)
    all_rewards = np.zeros(N_DATASET)

    for i in range(N_DATASET):
        npz_file = npz_files[i]
        print(i)
        file_path = os.path.join(input_dir, npz_file)
        data = np.load(file_path)
        all_states[i,:] = data["state"]
        all_action_idx[i] = data["action"]
        all_masks[i] = 1 - data['is_terminal']
        all_rewards[i] = reward_from_events(data['events'])


    np.savez_compressed("./expert_data.npz", states=all_states, actions=all_action_idx, masks = all_masks, rewards = all_rewards)


def reward_from_events(events: List[str]) -> int:

    game_rewards = {
        # SPECIAL EVENTS
        own_e.CONSTANT_PENALTY: -0.001,
        own_e.WON_ROUND: 10,
        own_e.BOMBED_1_TO_2_CRATES: 0,
        own_e.BOMBED_3_TO_5_CRATES: 0.5,
        own_e.BOMBED_5_PLUS_CRATES: 0.5,
        own_e.GOT_IN_LOOP: -0.3,
        own_e.ESCAPING: 0.03,
        own_e.OUT_OF_DANGER: 0.05,
        own_e.NOT_ESCAPING: -0.01,
        own_e.CLOSER_TO_COIN: 0.05,
        own_e.AWAY_FROM_COIN: -0.02,
        own_e.CLOSER_TO_CRATE: 0.01,
        own_e.AWAY_FROM_CRATE: -0.05,
        own_e.SURVIVED_STEP: 0,
        own_e.DESTROY_TARGET: 0.03,
        own_e.MISSED_TARGET: -0.01,
        own_e.WAITED_NECESSARILY: 0.05,
        own_e.WAITED_UNNECESSARILY: -2,
        own_e.CLOSER_TO_PLAYERS: 0.02,
        own_e.AWAY_FROM_PLAYERS: -0.01,
        own_e.SMART_BOMB_DROPPED: 0.7,
        own_e.DUMB_BOMB_DROPPED: -0.5,

        # DEFAULT EVENTS
        e.INVALID_ACTION: -1,
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,
        e.CRATE_DESTROYED: 0.01,
        e.COIN_FOUND: 0,
        e.COIN_COLLECTED: 3,
        e.KILLED_OPPONENT: 6,
        e.KILLED_SELF: -8,
        e.GOT_KILLED: -10,
        e.OPPONENT_ELIMINATED: 0,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum

if __name__ == "__main__":

    main()