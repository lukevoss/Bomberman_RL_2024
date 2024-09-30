import os
from typing import List

import numpy as np
from tqdm import tqdm  

import events as e
import agent_code.atom.own_events as own_e
from agent_code.atom.add_own_events import reward_from_events

def main():

    input_dir = "./agent_code/data_generator/data"
    npz_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]
    N_DATASET = 223000
    FEATURE_VECTOR_SIZE = 20

    all_states = np.zeros((N_DATASET,FEATURE_VECTOR_SIZE))
    all_new_states = np.zeros((N_DATASET,FEATURE_VECTOR_SIZE))
    all_action_idx = np.zeros(N_DATASET)
    all_masks = np.zeros(N_DATASET)
    all_rewards = np.zeros(N_DATASET)

    for i in tqdm(range(N_DATASET), desc="Loading .npz files"):
        npz_file = npz_files[i]
        file_path = os.path.join(input_dir, npz_file)
        data = np.load(file_path, allow_pickle=True)
        all_states[i,:] = data["state"]
        all_action_idx[i] = data["action"]
        all_masks[i] = 1 - data['is_terminal']
        all_rewards[i] = reward_from_events(data['events'])


    np.savez_compressed("./expert_data_ppo.npz", states=all_states, actions=all_action_idx, rewards = all_rewards, masks = all_masks)




if __name__ == "__main__":

    main()