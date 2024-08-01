import unittest

import numpy as np
import torch

from agent_code.utils import *
from agent_code.feature_extraction import state_to_features


class TestingFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.game_state = {
            "field": EMPTY_FIELD,
            "bombs": [],
            "explosion_map": np.zeros((s.ROWS, s.COLS)),
            "coins": [],
            "self": ('test_agent', 0, 1, (1, 1)),
            "others": [],
            "step": 0,
            "round": 0,
            "user_input": None
        }

    def test_empty_state(self):
        feature_vector = state_to_features(
            self.game_state, max_opponents_score=1)
        ground_truth = torch.zeros(30)
        ground_truth[20] = 1
        ground_truth[22] = 1
        ground_truth[25] = 1
        ground_truth[26] = 1
        self.assertTrue(torch.equal(feature_vector, ground_truth))

    def test_non_empty_state(self):
        state = copy.deepcopy(self.game_state)
        state["field"][5, 1] = CRATE
        state["bombs"] = [((1, 1), 3)]
        state["coins"] = [(1, 10)]
        state["others"] = [('opponent', 0, 1, (4, 1))]
        state["self"] = ('test_agent', 4, 0, (1, 1))

        feature_vector = state_to_features(
            state, max_opponents_score=0)
        ground_truth = torch.zeros(30)
        ground_truth[1] = 1
        ground_truth[1+5] = 1
        ground_truth[4+10] = 1
        ground_truth[1+15] = 1

        ground_truth[20] = 1
        ground_truth[21] = 0.25
        ground_truth[22] = 1
        ground_truth[23] = 0.25
        ground_truth[24] = 0.25

        ground_truth[25] = 1
        ground_truth[26] = 0
        ground_truth[27] = 1
        ground_truth[28] = 1/3
        ground_truth[29] = 1
        self.assertTrue(torch.equal(feature_vector, ground_truth))


if __name__ == '__main__':
    unittest.main()
