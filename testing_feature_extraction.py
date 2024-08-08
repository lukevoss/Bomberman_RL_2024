import unittest

import numpy as np
import torch

from agent_code.utils import *
from agent_code.feature_extraction import state_to_features_large, state_to_very_small_features

# TODO: Test very small features



class TestingLargeFeatureExtraction(unittest.TestCase):
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
        feature_vector = state_to_features_large(
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

        feature_vector = state_to_features_large(
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

    def test_direction_is_up(self):
        state = copy.deepcopy(self.game_state)
        state["field"][3, 6] = CRATE
        state["field"][2, 15] = CRATE
        state["bombs"] = [((1, 15), 3)]
        state["coins"] = [(3, 7)]
        state["others"] = [('opponent', 0, 1, (1, 7))]
        state["self"] = ('test_agent', 4, 0, (1, 15))
        feature_vector = state_to_features_large(
            state, max_opponents_score=0)
        ground_truth = torch.zeros(30)

        ground_truth[0] = 1 #Coin
        ground_truth[9] = 1 
        ground_truth[10] = 1
        ground_truth[15] = 1

        #Danger in each direction:
        ground_truth[20] = 0.25
        ground_truth[21] = 1
        ground_truth[22] = 1
        ground_truth[23] = 1
        ground_truth[24] = 0.25

        ground_truth[25] = 1
        ground_truth[26] = 0
        ground_truth[27] = 0
        ground_truth[28] = 1/3
        ground_truth[29] = 1
        print(feature_vector)
        self.assertTrue(torch.equal(feature_vector, ground_truth))


# class TestingVerySmallFeatureExtraction(unittest.TestCase):
#     def setUp(self):
#         self.game_state = {
#             "field": EMPTY_FIELD,
#             "bombs": [],
#             "explosion_map": np.zeros((s.ROWS, s.COLS)),
#             "coins": [],
#             "self": ('test_agent', 0, 1, (1, 1)),
#             "others": [],
#             "step": 0,
#             "round": 0,
#             "user_input": None
#         }

#     def test_empty_state(self):
#         feature_vector = state_to_features_large(
#             self.game_state, max_opponents_score=1)
#         ground_truth = torch.zeros(30)
#         ground_truth[20] = 1
#         ground_truth[22] = 1
#         ground_truth[25] = 1
#         ground_truth[26] = 1
#         self.assertTrue(torch.equal(feature_vector, ground_truth))

#     def test_non_empty_state(self):
#         state = copy.deepcopy(self.game_state)
#         state["field"][5, 1] = CRATE
#         state["bombs"] = [((1, 1), 3)]
#         state["coins"] = [(1, 10)]
#         state["others"] = [('opponent', 0, 1, (4, 1))]
#         state["self"] = ('test_agent', 4, 0, (1, 1))

#         feature_vector = state_to_features_large(
#             state, max_opponents_score=0)
#         ground_truth = torch.zeros(30)
#         ground_truth[1] = 1
#         ground_truth[1+5] = 1
#         ground_truth[4+10] = 1
#         ground_truth[1+15] = 1

#         ground_truth[20] = 1
#         ground_truth[21] = 0.25
#         ground_truth[22] = 1
#         ground_truth[23] = 0.25
#         ground_truth[24] = 0.25

#         ground_truth[25] = 1
#         ground_truth[26] = 0
#         ground_truth[27] = 1
#         ground_truth[28] = 1/3
#         ground_truth[29] = 1
#         self.assertTrue(torch.equal(feature_vector, ground_truth))

#     def test_direction_is_up(self):
#         state = copy.deepcopy(self.game_state)
#         state["field"][3, 6] = CRATE
#         state["field"][2, 15] = CRATE
#         state["bombs"] = [((1, 15), 3)]
#         state["coins"] = [(3, 7)]
#         state["others"] = [('opponent', 0, 1, (1, 7))]
#         state["self"] = ('test_agent', 4, 0, (1, 15))
#         feature_vector = state_to_features_large(
#             state, max_opponents_score=0)
#         ground_truth = torch.zeros(30)

#         ground_truth[0] = 1 #Coin
#         ground_truth[9] = 1 
#         ground_truth[10] = 1
#         ground_truth[15] = 1

#         #Danger in each direction:
#         ground_truth[20] = 0.25
#         ground_truth[21] = 1
#         ground_truth[22] = 1
#         ground_truth[23] = 1
#         ground_truth[24] = 0.25

#         ground_truth[25] = 1
#         ground_truth[26] = 0
#         ground_truth[27] = 0
#         ground_truth[28] = 1/3
#         ground_truth[29] = 1
#         print(feature_vector)
#         self.assertTrue(torch.equal(feature_vector, ground_truth))



if __name__ == '__main__':
    unittest.main()
