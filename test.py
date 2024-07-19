import os
import unittest
from time import time
from agent_code.ppo_agent.callbacks import normalize_state, state_to_features
import numpy as np
from main import main


class MainTestCase(unittest.TestCase):
    def test_play(self):
        start_time = time()
        main(["play", "--n-rounds", "1", "--no-gui"])
        # Assert that log exists
        self.assertTrue(os.path.isfile("logs/game.log"))
        # Assert that game log way actually written
        self.assertGreater(os.path.getmtime("logs/game.log"), start_time)

    def test_normalize_state(self):
        state = {}

        #### Test flip left/right ####
        field = np.zeros((17,17))
        field[0,0] = 1
        test_agent = ('test_agent', 0, True, (16,0))

        state['field'] = field
        state['self'] = test_agent
        state['bombs'] = [((0,0),3)]
        state['explosion_map'] = field
        state['coins'] = [(0,0)]
        state['others'] = [test_agent]
        flip_lr_state = state.copy()
        test_state = state.copy
        action_map, reverse_action_map = normalize_state(flip_lr_state)

        flipped_field = np.zeros((17,17))
        flipped_field[16,0] = 1
        flipped_bombs = [((16,0),3)]
        flipped_coins = [(16,0)]
        flipped_agent = ('test_agent', 0, True, (0,0))

        self.assertTrue(np.array_equal(flip_lr_state['field'], flipped_field))
        self.assertTrue(flip_lr_state['bombs'] == flipped_bombs)
        self.assertTrue(flip_lr_state['coins'] == flipped_coins)
        self.assertTrue(flip_lr_state['others'] == [flipped_agent])
        self.assertTrue(flip_lr_state['self'] == flipped_agent)
        
        self.assertTrue(action_map('LEFT') == 'RIGHT' == reverse_action_map('LEFT'))
        self.assertTrue(action_map('RIGHT') == 'LEFT' == reverse_action_map('RIGHT'))
        self.assertTrue(action_map('UP') == 'UP' == reverse_action_map('UP'))
        self.assertTrue(action_map('DOWN') == 'DOWN' == reverse_action_map('DOWN'))

        ### Test flip up/down with flip left/right
        test_agent = ('test_agent', 0, True, (16,16))
        flip_lr_ud_state = state.copy()
        flip_lr_ud_state['self'] = test_agent
        flip_lr_ud_state['others'] = [test_agent]

        action_map, reverse_action_map = normalize_state(flip_lr_ud_state)

        flipped_field = np.zeros((17,17))
        flipped_field[16,16] = 1
        flipped_bombs = [((16,16),3)]
        flipped_coins = [(16,16)]
        flipped_agent = ('test_agent', 0, True, (0,0))
        
        self.assertTrue(np.array_equal(flip_lr_ud_state['field'], flipped_field))
        self.assertTrue(flip_lr_ud_state['bombs'] == flipped_bombs)
        self.assertTrue(flip_lr_ud_state['coins'] == flipped_coins)
        self.assertTrue(flip_lr_ud_state['others'] == [flipped_agent])
        self.assertTrue(flip_lr_ud_state['self'] == flipped_agent)
        
        self.assertTrue(action_map('LEFT') == 'RIGHT' == reverse_action_map('LEFT'))
        self.assertTrue(action_map('RIGHT') == 'LEFT' == reverse_action_map('RIGHT'))
        self.assertTrue(action_map('UP') == 'DOWN' == reverse_action_map('UP'))
        self.assertTrue(action_map('DOWN') == 'UP'== reverse_action_map('DOWN'))

        ### Test flip diagonal
        field = np.zeros((17,17))
        field[1,0] = 1
        test_agent = ('test_agent', 0, True, (0,1))
        flip_diag_state = {}
        flip_diag_state['self'] = test_agent
        flip_diag_state['others'] = [test_agent]
        flip_diag_state['field'] = field
        flip_diag_state['bombs'] = [((1,0),3)]
        flip_diag_state['explosion_map'] = field
        flip_diag_state['coins'] = [(1,0)]

        action_map, reverse_action_map = normalize_state(flip_diag_state)

        flipped_field = np.zeros((17,17))
        flipped_field[0,1] = 1
        flipped_bombs = [((0,1),3)]
        flipped_coins = [(0,1)]
        flipped_agent = ('test_agent', 0, True, (1,0))
        
        self.assertTrue(np.array_equal(flip_diag_state['field'], flipped_field))
        self.assertTrue(flip_diag_state['bombs'] == flipped_bombs)
        self.assertTrue(flip_diag_state['coins'] == flipped_coins)
        self.assertTrue(flip_diag_state['others'] == [flipped_agent])
        self.assertTrue(flip_diag_state['self'] == flipped_agent)

        self.assertTrue(action_map('LEFT') == 'DOWN' == reverse_action_map('LEFT'))
        self.assertTrue(action_map('DOWN') == 'LEFT' == reverse_action_map('DOWN'))
        self.assertTrue(action_map('UP') == 'RIGHT'== reverse_action_map('UP'))
        self.assertTrue(action_map('RIGHT') == 'UP'== reverse_action_map('RIGHT'))
        
        ### Test all flips together
        field = np.zeros((17,17))
        field[15,16] = 1
        test_agent = ('test_agent', 0, True, (16,15))
        flip_all_state = {}
        flip_all_state['self'] = test_agent
        flip_all_state['others'] = [test_agent]
        flip_all_state['field'] = field
        flip_all_state['bombs'] = [((15,16),3)]
        flip_all_state['explosion_map'] = field
        flip_all_state['coins'] = [(15,16)]

        action_map, reverse_action_map = normalize_state(flip_all_state)

        flipped_field = np.zeros((17,17))
        flipped_field[0,1] = 1
        flipped_bombs = [((0,1),3)]
        flipped_coins = [(0,1)]
        flipped_agent = ('test_agent', 0, True, (1,0))
        
        self.assertTrue(np.array_equal(flip_all_state['field'], flipped_field))
        self.assertTrue(flip_all_state['bombs'] == flipped_bombs)
        self.assertTrue(flip_all_state['coins'] == flipped_coins)
        self.assertTrue(flip_all_state['others'] == [flipped_agent])
        self.assertTrue(flip_all_state['self'] == flipped_agent)
        
        self.assertTrue(action_map('LEFT') == 'UP' == reverse_action_map('LEFT'))
        self.assertTrue(action_map('UP') == 'LEFT' == reverse_action_map('UP'))
        self.assertTrue(action_map('RIGHT') == 'DOWN'== reverse_action_map('RIGHT'))
        self.assertTrue(action_map('DOWN') == 'RIGHT'== reverse_action_map('DOWN'))
        
        ### Test up/down with diagonal flips together
        field = np.zeros((17,17))
        field[0,0] = 1
        test_agent = ('test_agent', 0, True, (0,15))
        flip_ud_diag_state = {}
        flip_ud_diag_state['self'] = test_agent
        flip_ud_diag_state['others'] = [test_agent]
        flip_ud_diag_state['field'] = field
        flip_ud_diag_state['bombs'] = [((0,0),3)]
        flip_ud_diag_state['explosion_map'] = field
        flip_ud_diag_state['coins'] = [(0,0)]

        action_map, reverse_action_map = normalize_state(flip_ud_diag_state)

        flipped_field = np.zeros((17,17))
        flipped_field[16,0] = 1
        flipped_bombs = [((16,0),3)]
        flipped_coins = [(16,0)]
        flipped_agent = ('test_agent', 0, True, (1,0))
        
        self.assertTrue(np.array_equal(flip_ud_diag_state['field'], flipped_field))
        self.assertTrue(flip_ud_diag_state['bombs'] == flipped_bombs)
        self.assertTrue(flip_ud_diag_state['coins'] == flipped_coins)
        self.assertTrue(flip_ud_diag_state['others'] == [flipped_agent])
        self.assertTrue(flip_ud_diag_state['self'] == flipped_agent)

        self.assertTrue(action_map('LEFT') == 'DOWN' )
        self.assertTrue(action_map('RIGHT') == 'UP')
        self.assertTrue(action_map('UP') == 'LEFT')
        self.assertTrue(action_map('DOWN') == 'RIGHT')

        self.assertTrue(reverse_action_map('LEFT') == 'UP' )
        self.assertTrue(reverse_action_map('RIGHT') == 'DOWN')
        self.assertTrue(reverse_action_map('UP') == 'RIGHT')
        self.assertTrue(reverse_action_map('DOWN') == 'LEFT')

    def test_state_to_feature(self):
        pass # TODO

if __name__ == '__main__':
    unittest.main()
