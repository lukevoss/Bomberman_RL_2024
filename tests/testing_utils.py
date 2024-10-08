"""
Keep in mind that game state is turned making it quite complicated
Movement from game_state to GUI
Gui[x,y] = game_state[y,x]

Down -> Right
Up -> Left
Left -> Up
Right -> Down

# TODO: Testing stop criterion and object exists
"""
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from agent_code.utils import *


class TestingGameState(unittest.TestCase):
    def setUp(self):
        self.state = GameState(
            field=EMPTY_FIELD,
            bombs=[],
            explosion_map=np.zeros((s.ROWS, s.COLS)),
            coins=[],
            self=('test_agent', 0, 1, (1, 1)),
            others=[],
            step=0,
            round=0,
            user_input=None
        )

    def test_next(self):
        old_game_state = copy.deepcopy(self.state)
        # Valid move action
        next_game_state = old_game_state.next("RIGHT")
        self.assertEqual(next_game_state.self[3], (2, 1))

        # Invalid action agains wall
        next_game_state = old_game_state.next("UP")
        self.assertIsNone(next_game_state)

        # Wait action
        next_game_state = old_game_state.next("WAIT")
        self.assertEqual(next_game_state.self[3], old_game_state.self[3])

        # Valid bomb action
        next_game_state = old_game_state.next("BOMB")
        self.assertEqual(next_game_state.bombs, [((1, 1), s.BOMB_TIMER-1)])

        # Invalid bomb action
        old_game_state.self = ('test_agent', 0, 0, (1, 1))
        next_game_state = old_game_state.next("BOMB")
        self.assertIsNone(next_game_state)
        old_game_state.self = ('test_agent', 0, 1, (1, 1))

        # Bomb explodes and clear one create
        old_game_state.bombs = [((3, 2), 0)]
        old_game_state.field[3, 1] = 1

        next_game_state = old_game_state.next("WAIT")

        blast_idx = [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5)]
        for x, y in blast_idx:
            old_game_state.explosion_map[x][y] = 1
        assert_array_equal(next_game_state.field, self.state.field)
        assert_array_equal(next_game_state.explosion_map,
                           old_game_state.explosion_map)

        # Agent walks into explosion
        old_game_state.self = ('test_agent', 0, 1, (2, 1))
        next_game_state = old_game_state.next("RIGHT")
        self.assertIsNone(next_game_state)

        old_game_state = copy.deepcopy(self.state)

        # player dies
        old_game_state.bombs = [((1, 3), 0)]
        next_game_state = old_game_state.next("WAIT")
        self.assertIsNone(next_game_state)

        # player escapes
        next_game_state = old_game_state.next("RIGHT")
        blast_idx = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
                     (1, 6), (2, 3), (3, 3), (4, 3)]
        for x, y in blast_idx:
            old_game_state.explosion_map[x, y] = 1
        assert_array_equal(next_game_state.explosion_map,
                           old_game_state.explosion_map)
        self.assertEqual(next_game_state.self[3], (2, 1))

        old_game_state.explosion_map = self.state.explosion_map

        # Bomb timer one less
        old_game_state.bombs = [((1, 1), 3)]
        next_game_state = old_game_state.next("WAIT")
        bomb = next_game_state.bombs[0]
        self.assertEqual(bomb[1], 2)

    def test_not_escaping_danger(self):
        self.assertTrue(self.state.not_escaping_danger('WAIT'))
        self.assertTrue(self.state.not_escaping_danger('BOMB'))
        self.assertFalse(self.state.not_escaping_danger('LEFT'))

    def test_is_escaping_danger(self):
        state = copy.deepcopy(self.state)
        state.bombs = [((1, 5), 3)]
        state.self = ('test_agent', 0, 1, (1, 3))
        agent_coords = state.self[3]
        sorted_dangerous_bombs = state.sort_and_filter_out_dangerous_bombs(
            agent_coords)

        # Out of bomb reach
        self.assertTrue(state.is_escaping_danger(
            "RIGHT", sorted_dangerous_bombs))
        # Closer to Bomb
        self.assertFalse(state.is_escaping_danger(
            "DOWN", sorted_dangerous_bombs))
        # Further from bomb reach
        self.assertTrue(state.is_escaping_danger(
            'UP', sorted_dangerous_bombs))
        # Invalid action
        self.assertFalse(state.is_escaping_danger(
            'LEFT', sorted_dangerous_bombs))

    def test_has_escaped_danger(self):
        state = copy.deepcopy(self.state)
        state.bombs = [((1, 5), 3)]
        state.self = ('test_agent', 0, 1, (1, 3))
        # Escaped behind wall
        self.assertTrue(state.has_escaped_danger('RIGHT'))
        # Invalid action
        self.assertFalse(state.has_escaped_danger('LEFT'))
        # Not Escaped
        self.assertFalse(state.has_escaped_danger('WAIT'))

        # Run into another danger
        state.explosion_map[2][3] = 1
        self.assertFalse(state.has_escaped_danger('RIGHT'))
        state.explosion_map[2][3] = 0
        # Ran out of Reach
        state.self = ('test_agent', 0, 1, (1, 2))
        self.assertTrue(state.has_escaped_danger('UP'))

    def test_is_valid_movement(self):
        state = copy.deepcopy(self.state)

        self.assertTrue(state.is_valid_movement((1, 1)))
        self.assertFalse(state.is_valid_movement((0, 0)))
        self.assertFalse(state.is_valid_movement((2, 2)))

        # Bomb on field
        state.bombs = [((1, 1), 3)]
        self.assertFalse(state.is_valid_movement((1, 1)))

        # Opponent on field
        state.others = [('other1', 0, 1, (1, 2))]
        self.assertFalse(state.is_valid_movement((1, 2)))

    def test_is_dangerous(self):
        # Setup for the test
        state = copy.deepcopy(self.state)
        state.bombs = [((1, 2), 2)]
        state.explosion_map[3, 3] = 1

        # Test when step is in reach of bomb
        self.assertTrue(state.is_dangerous((1, 1)))

        # Test step out of reach of bomb
        self.assertFalse(state.is_dangerous((2, 1)))

        # Test when step on explosion
        self.assertTrue(state.is_dangerous((3, 3)))

    def test_is_danger_all_around(self):
        state = copy.deepcopy(self.state)
        state.bombs = [((2, 1), 2)]
        state.self = ('test_agent', 0, 1, (1, 2))
        agent_coords = (1,2)
        state.explosion_map[1, 3] = 1

        # Waiting because of bomb and explosion
        self.assertTrue(state.is_danger_all_around(agent_coords))

        # No need to Wait
        state.explosion_map[1, 3] = 0
        self.assertFalse(state.is_danger_all_around(agent_coords))

        # Waiting because of only bombs
        state.field[1, 3] = 1
        self.assertTrue(state.is_danger_all_around(agent_coords))


    def test_simulate_own_bomb(self):
        state = copy.deepcopy(self.state)
        state.field[3, 2] = CRATE
        state.self = ('test_agent', 0, 1, (3, 3))

        # Destroyes Crate, can reach safety
        can_reach_safety, is_effective = state.simulate_own_bomb()
        self.assertTrue(can_reach_safety)
        self.assertTrue(is_effective)

        # Destroyes Opponent, no crate, can reach safety
        state.field[3, 2] = FREE
        state.others = [('opponent1', 3, 1, (3, 2))]
        can_reach_safety, is_effective = state.simulate_own_bomb()
        self.assertTrue(can_reach_safety)
        self.assertTrue(is_effective)

        # Destroyes nothing, can reach safety
        state.others = []
        can_reach_safety, is_effective = state.simulate_own_bomb()
        self.assertTrue(can_reach_safety)
        self.assertFalse(is_effective)

        # Destroyes crate, can't reach safety
        state.self = ('test_agent', 0, 1, (1, 1))
        state.field[2, 1] = CRATE
        state.field[1, 2] = CRATE
        can_reach_safety, is_effective = state.simulate_own_bomb()
        self.assertFalse(can_reach_safety)
        self.assertTrue(is_effective)

    def test_get_action_idx_to_closest_thing(self):
        state = copy.deepcopy(self.state)

        ###########  Coin  ############
        # Two coins, should be DOWN, so 1
        state.self = ('test_agent', 0, 1, (3, 3))
        state.coins = [(3, 4), (7, 7)]
        action_idx = state.get_action_idx_to_closest_thing('coin')
        self.assertEqual(action_idx, 1)

        # Two coin, same distance
        state.coins = [(3, 4), (4, 3)]
        action_idx = state.get_action_idx_to_closest_thing('coin')
        self.assertTrue(action_idx == 1 or action_idx == 3)

        # One coin around corner
        state.coins = [(5, 4)]
        action_idx = state.get_action_idx_to_closest_thing('coin')
        self.assertEqual(action_idx, 3)

        # Two coins one close, but blocked by crates, one free
        state.coins = [(3, 13), (6, 3)]
        state.field[5, 3] = CRATE
        state.field[7, 3] = CRATE
        action_idx = state.get_action_idx_to_closest_thing('coin')
        self.assertEqual(action_idx, 1)

        # No way exists
        state.coins = [(6, 3)]
        action_idx = state.get_action_idx_to_closest_thing('coin')
        self.assertIsNone(action_idx)

        # Walk around crates
        state = copy.deepcopy(self.state)
        state.self = ('test_agent', 0, 1, (3, 3))
        state.coins = [(5, 3)]
        state.field[4, 3] = CRATE
        state.field[6, 3] = CRATE
        state.field[5, 4] = CRATE
        action_idx = state.get_action_idx_to_closest_thing('coin')
        self.assertEqual(action_idx, 0)

        ###########  Crates  ############
        state = copy.deepcopy(self.state)

        # One Crate
        state.field[1, 3] = 1
        action_idx = state.get_action_idx_to_closest_thing('crate')
        self.assertEqual(action_idx, 1)

        ###########  Opponents  ############
        state = copy.deepcopy(self.state)

        # Opponent in blast range
        state.others = [('opponent', 0, 1, (1, 3))]
        action_idx = state.get_action_idx_to_closest_thing('opponent')
        self.assertEqual(action_idx, 4)

        # Next to opponent
        action_idx = state.get_action_idx_to_closest_thing('next_to_opponent')
        self.assertEqual(action_idx, 1)

        # Opponent not in blast range
        state.others = [('opponent', 0, 1, (1, 7))]
        action_idx = state.get_action_idx_to_closest_thing('opponent')
        self.assertEqual(action_idx, 1)

        ###########  Savety  ############
        state = copy.deepcopy(self.state)

        # Way to safety is towards bomb
        state.bombs = [((1, 4), 3)]
        state.field[2, 1] = CRATE
        action_idx = state.get_action_idx_to_closest_thing('safety')
        self.assertEqual(action_idx, 1)

        # Agent is on top of bomb
        state.field[15,3] == CRATE
        state.bombs = [((15,2),3)]
        state.self = ('test_agent', 0, 0, (15, 2))
        action_idx = state.get_action_idx_to_closest_thing('safety')
        self.assertEqual(action_idx,ACTIONS.index("UP"))


    def test_get_danger_in_each_direction(self):
        state = copy.deepcopy(self.state)

        # No Danger, but invalid moves
        danger = state.get_danger_in_each_direction((1, 1))
        ground_truth_danger = [1, 0, 1, 0, 0]
        assert_array_equal(danger, ground_truth_danger)

        # One bomb one explosion
        state.self = ('test_agent', 0, 1, (3, 3))
        state.explosion_map[3, 4] = 1
        state.bombs = [((5, 3), 3)]
        danger = state.get_danger_in_each_direction((3, 3))
        ground_truth_danger = [0, 1, 0.25, 0.25, 0.25]
        assert_array_equal(danger, ground_truth_danger)

        # Two bombs with different timers
        state.bombs = [((5, 3), 2), ((3, 5), 0)]
        state.explosion_map[3, 4] = 0
        danger = state.get_danger_in_each_direction((3, 3))
        ground_truth_danger = [1, 1, 0.5, 0.5, 1]
        assert_array_equal(danger, ground_truth_danger)

    def test_is_perfect_bomb_spot(self):
        state = copy.deepcopy(self.state)

        # Opponent at the edge of field, not in deadend
        state.others = [('test_agent', 0, 1, (15, 15))]
        self.assertFalse(state.is_perfect_bomb_spot((14, 15)))

        # Can destroy 4 crates
        state.field[1, 2] = CRATE
        state.field[1, 3] = CRATE
        state.field[1, 4] = CRATE
        state.field[2, 1] = CRATE
        self.assertTrue(state.is_perfect_bomb_spot((1, 1)))

        # Can only destroy 3 crates
        self.assertFalse(state.is_perfect_bomb_spot((1, 5)))
        state.field[2, 1] = FREE

        # Opponent in deadend, can't move
        state.others = [('opponent', 0, 1, (2, 1))]
        self.assertTrue(state.is_perfect_bomb_spot((3, 1)))

        # Opponent in deadend, can move
        state.self = ('test_agent', 0, 1, (3, 3))
        self.assertTrue(state.is_perfect_bomb_spot((3, 1)))

        # Opponent not in deadend
        state.field[1, 2] = FREE
        self.assertFalse(state.is_perfect_bomb_spot((3, 1)))

        

    def test_sort_and_filter_out_dangerous_bombs(self):
        state = copy.deepcopy(self.state)

        # Sort bombs with many bombs present
        state.bombs = [((1, 2), 0), ((2, 2), 0), ((3, 3), 0),
                       ((1, 4), 0), ((1, 1), 0)]
        sorted_dangerous_bombs = state.sort_and_filter_out_dangerous_bombs(
            (1, 1))
        expected_bombs = [((1, 1), 0), ((1, 2), 0), ((1, 4), 0)]
        self.assertEqual(sorted_dangerous_bombs, expected_bombs)

        # Sort with no bombs
        state.bombs = []
        self.assertEqual(state.sort_and_filter_out_dangerous_bombs((1, 1)), [])

        # Sort with bombs, but none dangerous
        state.bombs = [((1, 1), 0), ((2, 2), 0), ((3, 3), 0)]
        self.assertEqual(state.sort_and_filter_out_dangerous_bombs((5, 5)), [])

    def test_sort_coins_by_distance(self):
        state = copy.deepcopy(self.state)

        state.coins = [(5, 5), (1, 1), (3, 3), (2, 2), (4, 4)]
        agent_coords = (0, 0)
        sorted_coins = state.get_coins_sorted_by_distance(agent_coords)
        self.assertEqual(sorted_coins, [
                         (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])

        # Coins same distance
        state.coins = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        sorted_coins = state.get_coins_sorted_by_distance(agent_coords)
        # Check if sorted by distance and keeps tuple order
        self.assertListEqual(
            sorted_coins, [(1, 0), (0, 1), (-1, 0), (0, -1)])

        # No Coins
        state.coins = []
        self.assertEqual(state.get_coins_sorted_by_distance(agent_coords), [])

        # One Coin
        state.coins = [(2, 2)]
        self.assertEqual(state.get_coins_sorted_by_distance(
            agent_coords), [(2, 2)])

    def test_sort_opponents_by_distance(self):
        state = copy.deepcopy(self.state)

        agent_coords = (0, 0)
        state.others = [('opponent1', 4, 0, (5, 5)),
                        ('opponent2', 3, 1, (1, 1)),
                        ('opponent3', 0, 0, (3, 3))]
        sorted_opponents = state.get_opponents_sorted_by_distance(agent_coords)
        expected_order = [('opponent2', 3, 1, (1, 1)),
                          ('opponent3', 0, 0, (3, 3)),
                          ('opponent1', 4, 0, (5, 5))]
        self.assertEqual(sorted_opponents, expected_order)

        # No opponents
        state.others = []
        self.assertEqual(
            state.get_opponents_sorted_by_distance(agent_coords), [])

        # One opponent
        state.others = [('opponent1', 4, 0, (2, 2))]
        self.assertEqual(state.get_opponents_sorted_by_distance(
            agent_coords), state.others)

    def test_find_closest_create(self):
        state = copy.deepcopy(self.state)
        # Test Crate Directly Adjacent
        state.field[2, 3] = CRATE
        self.assertEqual(state.find_closest_crate((1, 1)), (2, 3))
        self.assertEqual(state.find_closest_crate((2, 1)), (2, 3))

        # Test No Crates
        state.field[2][3] = 0
        self.assertIsNone(state.find_closest_crate((1, 1)))

        # Test Multiple Crates Same Distance
        state.field[2, 3] = CRATE
        state.field[3, 3] = CRATE
        self.assertEqual(state.find_closest_crate((1, 1)), (2, 3))
        self.assertEqual(state.find_closest_crate((3, 4)), (3, 3))

    def test_is_dangerous_bomb(self):
        state = copy.deepcopy(self.state)

        # Bomb in same row, clear path, within distance
        self.assertTrue(state._is_dangerous_bomb((1, 1), (1, 4)))

        # Bomb in same column, clear path, within distance
        self.assertTrue(state._is_dangerous_bomb((1, 1), (4, 1)))

        # Bomb in same row, more than 3 cells away
        self.assertFalse(state._is_dangerous_bomb((1, 1), (1, 5)))

        # Bomb in same row, path blocked
        self.assertFalse(state._is_dangerous_bomb((1, 2), (3, 2)))

        # Bomb not in same row or column
        self.assertFalse(state._is_dangerous_bomb((1, 1), (3, 3)))

        # Bomb goes through crates
        state.field[1, 2] = CRATE
        self.assertTrue(state._is_dangerous_bomb((1, 1), (1, 3)))

    def test_is_wall_free_path(self):
        state = copy.deepcopy(self.state)

        self.assertTrue(state._is_wall_free_path((1, 1), (1, 3)))
        self.assertTrue(state._is_wall_free_path((1, 1), (3, 1)))
        self.assertFalse(state._is_wall_free_path((1, 1), (2, 3)))
        # Same position as bomb
        self.assertTrue(state._is_wall_free_path((1, 1), (1, 1)))
        # Blocked by wall vertically
        self.assertFalse(state._is_wall_free_path((1, 2), (3, 2)))
        # Blocked by wall horizontally
        self.assertFalse(state._is_wall_free_path((2, 1), (2, 3)))


class TestingUtils(unittest.TestCase):

    def test_march_forward(self):
        self.assertEqual(march_forward((0, 0), 'LEFT'), (-1, 0))
        self.assertEqual(march_forward((0, 0), 'RIGHT'), (1, 0))
        self.assertEqual(march_forward((0, 0), 'UP'), (0, -1))
        self.assertEqual(march_forward((0, 0), 'DOWN'), (0, 1))
        self.assertEqual(march_forward((0, 0), 'WAIT'), (0, 0))

    def test_manhattan_distance(self):
        self.assertEqual(manhatten_distance((0, 0), (0, 0)), 0)
        self.assertEqual(manhatten_distance((0, 0), (5, 0)), 5)
        self.assertEqual(manhatten_distance((0, 0), (0, 5)), 5)
        self.assertEqual(manhatten_distance((3, 4), (7, 8)), 8)
        self.assertEqual(manhatten_distance((-1, -1), (1, 1)), 4)

    def test_increased_distance(self):
        self.assertFalse(increased_distance((0, 1), (0, 0), (0, 0)))
        self.assertTrue(increased_distance((0, 1), (0, 2), (0, 0)))
        self.assertTrue(increased_distance((0, 1), (1, 1), (0, 0)))

    def test_decreased_distance(self):
        self.assertTrue(decreased_distance((0, 1), (0, 0), (0, 0)))
        self.assertFalse(decreased_distance((0, 1), (0, 2), (0, 0)))
        self.assertFalse(decreased_distance((0, 1), (1, 1), (0, 0)))

    def test_got_in_loop(self):
        agent_coord_history = deque(
            [(0, 0), (0, 1), (0, 0), (0, 1), (0, 0), (1, 0)], 6)
        self.assertTrue(got_in_loop((1, 0), "LEFT", agent_coord_history))
        self.assertFalse(got_in_loop((0, 0), "RIGHT", agent_coord_history))
        self.assertFalse(got_in_loop((0, 0), "DOWN", agent_coord_history))

    def test_has_destroyed_target(self):
        events = ['KILLED_OPPONENT']
        self.assertTrue(has_destroyed_target(events))

        events = ['CRATE_DESTROYED']
        self.assertTrue(has_destroyed_target(events))


    def test_is_in_game_grid(self):
        self.assertTrue(is_in_game_grid((1, 1)))
        self.assertFalse(is_in_game_grid((17, 17)))
        self.assertFalse(is_in_game_grid((-1, 1)))


if __name__ == '__main__':
    unittest.main()
