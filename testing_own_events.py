"""
Keep in mind that game state is turned making it quite complicated
Movement from game_state to GUI
Down -> Right
Up -> Left
Left -> Up
Right -> Down
"""

import unittest
import numpy as np
from numpy.testing import assert_array_equal

from agent_code.own_events import *


class TestOwnEvents(unittest.TestCase):
    def setUp(self):
        # Common field setup for the tests
        self.field_small = np.array([
            [0, 0, 0],
            [0, -1, 0],
            [0, 0, 0]
        ])
        self.field = np.array([
            [0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0],
            [0, 0, 0, -1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        self.game_field = np.array([
            [-1, -1, -1, -1, -1, -1, -1],
            [-1,  0,  0,  0,  0,  0, -1],
            [-1,  0, -1,  0,  0,  0, -1],
            [-1,  0,  0,  0, -1,  0, -1],
            [-1,  0,  0,  0,  0,  0, -1],
            [-1,  0,  0,  0,  1,  1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
        ])
        self.opponents = [
            ('opponent1', 4, 0, (5, 5)),
            ('opponent2', 3, 1, (1, 1)),
            ('opponent3', 0, 0, (3, 3))
        ]
        self.explosion_map = np.zeros((7, 7))

    def test_march_forward(self):
        self.assertEqual(march_forward(0, 0, 'LEFT'), (-1, 0))
        self.assertEqual(march_forward(0, 0, 'RIGHT'), (1, 0))
        self.assertEqual(march_forward(0, 0, 'UP'), (0, -1))
        self.assertEqual(march_forward(0, 0, 'DOWN'), (0, 1))
        self.assertEqual(march_forward(0, 0, 'WAIT'), (0, 0))

    def test_is_in_explosion(self):
        explosion_map = np.array([[0, 0],
                                  [2, 1]])
        self.assertTrue(is_in_explosion(1, 1, explosion_map))
        self.assertTrue(is_in_explosion(1, 0, explosion_map))
        self.assertFalse(is_in_explosion(0, 0, explosion_map))

    def test_is_clear_path(self):
        self.assertTrue(is_clear_path(0, 0, (0, 2), self.field_small))
        self.assertTrue(is_clear_path(0, 0, (2, 0), self.field_small))
        self.assertFalse(is_clear_path(0, 0, (2, 1), self.field_small))
        # Same position as bomb
        self.assertTrue(is_clear_path(0, 0, (0, 0), self.field_small))
        # Blocked by wall vertically
        self.assertFalse(is_clear_path(0, 1, (2, 1), self.field_small))
        # Blocked by wall horizontally
        self.assertFalse(is_clear_path(1, 0, (1, 2), self.field_small))

    def test_is_dangerous_bomb(self):
        # Bomb in same row, clear path, within distance
        self.assertTrue(is_dangerous_bomb(0, 1, (0, 3), self.field))
        # Bomb in same column, clear path, within distance
        self.assertTrue(is_dangerous_bomb(1, 2, (3, 2), self.field))
        # Bomb in same row, exactly 3 cells away
        self.assertTrue(is_dangerous_bomb(0, 0, (0, 3), self.field))
        # Bomb in same row, more than 3 cells away
        self.assertFalse(is_dangerous_bomb(0, 0, (0, 4), self.field))
        # Bomb in same row, path blocked
        self.assertFalse(is_dangerous_bomb(1, 0, (1, 2), self.field))
        # Bomb not in same row or column
        self.assertFalse(is_dangerous_bomb(2, 2, (4, 4), self.field))
        # Bomb goes through crates
        self.assertTrue(is_dangerous_bomb(3, 1, (3, 3), self.field))

    def test_filter_dangerous_bombs(self):
        bombs = [(0, 3), (1, 0), (3, 2), (4, 4)]
        # One bomb in the same row and two in the same column, only two are dangerous
        dangerous_bombs = filter_dangerous_bombs(0, 0, bombs, self.field)
        self.assertEqual(len(dangerous_bombs), 2)
        self.assertTrue((0, 3) in dangerous_bombs)
        self.assertTrue((1, 0) in dangerous_bombs)

        # No bombs are dangerous
        bombs = [(4, 4), (3, 0)]
        dangerous_bombs = filter_dangerous_bombs(0, 1, bombs, self.field)
        self.assertEqual(len(dangerous_bombs), 0)

        # All bombs are dangerous
        bombs = [(0, 0), (0, 1), (0, 2)]
        dangerous_bombs = filter_dangerous_bombs(0, 0, bombs, self.field)
        self.assertEqual(len(dangerous_bombs), 3)

    def test_manhattan_distance(self):
        self.assertEqual(manhatten_distance(0, 0, 0, 0), 0)
        self.assertEqual(manhatten_distance(0, 0, 5, 0), 5)
        self.assertEqual(manhatten_distance(0, 0, 0, 5), 5)
        self.assertEqual(manhatten_distance(3, 4, 7, 8), 8)
        self.assertEqual(manhatten_distance(-1, -1, 1, 1), 4)

    def test_sort_objects_by_distance(self):
        objects = [(5, 5), (1, 1), (3, 3), (2, 2), (4, 4)]
        x_agent, y_agent = 0, 0
        sorted_objects = sort_objects_by_distance(x_agent, y_agent, objects)
        self.assertEqual(sorted_objects, [
                         (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])

        objects_same_distance = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        sorted_objects = sort_objects_by_distance(0, 0, objects_same_distance)
        # Check if sorted by distance and keeps tuple order
        self.assertListEqual(
            sorted_objects, [(1, 0), (0, 1), (-1, 0), (0, -1)])

        self.assertEqual(sort_objects_by_distance(0, 0, []), None)
        self.assertEqual(sort_objects_by_distance(0, 0, [(2, 2)]), [(2, 2)])

    def test_sort_opponents_by_distance(self):

        x_agent, y_agent = 0, 0
        sorted_opponents = sort_opponents_by_distance(
            x_agent, y_agent, self.opponents)
        expected_order = [('opponent2', 3, 1, (1, 1)),
                          ('opponent3', 0, 0, (3, 3)),
                          ('opponent1', 4, 0, (5, 5))]
        self.assertEqual(sorted_opponents, expected_order)

        self.assertEqual(sort_opponents_by_distance(0, 0, []), None)

        single_opponent = [('opponent1', 4, 0, (2, 2))]
        self.assertEqual(sort_opponents_by_distance(
            0, 0, single_opponent), single_opponent)

    def test_filter_and_sort_bombs(self):
        bombs = [(0, 1), (1, 1), (2, 2), (0, 3), (0, 0)]
        sorted_dangerous_bombs = filter_and_sort_bombs(0, 0, bombs, self.field)
        expected_bombs = [(0, 0), (0, 1), (0, 3)]
        self.assertEqual(sorted_dangerous_bombs, expected_bombs)

        self.assertEqual(filter_and_sort_bombs(0, 0, [], self.field), None)
        self.assertEqual(filter_and_sort_bombs(
            3, 3, [((0, 0)), ((1, 1)), ((2, 2))], self.field), None)

    def test_is_in_danger(self):
        # Setup for the test
        # 1 represents danger at (1, 1)
        explosion_map = np.array([[0, 0], [0, 1]])
        dangerous_bombs = [(0, 1), (1, 0)]

        # Test when agent is on a dangerous position
        self.assertTrue(is_in_danger(1, 1, explosion_map, []))

        # Test when there are dangerous bombs listed
        self.assertTrue(is_in_danger(0, 0, explosion_map, dangerous_bombs))

        # Test when agent is not on a dangerous position and no bombs are listed
        self.assertFalse(is_in_danger(0, 0, explosion_map, []))

        # Edge case: testing boundary of the explosion_map
        self.assertFalse(is_in_danger(1, 0, explosion_map, []))

    def test_has_highest_score(self):
        self.assertTrue(has_highest_score(self.opponents, 5))
        self.assertFalse(has_highest_score(self.opponents, 2))
        self.assertTrue(has_highest_score([], 2))

    def test_has_won_the_game(self):
        # Win at the end of the round
        single_opponent = [('opponent2', 3, 1, (1, 1))]
        self.assertTrue(has_won_the_game(
            self.opponents, 5, [], MAX_STEPS))
        # Player got killed
        self.assertFalse(has_won_the_game(
            self.opponents, 5, ['GOT_KILLED'], 50))
        # Player killed themselves with more than one opponent alive
        self.assertFalse(has_won_the_game(
            self.opponents, 5, ['KILLED_SELF'], 50))
        # Player killed themselves with no opponents left
        self.assertTrue(has_won_the_game(
            single_opponent, 5, ['KILLED_SELF'], 50))
        # Check for invalid game state exception
        with self.assertRaises(ValueError):
            has_won_the_game(single_opponent, 2, [], 50)

    def test_not_escaping_danger(self):
        self.assertTrue(not_escaping_danger('WAIT'))
        self.assertTrue(not_escaping_danger('BOMB'))
        self.assertFalse(not_escaping_danger('LEFT'))

    def test_is_escaping_danger(self):
        sorted_dangerous_bombs = [(1, 3)]
        # Out of bomb reach
        self.assertTrue(is_escaping_danger(
            1, 1, 'RIGHT', self.game_field, sorted_dangerous_bombs))
        # Closer to Bomb
        self.assertFalse(is_escaping_danger(
            1, 1, 'DOWN', self.game_field, sorted_dangerous_bombs))
        # Further from bomb reach
        self.assertTrue(is_escaping_danger(
            1, 2, 'UP', self.game_field, sorted_dangerous_bombs))
        # Invalid action
        self.assertFalse(is_escaping_danger(
            1, 1, 'LEFT', self.game_field, sorted_dangerous_bombs))

    def test_has_escaped_danger(self):
        sorted_dangerous_bombs = [(1, 3)]

        # Escaped behind wall
        self.assertTrue(has_escaped_danger(1, 1, 'RIGHT', self.game_field,
                        sorted_dangerous_bombs, self.explosion_map))
        # Invalid action
        self.assertFalse(has_escaped_danger(1, 1, 'LEFT', self.game_field,
                                            sorted_dangerous_bombs, self.explosion_map))
        # Not Escaped
        self.assertFalse(has_escaped_danger(
            1, 1, 'WAIT', self.game_field, sorted_dangerous_bombs, self.explosion_map))

        # Run into another danger
        self.explosion_map[2][1] = 1
        self.assertFalse(has_escaped_danger(
            1, 1, 'RIGHT', self.game_field, sorted_dangerous_bombs, self.explosion_map))
        self.explosion_map[2][1] = 0
        # Ran out of Reach
        self.assertTrue(has_escaped_danger(
            1, 2, 'UP', self.game_field, [(1, 5)], self.explosion_map))

    def test_is_valid_action(self):
        self.assertTrue(is_valid_action(1, 1, self.game_field))
        self.assertFalse(is_valid_action(0, 0, self.game_field))
        self.assertFalse(is_valid_action(5, 5, self.game_field))

    def test_is_save_step(self):
        sorted_dangerous_bombs = [(1, 3)]
        self.explosion_map[2][1] = 1

        # In reach of Bomb
        self.assertFalse(is_save_step(
            1, 1, self.game_field, self.explosion_map, sorted_dangerous_bombs))

        # On Explosion
        self.assertFalse(is_save_step(
            2, 1, self.game_field, self.explosion_map, sorted_dangerous_bombs))

        # Save Step
        self.assertTrue(is_save_step(
            5, 1, self.game_field, self.explosion_map, sorted_dangerous_bombs))

        # Invalid step
        self.assertFalse(is_save_step(
            0, 0, self.game_field, self.explosion_map, sorted_dangerous_bombs))

    def test_increased_distance(self):
        self.assertFalse(increased_distance(0, 1, 0, 0, 0, 0))
        self.assertTrue(increased_distance(0, 1, 0, 2, 0, 0))
        self.assertTrue(increased_distance(0, 1, 1, 1, 0, 0))

    def test_decreased_distance(self):
        self.assertTrue(decreased_distance(0, 1, 0, 0, 0, 0))
        self.assertFalse(decreased_distance(0, 1, 0, 2, 0, 0))
        self.assertFalse(decreased_distance(0, 1, 1, 1, 0, 0))

    def test_got_in_loop(self):
        agent_coord_history = deque(
            [(0, 0), (0, 1), (0, 0), (0, 1), (0, 0), (1, 0)], 6)
        self.assertTrue(got_in_loop(0, 0, agent_coord_history))
        self.assertFalse(got_in_loop(0, 1, agent_coord_history))
        self.assertFalse(got_in_loop(1, 0, agent_coord_history))

    def test_waited_necessary(self):
        field = np.array([
            [-1, -1, -1, -1, -1],
            [-1,  1,  0,  0, -1],
            [-1,  0, -1,  0, -1],
            [-1,  0,  0,  0, -1],
            [-1, -1, -1, -1, -1]
        ])
        explosion_map = np.array([
            [-1, -1, -1, -1, -1],
            [-1,  0,  0,  1, -1],
            [-1,  0,  0,  0, -1],
            [-1,  0,  0,  0, -1],
            [-1, -1, -1, -1, -1]
        ])
        bombs = [(3, 3)]
        # Waiting because of bomb
        self.assertTrue(waited_necessarily(2, 1, field, explosion_map, bombs))
        # Waiting because of explosion
        self.assertTrue(waited_necessarily(1, 2, field, explosion_map, bombs))
        # No need to Wait
        bombs = None
        self.assertFalse(waited_necessarily(2, 1, field, explosion_map, bombs))

    def test_find_closest_create(self):
        # Test Crate Directly Adjacent
        field = [
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ]
        self.assertEqual(find_closest_crate(1, 1, field), (1, 2))
        self.assertEqual(find_closest_crate(0, 0, field), (1, 2))

        # Test No Crates
        field[1][2] = 0
        self.assertIsNone(find_closest_crate(1, 1, field))

        # Test Multiple Crates Same Distance
        field_multiple = [
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1]
        ]
        self.assertEqual(find_closest_crate(1, 1, field_multiple), (1, 0))
        self.assertEqual(find_closest_crate(0, 2, field_multiple), (0, 0))

    def test_has_destroyed_target(self):
        events = ['BOMB_EXPLODED', 'KILLED_OPPONENT']
        self.assertTrue(has_destroyed_target(events))

        events = ['BOMB_EXPLODED', 'CRATE_DESTROYED']
        self.assertTrue(has_destroyed_target(events))

        events = ['CRATE_DESTROYED']
        self.assertFalse(has_destroyed_target(events))

    def test_is_in_game_grid(self):
        self.assertTrue(is_in_game_grid(0, 0, 3, 3))
        self.assertFalse(is_in_game_grid(3, 3, 3, 3))
        self.assertFalse(is_in_game_grid(-1, 0, 3, 3))

    def test_simulate_bomb_explosion(self):
        bomb_simulated_field = np.array([
            [-1, -1, -1, -1, -1, -1, -1],
            [-1,  0,  0,  0,  0,  0, -1],
            [-1,  0, -1,  2,  0,  0, -1],
            [-1,  0,  0,  2, -1,  0, -1],
            [-1,  0,  0,  2,  0,  0, -1],
            [-1,  2,  2,  2,  2,  2, -1],
            [-1, -1, -1, -1, -1, -1, -1],
        ])
        test_simulated_field, test_n_crates = simulate_bomb_explosion(
            5, 3, self.game_field)
        assert_array_equal(test_simulated_field, bomb_simulated_field)
        self.assertEqual(test_n_crates, 2)

        bomb_simulated_field = np.array([
            [-1, -1, -1, -1, -1, -1, -1],
            [-1,  0,  0,  2,  0,  0, -1],
            [-1,  0, -1,  2,  2,  2, -1],
            [-1,  0,  0,  2, -1,  0, -1],
            [-1,  0,  0,  2,  0,  0, -1],
            [-1,  0,  0,  2,  1,  1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
        ])
        test_simulated_field, test_n_crates = simulate_bomb_explosion(
            2, 3, self.game_field)
        assert_array_equal(test_simulated_field, bomb_simulated_field)
        self.assertEqual(test_n_crates, 0)


if __name__ == '__main__':
    unittest.main()
