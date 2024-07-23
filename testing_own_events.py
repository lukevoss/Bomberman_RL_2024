import unittest
import numpy as np

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
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])

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
        opponents = [
            ('opponent1', 4, 0, (5, 5)),
            ('opponent2', 3, 1, (1, 1)),
            ('opponent3', 0, 0, (3, 3))
        ]
        x_agent, y_agent = 0, 0
        sorted_opponents = sort_opponents_by_distance(
            x_agent, y_agent, opponents)
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
        sorted_dangerous_bombs = filter_and_sort_bombs(0, 0, bombs)
        expected_bombs = [(0, 0), (0, 1), (0, 3)]
        self.assertEqual(sorted_dangerous_bombs, expected_bombs)

        self.assertEqual(filter_and_sort_bombs(0, 0, []), None)
        self.assertEqual(filter_and_sort_bombs(
            3, 3, [((0, 0)), ((1, 1)), ((2, 2))]), None)


if __name__ == '__main__':
    unittest.main()
