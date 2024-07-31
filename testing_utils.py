"""
Keep in mind that game state is turned making it quite complicated
Movement from game_state to GUI
Gui[x,y] = game_state[y,x]

Down -> Right
Up -> Left
Left -> Up
Right -> Down

# TODO Update simualte_bomb and simulate bomb explosion -> get blast coords
# TODO Add is effective
"""

from agent_code.utils import *
import unittest
import numpy as np
from numpy.testing import assert_array_equal

EMPTY_FIELD = np.array([
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
    [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
    [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
    [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
    [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
    [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
    [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
    [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
    [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
    [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
    [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
    [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
    [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
    [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
    [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
])


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

    def test_waited_necessary(self):
        state = copy.deepcopy(self.state)
        state.bombs = [((2, 1), 2)]
        state.self = ('test_agent', 0, 1, (1, 2))
        state.explosion_map[1, 3] = 1

        # Waiting because of bomb and explosion
        self.assertTrue(state.waited_necessarily())
        state.explosion_map[1, 3] = 0

        # No need to Wait
        self.assertFalse(state.waited_necessarily())

        # Waiting because of only bombs
        state.field[1, 3] = 1
        self.assertTrue(state.waited_necessarily())

    def test_is_save_step(self):
        state = copy.deepcopy(self.state)
        state.bombs = [((1, 3), 1)]
        state.explosion_map[2][1] = 1

        # In reach of Bomb
        self.assertFalse(state.is_save_step((1, 1)))

        # On Explosion
        self.assertFalse(state.is_save_step((2, 1)))

        # Save Step
        self.assertTrue(state.is_save_step((5, 1)))

        # Invalid step
        self.assertFalse(state.is_save_step((0, 0)))

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
        self.assertEqual(action_idx, 4)

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


"""
class TestingUtils(unittest.TestCase):
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
        self.small_game_field = np.array([
            [-1, -1, -1, -1, -1],
            [-1,  1,  0,  0, -1],
            [-1,  0, -1,  0, -1],
            [-1,  0,  1,  0, -1],
            [-1, -1, -1, -1, -1]
        ])
        self.opponents = [
            ('opponent1', 4, 0, (5, 5)),
            ('opponent2', 3, 1, (1, 1)),
            ('opponent3', 0, 0, (3, 3))
        ]
        self.explosion_map = np.zeros((7, 7))

    def test_march_forward(self):
        self.assertEqual(march_forward((0, 0), 'LEFT'), (-1, 0))
        self.assertEqual(march_forward((0, 0), 'RIGHT'), (1, 0))
        self.assertEqual(march_forward((0, 0), 'UP'), (0, -1))
        self.assertEqual(march_forward((0, 0), 'DOWN'), (0, 1))
        self.assertEqual(march_forward((0, 0), 'WAIT'), (0, 0))

    def test_is_in_explosion(self):
        explosion_map = np.array([[0, 0],
                                  [2, 1]])
        self.assertTrue(is_in_explosion((1, 1), explosion_map))
        self.assertTrue(is_in_explosion((1, 0), explosion_map))
        self.assertFalse(is_in_explosion((0, 0), explosion_map))

    def test_is_wall_free_path(self):
        self.assertTrue(is_wall_free_path((0, 0), (0, 2), self.field_small))
        self.assertTrue(is_wall_free_path((0, 0), (2, 0), self.field_small))
        self.assertFalse(is_wall_free_path((0, 0), (2, 1), self.field_small))
        # Same position as bomb
        self.assertTrue(is_wall_free_path((0, 0), (0, 0), self.field_small))
        # Blocked by wall vertically
        self.assertFalse(is_wall_free_path((0, 1), (2, 1), self.field_small))
        # Blocked by wall horizontally
        self.assertFalse(is_wall_free_path((1, 0), (1, 2), self.field_small))

    def test_is_dangerous_bomb(self):
        # Bomb in same row, clear path, within distance
        self.assertTrue(is_dangerous_bomb((0, 1), (0, 3), self.field))
        # Bomb in same column, clear path, within distance
        self.assertTrue(is_dangerous_bomb((1, 2), (3, 2), self.field))
        # Bomb in same row, exactly 3 cells away
        self.assertTrue(is_dangerous_bomb((0, 0), (0, 3), self.field))
        # Bomb in same row, more than 3 cells away
        self.assertFalse(is_dangerous_bomb((0, 0), (0, 4), self.field))
        # Bomb in same row, path blocked
        self.assertFalse(is_dangerous_bomb((1, 0), (1, 2), self.field))
        # Bomb not in same row or column
        self.assertFalse(is_dangerous_bomb((2, 2), (4, 4), self.field))
        # Bomb goes through crates
        self.assertTrue(is_dangerous_bomb((3, 1), (3, 3), self.field))

    def test_filter_dangerous_bombs(self):
        bombs = [(0, 3), (1, 0), (3, 2), (4, 4)]
        # One bomb in the same row and two in the same column, only two are dangerous
        dangerous_bombs = filter_dangerous_bombs((0, 0), bombs, self.field)
        self.assertEqual(len(dangerous_bombs), 2)
        self.assertTrue((0, 3) in dangerous_bombs)
        self.assertTrue((1, 0) in dangerous_bombs)

        # No bombs are dangerous
        bombs = [(4, 4), (3, 0)]
        dangerous_bombs = filter_dangerous_bombs((0, 1), bombs, self.field)
        self.assertEqual(len(dangerous_bombs), 0)

        # All bombs are dangerous
        bombs = [(0, 0), (0, 1), (0, 2)]
        dangerous_bombs = filter_dangerous_bombs((0, 0), bombs, self.field)
        self.assertEqual(len(dangerous_bombs), 3)

    def test_manhattan_distance(self):
        self.assertEqual(manhatten_distance((0, 0), (0, 0)), 0)
        self.assertEqual(manhatten_distance((0, 0), (5, 0)), 5)
        self.assertEqual(manhatten_distance((0, 0), (0, 5)), 5)
        self.assertEqual(manhatten_distance((3, 4), (7, 8)), 8)
        self.assertEqual(manhatten_distance((-1, -1), (1, 1)), 4)

    def test_sort_objects_by_distance(self):
        objects = [(5, 5), (1, 1), (3, 3), (2, 2), (4, 4)]
        agent_coords = (0, 0)
        sorted_objects = sort_objects_by_distance(agent_coords, objects)
        self.assertEqual(sorted_objects, [
                         (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])

        objects_same_distance = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        sorted_objects = sort_objects_by_distance(
            agent_coords, objects_same_distance)
        # Check if sorted by distance and keeps tuple order
        self.assertListEqual(
            sorted_objects, [(1, 0), (0, 1), (-1, 0), (0, -1)])

        self.assertEqual(sort_objects_by_distance(agent_coords, []), [])
        self.assertEqual(sort_objects_by_distance(
            agent_coords, [(2, 2)]), [(2, 2)])

    def test_sort_opponents_by_distance(self):

        agent_coords = (0, 0)
        sorted_opponents = sort_opponents_by_distance(
            agent_coords, self.opponents)
        expected_order = [('opponent2', 3, 1, (1, 1)),
                          ('opponent3', 0, 0, (3, 3)),
                          ('opponent1', 4, 0, (5, 5))]
        self.assertEqual(sorted_opponents, expected_order)

        self.assertEqual(sort_opponents_by_distance(agent_coords, []), [])

        single_opponent = [('opponent1', 4, 0, (2, 2))]
        self.assertEqual(sort_opponents_by_distance(
            agent_coords, single_opponent), single_opponent)

    def test_sort_and_filter_out_dangerous_bombs(self):
        bombs = [(0, 1), (1, 1), (2, 2), (0, 3), (0, 0)]
        sorted_dangerous_bombs = sort_and_filter_out_dangerous_bombs(
            (0, 0), bombs, self.field)
        expected_bombs = [(0, 0), (0, 1), (0, 3)]
        self.assertEqual(sorted_dangerous_bombs, expected_bombs)

        self.assertEqual(sort_and_filter_out_dangerous_bombs(
            (0, 0), [], self.field), [])
        self.assertEqual(sort_and_filter_out_dangerous_bombs(
            (3, 3), [((0, 0)), ((1, 1)), ((2, 2))], self.field), [])

    

    def test_has_highest_score(self):
        self.assertTrue(has_highest_score(self.opponents, 5))
        self.assertFalse(has_highest_score(self.opponents, 2))
        self.assertTrue(has_highest_score([], 2))

    def test_has_won_the_game(self):
        # Win at the end of the round
        single_opponent = [('opponent2', 3, 1, (1, 1))]
        self.assertTrue(has_won_the_round(
            self.opponents, 5, [], MAX_STEPS))
        # Player got killed
        self.assertFalse(has_won_the_round(
            self.opponents, 5, ['GOT_KILLED'], 50))
        # Player killed themselves with more than one opponent alive
        self.assertFalse(has_won_the_round(
            self.opponents, 5, ['KILLED_SELF'], 50))
        # Player killed themselves with no opponents left
        self.assertTrue(has_won_the_round(
            single_opponent, 5, ['KILLED_SELF'], 50))
        # Check for invalid game state exception
        with self.assertRaises(ValueError):
            has_won_the_round(single_opponent, 2, [], 50)

    

    

    

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
        self.assertTrue(got_in_loop((0, 0), agent_coord_history))
        self.assertFalse(got_in_loop((0, 1), agent_coord_history))
        self.assertFalse(got_in_loop((1, 0), agent_coord_history))

    

    def test_find_closest_create(self):
        # Test Crate Directly Adjacent
        field = np.array([
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        self.assertEqual(find_closest_crate((1, 1), field), (1, 2))
        self.assertEqual(find_closest_crate((0, 0), field), (1, 2))

        # Test No Crates
        field[1][2] = 0
        self.assertIsNone(find_closest_crate((1, 1), field))

        # Test Multiple Crates Same Distance
        field_multiple = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        self.assertEqual(find_closest_crate((1, 1), field_multiple), (1, 0))
        self.assertEqual(find_closest_crate((0, 2), field_multiple), (0, 0))

    def test_has_destroyed_target(self):
        events = ['BOMB_EXPLODED', 'KILLED_OPPONENT']
        self.assertTrue(has_destroyed_target(events))

        events = ['BOMB_EXPLODED', 'CRATE_DESTROYED']
        self.assertTrue(has_destroyed_target(events))

        events = ['CRATE_DESTROYED']
        self.assertFalse(has_destroyed_target(events))

    def test_is_in_game_grid(self):
        self.assertTrue(is_in_game_grid((0, 0), 3, 3))
        self.assertFalse(is_in_game_grid((3, 3), 3, 3))
        self.assertFalse(is_in_game_grid((-1, 0), 3, 3))

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
            (5, 3), self.game_field)
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
            (2, 3), self.game_field)
        assert_array_equal(test_simulated_field, bomb_simulated_field)
        self.assertEqual(test_n_crates, 0)

    def test_path_to_safety_exists(self):
        bombs = []
        opponents = []
        bomb_simulated_field = np.array([
            [-1, -1, -1, -1, -1],
            [-1,  1,  0,  2, -1],
            [-1,  0, -1,  2, -1],
            [-1,  2,  2,  2, -1],
            [-1, -1, -1, -1, -1]
        ])
        self.assertTrue(path_to_safety_exists(
            (3, 3), bomb_simulated_field, self.small_game_field, opponents, bombs))
        # bomb and opponent in the way
        bombs = [(3, 1)]
        opponents = [(1, 3)]
        self.assertFalse(path_to_safety_exists(
            (3, 3), bomb_simulated_field, self.small_game_field, opponents, bombs))

        bombs = []
        opponents = []
        bomb_simulated_field_no_path = np.array([
            [-1, -1, -1, -1, -1],
            [-1,  2,  0,  0, -1],
            [-1,  2, -1,  0, -1],
            [-1,  2,  2,  2, -1],
            [-1, -1, -1, -1, -1]
        ])
        self.assertFalse(path_to_safety_exists(
            (3, 1), bomb_simulated_field_no_path, self.small_game_field, opponents, bombs))

    def test_potentially_destroying_opponent(self):
        opponents = [('opponent1', 3, 1, (3, 1)),
                     ('opponent2', 3, 1, (1, 2))]
        bomb_simulated_field = np.array([
            [-1, -1, -1, -1, -1],
            [-1,  1,  0,  2, -1],
            [-1,  0, -1,  2, -1],
            [-1,  2,  2,  2, -1],
            [-1, -1, -1, -1, -1]
        ])
        # Destroyes one opponent trough crate
        self.assertTrue(potentially_destroying_opponent(
            bomb_simulated_field, opponents))

        # Destroyes no opponent
        opponents = [('opponent1', 3, 1, (2, 1)),
                     ('opponent2', 3, 1, (1, 2))]
        self.assertFalse(potentially_destroying_opponent(
            bomb_simulated_field, opponents))

        # Destroyes two opponent
        opponents = [('opponent1', 3, 1, (3, 1)),
                     ('opponent2', 3, 1, (3, 3))]
        self.assertTrue(potentially_destroying_opponent(
            bomb_simulated_field, opponents))

        # No opponent present
        opponents = []
        self.assertFalse(potentially_destroying_opponent(
            bomb_simulated_field, opponents))

    
"""

if __name__ == '__main__':
    unittest.main()
