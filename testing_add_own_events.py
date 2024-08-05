import unittest
import copy
import numpy as np

import own_events as own_e
from agent_code.utils import *
from agent_code.add_own_events import add_own_events

#TODO test got into Danger

class TestingAddOwnEvents(unittest.TestCase):
    def setUp(self):
        self.old_game_state = {
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
        self.events_src = []
        self.end_of_round = False
        self.agent_coord_history = [(1, 1)]
        self.max_opponents_score = 0

    def test_survived_step_event(self):
        result = add_own_events(self.old_game_state, 'UP', self.events_src,
                                self.end_of_round, self.agent_coord_history, self.max_opponents_score)
        self.assertIn(own_e.SURVIVED_STEP, result)

    def test_won_round(self):
        state = copy.deepcopy(self.old_game_state)
        end_of_round = True
        state['self'] = ('test_agent', 10, 1, (1, 1))
        max_opponents_score = 5

        # Won game
        result = add_own_events(state, 'UP', self.events_src,
                                end_of_round, self.agent_coord_history, max_opponents_score)
        self.assertIn(own_e.WON_ROUND, result)

        # Lost
        self.max_opponents_score = 11
        result = add_own_events(state, 'UP', self.events_src,
                                self.end_of_round, self.agent_coord_history, self.max_opponents_score)
        self.assertNotIn(own_e.WON_ROUND, result)

    def test_escaping(self):
        state = copy.deepcopy(self.old_game_state)

        # Is escaping
        state["bombs"] = [((1, 1), 3)]
        result = add_own_events(state, 'DOWN', self.events_src,
                                self.end_of_round, self.agent_coord_history, self.max_opponents_score)
        self.assertIn(own_e.ESCAPING, result)

        # Is not escaping
        state["bombs"] = [((1, 3), 3)]
        result = add_own_events(state, 'DOWN', self.events_src,
                                self.end_of_round, self.agent_coord_history, self.max_opponents_score)
        self.assertIn(own_e.NOT_ESCAPING, result)

        # Has ecaped
        result = add_own_events(state, 'RIGHT', self.events_src,
                                self.end_of_round, self.agent_coord_history, self.max_opponents_score)
        self.assertIn(own_e.OUT_OF_DANGER, result)

    def test_waited_necessarily(self):
        state = copy.deepcopy(self.old_game_state)
        state['explosion_map'][2, 1] = 1
        state['explosion_map'][1, 2] = 1
        result = add_own_events(state, 'WAIT', self.events_src,
                                self.end_of_round, self.agent_coord_history, self.max_opponents_score)
        self.assertIn(own_e.WAITED_NECESSARILY, result)

    def test_bomb_dropped(self):
        state = copy.deepcopy(self.old_game_state)

        # Dumb bomb dropped
        state['field'][2, 1] = CRATE
        state['field'][1, 2] = CRATE
        result = add_own_events(state, 'BOMB', self.events_src,
                                self.end_of_round, self.agent_coord_history, self.max_opponents_score)
        self.assertIn(own_e.DUMB_BOMB_DROPPED, result)

        # Smart bomb dropped
        state['field'][1, 2] = FREE
        result = add_own_events(state, 'BOMB', self.events_src,
                                self.end_of_round, self.agent_coord_history, self.max_opponents_score)
        self.assertIn(own_e.SMART_BOMB_DROPPED, result)

    def test_got_in_loop(self):
        state = copy.deepcopy(self.old_game_state)
        state["self"] = ('test_agent', 10, 1, (2, 1))
        agent_coord_history = deque(
            [(1, 1), (1, 2), (1, 1), (1, 2), (1, 1), (2, 1)], 6)
        result = add_own_events(state, 'LEFT', self.events_src,
                                self.end_of_round, agent_coord_history, self.max_opponents_score)
        self.assertIn(own_e.GOT_IN_LOOP, result)

    def test_closer_to_coin(self):
        state = copy.deepcopy(self.old_game_state)
        state['coins'] = [(2, 1)]

        # Closer to Coin
        result = add_own_events(state, 'RIGHT', self.events_src,
                                self.end_of_round, self.agent_coord_history, self.max_opponents_score)
        self.assertIn(own_e.CLOSER_TO_COIN, result)
        self.assertNotIn(own_e.AWAY_FROM_COIN, result)

        # Away from Coin
        result = add_own_events(state, 'DOWN', self.events_src,
                                self.end_of_round, self.agent_coord_history, self.max_opponents_score)
        self.assertIn(own_e.AWAY_FROM_COIN, result)
        self.assertNotIn(own_e.CLOSER_TO_COIN, result)

    def test_closer_to_opponent(self):
        state = copy.deepcopy(self.old_game_state)
        state['others'] = [('opponent', 10, 1, (2, 1))]

        # Closer to Opponent
        result = add_own_events(state, 'RIGHT', self.events_src,
                                self.end_of_round, self.agent_coord_history, self.max_opponents_score)
        self.assertIn(own_e.CLOSER_TO_PLAYERS, result)
        self.assertNotIn(own_e.AWAY_FROM_PLAYERS, result)

        # Away from Opponent
        result = add_own_events(state, 'DOWN', self.events_src,
                                self.end_of_round, self.agent_coord_history, self.max_opponents_score)
        self.assertIn(own_e.AWAY_FROM_PLAYERS, result)
        self.assertNotIn(own_e.CLOSER_TO_PLAYERS, result)

    def test_closer_to_crate(self):
        state = copy.deepcopy(self.old_game_state)
        state['field'][3, 1] = CRATE

        # Closer to Crate
        result = add_own_events(state, 'RIGHT', self.events_src,
                                self.end_of_round, self.agent_coord_history, self.max_opponents_score)
        self.assertIn(own_e.CLOSER_TO_CRATE, result)
        self.assertNotIn(own_e.AWAY_FROM_CRATE, result)

        # Away from Crate
        result = add_own_events(state, 'DOWN', self.events_src,
                                self.end_of_round, self.agent_coord_history, self.max_opponents_score)
        self.assertIn(own_e.AWAY_FROM_CRATE, result)
        self.assertNotIn(own_e.CLOSER_TO_CRATE, result)

    def test_destroy_target(self):
        state = copy.deepcopy(self.old_game_state)

        # Destroyed target
        self.events_src = [e.CRATE_DESTROYED,
                           e.CRATE_DESTROYED, e.CRATE_DESTROYED, e.BOMB_EXPLODED]
        result = add_own_events(state, 'UP', self.events_src,
                                self.end_of_round, self.agent_coord_history, self.max_opponents_score)
        self.assertIn(own_e.DESTROY_TARGET, result)
        self.assertNotIn(own_e.MISSED_TARGET, result)

        # Missed Target
        self.events_src = [e.BOMB_EXPLODED]
        result = add_own_events(state, 'UP', self.events_src,
                                self.end_of_round, self.agent_coord_history, self.max_opponents_score)
        self.assertIn(own_e.MISSED_TARGET, result)
        self.assertNotIn(own_e.DESTROY_TARGET, result)

        # No Bomb exploded
        self.events_src = []
        result = add_own_events(state, 'UP', self.events_src,
                                self.end_of_round, self.agent_coord_history, self.max_opponents_score)
        self.assertNotIn(own_e.MISSED_TARGET, result)
        self.assertNotIn(own_e.DESTROY_TARGET, result)

    def test_bombed_1_to_2_crates(self):
        state = copy.deepcopy(self.old_game_state)
        self.events_src = [e.CRATE_DESTROYED, e.CRATE_DESTROYED]
        result = add_own_events(state, 'UP', self.events_src,
                                self.end_of_round, self.agent_coord_history, self.max_opponents_score)
        self.assertIn(own_e.BOMBED_1_TO_2_CRATES, result)

    def test_bombed_3_to_5_crates(self):
        state = copy.deepcopy(self.old_game_state)
        self.events_src = [e.CRATE_DESTROYED] * 4
        result = add_own_events(state, 'UP', self.events_src,
                                self.end_of_round, self.agent_coord_history, self.max_opponents_score)
        self.assertIn(own_e.BOMBED_3_TO_5_CRATES, result)

    def test_bombed_5_plus_crates(self):
        state = copy.deepcopy(self.old_game_state)
        self.events_src = [e.CRATE_DESTROYED] * 6
        result = add_own_events(state, 'UP', self.events_src,
                                self.end_of_round, self.agent_coord_history, self.max_opponents_score)
        self.assertIn(own_e.BOMBED_5_PLUS_CRATES, result)


if __name__ == '__main__':
    unittest.main()
