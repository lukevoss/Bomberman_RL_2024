from typing import List, Tuple, Any, Dict

import events as e
from settings import MAX_STEPS


# Dense Rewards
CONSTANT_PENALTY = "Constant Penalty"
WON_GAME = "Won the game"
BOMBED_1_TO_2_CRATES = "BOMBED_1_TO_2_CRATES"
BOMBED_3_TO_5_CRATES = "BOMBED_3_TO_5_CRATES"
BOMBED_5_PLUS_CRATES = "BOMBED_5_PLUS_CRATES"
GET_IN_LOOP = "GET_IN_LOOP"
PLACEHOLDER_EVENT = "PLACEHOLDER"
ESCAPE = "ESCAPE"
NOT_ESCAPE = "NOT_ESCAPE"
CLOSER_TO_COIN = "CLOSER_TO_COIN"
AWAY_FROM_COIN = "AWAY_FROM_COIN"
CLOSER_TO_CRATE = "CLOSER_TO_CRATE"
AWAY_FROM_CRATE = "AWAY_FROM_CRATE"
SURVIVED_STEP = "SURVIVED_STEP"
DESTROY_TARGET = "DESTROY_TARGET"
MISSED_TARGET = "MISSED_TARGET"
WAITED_NECESSARILY = "WAITED_NECESSARILY"
WAITED_UNNECESSARILY = "WAITED_UNNECESSARILY"
CLOSER_TO_PLAYERS = "CLOSER_TO_PLAYERS"
AWAY_FROM_PLAYERS = "AWAY_FROM_PLAYERS"

############# event shaping #####################


# def check_position(x, y, arena, object):
#     # Check whether the current position is of object type.
#     if object == 'crate':
#         return arena[x, y] == 1
#     elif object == 'free':
#         return arena[x, y] == 0
#     elif object == 'wall':
#         return arena[x, y] == -1


def march_forward(x, y, direction):
    # Forward in direction.
    if direction == 'LEFT':
        x -= 1
    elif direction == 'RIGHT':
        x += 1
    elif direction == 'UP':
        y -= 1
    elif direction == 'DOWN':
        y += 1
    return x, y


def is_in_explosion(explosion_map, x_agent, y_agent):
    return explosion_map[x_agent, y_agent] != 0


def is_clear_path(x_agent: int, y_agent: int, bomb: Tuple[int, int], field: List[List[int]]) -> bool:
    if x_agent == bomb[0]:  # Same column
        step = 1 if y_agent < bomb[1] else -1
        for y in range(y_agent + step, bomb[1], step):
            if field[x_agent][y] == -1:
                return False
    elif y_agent == bomb[1]:  # Same row
        step = 1 if x_agent < bomb[0] else -1
        for x in range(x_agent + step, bomb[0], step):
            if field[x][y_agent] == -1:
                return False
    return True


def is_dangerous_bomb(x_agent: int, y_agent: int, bomb: Tuple[int, int], field: List[List[int]]) -> bool:
    """Check if a bomb is dangerous and has a clear path to the agent."""
    return (bomb[0] == x_agent or bomb[1] == y_agent) and abs(bomb[0] - x_agent) + abs(bomb[1] - y_agent) <= 3 and is_clear_path(x_agent, y_agent, bomb, field)


def filter_dangerous_bombs(x_agent: int, y_agent: int, bombs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Filters bombs that are in the same row or column as the agent and within a distance of 3."""
    return [
        bomb for bomb in bombs
        if is_dangerous_bomb(x_agent, y_agent, bomb)
    ]


def sort_bombs_by_distance(x_agent: int, y_agent: int, bombs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sorts bombs by Manhattan distance to the agent."""
    return sorted(bombs, key=lambda bomb: abs(bomb[0] - x_agent) + abs(bomb[1] - y_agent))


def filter_and_sort_bombs(x_agent, y_agent, bombs):
    """
    Filters and sorts bombs by their Manhattan distance to the agent's position, considering only those
    bombs that are either in the same row (y coordinate) or the same column (x coordinate) as the agent and having a distance of 3 or less, thus being a potential danger.
    """
    dangerous_bombs = filter_dangerous_bombs(x_agent, y_agent, bombs)

    if dangerous_bombs:
        dangerous_bombs = sort_bombs_by_distance(
            x_agent, y_agent, dangerous_bombs)

    return dangerous_bombs


def is_in_danger(x_agent, y_agent, explosion_map, sorted_dangerous_bombs):
    """
    Function checks if given agent position is dangerous
    """
    if is_in_explosion(explosion_map, x_agent, y_agent):
        return True
    if sorted_dangerous_bombs:
        return True

    return False


def has_highest_score(living_opponents, score_self):
    if living_opponents:
        score_opponents = [opponent[1] for opponent in living_opponents]
        if all(score_self > score for score in score_opponents):
            return True
        else:
            return False
    else:
        # TODO: How to see scores of killed opponents
        return True


def has_won_the_game(living_opponents, score_self, events, steps_of_round):
    # TODO: Can killed agent still win?
    # TODO: How can we access killed opponents for score

    if steps_of_round == MAX_STEPS:
        return has_highest_score(living_opponents, score_self)
    elif e.GOT_KILLED in events:
        return False
    elif e.KILLED_SELF in events:
        if len(living_opponents) > 1:
            return False
        else:
            return has_highest_score(living_opponents, score_self)
    else:
        raise ValueError("Invalid game state or undefined events")


def not_escaping_danger(self_action):
    return self_action == 'WAIT' or self_action == 'BOMB'


def is_escaping_danger(x_agent, y_agent, self_action, field, sorted_dangerous_bombs):
    x_new, y_new = march_forward(x_agent, y_agent, self_action)
    if valid_action(x_new, y_new, field):
        if sorted_dangerous_bombs:
            closest_bomb = sorted_dangerous_bombs[0]
            if distance_increased(closest_bomb[0], closest_bomb[1], x_agent, y_agent, x_new, y_new):
                return True
            else:
                return False
    else:
        return False


def valid_action(x_new, y_new, field):
    return field[x_new, y_new] == 0


def distance_increased(x_bomb, y_bomb, x_agent, y_agent, x_new, y_new):
    return ((abs(x_bomb - x_new) > abs(x_bomb - x_agent)) or
            ((y_bomb - y_new) > abs(y_bomb - y_agent)))


def add_own_events(self, old_game_state, self_action, events_src, end_of_round) -> list:

    # events = copy.deepcopy(events_src)
    events = events_src.copy()
    events.append(CONSTANT_PENALTY)

    field = old_game_state['field']
    x_agent, y_agent = old_game_state['self'][3]
    bombs = [xy for (xy, t) in old_game_state['bombs']]
    sorted_dangerous_bombs = filter_and_sort_bombs(x_agent, y_agent, bombs)
    living_opponents = old_game_state['others']
    coins = old_game_state['coins']
    explosion_map = old_game_state['explosion_map']
    score_self = old_game_state['self'][1]
    steps_of_round = old_game_state['steps']

    if end_of_round:
        if has_won_the_game(living_opponents, score_self, events, steps_of_round):
            events.append(WON_GAME)

    if is_in_danger(x_agent, y_agent, explosion_map, sorted_dangerous_bombs):
        if not_escaping_danger(self_action):
            events.append(NOT_ESCAPE)
        elif is_escaping_danger(x_agent, y_agent, self_action, field, sorted_dangerous_bombs):
            events.append(ESCAPE)
        else:
            events.append(NOT_ESCAPE)

    else:
        # Check if in loop
        self.loop_count = self.agent_coord_history.count((x_agent, y_agent))

        # If the agent gets caught in a loop, he will be punished.
        if self.loop_count > 2:
            events.append(GET_IN_LOOP)

        if self_action == 'WAIT':
            # Reward the agent if waiting is necessary.
            if (is_in_danger(x_agent+1, y_agent, field, explosion_map, bombs) or
                is_in_danger(x_agent-1, y_agent, field, explosion_map, bombs) or
                is_in_danger(x_agent, y_agent+1, field, explosion_map, bombs) or
                    is_in_danger(x_agent, y_agent-1, field, explosion_map, bombs)):
                events.append(WAITED_NECESSARILY)
            else:
                events.append(WAITED_UNNECESSARILY)

        elif self_action != 'BOMB':
            x_new, y_new = march_forward(x_agent, y_agent, self_action)
            for opponent in living_opponents:
                x_opponent, y_opponent = opponent[3]

                # Check if distance decreased
                if ((abs(x_opponent - x_new) > abs(x_opponent - x_agent)) or
                        ((y_opponent - y_new) > abs(y_opponent - y_agent))):
                    events.append(CLOSER_TO_PLAYERS)
                else:
                    events.append(AWAY_FROM_PLAYERS)

            for x_coin, y_coin in coins:
                # Check if distance decreased
                if ((abs(x_coin - x_new) > abs(x_coin - x_agent)) or
                        ((y_coin - y_new) > abs(y_coin - y_agent))):
                    events.append(CLOSER_TO_COIN)
                else:
                    events.append(AWAY_FROM_COIN)

    return events


def reward_from_events(self, events: List[str]) -> int:
    """
    Calculate the Rewards sum from all current events in the game

        Parameter:
            events (list[str]) = List of occured events

        Return:
            reward_sum [float] = Sum of all reward for occured events

    Author: Luke Voss
    """
    # Base rewards:

    aggressive_action = 0.3
    coin_action = 0.2
    escape = 0.6
    waiting = 0.5

    game_rewards = {
        # SPECIAL EVENTS
        ESCAPE: escape,
        NOT_ESCAPE: -escape,
        WAITED_NECESSARILY: waiting,
        WAITED_UNNECESSARILY: -waiting,
        CLOSER_TO_PLAYERS: aggressive_action,
        AWAY_FROM_PLAYERS: -aggressive_action,
        CLOSER_TO_COIN: coin_action,
        AWAY_FROM_COIN: -coin_action,
        CONSTANT_PENALTY: -0.001,
        WON_GAME: 10,
        GET_IN_LOOP: -0.025 * self.loop_count,

        # DEFAULT EVENTS
        e.INVALID_ACTION: -1,

        # bombing
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,

        # crates, coins
        e.CRATE_DESTROYED: 0.4,
        e.COIN_FOUND: 0,
        e.COIN_COLLECTED: 2,

        # kills
        e.KILLED_OPPONENT: 5,
        # TODO: make killed self positiv since its better to kill himself, than to get killed
        e.KILLED_SELF: -10,
        e.GOT_KILLED: -10,
        e.OPPONENT_ELIMINATED: 0,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
