from typing import List, Tuple, Any, Dict
from collections import deque

import events as e
from settings import MAX_STEPS, BOMB_POWER, BOMB_TIMER


# Dense Rewards
CONSTANT_PENALTY = "CONSTANT_PENALTY"
WON_GAME = "WON_GAME"
BOMBED_1_TO_2_CRATES = "BOMBED_1_TO_2_CRATES"
BOMBED_3_TO_5_CRATES = "BOMBED_3_TO_5_CRATES"
BOMBED_5_PLUS_CRATES = "BOMBED_5_PLUS_CRATES"
GOT_IN_LOOP = "GOT_IN_LOOP"
ESCAPING = "ESCAPING"
OUT_OF_DANGER = "OUT_OF_DANGER"
NOT_ESCAPING = "NOT_ESCAPING"
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
SMART_BOMB_DROPPED = "SMART_BOMB_DROPPED"
DUMB_BOMB_DROPPED = "DUMB_BOMB_DROPPED"

############# event shaping #####################

DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
UNSAFE_FIELD = 2


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


def is_in_explosion(x_agent, y_agent, explosion_map) -> bool:
    return explosion_map[x_agent, y_agent] != 0


def is_clear_path(x_agent: int, y_agent: int, bomb: Tuple[int, int], field) -> bool:
    """
    Determines if there is a clear path (no obstacles) between an agent's position and a bomb's position
    on a given field. The field is represented as a 2D list where -1 indicates an obstacle.
    """
    def clear_path(start: int, end: int, fixed: int, is_row_fixed: bool) -> bool:
        """ Helper function to check for obstacles in a row or column. """
        step = 1 if start < end else -1
        for i in range(start + step, end, step):
            if (field[fixed][i] if is_row_fixed else field[i][fixed]) == -1:
                return False
        return True

    if x_agent == bomb[0]:  # Same column
        return clear_path(y_agent, bomb[1], x_agent, True)
    elif y_agent == bomb[1]:  # Same row
        return clear_path(x_agent, bomb[0], y_agent, False)


def is_dangerous_bomb(x_agent: int, y_agent: int, bomb: Tuple[int, int], field) -> bool:
    """Check if a bomb is dangerous and has a clear path to the agent."""
    return (bomb[0] == x_agent or bomb[1] == y_agent) and abs(bomb[0] - x_agent) + abs(bomb[1] - y_agent) <= 3 and is_clear_path(x_agent, y_agent, bomb, field)


def filter_dangerous_bombs(x_agent: int, y_agent: int, bombs: List[Tuple[int, int]], field) -> List[Tuple[int, int]]:
    """Filters bombs that are in the same row or column as the agent and within a distance of 3."""
    return [
        bomb for bomb in bombs
        if is_dangerous_bomb(x_agent, y_agent, bomb, field)
    ]


def manhatten_distance(x1, y1, x2, y2) -> int:
    return abs(x1 - x2) + abs(y1 - y2)


def sort_objects_by_distance(x_agent: int, y_agent: int, objects: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sorts one type of objects by Manhattan distance to the agent."""
    if objects:
        return sorted(objects, key=lambda object: manhatten_distance(object[0], object[1], x_agent, y_agent))


def sort_opponents_by_distance(x_agent: int, y_agent: int, living_opponents: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sorts living opponents by Manhattan distance to the agent."""
    if living_opponents:
        return sorted(living_opponents, key=lambda opponent: manhatten_distance(opponent[3][0], opponent[3][1], x_agent, y_agent))


def filter_and_sort_bombs(x_agent: int, y_agent: int, bombs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Filters and sorts bombs by their Manhattan distance to the agent's position, considering only those
    bombs that are either in the same row (y coordinate) or the same column (x coordinate) as the agent and having a distance of 3 or less, thus being a potential danger.
    """
    dangerous_bombs = filter_dangerous_bombs(x_agent, y_agent, bombs)

    dangerous_bombs = sort_objects_by_distance(
        x_agent, y_agent, dangerous_bombs)

    return dangerous_bombs


def is_in_danger(x_agent: int, y_agent: int, explosion_map, sorted_dangerous_bombs: List[Tuple[int, int]]) -> bool:
    """
    Function checks if given agent position is dangerous
    """
    if is_in_explosion(x_agent, y_agent, explosion_map):
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


def has_escaped_danger(x_agent, y_agent, self_action, field, sorted_dangerous_bombs, explosion_map):
    x_new, y_new = march_forward(x_agent, y_agent, self_action)
    if valid_action(x_new, y_new, field):
        if is_in_danger(x_new, y_new, explosion_map, sorted_dangerous_bombs):
            return False
        else:
            return True
    else:
        return False


def valid_action(x_new, y_new, field):
    return field[x_new, y_new] == 0


def is_in_danger_or_invalid_action(x_agent, y_agent, field, explosion_map, sorted_dangerous_bombs):
    return (is_in_danger((x_agent, y_agent, field, explosion_map, sorted_dangerous_bombs))
            or not valid_action(x_agent, y_agent, field))


def distance_increased(x_bomb, y_bomb, x_agent, y_agent, x_new, y_new):
    return ((abs(x_bomb - x_new) > abs(x_bomb - x_agent)) or
            ((y_bomb - y_new) > abs(y_bomb - y_agent)))


def got_in_loop(self, x_agent, y_agent):
    self.loop_count = self.agent_coord_history.count((x_agent, y_agent))
    return self.loop_count > 2


def waited_necessarily(x_agent, y_agent, field, explosion_map, sorted_dangerous_bombs):
    """
    Check if there is an explosion or danger around agent
    """
    return (is_in_danger_or_invalid_action(x_agent+1, y_agent, field, explosion_map, sorted_dangerous_bombs) and
            is_in_danger_or_invalid_action(x_agent-1, y_agent, field, explosion_map, sorted_dangerous_bombs) and
            is_in_danger_or_invalid_action(x_agent, y_agent+1, field, explosion_map, sorted_dangerous_bombs) and
            is_in_danger_or_invalid_action(x_agent, y_agent-1, field, explosion_map, sorted_dangerous_bombs))


def decreased_distance(x_old, y_old, x_new, y_new, x_obj, y_obj):
    return (manhatten_distance(x_new, y_new, x_obj, y_obj)
            < manhatten_distance(x_old, y_old, x_obj, y_obj))


def increased_distance(x_old, y_old, x_new, y_new, x_obj, y_obj):
    return (manhatten_distance(x_new, y_new, x_obj, y_obj)
            > manhatten_distance(x_old, y_old, x_obj, y_obj))


def find_closest_crate(x_agent, y_agent, field):
    """ 
    Breadth First Search for efficiant search of closest create 
    """
    rows, cols = len(field), len(field[0])
    queue = deque([(x_agent, y_agent)])
    visited = set((x_agent, y_agent))

    while queue:
        x, y = queue.popleft()

        # Return the position as soon as we find a "1"
        if field[x][y] == 1:
            return (x, y)

        # Explore the four possible directions
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))

    return None  # In case there's no "1" in the matrix


def has_destroyed_target(events):
    if e.BOMB_EXPLODED in events:
        return e.KILLED_OPPONENT in events or e.CRATE_DESTROYED in events


def has_missed_target(events):
    if e.BOMB_EXPLODED in events:
        return not (e.KILLED_OPPONENT in events or e.CRATE_DESTROYED in events)


def is_in_game_grid(x, y, max_row, max_col):
    return 0 <= x < max_row and 0 <= y < max_col


def simulate_bomb_explosion(x_agent, y_agent, field):
    bomb_simulated_field = field.copy()
    max_row, max_col = len(field), len(field[0])
    number_of_destroying_crates = 0

    for dx, dy in DIRECTIONS:
        for dist in range(1, BOMB_POWER + 1):
            nx, ny = x_agent + dx * dist, y_agent + dy * dist
            if is_in_game_grid(nx, ny, max_row, max_col) and bomb_simulated_field[nx][ny] != -1:
                bomb_simulated_field[nx][ny] = UNSAFE_FIELD  # Mark as unsafe
                if bomb_simulated_field[nx][ny] == 1:
                    number_of_destroying_crates += 1
            else:
                break
    bomb_simulated_field[x_agent][y_agent] = UNSAFE_FIELD
    return bomb_simulated_field, number_of_destroying_crates


def path_to_safety_exists(x_agent, y_agent, bomb_simulated_field, field):
    max_row, max_col = len(field), len(field[0])
    queue = deque([(x_agent, y_agent)])
    visited = set([(x_agent, y_agent)])
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy

            if is_in_game_grid(nx, ny, max_row, max_col) and (nx, ny) not in visited:
                if bomb_simulated_field[nx][ny] == 0:
                    return True
                if field[nx][ny] == 0:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    return False  # No safe path exists


def potentially_destroying_opponent(bomb_simulated_field, sorted_living_opponents):
    for opponent in sorted_living_opponents:
        x_opponent, y_opponent = opponent[3]
        if bomb_simulated_field[x_opponent][y_opponent] == UNSAFE_FIELD:
            return True
    return False


def simulate_bomb(x_agent, y_agent, field, sorted_living_opponents):
    """ Simulate the bomb explosion and evaluate its effects. """
    simulated_field, num_destroyed_crates = simulate_bomb_explosion(
        x_agent, y_agent, field)
    can_reach_safety = path_to_safety_exists(x_agent, y_agent, simulated_field)
    could_hit_opponent = potentially_destroying_opponent(
        simulated_field, sorted_living_opponents)
    is_effective = num_destroyed_crates > 0 or could_hit_opponent

    return can_reach_safety, is_effective


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
    else:
        events.append(SURVIVED_STEP)

    if is_in_danger(x_agent, y_agent, explosion_map, sorted_dangerous_bombs):
        if not_escaping_danger(self_action):
            events.append(NOT_ESCAPING)
        elif has_escaped_danger(x_agent, y_agent, self_action, field, sorted_dangerous_bombs, explosion_map):
            events.append(OUT_OF_DANGER)
        elif is_escaping_danger(x_agent, y_agent, self_action, field, sorted_dangerous_bombs):
            events.append(ESCAPING)
        else:
            events.append(NOT_ESCAPING)

    else:  # Only punish loops if not in danger
        if got_in_loop(self, x_agent, y_agent):
            events.append(GOT_IN_LOOP)

    if self_action == 'WAIT':
        # Reward the agent if waiting is necessary.
        if waited_necessarily(x_agent, y_agent, field, explosion_map, sorted_dangerous_bombs):
            events.append(WAITED_NECESSARILY)
        else:
            events.append(WAITED_UNNECESSARILY)

    elif self_action == 'BOMB':
        can_reach_safety, is_effective = simulate_bomb(
            x_agent, y_agent, field, sorted_living_opponents)
        if not can_reach_safety:
            events.append(DUMB_BOMB_DROPPED)
        elif is_effective:
            events.append(SMART_BOMB_DROPPED)

    else:
        x_new, y_new = march_forward(x_agent, y_agent, self_action)

        sorted_living_opponents = sort_opponents_by_distance(
            x_agent, y_agent, living_opponents)
        if sorted_living_opponents:
            closest_opponent = sorted_living_opponents[0]
            x_opponent, y_opponent = closest_opponent[3]

            if decreased_distance(x_agent, y_agent, x_new, y_new, x_opponent, y_opponent):
                events.append(CLOSER_TO_PLAYERS)
            elif increased_distance(x_agent, y_agent, x_new, y_new, x_opponent, y_opponent):
                events.append(AWAY_FROM_PLAYERS)

        sorted_coins = sort_objects_by_distance(x_agent, y_agent, coins)
        if sorted_coins:
            closest_coin = sorted_coins[0]
            x_coin, y_coin = closest_coin

            if decreased_distance(x_agent, y_agent, x_new, y_new, x_coin, y_coin):
                events.append(CLOSER_TO_COIN)
            elif increased_distance(x_agent, y_agent, x_new, y_new, x_coin, y_coin):
                events.append(AWAY_FROM_COIN)

        closest_crate = find_closest_crate(x_agent, y_agent, field)
        if closest_crate:
            x_crate, y_crate = closest_crate
            if decreased_distance(x_agent, y_agent, x_new, y_new, x_crate, y_crate):
                events.append(CLOSER_TO_CRATE)
            elif increased_distance(x_agent, y_agent, x_new, y_new, x_crate, y_crate):
                events.append(AWAY_FROM_CRATE)

    if has_destroyed_target(events):
        events.append(DESTROY_TARGET)
    elif has_missed_target(events):
        events.append(MISSED_TARGET)

    if e.CRATE_DESTROYED in events:
        number_of_crates_destroyed = events.count(e.CRATE_DESTROYED)
        if number_of_crates_destroyed > 2:
            events.append(BOMBED_3_TO_5_CRATES)
        elif number_of_crates_destroyed > 5:
            events.append(BOMBED_5_PLUS_CRATES)
        else:
            events.append(BOMBED_1_TO_2_CRATES)

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
        ESCAPING: escape,
        NOT_ESCAPING: -escape,
        WAITED_NECESSARILY: waiting,
        WAITED_UNNECESSARILY: -waiting,
        CLOSER_TO_PLAYERS: aggressive_action,
        AWAY_FROM_PLAYERS: -aggressive_action,
        CLOSER_TO_COIN: coin_action,
        AWAY_FROM_COIN: -coin_action,
        CONSTANT_PENALTY: -0.001,
        WON_GAME: 10,
        GOT_IN_LOOP: -0.025 * self.loop_count,

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
        self.logger.info(f"Awarded {reward_sum} for events {
                         ', '.join(events)}")
        return reward_sum
