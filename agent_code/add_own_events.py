from typing import List, Tuple
from collections import deque

import events as e
import own_events as own_e
from settings import MAX_STEPS, BOMB_POWER


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
    def _is_clear_path(start: int, end: int, fixed: int, is_row_fixed: bool) -> bool:
        """ Helper function to check for obstacles in a row or column. """
        step = 1 if start < end else -1
        for i in range(start + step, end, step):
            if (field[fixed][i] if is_row_fixed else field[i][fixed]) == -1:
                return False
        return True

    if x_agent == bomb[0]:  # Same column
        return _is_clear_path(y_agent, bomb[1], x_agent, True)
    elif y_agent == bomb[1]:  # Same row
        return _is_clear_path(x_agent, bomb[0], y_agent, False)


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


def filter_and_sort_bombs(x_agent: int, y_agent: int, bombs: List[Tuple[int, int]], field) -> List[Tuple[int, int]]:
    """
    Filters and sorts bombs by their Manhattan distance to the agent's position, considering only those
    bombs that are either in the same row (y coordinate) or the same column (x coordinate) as the agent and having a distance of 3 or less, thus being a potential danger.
    """
    if bombs:
        dangerous_bombs = filter_dangerous_bombs(
            x_agent, y_agent, bombs, field)

        dangerous_bombs = sort_objects_by_distance(
            x_agent, y_agent, dangerous_bombs)

        return dangerous_bombs
    else:
        return None


def is_in_danger(x_agent: int, y_agent: int, explosion_map, sorted_dangerous_bombs: List[Tuple[int, int]]) -> bool:
    """
    Function checks if given agent position is dangerous
    """
    return is_in_explosion(x_agent, y_agent, explosion_map) or sorted_dangerous_bombs


def has_highest_score(living_opponents, score_self: int) -> bool:
    if living_opponents:
        score_opponents = [opponent[1] for opponent in living_opponents]
        return all(score_self > score for score in score_opponents)
    else:
        # TODO: How to see scores of killed opponents
        return True


def has_won_the_game(living_opponents, score_self: int, events: List[str], steps_of_round: int) -> bool:
    """
    Determine if the player has won the game based on the current game state.
    # TODO: Can killed agent still win?
    # TODO: How can we access killed opponents for score
    """
    if steps_of_round == MAX_STEPS:
        return has_highest_score(living_opponents, score_self)
    elif e.GOT_KILLED in events:
        return False
    elif e.KILLED_SELF in events:
        return len(living_opponents) == 1 and has_highest_score(living_opponents, score_self)
    raise ValueError("Invalid game state or undefined events")


def not_escaping_danger(self_action):
    return self_action == 'WAIT' or self_action == 'BOMB'


def is_escaping_danger(x_agent, y_agent, self_action, field, sorted_dangerous_bombs):
    x_new, y_new = march_forward(x_agent, y_agent, self_action)
    if is_valid_action(x_new, y_new, field):
        if sorted_dangerous_bombs:
            closest_bomb = sorted_dangerous_bombs[0]
            return increased_distance(x_agent, y_agent, x_new, y_new, closest_bomb[0], closest_bomb[1])
    else:
        return False


def has_escaped_danger(x_agent, y_agent, self_action, field, bombs, explosion_map):
    x_new, y_new = march_forward(x_agent, y_agent, self_action)
    if is_valid_action(x_new, y_new, field):
        sorted_dangerous_bombs = filter_and_sort_bombs(
            x_new, y_new, bombs, field)
        return not is_in_danger(x_new, y_new, explosion_map, sorted_dangerous_bombs)
    else:
        return False


def is_valid_action(x_new, y_new, field) -> bool:
    """
    Check whether the action is possible or not.
    Expects walls (-1) around game field!!
    """
    return field[x_new, y_new] == 0


def is_save_step(x_agent, y_agent, field, explosion_map, bombs):
    sorted_dangerous_bombs = filter_and_sort_bombs(
        x_agent, y_agent, bombs, field)
    return (is_valid_action(x_agent, y_agent, field) and
            not is_in_danger(x_agent, y_agent, explosion_map, sorted_dangerous_bombs))


def got_in_loop(x_agent, y_agent, agent_coord_history):
    loop_count = agent_coord_history.count((x_agent, y_agent))
    return loop_count > 2


def waited_necessarily(x_agent, y_agent, field, explosion_map, bombs):
    """
    Check if there is an explosion or danger around agent
    """
    return (not is_save_step(x_agent+1, y_agent, field, explosion_map, bombs) and
            not is_save_step(x_agent-1, y_agent, field, explosion_map, bombs) and
            not is_save_step(x_agent, y_agent+1, field, explosion_map, bombs) and
            not is_save_step(x_agent, y_agent-1, field, explosion_map, bombs))


def decreased_distance(x_old, y_old, x_new, y_new, x_obj, y_obj):
    return (manhatten_distance(x_new, y_new, x_obj, y_obj)
            < manhatten_distance(x_old, y_old, x_obj, y_obj))


def increased_distance(x_old, y_old, x_new, y_new, x_obj, y_obj):
    return (manhatten_distance(x_new, y_new, x_obj, y_obj)
            > manhatten_distance(x_old, y_old, x_obj, y_obj))


def find_closest_crate(x_agent, y_agent, field):
    """ 
    Breadth First Search for efficiant search of closest crate 
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
            next_x, next_y = x + dx, y + dy
            if 0 <= next_x < rows and 0 <= next_y < cols and (next_x, next_y) not in visited:
                visited.add((next_x, next_y))
                queue.append((next_x, next_y))

    return None  # In case there's no "1" in the matrix


def has_destroyed_target(events):
    if e.BOMB_EXPLODED in events:
        return e.KILLED_OPPONENT in events or e.CRATE_DESTROYED in events


def is_in_game_grid(x, y, max_row, max_col):
    return 0 <= x < max_row and 0 <= y < max_col


def simulate_bomb_explosion(x_bomb, y_bomb, field):
    bomb_simulated_field = field.copy()
    max_row, max_col = len(field), len(field[0])
    number_of_destroying_crates = 0

    for dx, dy in DIRECTIONS:
        for dist in range(1, BOMB_POWER + 1):
            next_x, next_y = x_bomb + dx * dist, y_bomb + dy * dist
            if is_in_game_grid(next_x, next_y, max_row, max_col) and bomb_simulated_field[next_x][next_y] != -1:
                if bomb_simulated_field[next_x][next_y] == 1:
                    number_of_destroying_crates += 1
                bomb_simulated_field[next_x][next_y] = UNSAFE_FIELD
            else:
                break
    bomb_simulated_field[x_bomb][y_bomb] = UNSAFE_FIELD
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
    if sorted_living_opponents:
        for opponent in sorted_living_opponents:
            x_opponent, y_opponent = opponent[3]
            if bomb_simulated_field[x_opponent][y_opponent] == UNSAFE_FIELD:
                return True
    return False


def simulate_bomb(x_bomb, y_bomb, field, sorted_living_opponents):
    """ Simulate the bomb explosion and evaluate its effects. """
    simulated_field, num_destroyed_crates = simulate_bomb_explosion(
        x_bomb, y_bomb, field)
    can_reach_safety = path_to_safety_exists(
        x_bomb, y_bomb, simulated_field, field)
    could_hit_opponent = potentially_destroying_opponent(
        simulated_field, sorted_living_opponents)
    is_effective = num_destroyed_crates > 0 or could_hit_opponent

    return can_reach_safety, is_effective


def add_own_events(self, old_game_state, self_action, events_src, end_of_round, agent_coord_history) -> list:

    # events = copy.deepcopy(events_src)
    events = events_src.copy()
    events.append(own_e.CONSTANT_PENALTY)

    field = old_game_state['field']
    x_agent, y_agent = old_game_state['self'][3]
    is_bomb_possible = old_game_state['self'][2]
    bombs = [xy for (xy, t) in old_game_state['bombs']]
    sorted_dangerous_bombs = filter_and_sort_bombs(x_agent, y_agent, bombs)
    living_opponents = old_game_state['others']
    sorted_living_opponents = sort_opponents_by_distance(
        x_agent, y_agent, living_opponents)
    coins = old_game_state['coins']
    explosion_map = old_game_state['explosion_map']
    score_self = old_game_state['self'][1]
    steps_of_round = old_game_state['steps']

    if end_of_round:
        if has_won_the_game(living_opponents, score_self, events, steps_of_round):
            events.append(own_e.WON_GAME)
    else:
        events.append(own_e.SURVIVED_STEP)

    if is_in_danger(x_agent, y_agent, explosion_map, sorted_dangerous_bombs):
        if not_escaping_danger(self_action):
            events.append(own_e.NOT_ESCAPING)
        elif has_escaped_danger(x_agent, y_agent, self_action, field, bombs, explosion_map):
            events.append(own_e.OUT_OF_DANGER)
        elif is_escaping_danger(x_agent, y_agent, self_action, field, sorted_dangerous_bombs):
            events.append(own_e.ESCAPING)
        else:
            events.append(own_e.NOT_ESCAPING)

    if self_action == 'WAIT':
        # Reward the agent if waiting is necessary.
        if waited_necessarily(x_agent, y_agent, field, explosion_map, bombs):
            events.append(own_e.WAITED_NECESSARILY)
        else:
            events.append(own_e.WAITED_UNNECESSARILY)

    elif self_action == 'BOMB':
        if is_bomb_possible:
            can_reach_safety, is_effective = simulate_bomb(
                x_agent, y_agent, field, sorted_living_opponents)
            if not can_reach_safety:
                events.append(own_e.DUMB_BOMB_DROPPED)
            elif is_effective:
                events.append(own_e.SMART_BOMB_DROPPED)
        else:
            events.append(own_e.DUMB_BOMB_DROPPED)

    else:
        if got_in_loop(x_agent, y_agent, agent_coord_history):
            events.append(own_e.GOT_IN_LOOP)

        x_new, y_new = march_forward(x_agent, y_agent, self_action)
        if sorted_living_opponents:
            closest_opponent = sorted_living_opponents[0]
            x_opponent, y_opponent = closest_opponent[3]

            if decreased_distance(x_agent, y_agent, x_new, y_new, x_opponent, y_opponent):
                events.append(own_e.CLOSER_TO_PLAYERS)
            elif increased_distance(x_agent, y_agent, x_new, y_new, x_opponent, y_opponent):
                events.append(own_e.AWAY_FROM_PLAYERS)

        sorted_coins = sort_objects_by_distance(x_agent, y_agent, coins)
        if sorted_coins:
            closest_coin = sorted_coins[0]
            x_coin, y_coin = closest_coin

            if decreased_distance(x_agent, y_agent, x_new, y_new, x_coin, y_coin):
                events.append(own_e.CLOSER_TO_COIN)
            elif increased_distance(x_agent, y_agent, x_new, y_new, x_coin, y_coin):
                events.append(own_e.AWAY_FROM_COIN)

        closest_crate = find_closest_crate(x_agent, y_agent, field)
        if closest_crate:
            x_crate, y_crate = closest_crate
            if decreased_distance(x_agent, y_agent, x_new, y_new, x_crate, y_crate):
                events.append(own_e.CLOSER_TO_CRATE)
            elif increased_distance(x_agent, y_agent, x_new, y_new, x_crate, y_crate):
                events.append(own_e.AWAY_FROM_CRATE)

    if has_destroyed_target(events):
        events.append(own_e.DESTROY_TARGET)
    else:
        events.append(own_e.MISSED_TARGET)

    if e.CRATE_DESTROYED in events:
        number_of_crates_destroyed = events.count(e.CRATE_DESTROYED)
        if number_of_crates_destroyed > 2:
            events.append(own_e.BOMBED_3_TO_5_CRATES)
        elif number_of_crates_destroyed > 5:
            events.append(own_e.BOMBED_5_PLUS_CRATES)
        else:
            events.append(own_e.BOMBED_1_TO_2_CRATES)

    return events
