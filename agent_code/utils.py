from typing import List, Tuple
from collections import deque
import events as e

from settings import MAX_STEPS, BOMB_POWER

DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
UNSAFE_FIELD = 2
CRATE = 1
WALL = -1
FREE = 0


def march_forward(coords, direction):
    x, y = coords
    # Forward in direction.
    if direction == 'LEFT':
        x -= 1
    elif direction == 'RIGHT':
        x += 1
    elif direction == 'UP':
        y -= 1
    elif direction == 'DOWN':
        y += 1
    return (x, y)


def is_in_explosion(agent_coords, explosion_map) -> bool:
    return explosion_map[agent_coords] != 0


def is_wall_free_path(agent_coords: Tuple[int, int], bomb: Tuple[int, int], field) -> bool:
    """
    Determines if there is a clear path (no obstacles) between an agent's position and a bomb's position
    on a given field. The field is represented as a 2D list where only -1 indicates an obstacle.
    """
    def _is_wall_free_path(start: int, end: int, fixed: int, is_row_fixed: bool) -> bool:
        """ Helper function to check for obstacles in a row or column. """
        step = 1 if start < end else -1
        for i in range(start + step, end, step):
            if (field[fixed][i] if is_row_fixed else field[i][fixed]) == WALL:
                return False
        return True

    if agent_coords[0] == bomb[0]:  # Same column
        return _is_wall_free_path(agent_coords[1], bomb[1], agent_coords[0], True)
    elif agent_coords[1] == bomb[1]:  # Same row
        return _is_wall_free_path(agent_coords[0], bomb[0], agent_coords[1], False)


def is_dangerous_bomb(agent_coords: Tuple[int, int], bomb: Tuple[int, int], field) -> bool:
    """Check if a bomb is dangerous and has a clear path to the agent."""
    return ((bomb[0] == agent_coords[0] or bomb[1] == agent_coords[1]) and
            abs(bomb[0] - agent_coords[0]) + abs(bomb[1] - agent_coords[1]) <= 3 and
            is_wall_free_path(agent_coords, bomb, field))


def filter_dangerous_bombs(agent_coords: Tuple[int, int], bombs: List[Tuple[int, int]], field) -> List[Tuple[int, int]]:
    """Filters bombs that are in the same row or column as the agent and within a distance of 3."""
    return [
        bomb for bomb in bombs
        if is_dangerous_bomb(agent_coords, bomb, field)
    ]


def manhatten_distance(coords_1, coords_2) -> int:
    return abs(coords_1[0] - coords_2[0]) + abs(coords_1[1] - coords_2[1])


def sort_objects_by_distance(agent_coords: Tuple[int, int], objects: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sorts one type of objects by Manhattan distance to the agent."""
    return sorted(objects, key=lambda object: manhatten_distance(object, agent_coords))


def sort_opponents_by_distance(agent_coords: Tuple[int, int], living_opponents: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sorts living opponents by Manhattan distance to the agent."""
    return sorted(living_opponents, key=lambda opponent: manhatten_distance(opponent[3], agent_coords))


def sort_and_filter_out_dangerous_bombs(agent_coords: Tuple[int, int], bombs: List[Tuple[int, int]], field) -> List[Tuple[int, int]]:
    """
    Filters and sorts bombs by their Manhattan distance to the agent's position, considering only those
    bombs that are either in the same row (y coordinate) or the same column (x coordinate) as the agent and 
    having a distance of 3 or less and not beeing blocked by walls, thus being a potential danger.
    """
    dangerous_bombs = filter_dangerous_bombs(
        agent_coords, bombs, field)

    dangerous_bombs = sort_objects_by_distance(
        agent_coords, dangerous_bombs)

    return dangerous_bombs


def is_dangerous(step_coords: Tuple[int, int], explosion_map, sorted_dangerous_bombs: List[Tuple[int, int]]) -> bool:
    """
    Function checks if given position is dangerous
    """
    return is_in_explosion(step_coords, explosion_map) or sorted_dangerous_bombs


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


def is_escaping_danger(agent_coords, self_action, field, opponents, sorted_dangerous_bombs):
    new_agents_coords = march_forward(agent_coords, self_action)
    if is_valid_action(new_agents_coords, field, opponents, sorted_dangerous_bombs):
        if sorted_dangerous_bombs:
            closest_bomb = sorted_dangerous_bombs[0]
            return increased_distance(agent_coords, new_agents_coords, closest_bomb)
    else:
        return False


def has_escaped_danger(agent_coords, self_action, field, opponents, bombs, explosion_map):
    new_agents_coords = march_forward(agent_coords, self_action)
    if is_valid_action(new_agents_coords, field, opponents, bombs):
        sorted_dangerous_bombs = sort_and_filter_out_dangerous_bombs(
            new_agents_coords, bombs, field)
        return not is_dangerous(new_agents_coords, explosion_map, sorted_dangerous_bombs)
    else:
        return False


def is_valid_action(step_coords, field, opponents, bombs) -> bool:
    """
    Check whether the action is possible or not.
    Expects walls (-1) around game field!!
    """
    return (field[step_coords] == FREE and
            (not step_coords in opponents) and
            (not step_coords in bombs))


def is_save_step(agent_coords, field, opponents, explosion_map, bombs):
    sorted_dangerous_bombs = sort_and_filter_out_dangerous_bombs(
        agent_coords, bombs, field)
    return (is_valid_action(agent_coords, field, opponents, bombs) and
            not is_dangerous(agent_coords, explosion_map, sorted_dangerous_bombs))


def got_in_loop(agent_coords, agent_coord_history):
    loop_count = agent_coord_history.count(agent_coords)
    return loop_count > 2


def waited_necessarily(agent_coords, field, opponents, explosion_map, bombs):
    """
    Check if there is an explosion or danger around agent
    """
    x_agent, y_agent = agent_coords
    return (not is_save_step((x_agent+1, y_agent), field, opponents, explosion_map, bombs) and
            not is_save_step((x_agent-1, y_agent), field, opponents, explosion_map, bombs) and
            not is_save_step((x_agent, y_agent+1), field, opponents, explosion_map, bombs) and
            not is_save_step((x_agent, y_agent-1), field, opponents, explosion_map, bombs))


def decreased_distance(old_coords, new_coords, object_coords):
    return (manhatten_distance(new_coords, object_coords)
            < manhatten_distance(old_coords, object_coords))


def increased_distance(old_coords, new_coords, object_coords):
    return (manhatten_distance(new_coords, object_coords)
            > manhatten_distance(old_coords, object_coords))


def find_closest_crate(agent_coords, field):
    """ 
    Breadth First Search for efficiant search of closest crate 
    """
    rows, cols = len(field), len(field[0])
    queue = deque([agent_coords])
    visited = set(agent_coords)

    while queue:
        coords = queue.popleft()

        if field[coords] == CRATE:
            return coords

        # Explore the four possible directions
        for direction in DIRECTIONS:
            next_coords = (coords[0] + direction[0], coords[1] + direction[1])
            if 0 <= next_coords[0] < rows and 0 <= next_coords[1] < cols and next_coords not in visited:
                visited.add(next_coords)
                queue.append(next_coords)

    return None


def has_destroyed_target(events):
    if e.BOMB_EXPLODED in events:
        return e.KILLED_OPPONENT in events or e.CRATE_DESTROYED in events


def is_in_game_grid(coords, max_row, max_col):
    return 0 <= coords[0] < max_row and 0 <= coords[1] < max_col


def simulate_bomb_explosion(bomb, field):
    bomb_simulated_field = field.copy()
    max_row, max_col = len(field), len(field[0])
    number_of_destroying_crates = 0

    for direction in DIRECTIONS:
        for dist in range(1, BOMB_POWER + 1):
            next_coords = (bomb[0] + direction[0] * dist,
                           bomb[1] + direction[1] * dist)
            if is_in_game_grid(next_coords, max_row, max_col) and bomb_simulated_field[next_coords] != WALL:
                if bomb_simulated_field[next_coords] == CRATE:
                    number_of_destroying_crates += 1
                bomb_simulated_field[next_coords] = UNSAFE_FIELD
            else:
                break
    bomb_simulated_field[bomb] = UNSAFE_FIELD
    return bomb_simulated_field, number_of_destroying_crates


def path_to_safety_exists(agent_coords, bomb_simulated_field, field, living_opponents, bombs):
    """
    Gives if there exist a path to safety based on the given agents coordinates and the simulated bombs field. 
    Bombs and opponents are considered as not vanishing and not moving.
    TODO Include vanishing bombs?
    """
    max_row, max_col = len(field), len(field[0])
    queue = deque([agent_coords])
    visited = set([agent_coords])
    while queue:
        coords = queue.popleft()
        for direction in DIRECTIONS:
            new_coords = coords[0] + direction[0], coords[1] + direction[1]

            if (is_in_game_grid(new_coords, max_row, max_col) and
                new_coords not in visited and
                new_coords not in living_opponents and
                    new_coords not in bombs):
                if bomb_simulated_field[new_coords] == FREE:
                    return True
                if field[new_coords] == FREE:
                    visited.add(new_coords)
                    queue.append(new_coords)
    return False  # No safe path exists


def potentially_destroying_opponent(bomb_simulated_field, sorted_living_opponents):
    if sorted_living_opponents:
        for opponent in sorted_living_opponents:
            opponent_coords = opponent[3]
            if bomb_simulated_field[opponent_coords] == UNSAFE_FIELD:
                return True
    return False


def simulate_bomb(bomb, field, sorted_living_opponents, bombs):
    """ Simulate the bomb explosion and evaluate its effects. """
    simulated_field, num_destroyed_crates = simulate_bomb_explosion(
        bomb, field)
    can_reach_safety = path_to_safety_exists(
        bomb, simulated_field, field, sorted_living_opponents, bombs)
    could_hit_opponent = potentially_destroying_opponent(
        simulated_field, sorted_living_opponents)
    is_effective = num_destroyed_crates > 0 or could_hit_opponent

    return can_reach_safety, is_effective
