from typing import List, Tuple, Optional
from collections import deque
from dataclasses import dataclass
import copy

import numpy as np

import events as e
import settings as s

MOVEMENT_DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)
                       ]  # UP, DOWN, LEFT, RIGHT
DIRECTIONS_AND_WAIT = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]
MOVEMENT_ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
MOVEMENT = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}

UNSAFE_FIELD = 2
CRATE = 1
WALL = -1
FREE = 0

# TODO TypeDict?


@dataclass
class GameState:
    field: np.ndarray
    bombs: List[Tuple[Tuple[int, int], int]]
    explosion_map: np.ndarray
    coins: List[Tuple[int, int]]
    self: Tuple[str, int, bool, Tuple[int, int]]
    others: List[Tuple[str, int, bool, Tuple[int, int]]]
    step: int
    round: int
    user_input: Optional[str]

    def next(self, action: str) -> Optional['GameState']:
        """
        Advances the game state by one action performed by the player. 
        Returns None if action is invalid or agent dies
        """
        next_game_state = copy.deepcopy(self)
        if not next_game_state._process_player_action(action):
            return None
        next_game_state._update_bombs()
        next_game_state._evaluate_explosions()
        next_game_state._update_explosions()
        return next_game_state

    def _process_player_action(self, action: str) -> bool:
        name, score, bomb, agent_coords = self.self

        if action in MOVEMENT:
            direction = MOVEMENT[action]
            new_coords = move_in_direction(agent_coords, direction)
            if self._is_valid_action(new_coords):
                self.self = (name, score, bomb, new_coords)
            else:
                return False
        elif action == "WAIT":
            pass
        elif action == "BOMB":
            if bomb:
                self.bombs.append((agent_coords, s.BOMB_TIMER))
            else:
                return False
        else:
            raise ValueError("Action, doesn't exist")

        return True

    def _is_valid_action(self, step_coords) -> bool:
        """
        Check whether the action is possible or not.
        Expects walls (-1) around game field!!
        # TODO Self.opponents and self.bombs incorrect format
        """
        return (self.field[step_coords] == FREE and
                (not step_coords in self.opponents) and
                (not step_coords in self.bombs))

    def _update_explosions(self):
        """
        Like in environment.py: self.update_explosions()
        """
        self["explosion_map"] = np.clip(
            self["explosion_map"] - 1, 0, None)

    def _update_bombs(self):
        """
        Like in environment.py: self.update_explosions()
        """
        self['field'] = np.array(self['field'])
        i = 0
        while i < len(self['bombs']):
            (bomb_coords, t) = self['bombs'][i]
            t -= 1
            if t < 0:
                self['bombs'].pop(i)
                all_blast_coords = get_blast_effected_coords(
                    blast_coords, game_state)
                for blast_coords in all_blast_coords:
                    self['field'][blast_coords] = 0
                    self["explosion_map"][blast_coords] = s.EXPLOSION_TIMER
            else:
                self['bombs'][i] = (bomb_coords, t)
                i += 1


def evaluate_explosions(game_state: GameState):
    """
    Like in environment.py: self.evaluate_explosions()
    """
    agent_coords = game_state['self'][3]
    if game_state["explosion_map"][agent_coords] != 0:
        return None  # Player dies
    others_copy = list(game_state['others'])  # Make a copy of the others list
    for opponent in others_copy:
        opponent_coords = opponent[3]
        if game_state['explosion_map'][opponent_coords] != 0:
            # Remove the opponent from the original list
            game_state['others'].remove(opponent)


def move_in_direction(coords, direction):
    return coords[0] + direction[0], coords[1] + direction[1]


def march_forward(coords, action: str):
    x, y = coords
    # Forward in direction.
    if action == 'LEFT':
        x -= 1
    elif action == 'RIGHT':
        x += 1
    elif action == 'UP':
        y -= 1
    elif action == 'DOWN':
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


def is_dangerous_bomb(agent_coords: Tuple[int, int], bomb_coords: Tuple[int, int], field) -> bool:
    """Check if a bomb is dangerous and has a clear path to the agent."""
    return ((bomb_coords[0] == agent_coords[0] or bomb_coords[1] == agent_coords[1]) and
            abs(bomb_coords[0] - agent_coords[0]) + abs(bomb_coords[1] - agent_coords[1]) <= 3 and
            is_wall_free_path(agent_coords, bomb_coords, field))


def filter_dangerous_bombs(agent_coords: Tuple[int, int], game_state: GameState) -> List[Tuple[int, int]]:
    """Filters bombs that are in the same row or column as the agent and within a distance of 3."""
    return [
        bomb[0] for bomb in game_state.bombs
        if is_dangerous_bomb(agent_coords, bomb, game_state.field)
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
    if steps_of_round == s.MAX_STEPS:
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


def is_about_to_explode(sorted_dangerous_bombs):
    if sorted_dangerous_bombs:
        for bomb in sorted_dangerous_bombs:
            if bomb[1] == 0:
                return True
    return False


def is_save_step(new_coords, game_state: GameState):
    sorted_dangerous_bombs = sort_and_filter_out_dangerous_bombs(
        new_coords, game_state)
    return (is_valid_action(new_coords, game_state) and
            not is_dangerous(new_coords, game_state.explosion_map, sorted_dangerous_bombs))


def is_coin(coords, game_state: GameState):
    return coords in game_state.coins


def is_near_crate(coords, game_state: GameState) -> bool:
    """Return True if the given coordinate is near a crate."""
    for direction in MOVEMENT_DIRECTIONS:
        new_coords = coords[0]+direction[0], coords[1] + direction[1]
        if game_state.field[new_coords] == 1:
            return True
    return False


def is_opponent_in_blast_range(coords, game_state: GameState) -> bool:
    """Return True if the player is within blast range of the enemy."""
    for opponent in game_state.others:
        if opponent[3] in get_blast_effected_coords(coords, game_state.field):
            return True
    return False


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

    TODO add opponents and bombs?
    """
    rows, cols = len(field), len(field[0])
    queue = deque([agent_coords])
    visited = set(agent_coords)

    while queue:
        coords = queue.popleft()

        if field[coords] == CRATE:
            return coords

        # Explore the four possible directions
        for direction in MOVEMENT_DIRECTIONS:
            next_coords = (coords[0] + direction[0], coords[1] + direction[1])
            if (0 <= next_coords[0] < rows and 0 <= next_coords[1] < cols and
                    next_coords not in visited):
                visited.add(next_coords)
                queue.append(next_coords)

    return None


def has_destroyed_target(events):
    if e.BOMB_EXPLODED in events:
        return e.KILLED_OPPONENT in events or e.CRATE_DESTROYED in events


def is_in_game_grid(coords, max_row, max_col):
    return 0 <= coords[0] < max_row and 0 <= coords[1] < max_col


def get_blast_effected_coords(bomb_coords, field) -> List[tuple[int, int]]:
    """
    Calculate all coordinates affected by a bomb's blast.
    """
    # TODO necesarry?
    if field[bomb_coords] == WALL:
        return []

    max_row, max_col = len(field), len(field[0])

    blast_coords = [bomb_coords]
    for direction in MOVEMENT_DIRECTIONS:
        for i in range(1, s.BOMB_POWER + 1):
            new_coords = bomb_coords[0] + direction[0] * \
                i, bomb_coords[1] + direction[1] * i
            if not is_in_game_grid(new_coords, max_row, max_col) or field[new_coords] == WALL:
                break
            blast_coords.append(new_coords)

    return blast_coords


def path_to_safety_exists(agent_coords, bomb_blast_coords, field, living_opponents, bombs):
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
        for direction in MOVEMENT_DIRECTIONS:
            new_coords = coords[0] + direction[0], coords[1] + direction[1]

            if (is_in_game_grid(new_coords, max_row, max_col) and
                new_coords not in visited and
                new_coords not in living_opponents and
                    new_coords not in bombs):
                if new_coords not in bomb_blast_coords:
                    return True
                if field[new_coords] == FREE:
                    visited.add(new_coords)
                    queue.append(new_coords)
    return False  # No safe path exists


def potentially_destroying_opponent(bomb_blast_coords, sorted_living_opponents):
    if sorted_living_opponents:
        for opponent in sorted_living_opponents:
            opponent_coords = opponent[3]
            if opponent_coords in bomb_blast_coords:
                return True
    return False


def get_number_of_destroying_crates(bomb_blast_coords, field):
    n_detroying_crates = 0
    for blast_coord in bomb_blast_coords:
        if field[blast_coord] == CRATE:
            n_detroying_crates += 1
    return n_detroying_crates


def simulate_bomb(bomb, field, sorted_living_opponents, bombs):
    """ Simulate the bomb explosion and evaluate its effects. """
    bomb_blast_coords = get_blast_effected_coords(bomb, field)
    can_reach_safety = path_to_safety_exists(
        bomb, bomb_blast_coords, field, sorted_living_opponents, bombs)
    could_hit_opponent = potentially_destroying_opponent(
        bomb_blast_coords, sorted_living_opponents)
    num_destroying_crates = get_number_of_destroying_crates(
        bomb_blast_coords, field)
    is_effective = num_destroying_crates > 0 or could_hit_opponent

    return can_reach_safety, is_effective


def find_shortest_path(start_coords, game_state: GameState, stop_criterion: function):
    """
    Returns the shortest path to one of the given goal coordinates, currently bombs and opponents block movements
    with next game state estimation
    # TODO put waiting into exploring?
    """
    queue = deque([start_coords, game_state])
    visited = set([start_coords])
    parent = {start_coords: None}

    while queue:
        current_coords, current_game_state = queue.popleft()
        if stop_criterion(current_coords, current_game_state):
            step = current_coords
            path = []
            while step != start_coords:
                path.append(step)
                step = parent[step]
            return path[::-1]

        for action in MOVEMENT_ACTIONS:
            next_game_state = get_next_game_state(action, current_game_state)
            if next_game_state != None:
                new_coords = next_game_state['self'][3]
                if new_coords not in visited:
                    queue.append(new_coords)
                    visited.add(new_coords)
                    parent[new_coords] = new_coords
    return []


def get_action_idx_from_coords(agent_coords, new_coords):
    direction = (new_coords[0]-agent_coords[0], new_coords[1]-agent_coords[1])
    return MOVEMENT_DIRECTIONS.index(direction)


def get_action_idx_to_closest_thing(game_state: GameState, stop_criterion: function):
    agent_coords = game_state.self[3]
    shortest_path = find_shortest_path(
        agent_coords, game_state, stop_criterion)

    if shortest_path:
        first_step_coords = shortest_path[0]
        return get_action_idx_from_coords(first_step_coords)
    else:
        return ACTIONS.index('WAIT')


def get_danger_in_each_direction(coords, game_state: GameState):
    danger_per_action = np.zeros(len(DIRECTIONS_AND_WAIT))
    for idx_action, direction in enumerate(DIRECTIONS_AND_WAIT):
        new_coords = coords[0] + direction[0], coords[1] + direction[1]
        sorted_dangerous_bombs = sort_and_filter_out_dangerous_bombs(
            new_coords, game_state)
        if game_state.explosion_map[new_coords] == 1:
            danger_per_action[idx_action] = EXTREME_DANGER
        for bomb_coords, timer in sorted_dangerous_bombs:
            blast_coords = get_blast_effected_coords(
                bomb_coords, game_state.field)
            if new_coords in blast_coords:
                match timer:
                    case 0:
                        danger_per_action[idx_action] = max(
                            danger_per_action[idx_action], EXTREME_DANGER)
                    case 1:
                        danger_per_action[idx_action] = max(
                            danger_per_action[idx_action], HIGH_DANGER)
                    case 2:
                        danger_per_action[idx_action] = max(
                            danger_per_action[idx_action], MEDIUM_DANGER)
                    case 3:
                        danger_per_action[idx_action] = max(
                            danger_per_action[idx_action], LOW_DANGER)

    return danger_per_action


def is_deadend(coords, game_state: GameState):
    count_free_tiles = 0
    for direction in MOVEMENT_DIRECTIONS:
        new_coords = move_in_direction(coords, direction)
        if game_state.field[new_coords] == FREE:
            count_free_tiles += 1
    return count_free_tiles <= 1


def get_possible_directions(agent_coords, game_state: GameState):
    possible_directions = []
    for direction in MOVEMENT_DIRECTIONS:
        new_coords = march_forward(agent_coords, direction)
        if is_valid_action(new_coords, game_state):
            possible_directions.append(direction)
    return possible_directions


def are_opposite_directions(directions):
    if directions == 2:
        direction_1 = directions[0]
        direction_2 = directions[1]
        return (direction_1[0] + direction_2[0] == 0 and
                direction_1[1] + direction_2[1] == 0)
    return False


def opponent_in_deadend(opponent, game_state):
    """
    Returns if opponent is in deadend (e.g only has two availabe opposite directions that 
    he can walk and one of them leads into a deadend)
    """
    opponent_coord = opponent[3]
    possible_directions = get_possible_directions(opponent_coord, game_state)
    if len(possible_directions) <= 2 and are_opposite_directions(possible_directions):
        for direction in possible_directions:
            new_coords = opponent_coord
            for i in range(0, s.BOMB_POWER):
                new_coords = move_in_direction(new_coords, direction)
                if is_deadend(new_coords, game_state):
                    return True
    return False


def would_surely_kill_opponent(bomb_blast_coords, game_state: GameState):
    for opponent in game_state.others:
        if opponent_in_deadend(opponent, game_state) and potentially_destroying_opponent(bomb_blast_coords, [opponent]):
            return True
    return False
