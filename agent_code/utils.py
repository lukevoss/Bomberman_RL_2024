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

EXTREME_DANGER = 1
HIGH_DANGER = 0.75
MEDIUM_DANGER = 0.5
LOW_DANGER = 0.25
NO_DANGER = 0


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
        if not next_game_state._evaluate_explosions():
            return None
        next_game_state._update_explosions()
        return next_game_state

    def not_escaping_danger(self, self_action: str):
        return self_action == 'WAIT' or self_action == 'BOMB'

    def is_escaping_danger(self, self_action: str, sorted_dangerous_bombs: List[Tuple[int, int]]):
        agent_coords = self.self[3]
        new_agents_coords = march_forward(agent_coords, self_action)
        if self.is_valid_movement(new_agents_coords):
            if sorted_dangerous_bombs:
                closest_bomb = sorted_dangerous_bombs[0]
                return increased_distance(agent_coords, new_agents_coords, closest_bomb)
            else:
                return True
        return False

    def has_escaped_danger(self, self_action: str) -> bool:
        agent_coords = self.self[3]
        new_agents_coords = march_forward(agent_coords, self_action)
        if self.is_valid_movement(new_agents_coords):
            return not self.is_dangerous(new_agents_coords)
        else:
            return False

    def is_valid_movement(self, step_coords: Tuple[int, int]) -> bool:
        """
        Check whether the movement is possible or not.
        Expects walls (-1) around game field!!
        """
        opponent_positions = {opponent[3] for opponent in self.others}
        bomb_positions = {bomb[0] for bomb in self.bombs}

        return (self.field[step_coords] == FREE and
                (not step_coords in opponent_positions) and
                (not step_coords in bomb_positions))

    def is_dangerous(self, step_coords: Tuple[int, int]) -> bool:
        """Function checks if given position is dangerous"""
        sorted_dangerous_bombs = self.sort_and_filter_out_dangerous_bombs(
            step_coords)
        return self._is_in_explosion(step_coords) or (sorted_dangerous_bombs != [])

    def waited_necessarily(self) -> bool:
        """
        Check if there is an explosion or danger around agent
        """
        x_agent, y_agent = self.self[3]
        return (not self.is_save_step((x_agent+1, y_agent)) and
                not self.is_save_step((x_agent-1, y_agent)) and
                not self.is_save_step((x_agent, y_agent+1)) and
                not self.is_save_step((x_agent, y_agent-1)))

    def is_save_step(self, new_coords: Tuple[int, int]) -> bool:
        return (self.is_valid_movement(new_coords) and
                not self.is_dangerous(new_coords))

    def simulate_own_bomb(self) -> Tuple[bool, bool]:
        """ Simulate the bomb explosion and evaluate its effects. """
        agent_coords = self.self[3]
        bomb_blast_coords = self._get_blast_effected_coords(agent_coords)
        can_reach_safety = self._path_to_safety_exists(agent_coords)
        could_hit_opponent = self._potentially_destroying_opponent(
            bomb_blast_coords)
        num_destroying_crates = self._get_number_of_destroying_crates(
            bomb_blast_coords)
        is_effective = num_destroying_crates > 0 or could_hit_opponent

        return can_reach_safety, is_effective

    def get_action_idx_to_closest_thing(self, thing: str) -> int:
        agent_coords = self.self[3]
        shortest_path = self._find_shortest_path(
            agent_coords, thing)

        if shortest_path:
            first_step_coords = shortest_path[0]
            return get_action_idx_from_coords(agent_coords, first_step_coords)
        else:
            return ACTIONS.index('WAIT')

    def get_danger_in_each_direction(self, coords: Tuple[int, int]) -> np.ndarray:
        danger_per_action = np.zeros(len(DIRECTIONS_AND_WAIT))
        for idx_action, direction in enumerate(DIRECTIONS_AND_WAIT):
            new_coords = coords[0] + direction[0], coords[1] + direction[1]
            sorted_dangerous_bombs = self.sort_and_filter_out_dangerous_bombs(
                new_coords)
            if self.explosion_map[new_coords] == 1:
                danger_per_action[idx_action] = EXTREME_DANGER
            for bomb_coords, timer in sorted_dangerous_bombs:
                blast_coords = self._get_blast_effected_coords(
                    bomb_coords)
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

    def is_perfect_bomb_spot(self, coords: Tuple[int, int]) -> bool:
        bomb_blast_coords = self._get_blast_effected_coords(coords)
        n_destroying_crates = self._get_number_of_destroying_crates(
            bomb_blast_coords)
        would_kill_opponent = self._would_surely_kill_opponent(
            bomb_blast_coords)
        return n_destroying_crates >= 4 or would_kill_opponent

    def sort_and_filter_out_dangerous_bombs(self, agent_coords: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Filters and sorts bombs by their Manhattan distance to the agent's position, considering only those
        bombs that are either in the same row (y coordinate) or the same column (x coordinate) as the agent and 
        having a distance of 3 or less and not beeing blocked by walls, thus being a potential danger.

        No need to handle Bomb available bool
        """
        dangerous_bombs = self._filter_dangerous_bombs(agent_coords)

        dangerous_bombs = sort_objects_by_distance(
            agent_coords, dangerous_bombs)

        return dangerous_bombs

    def sort_opponents(self, agent_coords: Tuple[int, int]) -> List[Tuple[int, int]]:
        opponent_positions = {opponent[3] for opponent in self.others}
        return sort_objects_by_distance(agent_coords, opponent_positions)

    def _process_player_action(self, action: str) -> bool:
        name, score, is_bomb_possible, agent_coords = self.self

        if action in MOVEMENT:
            direction = MOVEMENT[action]
            new_coords = move_in_direction(agent_coords, direction)
            if self.is_valid_movement(new_coords):
                self.self = (name, score, is_bomb_possible, new_coords)
            else:
                return False
        elif action == "WAIT":
            pass
        elif action == "BOMB":
            if is_bomb_possible:
                self.bombs.append((agent_coords, s.BOMB_TIMER))
            else:
                return False
        else:
            raise ValueError("Action, doesn't exist")

        return True

    def _update_explosions(self):
        """
        Like in environment.py: self.update_explosions()
        """
        self.explosion_map = np.clip(
            self.explosion_map - 1, 0, None)

    def _update_bombs(self):
        """
        Like in environment.py: self.update_explosions()
        No need to handle Bomb available bool
        """
        new_bombs = []
        for bomb_coords, t in self.bombs:
            t -= 1
            if t < 0:
                self._apply_bomb_effect(bomb_coords)
            else:
                new_bombs.append((bomb_coords, t))
        self.bombs = new_bombs

    def _apply_bomb_effect(self, bomb_coords):
        all_blast_coords = self._get_blast_effected_coords(bomb_coords)
        for blast_coords in all_blast_coords:
            self.field[blast_coords] = 0
            self.explosion_map[blast_coords] = s.EXPLOSION_TIMER

    def _evaluate_explosions(self) -> bool:
        """
        Like in environment.py: self.evaluate_explosions()
        """
        # Player dies
        agent_coords = self.self[3]
        if self.explosion_map[agent_coords] != 0:
            return False

        # Opponents die
        others_copy = list(self.others)
        for opponent in others_copy:
            opponent_coords = opponent[3]
            if self.explosion_map[opponent_coords] != 0:
                self.others.remove(opponent)
        return True

    def _get_blast_effected_coords(self, bomb_coords: Tuple[int, int]) -> List[tuple[int, int]]:
        """
        Calculate all coordinates affected by a bomb's blast.
        """
        blast_coords = [bomb_coords]
        for direction in MOVEMENT_DIRECTIONS:
            for i in range(1, s.BOMB_POWER + 1):
                new_coords = bomb_coords[0] + direction[0] * \
                    i, bomb_coords[1] + direction[1] * i
                if not is_in_game_grid(new_coords) or self.field[new_coords] == WALL:
                    break
                blast_coords.append(new_coords)

        return blast_coords

    def _filter_dangerous_bombs(self, agent_coords: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Filters bombs that are in the same row or column as the agent and within a distance of 3."""
        return [
            bomb[0] for bomb in self.bombs
            if self._is_dangerous_bomb(agent_coords, bomb[0])
        ]

    def _is_dangerous_bomb(self, agent_coords: Tuple[int, int], bomb_coords: Tuple[int, int]) -> bool:
        """Check if a bomb is dangerous and has a clear path to the agent."""
        return ((bomb_coords[0] == agent_coords[0] or bomb_coords[1] == agent_coords[1]) and
                abs(bomb_coords[0] - agent_coords[0]) + abs(bomb_coords[1] - agent_coords[1]) <= 3 and
                self._is_wall_free_path(agent_coords, bomb_coords))

    def _is_wall_free_path(self, agent_coords: Tuple[int, int], bomb_coords: Tuple[int, int]) -> bool:
        """
        Determines if there is a clear path (no obstacles) between an agent's position and a bomb's position
        on a given field. The field is represented as a 2D list where only -1 indicates an obstacle.
        """
        if agent_coords[0] == bomb_coords[0]:  # Same column
            return self._is_wall_free_direction(agent_coords[1], bomb_coords[1], agent_coords[0], True)
        elif agent_coords[1] == bomb_coords[1]:  # Same row
            return self._is_wall_free_direction(agent_coords[0], bomb_coords[0], agent_coords[1], False)

    def _is_wall_free_direction(self, start: int, end: int, fixed: int, is_row_fixed: bool) -> bool:
        """ Helper function to check for obstacles in a row or column. """
        step = 1 if start < end else -1
        for i in range(start + step, end, step):
            if (self.field[fixed][i] if is_row_fixed else self.field[i][fixed]) == WALL:
                return False
        return True

    def _is_in_explosion(self, coords: Tuple[int, int]) -> bool:
        return self.explosion_map[coords] != 0

    def find_closest_crate(self, agent_coords: Tuple[int, int]) -> Tuple[int, int]:
        """ 
        Breadth First Search for efficiant search of closest crate 
        Ignores opponents and bombs
        """
        rows, cols = len(self.field), len(self.field[0])
        queue = deque([agent_coords])
        visited = set(agent_coords)

        while queue:
            coords = queue.popleft()

            if self.field[coords] == CRATE:
                return coords

            # Explore the four possible directions
            for direction in MOVEMENT_DIRECTIONS:
                next_coords = move_in_direction(coords, direction)
                if (is_in_game_grid(next_coords, rows, cols) and
                        next_coords not in visited):
                    visited.add(next_coords)
                    queue.append(next_coords)

        return None

    def _path_to_safety_exists(self, agent_coords: Tuple[int, int]) -> bool:
        """
        Gives if there exist a path to safety based on the given agents coordinates and the simulated bombs field. 
        Based on next game state estimation
        """
        shortest_path = self._find_shortest_path(
            agent_coords, 'safety', place_bomb=True)
        return shortest_path != []

    def _find_shortest_path(self, start_coords: Tuple[int, int], thing: str, place_bomb=False):
        """
        Returns the shortest path to one of the given goal coordinates, currently bombs and opponents block movements
        with next game state estimation
        TODO put waiting into exploring?
        """
        starting_game_state = copy.deepcopy(self)
        if place_bomb:
            starting_game_state.bombs.append((start_coords, s.BOMB_TIMER))

        queue = deque([(start_coords, starting_game_state)])
        visited = set([start_coords])
        parent = {start_coords: None}
        stop_criterion = get_stop_criterion_for_thing(thing)

        while queue:
            current_coords, current_game_state = queue.popleft()
            if stop_criterion(current_game_state, current_coords):
                step = current_coords
                path = []
                while step != start_coords:
                    path.append(step)
                    step = parent[step]
                return path[::-1]

            for action in MOVEMENT_ACTIONS:
                next_game_state = current_game_state.next(action)
                if next_game_state != None:
                    new_coords = next_game_state.self[3]
                    if new_coords not in visited:
                        queue.append((new_coords, next_game_state))
                        visited.add(new_coords)
                        parent[new_coords] = current_coords
        return []

    def _potentially_destroying_opponent(self, bomb_blast_coords: Tuple[int, int]) -> bool:
        if self.others:
            for opponent in self.others:
                opponent_coords = opponent[3]
                if opponent_coords in bomb_blast_coords:
                    return True
        return False

    def _get_number_of_destroying_crates(self, bomb_blast_coords: Tuple[int, int]) -> int:
        n_detroying_crates = 0
        for blast_coord in bomb_blast_coords:
            if self.field[blast_coord] == CRATE:
                n_detroying_crates += 1
        return n_detroying_crates

    def _would_surely_kill_opponent(self, bomb_blast_coords):
        for opponent in self.others:
            if self._opponent_in_deadend(opponent) and self._potentially_destroying_opponent(bomb_blast_coords, [opponent]):
                return True
        return False

    def _opponent_in_deadend(self, opponent):
        """
        Returns if opponent is in deadend (e.g only has two availabe opposite directions that 
        he can walk and one of them leads into a deadend)
        """
        opponent_coord = opponent[3]
        possible_directions = self._get_possible_directions(opponent_coord)
        if len(possible_directions) <= 2 and self._are_opposite_directions(possible_directions):
            for direction in possible_directions:
                new_coords = opponent_coord
                for i in range(0, s.BOMB_POWER):
                    new_coords = move_in_direction(new_coords, direction)
                    if self._is_deadend(new_coords):
                        return True
        return False

    def _is_deadend(self, coords):
        count_free_tiles = 0
        for direction in MOVEMENT_DIRECTIONS:
            new_coords = move_in_direction(coords, direction)
            if self.field[new_coords] == FREE:
                count_free_tiles += 1
        return count_free_tiles <= 1

    def _get_possible_directions(self, agent_coords):
        possible_directions = []
        for direction in MOVEMENT_DIRECTIONS:
            new_coords = march_forward(agent_coords, direction)
            if self.is_valid_movement(new_coords):
                possible_directions.append(direction)
        return possible_directions

    def _are_opposite_directions(directions: List[Tuple[int, int]]) -> bool:
        if len(directions) == 2:
            direction_1 = directions[0]
            direction_2 = directions[1]
            return (direction_1[0] + direction_2[0] == 0 and
                    direction_1[1] + direction_2[1] == 0)
        return False


def move_in_direction(coords: Tuple[int, int], direction: Tuple[int, int]) -> Tuple[int, int]:
    return coords[0] + direction[0], coords[1] + direction[1]


def march_forward(coords, action: str) -> Tuple[int, int]:
    x, y = coords
    # Forward in direction.
    match action:
        case 'LEFT':
            x -= 1
        case 'RIGHT':
            x += 1
        case 'UP':
            y -= 1
        case 'DOWN':
            y += 1
    return (x, y)


def is_in_game_grid(coords: Tuple[int, int]) -> bool:
    return 0 <= coords[0] < s.ROWS and 0 <= coords[1] < s.COLS


def sort_objects_by_distance(agent_coords: Tuple[int, int], objects: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sorts one type of objects by Manhattan distance to the agent."""
    return sorted(objects, key=lambda object: manhatten_distance(object, agent_coords))


def manhatten_distance(coords_1, coords_2) -> int:
    return abs(coords_1[0] - coords_2[0]) + abs(coords_1[1] - coords_2[1])


def got_in_loop(agent_coords, agent_coord_history):
    loop_count = agent_coord_history.count(agent_coords)
    return loop_count > 2


def decreased_distance(old_coords, new_coords, object_coords):
    return (manhatten_distance(new_coords, object_coords)
            < manhatten_distance(old_coords, object_coords))


def increased_distance(old_coords, new_coords, object_coords):
    return (manhatten_distance(new_coords, object_coords)
            > manhatten_distance(old_coords, object_coords))


def has_destroyed_target(events):
    if e.BOMB_EXPLODED in events:
        return e.KILLED_OPPONENT in events or e.CRATE_DESTROYED in events


def get_action_idx_from_coords(agent_coords, new_coords):
    direction = (new_coords[0]-agent_coords[0], new_coords[1]-agent_coords[1])
    return MOVEMENT_DIRECTIONS.index(direction)


def get_stop_criterion_for_thing(thing: str):
    match thing:
        case 'coin':
            return is_coin
        case 'crate':
            return is_near_crate
        case 'opponent':
            return is_opponent_in_blast_range
        case 'safety':
            return is_out_of_danger
        case _:
            raise ValueError(f"Unrecognized criterion: {thing}")


def is_coin(game_state: GameState, coords: Tuple[int, int]) -> bool:
    return coords in game_state.coins


def is_near_crate(game_state: GameState, coords: Tuple[int, int]) -> bool:
    """Return True if the given coordinate is near a crate."""
    for direction in MOVEMENT_DIRECTIONS:
        new_coords = move_in_direction(coords, direction)
        if game_state.field[new_coords] == 1:
            return True
    return False


def is_opponent_in_blast_range(game_state: GameState, potential_bomb_coords: Tuple[int, int]) -> bool:
    """Return True if the player is within blast range of the enemy."""
    for opponent in game_state.others:
        if opponent[3] in game_state._get_blast_effected_coords(potential_bomb_coords):
            return True
    return False


def is_out_of_danger(game_state: GameState, new_coords: Tuple[int, int]) -> bool:
    """Returns true if the player is out of danger. Stop critereon version if is_save_step"""
    return (game_state.is_valid_movement(new_coords) and
            not game_state.is_dangerous(new_coords))