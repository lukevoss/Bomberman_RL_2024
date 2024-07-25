# TODO only give game state?

from utils import *
import events as e
import own_events as own_e


def add_own_events(old_game_state, self_action, events_src, end_of_round, agent_coord_history) -> list:

    # events = copy.deepcopy(events_src)
    events = events_src.copy()
    events.append(own_e.CONSTANT_PENALTY)

    field = old_game_state['field']
    agent_coords = old_game_state['self'][3]
    is_bomb_possible = old_game_state['self'][2]
    bombs = [xy for (xy, t) in old_game_state['bombs']]
    sorted_dangerous_bombs = sort_and_filter_out_dangerous_bombs(
        agent_coords, bombs)
    living_opponents = old_game_state['others']
    sorted_living_opponents = sort_opponents_by_distance(
        agent_coords, living_opponents)
    coins = old_game_state['coins']
    explosion_map = old_game_state['explosion_map']
    score_self = old_game_state['self'][1]
    steps_of_round = old_game_state['steps']

    if end_of_round:
        if has_won_the_game(living_opponents, score_self, events, steps_of_round):
            events.append(own_e.WON_GAME)
    else:
        events.append(own_e.SURVIVED_STEP)

    if is_dangerous(agent_coords, explosion_map, sorted_dangerous_bombs):
        if not_escaping_danger(self_action):
            events.append(own_e.NOT_ESCAPING)
        elif has_escaped_danger(agent_coords, self_action, field, living_opponents, bombs, explosion_map):
            events.append(own_e.OUT_OF_DANGER)
        elif is_escaping_danger(agent_coords, self_action, field, living_opponents, sorted_dangerous_bombs):
            events.append(own_e.ESCAPING)
        else:
            events.append(own_e.NOT_ESCAPING)

    if self_action == 'WAIT':
        # Reward the agent if waiting is necessary.
        if waited_necessarily(agent_coords, field, living_opponents, explosion_map, bombs):
            events.append(own_e.WAITED_NECESSARILY)
        else:
            events.append(own_e.WAITED_UNNECESSARILY)

    elif self_action == 'BOMB':
        if is_bomb_possible:
            can_reach_safety, is_effective = simulate_bomb(
                agent_coords, field, sorted_living_opponents, bombs)
            if not can_reach_safety:
                events.append(own_e.DUMB_BOMB_DROPPED)
            elif is_effective:
                events.append(own_e.SMART_BOMB_DROPPED)
        else:
            events.append(own_e.DUMB_BOMB_DROPPED)

    else:
        if got_in_loop(agent_coords, agent_coord_history):
            events.append(own_e.GOT_IN_LOOP)

        new_agents_coords = march_forward(agent_coords, self_action)
        if sorted_living_opponents:
            closest_opponent = sorted_living_opponents[0]
            opponent_coords = closest_opponent[3]

            if decreased_distance(agent_coords, new_agents_coords, opponent_coords):
                events.append(own_e.CLOSER_TO_PLAYERS)
            elif increased_distance(agent_coords, new_agents_coords, opponent_coords):
                events.append(own_e.AWAY_FROM_PLAYERS)

        sorted_coins = sort_objects_by_distance(agent_coords, coins)
        if sorted_coins:
            closest_coin = sorted_coins[0]
            coin_coords = closest_coin

            if decreased_distance(agent_coords, new_agents_coords, coin_coords):
                events.append(own_e.CLOSER_TO_COIN)
            elif increased_distance(agent_coords, new_agents_coords, coin_coords):
                events.append(own_e.AWAY_FROM_COIN)

        closest_crate = find_closest_crate(agent_coords, field)
        if closest_crate:
            crate_coords = closest_crate
            if decreased_distance(agent_coords, new_agents_coords, crate_coords):
                events.append(own_e.CLOSER_TO_CRATE)
            elif increased_distance(agent_coords, new_agents_coords, crate_coords):
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
