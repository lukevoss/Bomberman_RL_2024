from agent_code.utils import *
import events as e
import own_events as own_e


def add_own_events(old_game_state, self_action, events_src, end_of_round, agent_coord_history, max_opponents_score) -> list:

    state = GameState(**old_game_state)
    events = copy.deepcopy(events_src)
    events.append(own_e.CONSTANT_PENALTY)

    agent_coords = old_game_state['self'][3]
    is_bomb_possible = old_game_state['self'][2]
    sorted_dangerous_bombs = state.sort_and_filter_out_dangerous_bombs(
        agent_coords)
    sorted_opponents = state.get_opponents_sorted_by_distance(agent_coords)

    score_self = old_game_state['self'][1]

    if end_of_round:
        if score_self > max_opponents_score:
            events.append(own_e.WON_ROUND)
    else:
        events.append(own_e.SURVIVED_STEP)

    if state.is_dangerous(agent_coords):
        if state.not_escaping_danger(self_action):
            events.append(own_e.NOT_ESCAPING)
        elif state.has_escaped_danger(self_action):
            events.append(own_e.OUT_OF_DANGER)
        elif state.is_escaping_danger(self_action, sorted_dangerous_bombs):
            events.append(own_e.ESCAPING)
        else:
            events.append(own_e.NOT_ESCAPING)

    if self_action == 'WAIT':
        if state.waited_necessarily():
            events.append(own_e.WAITED_NECESSARILY)
        else:
            events.append(own_e.WAITED_UNNECESSARILY)

    elif self_action == 'BOMB':
        if is_bomb_possible:
            can_reach_safety, is_effective = state.simulate_own_bomb()
            if not can_reach_safety:
                events.append(own_e.DUMB_BOMB_DROPPED)
            elif is_effective:
                events.append(own_e.SMART_BOMB_DROPPED)
        else:
            events.append(own_e.DUMB_BOMB_DROPPED)

    else:
        if got_in_loop(agent_coords, self_action, agent_coord_history):
            events.append(own_e.GOT_IN_LOOP)

        new_agents_coords = march_forward(agent_coords, self_action)
        if sorted_opponents:
            closest_opponent = sorted_opponents[0]
            closest_opponent_coords = closest_opponent[3]

            if decreased_distance(agent_coords, new_agents_coords, closest_opponent_coords):
                events.append(own_e.CLOSER_TO_PLAYERS)
            elif increased_distance(agent_coords, new_agents_coords, closest_opponent_coords):
                events.append(own_e.AWAY_FROM_PLAYERS)

        sorted_coins = state.get_coins_sorted_by_distance(agent_coords)
        if sorted_coins:
            closest_coin = sorted_coins[0]
            coin_coords = closest_coin

            if decreased_distance(agent_coords, new_agents_coords, coin_coords):
                events.append(own_e.CLOSER_TO_COIN)
            elif increased_distance(agent_coords, new_agents_coords, coin_coords):
                events.append(own_e.AWAY_FROM_COIN)

        closest_crate = state.find_closest_crate(agent_coords)
        if closest_crate:
            crate_coords = closest_crate
            if decreased_distance(agent_coords, new_agents_coords, crate_coords):
                events.append(own_e.CLOSER_TO_CRATE)
            elif increased_distance(agent_coords, new_agents_coords, crate_coords):
                events.append(own_e.AWAY_FROM_CRATE)

    if e.BOMB_EXPLODED in events:
        if has_destroyed_target(events):
            events.append(own_e.DESTROY_TARGET)
        else:
            events.append(own_e.MISSED_TARGET)

    if e.CRATE_DESTROYED in events:
        number_of_crates_destroyed = events.count(e.CRATE_DESTROYED)
        if number_of_crates_destroyed > 5:
            events.append(own_e.BOMBED_5_PLUS_CRATES)
        elif number_of_crates_destroyed > 2:
            events.append(own_e.BOMBED_3_TO_5_CRATES)
        else:
            events.append(own_e.BOMBED_1_TO_2_CRATES)

    return events
