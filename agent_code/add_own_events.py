from agent_code.utils import *
import events as e
import own_events as own_e

GAME_REWARDS = {
        # Idea: If we want the action it should sum up to positive number, if not it should be negative
        # i.e.: CLOSER_TO_COIN + AWAY_FROM_PLAYER + AWAY_FROM_CRATE = 0.07, we want the agent to focus primarily on coins
        # SPECIAL EVENTS
        own_e.CONSTANT_PENALTY: -0.001,
        own_e.WON_ROUND: 0,#10,
        own_e.GOT_IN_LOOP: -0.05,
        own_e.ESCAPING: 0.2, 
        own_e.NOT_ESCAPING: -0.2,
        own_e.OUT_OF_DANGER: 1, 
        own_e.IS_IN_DANGER: 0,#-0.1,
        own_e.CLOSER_TO_COIN: 0.15,
        own_e.AWAY_FROM_COIN: -0.15,
        own_e.CLOSER_TO_CRATE: 0.04,
        own_e.AWAY_FROM_CRATE: -0.02,
        own_e.WAITED_NECESSARILY: 0.1,
        own_e.WAITED_UNNECESSARILY: -2,
        own_e.CLOSER_TO_PLAYERS: 0.08,
        own_e.AWAY_FROM_PLAYERS: -0.06, 
        own_e.SMART_BOMB_DROPPED: 0.5,
        own_e.DUMB_BOMB_DROPPED: -8,

        # DEFAULT EVENTS
        e.INVALID_ACTION: -10,
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,
        e.CRATE_DESTROYED: 0.01,
        e.COIN_FOUND: 0,
        e.COIN_COLLECTED: 3,
        e.KILLED_OPPONENT: 7,
        e.KILLED_SELF: 0,
        e.GOT_KILLED: -10,
        e.OPPONENT_ELIMINATED: 0,
    }


def add_own_events_q_learning(old_game_state, old_feature_vector, self_action, events_src, end_of_round, agent_coord_history, max_opponents_score) -> list:

    state = GameState(**old_game_state)
    events = copy.deepcopy(events_src)
    events.append(own_e.CONSTANT_PENALTY)

    agent_coords = old_game_state['self'][3]
    is_bomb_possible = old_game_state['self'][2]

    score_self = old_game_state['self'][1]

    if end_of_round:
        if score_self > max_opponents_score:
            events.append(own_e.WON_ROUND)


    direction_to_safety = old_feature_vector[0:5]
    if any(x == 1 for x in direction_to_safety):
        action_idx = np.argmax(direction_to_safety)
        if self_action == ACTIONS[action_idx]:
            events.append(own_e.ESCAPING)
        else:
            events.append(own_e.NOT_ESCAPING)

    if state.is_dangerous(agent_coords):
        if state.has_escaped_danger(self_action):
            events.append(own_e.OUT_OF_DANGER)

    new_agents_coords = march_forward(agent_coords,self_action)
    if state.is_dangerous(new_agents_coords):
        events.append(own_e.IS_IN_DANGER)
    

    if self_action == 'WAIT':
        if not state.is_dangerous(agent_coords) and state.is_danger_all_around(agent_coords):
            events.append(own_e.WAITED_NECESSARILY)
        else:
            events.append(own_e.WAITED_UNNECESSARILY)

    elif self_action == 'BOMB':
        if is_bomb_possible:
            can_reach_safety, is_effective = state.simulate_own_bomb()
            if can_reach_safety and is_effective:
                events.append(own_e.SMART_BOMB_DROPPED)
            else:
                events.append(own_e.DUMB_BOMB_DROPPED)

    else:
        if got_in_loop(agent_coords, self_action, agent_coord_history):
            events.append(own_e.GOT_IN_LOOP)

        new_agents_coords = march_forward(agent_coords, self_action)

        # Ignore invalid actions, that occured because other player moved to this field
        if e.INVALID_ACTION in events and state.is_valid_movement(new_agents_coords):
            events.remove(e.INVALID_ACTION)

        # Is closer to player?
        direction_to_opponent = old_feature_vector[10:14]
        if any(x == 1 for x in direction_to_opponent):
            action_idx = np.argmax(direction_to_opponent)
            if self_action == ACTIONS[action_idx]:
                events.append(own_e.CLOSER_TO_PLAYERS)
            else:
                events.append(own_e.AWAY_FROM_PLAYERS)

        # Is closer to coin?
        direction_to_coin = old_feature_vector[15:19]
        if any(x == 1 for x in direction_to_coin):
            action_idx = np.argmax(direction_to_coin)
            if self_action == ACTIONS[action_idx]:
                events.append(own_e.CLOSER_TO_COIN)
            else:
                events.append(own_e.AWAY_FROM_COIN)
        
        # Is closer to crate?
        direction_to_crate = old_feature_vector[5:9]
        if any(x == 1 for x in direction_to_crate):
            action_idx = np.argmax(direction_to_crate)
            if self_action == ACTIONS[action_idx]:
                events.append(own_e.CLOSER_TO_CRATE)
            else:
                events.append(own_e.AWAY_FROM_CRATE)


    return events

def add_own_events_ppo(old_game_state, self_action, events_src, end_of_round, agent_coord_history, max_opponents_score) -> list:

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



    if state.is_dangerous(agent_coords):
        if state.has_escaped_danger(self_action):
            events.append(own_e.OUT_OF_DANGER)
        elif state.is_escaping_danger(self_action, sorted_dangerous_bombs):
            events.append(own_e.ESCAPING)
        else:
            events.append(own_e.NOT_ESCAPING)

    new_agents_coords = march_forward(agent_coords,self_action)
    if state.is_dangerous(new_agents_coords):
        events.append(own_e.IS_IN_DANGER)
    

    if self_action == 'WAIT':
        if not state.is_dangerous(agent_coords) and state.is_danger_all_around(agent_coords):
            events.append(own_e.WAITED_NECESSARILY)
        else:
            events.append(own_e.WAITED_UNNECESSARILY)

    elif self_action == 'BOMB':
        if is_bomb_possible:
            can_reach_safety, is_effective = state.simulate_own_bomb()
            if can_reach_safety and is_effective:
                events.append(own_e.SMART_BOMB_DROPPED)
            else:
                events.append(own_e.DUMB_BOMB_DROPPED)

    else:
        if got_in_loop(agent_coords, self_action, agent_coord_history):
            events.append(own_e.GOT_IN_LOOP)

        new_agents_coords = march_forward(agent_coords, self_action)

        # Ignore invalid actions, that occured because other player moved to this field
        # TODO: Also with PPO necessary?
        if e.INVALID_ACTION in events and state.is_valid_movement(new_agents_coords):
            events.remove(e.INVALID_ACTION)


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


    return events



def reward_from_events(events: List[str]) -> int:

    reward_sum = 0
    for event in events:
        if event in GAME_REWARDS:
            reward_sum += GAME_REWARDS[event]
    return reward_sum
