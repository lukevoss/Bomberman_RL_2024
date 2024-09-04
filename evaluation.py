"""
This file is used to evaluate the performance of the agents.
It is called from the main.py file with the "--save-stats" argument.
"""

import os
import re
import numpy as np
from pprint import pprint
from collections import defaultdict
from matplotlib import pyplot as plt


def evaluate_performance(results, base_dir, log_file_name='game.log'):
    """
    Called when game is over, therefore we can safely also access the game.log file for some additional metrics.
    
    Returns nothing, but saves the performance metrics in a .json file as well as the plots in a .png file in a dedicated folder.
    """
    # Step 1: Get the game log file content
    log_file_content = get_game_log_file_content(os.path.join(base_dir, log_file_name))
    # Step 2: Parse the game log file
    parsed_log, last_steps = parse_game_log(log_file_content)
    # Step 3: Build the metrics from the parsed game log
    own_metrics = build_metrics_from_game_log(parsed_log, last_steps)
    # Step 4: Save the performance metrics in a .json file
    pass

def get_game_log_file_content(log_file):
    """
    Returns the content of the game log file.
    """
    with open(log_file, "r") as f:
        lines = f.readlines()
    return "".join(lines)

    
def parse_game_log(log_data):
    """
    Parse the game log file and return the metrics.
    """
    # Initialize the variables
    steps = defaultdict(dict)
    current_round = None
    current_step = None
    last_step_of_player = {}

    # Regular expressions to match the log lines
    round_pattern = re.compile(r"STARTING ROUND #(\d+)")
    step_pattern = re.compile(r"STARTING STEP (\d+)")
    action_pattern = re.compile(r"Agent <([^>]+)> chose action (\w+) in ([\d.]+)s\.")
    time_pattern = re.compile(r"Agent <([^>]+)> stayed within acceptable think time\.")
    bomb_drop_pattern = re.compile(r"Agent <([^>]+)> drops bomb at \(np.int64\((\d+)\), np.int64\((\d+)\)\)")
    explosion_pattern = re.compile(r"bomb at \(np.int64\((\d+)\), np.int64\((\d+)\)\) explodes")
    location_pattern = re.compile(r"Agent <([^>]+)> is now at \(np.int64\((\d+)\), np.int64\((\d+)\)\)")
    coin_pickup_pattern = re.compile(r"Agent <([^>]+)> picked up coin at \(np.int64\((\d+)\), np.int64\((\d+)\)\)")
    coin_found_pattern = re.compile(r"Coin found at \(np.int64\((\d+)\), np.int64\((\d+)\)\)")
    bomb_suicide_pattern = re.compile(r"Agent <([^>]+)> blown up by own bomb")
    bomb_kill_pattern = re.compile(r"Agent <([^>]+)> blown up by agent <([^>]+)>'s bomb")

    # Parse the log line by line
    for line in log_data.strip().split("\n"):
        # Check if the line indicates the start of a new round
        round_match = round_pattern.search(line)
        if round_match:
            current_round = int(round_match.group(1))
            steps[current_round] = {}
            last_step_of_player[current_round] = {}
            continue
        # Check if the line indicates the start of a new step
        step_match = step_pattern.search(line)
        if step_match:
            current_step = int(step_match.group(1))
            steps[current_round][current_step] = {}
            continue
        

        # Check if the line contains an agent's action
        action_match = action_pattern.search(line)
        if action_match:
            agent_name = action_match.group(1)
            action = action_match.group(2)
            time_taken = float(action_match.group(3))
            steps[current_round][current_step][agent_name] = {'action': action, 'time_taken': time_taken}

        # Check if the line indicates that the agent stayed within the allowed time
        time_match = time_pattern.search(line)
        if time_match:
            agent_name = time_match.group(1)
            if agent_name in steps[current_round][current_step]:
                steps[current_round][current_step][agent_name]['within_time'] = True
        
        # Check if the line contains a bomb drop action
        bomb_drop_match = bomb_drop_pattern.search(line)
        if bomb_drop_match:
            agent_name = bomb_drop_match.group(1)
            steps[current_round][current_step][agent_name]['bomb_dropped'] = True
            steps[current_round][current_step][agent_name]['bomb_position'] = [int(bomb_drop_match.group(2)), int(bomb_drop_match.group(3))]

        # Check if the line contains a bomb explosion
        explosion_match = explosion_pattern.search(line)
        if explosion_match:
            if 'explosions' not in steps[current_round][current_step]:
                steps[current_round][current_step]['explosions'] = []
            steps[current_round][current_step]['explosions'].append([int(explosion_match.group(1)), int(explosion_match.group(2))])
        
        # Check if the line contains the location of an agent
        location_match = location_pattern.search(line)
        if location_match:
            agent_name = location_match.group(1)
            steps[current_round][current_step][agent_name]['location'] = [int(location_match.group(2)), int(location_match.group(3))]
            last_step_of_player[current_round][agent_name] = current_step
            
        # Check if the line contains a coin pickup action
        coin_pickup_match = coin_pickup_pattern.search(line)
        if coin_pickup_match:
            print("Coin pickup")
            agent_name = coin_pickup_match.group(1)
            steps[current_round][current_step][agent_name]['collected_coin_position'] = [int(coin_pickup_match.group(2)), int(coin_pickup_match.group(3))]
            last_step_of_player[current_round][agent_name] = current_step
            
        # Check if the line contains a coin found action
        coin_found_match = coin_found_pattern.search(line)
        if coin_found_match:
            steps[current_round][current_step]['coin_found'] = [int(coin_found_match.group(1)), int(coin_found_match.group(2))]
            
        # Check if the line contains a bomb suicide action
        bomb_suicide_match = bomb_suicide_pattern.search(line)
        if bomb_suicide_match:
            agent_name = bomb_suicide_match.group(1)
            steps[current_round][current_step][agent_name]['bomb_suicide'] = True
            last_step_of_player[current_round][agent_name] = current_step
        
        # Check if the line contains a bomb kill action
        bomb_kill_match = bomb_kill_pattern.search(line)
        if bomb_kill_match:
            agent_name_killed = bomb_kill_match.group(1)
            agent_name_killer = bomb_kill_match.group(2)
            steps[current_round][current_step][agent_name_killed]['killed_by'] = agent_name_killer
            steps[current_round][current_step][agent_name_killer]['killed'] = agent_name_killed

    return dict(steps), last_step_of_player


def build_metrics_from_game_log(parsed_log, last_steps):
    """
    Build the metrics from the parsed game log.
    """
    # build_time_metrics(parsed_log, last_steps)
    # build_action_metrics(parsed_log, last_steps)
    # build_bomb_metrics(parsed_log, last_steps)
    # build_location_metrics(parsed_log, last_steps)
    # build_coin_metrics(parsed_log, last_steps)
    build_kill_metrics(parsed_log, last_steps)


def build_time_metrics(parsed_log, last_steps):
    """
    Build the time metrics from the parsed game log.
    """
    agents = { round: {} for round in range(1, len(parsed_log.keys()) + 1) }
    for round, steps in parsed_log.items():
        for step, events in steps.items():
            for agent_name, event in events.items():
                if agent_name == 'explosions':
                    continue
                if agent_name not in agents:
                    agents[round][agent_name] = {}
                if 'time_taken' in event:
                    agents[round][agent_name]['total_time'] = agents[round][agent_name].get('total_time', 0) + event['time_taken']
                if 'within_time' in event:
                    agents[round][agent_name]['not_within_time'] = agents[round][agent_name].get('not_within_time', 0) + (event['within_time'] == False)
        for agent_name in agents[round]:
            agents[round][agent_name]['average_time'] = agents[round][agent_name]['total_time'] / last_steps[round][agent_name]
    pprint(agents)


def build_action_metrics(parsed_log, last_steps):
    """
    Build the action metrics from the parsed game log.
    """
    agents = { round: {} for round in range(1, len(parsed_log.keys()) + 1) }
    for round, steps in parsed_log.items():
        for step, events in steps.items():
            for agent_name, event in events.items():
                if agent_name == 'explosions':
                    continue
                if agent_name not in agents[round]:
                    agents[round][agent_name] = {}
                if 'action' in event:
                    agents[round][agent_name][event['action']] = agents[round][agent_name].get(event['action'], 0) + 1
        for agent_name in agents[round]:
            for action in agents[round][agent_name]:
                agents[round][agent_name][action] /= last_steps[round][agent_name]
    pprint(agents)


def build_bomb_metrics(parsed_log, last_steps):
    """
    Build the bomb metrics from the parsed game log.
    """
    rounds = len(parsed_log.keys())
    bombs = [np.zeros((15, 15)) for _ in range(rounds)]
    for round, steps in parsed_log.items():
        amount_of_bombs = 0
        for step, events in steps.items():
            if 'explosions' in events:
                for explosion in events['explosions']:
                    bombs[round - 1][explosion[0] - 1, explosion[1] - 1] += 1
                    amount_of_bombs += 1
        bombs[round - 1] /= amount_of_bombs
        pprint(bombs[round - 1])
        
        # Plot the heatmap
        plt.imshow(bombs[round - 1], cmap='hot', interpolation='nearest')
        plt.savefig(f'results/bombs_{round}.png')
    
    
def build_location_metrics(parsed_log, last_steps):
    """
    Build the location metrics from the parsed game log.
    """
    unique_locations_per_agent_per_round = { round: {} for round in range(1, len(parsed_log.keys()) + 1) }
    locations = { round: {} for round in range(1, len(parsed_log.keys()) + 1) }
    start_locations = { round: {} for round in range(1, len(parsed_log.keys()) + 1) }
    stop_locations = { round: {} for round in range(1, len(parsed_log.keys()) + 1) }
    for round, steps in parsed_log.items():
        for step, events in steps.items():
            for agent_name, event in events.items():
                # When only one is left, we don't need to track the location anymore
                if len(events) == 1:
                    break
                if agent_name == 'explosions':
                    continue
                if 'location' in event:
                    if agent_name not in locations[round]:
                        locations[round][agent_name] = np.zeros((15, 15))
                        start_locations[round][agent_name] = event['location']
                        stop_locations[round][agent_name] = event['location']
                        unique_locations_per_agent_per_round[round][agent_name] = set()
                    # get stop location
                    if step == last_steps[round][agent_name]:
                        stop_locations[round][agent_name] = event['location']
                    locations[round][agent_name][event['location'][0] - 1, event['location'][1] - 1] += 1
                    unique_locations_per_agent_per_round[round][agent_name].add((event['location'][0], event['location'][1]))
        for agent_name in locations[round]:
            locations[round][agent_name] /= last_steps[round][agent_name]
                
        
        # Plot the heatmaps in a 2x2 layout
        figure, axes = plt.subplots(len(locations[round]) // 2, len(locations[round]) // 2, figsize=(12, 12))
        for i, (agent_name, location) in enumerate(locations[round].items()):
            row = i // (len(locations[round]) // 2)
            col = i % (len(locations[round]) // 2)
            axes[row, col].imshow(location, cmap='hot', interpolation='nearest')
            # Mark the starting location for each agent
            axes[row, col].scatter(start_locations[round][agent_name][1] - 1, start_locations[round][agent_name][0] - 1, c='green', s=200)
            # Mark the stopping location for each agent
            axes[row, col].scatter(stop_locations[round][agent_name][1] - 1, stop_locations[round][agent_name][0] - 1, c='blue', s=200)
            axes[row, col].set_title(agent_name + "'s locations in " + str(last_steps[round][agent_name]) + " steps")
        cbar = figure.colorbar(axes[len(locations[round]) // 2 - 1][len(locations[round]) // 2 - 1].imshow(location, cmap='hot', interpolation='nearest'), ax=axes, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Percentage (%)')
        # Add legend to figure
        figure.legend(['Start', 'Stop'], loc='upper right')
        plt.savefig(f'results/locations_{round}.png')
        
    # Print the unique locations
    for round, agents in unique_locations_per_agent_per_round.items():
        print(f"Round {round}:")
        for agent_name, unique_locations in agents.items():
            print(f"{agent_name}: {len(unique_locations)} unique locations in {last_steps[round][agent_name]} steps ({len(unique_locations) / last_steps[round][agent_name] * 100:.2f}%)")
        print()
        
    # Print average unique locations per agent
    unique_locations_per_agent = {agent_name: [0, 0] for agent_name in unique_locations_per_agent_per_round[1]}
    
    for round, agents in unique_locations_per_agent_per_round.items():
        for agent_name, unique_locations in agents.items():
            unique_locations_per_agent[agent_name][0] += len(unique_locations)
            unique_locations_per_agent[agent_name][1] += last_steps[round][agent_name]
            
    for agent_name, (unique_locations, steps) in unique_locations_per_agent.items():
        print(f"{agent_name}: {unique_locations} unique locations in {steps} steps ({unique_locations / steps * 100:.2f}%)")


def build_coin_metrics(parsed_log, last_steps):
    """
    Build the coin metrics from the parsed game log.
    """
    agents = { round: {} for round in range(1, len(parsed_log.keys()) + 1) }
    for round, steps in parsed_log.items():
        for step, events in steps.items():
            for agent_name, event in events.items():
                if agent_name in ['explosions', 'coin_found']:
                    continue
                if agent_name not in agents[round]:
                    agents[round][agent_name] = 0
                if 'collected_coin_position' in event:
                    agents[round][agent_name] += 1
                    
        agents[round]['total_coins'] = sum([1 for event in steps.values() if 'coin_found' in event])
    pprint(agents)
    
    # Build metrics for the coins
    # How long did it take to find the coins?
    coins = { round: {} for round in range(1, len(parsed_log.keys()) + 1) }
    for round, steps in parsed_log.items():
        for step, events in steps.items():
            if 'coin_found' in events:
                coins[round][step] = events['coin_found']
            for agent_name, event in events.items():
                if 'collected_coin_position' in event:
                    coins[round][step] = event['collected_coin_position']
                    coins[round][step].append(agent_name)
                    break
    pprint(coins)
    
    # Go over all coins and calculate the difference between the coin being found and the coin being picked up on average and per agent
    time_to_coin = { round: {} for round in range(1, len(parsed_log.keys()) + 1) }
    for round, coins_per_round in coins.items():
        for step, coin_info in coins_per_round.items():
            coin_pos_str = str(coin_info[0]) + '-' + str(coin_info[1])
            if coin_pos_str not in time_to_coin[round]:
                time_to_coin[round][coin_pos_str] = [step, None]
            else:
                time_to_coin[round][coin_pos_str][1] = step
                time_to_coin[round][coin_pos_str].append(coin_info[2])
    for round, coins_per_round in time_to_coin.items():
        for coin_pos_str, steps in coins_per_round.items():
            if steps[1] is not None:
                time_to_coin[round][coin_pos_str].append(steps[1] - steps[0])
            else:
                time_to_coin[round][coin_pos_str].append(None)
    pprint(time_to_coin)
    
    # Get average steps to coin per round per agent
    average_steps_to_coin = { round: {} for round in range(1, len(parsed_log.keys()) + 1) }
    for round, coins_per_round in time_to_coin.items():
        for coin_pos_str, steps in coins_per_round.items():
            if steps[2] == None:
                continue
            if steps[2] not in average_steps_to_coin[round]:
                average_steps_to_coin[round][steps[2]] = []
            if len(steps) == 4:
                average_steps_to_coin[round][steps[2]].append(steps[3])
    for round, coins_per_round in average_steps_to_coin.items():
        for agent_name, steps in coins_per_round.items():
            average_steps_to_coin[round][agent_name] = np.mean(steps)
            
    pprint(average_steps_to_coin)
    
    # Get average over all rounds per agent
    average_steps_to_coin_overall = {}
    for round, coins_per_round in average_steps_to_coin.items():
        for agent_name, steps in coins_per_round.items():
            if agent_name not in average_steps_to_coin_overall:
                average_steps_to_coin_overall[agent_name] = []
            average_steps_to_coin_overall[agent_name].append(steps)
    for agent_name, steps in average_steps_to_coin_overall.items():
        average_steps_to_coin_overall[agent_name] = np.mean(steps)
    pprint(average_steps_to_coin_overall)
    
    # Plot the average steps to coin per agent
    plt.figure()
    plt.bar(average_steps_to_coin_overall.keys(), average_steps_to_coin_overall.values())
    plt.xlabel('Agent')
    plt.ylabel('Average steps to coin')
    plt.title('Average steps to coin per agent')
    
    plt.savefig('results/average_steps_to_coin.png')



def build_kill_metrics(parsed_log, last_steps):
    pass


evaluate_performance(None, 'logs', 'game.log')
