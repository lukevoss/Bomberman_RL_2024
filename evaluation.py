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


def evaluate_performance(results, base_dir):
    """
    Called when game is over, therefore we can safely also access the game.log file for some additional metrics.
    
    Returns nothing, but saves the performance metrics in a .json file as well as the plots in a .png file in a dedicated folder.
    """
    # Step 1: Get the game log file content
    log_file_content = get_game_log_file_content(os.path.join(base_dir, 'logs/game.log'))
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
    current_step = None
    last_step_of_player = {}

    # Regular expressions to match the log lines
    step_pattern = re.compile(r"STARTING STEP (\d+)")
    action_pattern = re.compile(r"Agent <([^>]+)> chose action (\w+) in ([\d.]+)s\.")
    time_pattern = re.compile(r"Agent <([^>]+)> stayed within acceptable think time\.")
    bomb_pattern = re.compile(r"Agent <([^>]+)> drops bomb at \(np.int64\((\d+)\), np.int64\((\d+)\)\)")
    explosion_pattern = re.compile(r"bomb at \(np.int64\((\d+)\), np.int64\((\d+)\)\) explodes")
    location_pattern = re.compile(r"Agent <([^>]+)> is now at \(np.int64\((\d+)\), np.int64\((\d+)\)\)")

    # Parse the log line by line
    for line in log_data.strip().split("\n"):
        # Check if the line indicates the start of a new step
        step_match = step_pattern.search(line)
        if step_match:
            current_step = int(step_match.group(1))
            continue

        # Check if the line contains an agent's action
        action_match = action_pattern.search(line)
        if action_match:
            agent_name = action_match.group(1)
            action = action_match.group(2)
            time_taken = float(action_match.group(3))
            steps[current_step][agent_name] = {'action': action, 'time_taken': time_taken}
            last_step_of_player[agent_name] = current_step

        # Check if the line indicates that the agent stayed within the allowed time
        time_match = time_pattern.search(line)
        if time_match:
            agent_name = time_match.group(1)
            if agent_name in steps[current_step]:
                steps[current_step][agent_name]['within_time'] = True
                last_step_of_player[agent_name] = current_step
        
        # Check if the line contains a bomb drop action
        bomb_match = bomb_pattern.search(line)
        if bomb_match:
            agent_name = bomb_match.group(1)
            steps[current_step][agent_name]['bomb_dropped'] = True
            steps[current_step][agent_name]['bomb_position'] = [int(bomb_match.group(2)), int(bomb_match.group(3))]
            last_step_of_player[agent_name] = current_step

        # Check if the line contains a bomb explosion
        explosion_match = explosion_pattern.search(line)
        if explosion_match:
            if 'explosions' not in steps[current_step]:
                steps[current_step]['explosions'] = []
            steps[current_step]['explosions'].append([int(explosion_match.group(1)), int(explosion_match.group(2))])
            # Do not update the last step of the player here, as the player might already be dead
        
        # Check if the line contains the location of an agent
        location_match = location_pattern.search(line)
        if location_match:
            agent_name = location_match.group(1)
            steps[current_step][agent_name]['location'] = [int(location_match.group(2)), int(location_match.group(3))]
            last_step_of_player[agent_name] = current_step

    return dict(steps), last_step_of_player


def build_metrics_from_game_log(parsed_log, last_steps):
    """
    Build the metrics from the parsed game log.
    """
    build_time_metrics(parsed_log, last_steps)
    build_action_metrics(parsed_log, last_steps)
    build_bomb_metrics(parsed_log, last_steps)


def build_time_metrics(parsed_log, last_steps):
    """
    Build the time metrics from the parsed game log.
    """
    agents = {}
    for step, events in parsed_log.items():
        for agent_name, event in events.items():
            if agent_name == 'explosions':
                continue
            if agent_name not in agents:
                agents[agent_name] = {}
            if 'time_taken' in event:
                agents[agent_name]['total_time'] = agents[agent_name].get('total_time', 0) + event['time_taken']
            if 'within_time' in event:
                agents[agent_name]['not_within_time'] = agents[agent_name].get('not_within_time', 0) + (event['within_time'] == False)
    for agent_name in agents:
        agents[agent_name]['average_time'] = agents[agent_name]['total_time'] / last_steps[agent_name]
    pprint(agents)
    

def build_action_metrics(parsed_log, last_steps):
    """
    Build the action metrics from the parsed game log.
    """
    agents = {}
    for step, events in parsed_log.items():
        for agent_name, event in events.items():
            if agent_name == 'explosions':
                continue
            if agent_name not in agents:
                agents[agent_name] = {}
            if 'action' in event:
                agents[agent_name][event['action']] = agents[agent_name].get(event['action'], 0) + 1
    for agent_name in agents:
        for action in agents[agent_name]:
            agents[agent_name][action] /= last_steps[agent_name]
    pprint(agents)


def build_bomb_metrics(parsed_log, last_steps):
    """
    Build the bomb metrics from the parsed game log.
    """
    bombs = np.zeros((15, 15))
    amount_of_bombs = 0
    for step, events in parsed_log.items():
        if 'explosions' in events:
            for explosion in events['explosions']:
                bombs[explosion[0] - 1, explosion[1] - 1] += 1
                amount_of_bombs += 1
    bombs /= amount_of_bombs
    pprint(bombs)
    
    # Plot the heatmap
    plt.imshow(bombs, cmap='hot', interpolation='nearest')
    plt.savefig('results/bombs.png')
    

evaluate_performance(None, '.')



    # performance = {}
    # for metric in METRICS:
    #     performance[metric] = get_performance_for_metric(metric, results, average=True)
    # plot = plot_results(performance)
    # return performance, plot









# def get_performance_for_metric(metric, results, average=False):
#     """
#     """
#     average_performance = {}
#     for agent_name, agent_results in results["by_agent"].items():
#         if metric in agent_results:
#             if average:
#                 average_performance[agent_name] = agent_results[metric] / agent_results["rounds"]
#             else:
#                 average_performance[agent_name] = agent_results[metric]
#         else:
#             average_performance[agent_name] = 0
#     return average_performance


# def plot_results(performance):
#     figure = plt.figure(figsize=(8, 20))
#     for metric in METRICS[:-1]:
#         plot_metric(figure, metric, performance)
#     plot_metric(figure, METRICS[-1], performance, print_axis_labels=True)
#     figure.tight_layout()
#     return figure


# def plot_metric(figure, metric, performance, print_axis_labels=False):
#     """
#     Plot the performance of the agents for a given metric into a figure.
#     """
#     ax = figure.add_subplot(len(METRICS), 1, METRICS.index(metric) + 1)
#     ax.set_title(metric)
#     ax.bar(performance[metric].keys(), performance[metric].values())
#     ax.set_ylabel(metric)
#     if print_axis_labels:
#         ax.set_xlabel("Agent")
#         ax.set_xticklabels(performance[metric].keys(), rotation=45)
#     else:
#         ax.set_xticklabels([])
#     ax.grid(axis="y")
