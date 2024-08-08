"""
This file is used to evaluate the performance of the agents.
It is called from the main.py file with the "--save-stats" argument.
"""

import os
from matplotlib import pyplot as plt


METRICS = [
    "bombs",
    "coins",
    "crates",
    "invalid",
    "kills",
    "moves",
    "score",
    "steps",
    "suicides", 
    "time"
]


def evaluate_performance(results):
    performance = {}
    for metric in METRICS:
        performance[metric] = get_performance_for_metric(metric, results, average=True)
    plot = plot_results(performance)
    return performance, plot


def get_performance_for_metric(metric, results, average=False):
    """
    """
    average_performance = {}
    for agent_name, agent_results in results["by_agent"].items():
        if metric in agent_results:
            if average:
                average_performance[agent_name] = agent_results[metric] / agent_results["rounds"]
            else:
                average_performance[agent_name] = agent_results[metric]
        else:
            average_performance[agent_name] = 0
    return average_performance


def plot_results(performance):
    figure = plt.figure(figsize=(8, 20))
    for metric in METRICS[:-1]:
        plot_metric(figure, metric, performance)
    plot_metric(figure, METRICS[-1], performance, print_axis_labels=True)
    figure.tight_layout()
    return figure


def plot_metric(figure, metric, performance, print_axis_labels=False):
    """
    Plot the performance of the agents for a given metric into a figure.
    """
    ax = figure.add_subplot(len(METRICS), 1, METRICS.index(metric) + 1)
    ax.set_title(metric)
    ax.bar(performance[metric].keys(), performance[metric].values())
    ax.set_ylabel(metric)
    if print_axis_labels:
        ax.set_xlabel("Agent")
        ax.set_xticklabels(performance[metric].keys(), rotation=45)
    else:
        ax.set_xticklabels([])
    ax.grid(axis="y")
