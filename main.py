#!/usr/bin/env python3
# The parts that aren't ready yet are commented out
import wcst
import models.deep_q as deep_q
import models.vanilla_q as vanilla_q
import models.sarsa as sarsa
from logbook import logbook

import torch
import torch.nn as nn
import numpy as np
import argparse
import inspect


# DONE low prio: move agents to subfolder, get all name strings from there
LIST_OF_AGENTS = ["vanilla_q", "sarsa", "deep_q"]
LIST_OF_POLICIES = ["value_max", "epsilon_greedy"]

DEFAULT_NETWORK = nn.Sequential(
    nn.Linear(5, 50), nn.ReLU(), nn.Linear(50, 50), nn.ReLU(), nn.Linear(50, 4)
)


def cli_args():
    """Parse arguments when running from command line.
    Returns a dictionary with structure { option : value }."""

    parser = argparse.ArgumentParser(
        description="Runs the Wisconsin Card Sorting Task (WCST) using the specified "
        "reinforcement learning (RL) agent. Default: vanilla_q"
    )

    # This is positional; last argument provided.
    parser.add_argument(
        "agent",
        nargs="?",
        default="vanilla_q",
        choices=LIST_OF_AGENTS,
        help="Agent to run the task. Allowed values: " + ", ".join(LIST_OF_AGENTS),
        metavar="",
    )

    parser.add_argument(
        "-e",
        "--episodes",
        default=100,
        help="Number of game episodes to run the agent. Default: 100",
    )

    parser.add_argument(
        "-p",
        "--policy",
        default="epsilon_greedy",
        choices=LIST_OF_POLICIES,
        help="Policy to use for value-based agent. Default: None (random). Allowed values: "
        + ", ".join(LIST_OF_POLICIES),
        metavar="",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output format of generated data. Default: None",
    )

    parser.add_argument(
        "-g",
        "--graphical_render",
        default=False,
        help="Graphical rendering. Default: False",
    )

    # returns dictionary of command line arguments
    # all of this is just for convenience
    return vars(parser.parse_args())


def main(agent="vanilla_q", **kwargs):

    # create and init environment
    env = wcst.WCST()
    env.reset()

    # set some parameters; agent must be hardcoded option
    agent_name = agent if agent in LIST_OF_AGENTS else None
    episodes = int(kwargs['episodes'])
    output = kwargs['output']

    # dynamically obtain accepted keywords from agent
    agent_keys = inspect.signature(eval(agent_name).Agent).parameters.keys()
    agent_args = {key: kwargs[key] for key in kwargs.keys() & agent_keys}

    # set default network if not otherwise specified
    if agent_name == "deep_q" and "q_network" not in agent_args:
        agent_args['q_network'] = DEFAULT_NETWORK

    # create agent
    agent = eval(agent_name).Agent(env, **agent_args)

    if agent_name == "vanilla_q":
        metadata = f"vani_q_ep_{episodes}_mem_{agent._streak_memory}_eps_{agent._epsilon}_step_{agent._step_size}"
    elif agent_name == "sarsa":
        metadata = f"sarsa_ep_{episodes}_mem_{agent._streak_memory}_eps_{agent._epsilon}_step_{agent._step_size}"
    elif agent_name == "deep_q":
        metadata = f"deep_q_ep_{episodes}_bs_{agent._batch_size}_lr_{agent._learning_rate}"

    # create uniform logbook
    log = logbook(agent, metadata)

    returns = agent.run(
        num_episodes=episodes,
        logbook=log
        )

    print(f"Return per episode: {returns}")

    if output == "csv":
        log.to_csv()

    return np.sum(returns)

if __name__ == "__main__":
    # Wrapper to parse cmd line args.
    # If you want you can run main() from elsewhere,
    # and specify your arguments there.
    cli_args = cli_args()
    main(**cli_args)
