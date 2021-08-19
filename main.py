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
        "-s",
        "--steps",
        default=None,
        help="Number of steps to run the agent through the environment. Default: No limit.",
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


def main(
    agent="vanilla_q",
    policy="epsilon_greedy",
    epsilon=0.05,
    memory=6,  # general
    num_episodes=100,
    num_steps=None,
    step_size=0.1,  # vanilla_q specific
    learning_rate=5e-3,
    batch_size=10,  # deep_q specific
    q_network=DEFAULT_NETWORK,
    output=None,
    graphical_render=False,
):
    # create and init environment
    env = wcst.WCST()
    env.reset()

    # specify agent
    if agent == "vanilla_q":

        agent = vanilla_q.Agent(
            env=env,
            policy=policy,
            memory=memory,
            epsilon=epsilon,
            step_size=step_size,
        )

        # set some identifiers
        metadata = (
            f"vani_q_ep_{num_episodes}_mem_{memory}_eps_{epsilon}_step_{step_size}"
        )
        # create uniform logbook
        log = logbook(agent, metadata)

        returns = vanilla_q.run(
            env=env,
            agent=agent,
            num_episodes=num_episodes,
            num_steps=num_steps,
            logbook=log,
        )

    elif agent == "deep_q":
        q_network = DEFAULT_NETWORK

        agent = deep_q.Agent(
            env=env,
            q_network=q_network,
            replay_capacity=100_000,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        # set some identifiers
        metadata = f"deep_q_ep_{num_episodes}_bs_{batch_size}_lr_{learning_rate}"

        # create uniform logbook
        log = logbook(agent, metadata)

        returns = deep_q.run(
            environment=env,
            agent=agent,
            num_episodes=num_episodes,
            logger_time_delta=1.0,
            logbook=log,
        )

    print(returns)

    if output == "csv":
        log.to_csv()


if __name__ == "__main__":
    # Wrapper to parse cmd line args.
    # If you want you can run main() from elsewhere,
    # and specify your arguments there.
    _cli_args = cli_args()

    agent = _cli_args["agent"]
    policy = _cli_args["policy"]
    episodes = int(_cli_args["episodes"])
    steps = _cli_args["steps"]
    if steps:
        steps = int(steps)
    output = _cli_args["output"]
    graphical_render = bool(_cli_args["graphical_render"])

    main(
        agent,
        policy,
        num_episodes=episodes,
        num_steps=steps,
        output=output,
        graphical_render=graphical_render,
    )
