#!/usr/bin/env python3
# The parts that aren't ready yet are commented out
import wcst
import models.deep_q as deep_q
import models.vanilla_q as vanilla_q
import models.sarsa as sarsa
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

from helper_functions import nn_loop

ROOT_DIR = Path(__file__).parent

# DONE low prio: move agents to subfolder, get all name strings from there
LIST_OF_AGENTS = ["vanilla_q", "sarsa", "deep_q"]
LIST_OF_POLICIES = ["value_max", "epsilon_greedy"]


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
        default=10000,
        help="Number of steps to run the agent through the environment. Default: 250",
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

def output_csv(df, metadata):
    """Writes a dataframe to root/gen_data/YY-MM-DD_HHMMSS.csv"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    datadir = ROOT_DIR.joinpath("gen_data")
    filepath = datadir.joinpath(f"{timestamp}_{metadata}.csv")

    # Create directory if needed
    os.makedirs(datadir, exist_ok=True)
    df.to_csv(filepath)


def create_agent(string, env, policy):
    """Creates a new agent object from the module specified in input string."""
    return eval(string).Agent(env, policy)

def vanilla_q_loop(env, agent, steps):
    for i in range(steps):

        # FIXME so this is a bit clunky and should be changed probably
        s = agent.get_state()
        a = agent._action
        ar = agent._rule
        er = agent._env.rule
        # q = agent._q.copy(deep=True)
        #
        # store cumulative and rolling rewards
        if agent._streak > 0:
            cr += 1
            rr.pop(0)
            rr.append(1)
        else:
            cr -= 1
            rr.pop(0)
            rr.append(-1)

        if output == "csv":
            df = df.append(
                {
                    "state": s,
                    "prev_action": a,
                    "agent_rule": ar,
                    "env_rule": er,
                    "current_reward": rr[-1],
                    "cumulative_reward": cr,
                    "rolling_reward": np.sum(rr),
                },
                # 'q_table' : q},
                ignore_index=True,
            )

        # render state of agent and environment
        # add print statements freely :)
        if i % 100 == 0:
            env.render(a, rr[-1], graphical_render=graphical_render)
            # agent.render()
            # print(f"agent rule: {ar}")
            # print(f"env rule: {er}")
            # print(f"current reward: {rr[-1]}")
            print(f"cumulative reward: {cr}")
            # print(f"rolling reward: {np.sum(rr)}")

        # update state of agent and environment
        agent.update()

def main(
    agent="vanilla_q", policy=None, steps=250, output=None, graphical_render=False
):
    # create and init environment
    env = wcst.WCST()
    env.reset()

    # create agent specified by cmd line option

    # create df to save some metadata
    # df = pd.DataFrame()

    # init reward counters
    # cr = 0
    # rr = [0] * 7

    if agent == "vanilla_q":
        agent = create_agent(agent, env, policy)
        vanilla_q_loop(env, agent, steps)
    elif agent == "deep_q":
        q_network = nn.Sequential(nn.Linear(1, 10),
                                nn.ReLU(),
                                nn.Linear(10, 50),
                                nn.ReLU(),
                                nn.Linear(50, env.action_space.n))

        # Build the trainable Q-learning agent
        agent = deep_q.Agent(
            env,
            q_network,
            replay_capacity=100_000,
            batch_size=1,
            learning_rate=1e-3)

        returns = nn_loop(
            environment=env,
            agent=agent,
            num_episodes=500,
            logger_time_delta=1.,
            log_loss=True)

        print(returns)

        # # @title Evaluating the agent (set $\epsilon=0$)
        # # Temporarily change epsilon to be more greedy; remember to change it back.
        # agent._epsilon = 0.0

        # # Record a few episodes.
        # frames = evaluate(environment, agent, evaluation_episodes=5)

        # # Change epsilon back.
        # agent._epsilon = epsilon

        # # Display the video of the episodes.
        # display_video(frames, frame_rate=6)

    # if output == "csv":
    #     metadata = f"{cli_args['agent'][:4]}_{policy[:4]}_{steps}st_e{agent._epsilon}_step{agent._step_size}_mem{agent._streak_memory}"
    #     output_csv(df, metadata)

    # return cr


if __name__ == "__main__":
    # now, this is just a wrapper to parse cmd line arguments
    # if you want you can run main() from another script or from notebook,
    # and specify arguments there
    cli_args = cli_args()

    agent = cli_args["agent"]
    policy = cli_args["policy"]
    steps = int(cli_args["steps"])
    output = cli_args["output"]
    graphical_render = bool(cli_args["graphical_render"])

    main(agent, policy, steps, output, graphical_render)

# # @title Training the NFQ Agent
# epsilon = 0.4 # @param {type:"number"}

# max_episode_length = 200

# # Create the environment.
# grid = build_gridworld_task(
#     task='simple',
#     observation_type=ObservationType.AGENT_GOAL_POS,
#     max_episode_length=max_episode_length)
# environment, environment_spec = setup_environment(grid)

# # Define the neural function approximator (aka Q network).

# @title Run loop  { form-width: "30%" }
# @markdown This function runs an agent in the environment for a number of
# @markdown episodes, allowing it to learn.

# @markdown *Double-click* to inspect the `run_loop` function.


