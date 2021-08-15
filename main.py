#!/usr/bin/env python3
# The parts that aren't ready yet are commented out
import wcst
import deep_q
import vanilla_q
# import torch
# import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).parent

#TODO low prio: move agents to subfolder, get all name strings from there
LIST_OF_AGENTS = ['vanilla_q', 'memory_q', 'deep_q']
LIST_OF_POLICIES = ['value_max', 'epsilon_greedy']

def cli_args():
    """Parse arguments when running from command line.
       Returns a dictionary with structure { option : value }."""

    parser = argparse.ArgumentParser(
        description="Runs the Wisconsin Card Sorting Task (WCST) using the specified "
        "reinforcement learning (RL) agent. Default: vanilla_q")

    # This is positional; last argument provided.
    parser.add_argument('agent', nargs='?', default='vanilla_q', choices=LIST_OF_AGENTS,
                        help='Agent to run the task. Allowed values: '+
                        ', '.join(LIST_OF_AGENTS), metavar='')

    parser.add_argument('-s', '--steps', default=250,
                        help='Number of steps to run the agent through the environment. Default: 250')

    parser.add_argument('-p', '--policy', default=None, choices=LIST_OF_POLICIES,
                        help='Policy to use for value-based agent. Default: None (random). Allowed values: '+
                        ', '.join(LIST_OF_POLICIES), metavar='')

    parser.add_argument('-o', '--output', default=None,
                        help='Output format of generated data. Default: None')

    # returns dictionary of command line arguments
    # all of this is just for convenience
    return vars(parser.parse_args())

def output_csv(df):
    """Writes a dataframe to root/gen_data/YY-MM-DD_HHMMSS.csv"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    datadir = ROOT_DIR.joinpath("gen_data")
    filepath = datadir.joinpath(timestamp + ".csv")

    # Create directory if needed
    os.makedirs(datadir, exist_ok=True)
    df.to_csv(filepath)

def create_agent(string, env, policy):
    """Creates a new agent object from the module specified in input string."""
    return eval(string).Agent(env, policy)

def main(agent='vanilla_q', policy=None, steps=250, output=None):
    # create and init environment
    env = wcst.WCST()
    env.reset()

    # create agent specified by cmd line option
    agent = create_agent(agent, env, policy)

    # create df to save some metadata
    df = pd.DataFrame()

    # init reward counters
    cr = 0
    rr = [0]*7

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

        df = df.append({'state' : s,
                   'prev_action' : a,
                   'agent_rule' : ar,
                   'env_rule' : er,
                   'cumulative_reward' : cr,
                   'rolling_reward' : np.sum(rr)},
                   # 'q_table' : q},
                   ignore_index=True)

        # render state of agent and environment
        # add print statements freely :)
        if i % 100 == 0:
            agent.render()
            print(f"agent rule: {ar}")
            print(f"env rule: {er}")
            print(f"current reward: {rr[-1]}")
            print(f"cumulative reward: {cr}")
            # print(f"rolling reward: {np.sum(rr)}")

        # update state of agent and environment
        agent.update()

    if output == 'csv':
        output_csv(df)

if __name__ == '__main__':
    # now, this is just a wrapper to parse cmd line arguments
    # if you want you can run main() from another script or from notebook,
    # and specify arguments there
    cli_args = cli_args()

    agent = cli_args['agent']
    policy = cli_args['policy']
    steps = int(cli_args['steps'])
    output = cli_args['output']

    main(agent, policy, steps, output)
