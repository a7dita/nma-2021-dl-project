#!/usr/bin/env python3
# The parts that aren't ready yet are commented out
import wcst
# import deep_q
import vanilla_q
# import torch
# import torch.nn as nn
import pandas as pd
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).parent

#TODO low prio: move agents to subfolder, get all name strings from there
LIST_OF_AGENTS = ['vanilla_q', 'memory_q', 'deep_q']

def cli_args():
    """Parse arguments when running from command line.
       Returns a dictionary with structure { option : value }."""

    parser = argparse.ArgumentParser(
        description="Runs the Wisconsin Card Sorting Task (WCST) using the specified "
        "reinforcement learning (RL) agent. Default: vanilla_q")

    parser.add_argument('agent', nargs='?', default='vanilla_q', choices=LIST_OF_AGENTS,
                        help='Agent to run the task. Allowed values: '+
                        ', '.join(LIST_OF_AGENTS), metavar='')

    # returns dictionary of command line arguments
    return vars(parser.parse_args())

def output_csv(df):
    """Writes a dataframe to root/gen_data/YY-MM-DD_HHMMSS.csv"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    datadir = ROOT_DIR.joinpath("gen_data")
    filepath = datadir.joinpath(timestamp + ".csv")

    # Create directory if needed
    os.makedirs(datadir, exist_ok=True)
    df.to_csv(filepath)

def create_agent(string):
    """Creates a new agent object from the module specified in input string."""
    return eval(string).Agent(env)

# TODO merge lol
# if __name__ == '__main__':
#     env = wcst.WCST()
#     env.reset()
#     # agent = vanilla_q.VanillaQ(env)

#     for i in range(250):
#         action = random.randint(1,4) # taking random actions on every step
#         action, reward, obs, done = env.step(action)
#         env.render()
#         print(action, reward, obs, done)

if __name__ == '__main__':
    # parse cmd line arguments
    cli_args = cli_args()

    # create and init environment
    env = wcst.WCST()
    env.reset()

    # create agent specified by cmd line option
    agent = create_agent(cli_args['agent'])

    # # create df to save some metadata
    # df = pd.DataFrame()

    # for i in range(250):

    #     s = agent._state
    #     a = agent._action
    #     rule = agent._env.rule
    #     q = agent._q.copy(deep=True)

    #     df = df.append({'state' : s,
    #                'prev_action' : a,
    #                'rule' : rule,
    #                'q_table' : q}, ignore_index=True)

    #     # update state of agent and environment
    #     agent.update()
    #     output_csv(df)
