#!/usr/bin/env python3
# The parts that aren't ready yet are commented out
import wcst
# import deep_q
import vanilla_q
# import torch
# import torch.nn as nn
import pandas as pd
import os
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).parent

# TODO merge lol
if __name__ == '__main__':
    env = wcst.WCST()
    env.reset()
    # agent = vanilla_q.VanillaQ(env)

    for i in range(250):
        action = random.randint(1,4) # taking random actions on every step
        action, reward, obs, done = env.step(action)
        env.render()
        print(action, reward, obs, done)

if __name__ == '__main__':
    env = wcst.WCST()
    env.reset()
    agent = vanilla_q.VanillaQ(env)

    df = pd.DataFrame()

    for i in range(250):

        s = agent._state
        a = agent._action
        rule = agent._env.rule
        q = agent._q.copy(deep=True)

        df = df.append({'state' : s,
                   'prev_action' : a,
                   'rule' : rule,
                   'q_table' : q}, ignore_index=True)

        # update state of agent and environment
        agent.update()

    # write some data to .csv
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    datadir = ROOT_DIR.joinpath("gen_data")
    filepath = datadir.joinpath(timestamp + ".csv")

    os.makedirs(datadir, exist_ok=True)
    df.to_csv(filepath)
