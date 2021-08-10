#!/usr/bin/env python3
# The parts that aren't ready yet are commented out

import wcst
# import deep_q
# import vanilla_q
import torch
import random
import torch.nn as nn

if __name__ == '__main__':
    env = wcst.WCST()
    # agent = vanilla_q.VanillaQ(env)

    for i in range(250):
        action = random.randint(1,4) # taking random actions on every step
        action, reward, obs, done = env.step(action)
        env.render()
        print(action, reward, obs, done)
