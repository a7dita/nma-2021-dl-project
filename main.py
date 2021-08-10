#!/usr/bin/env python3

import wcst
import deep_q
import vanilla_q
import torch
import torch.nn as nn

if __name__ == '__main__':
    env = wcst.WCST()
    agent = vanilla_q.VanillaQ(env)

    print(env)
    print(agent)
