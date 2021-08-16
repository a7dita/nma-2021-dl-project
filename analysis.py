#!/usr/bin/env python3
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_perseverative_error(csv_file: str):

    df = pd.read_csv(csv_file, sep=",")
    d = df[["current_reward", "agent_rule"]].to_numpy(dtype=int)

    switch_count = 0
    perseverative_error = 0

    for i in range(1, d.shape[0]):

        if d[i - 1][0] == -1:
            switch_count += 1

            if d[i - 1][1] == d[i][1]:
                perseverative_error += 1

    perseverative_error = perseverative_error / switch_count

    return perseverative_error


def get_set_loss_error(csv_file: str):

    df = pd.read_csv(csv_file, sep=",")
    d = df[["current_reward", "agent_rule"]].to_numpy(dtype=int)

    repeat_count = 0
    set_loss_error = 0

    for i in range(1, d.shape[0]):

        if d[i - 1][0] == 1:
            repeat_count += 1

            if d[i - 1][1] != d[i][1]:
                set_loss_error += 1

    set_loss_error = set_loss_error / repeat_count

    return set_loss_error
