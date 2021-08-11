import numpy as np


def create_rule_series(total_time):

    rule_series = []
    rules = [0, 1, 2]
    last_rule = None  # To begin with

    while len(rule_series) < total_time:

        available_rules = [x for x in rules if x != last_rule]
        current_rule = np.random.choice(available_rules)
        current_rep = np.random.randint(2, 6)
        rule_series += [current_rule] * current_rep
        last_rule = current_rule

    rule_series = rule_series[:total_time]

    return rule_series
