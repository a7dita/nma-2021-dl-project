from itertools import permutations
import numpy as np
import collections

def card_generator(li_values=[0, 1, 2, 3], length=3):
    """generate cards given a list of values and the length of tuple."""
    cards = list(permutations(li_values, length))

    return cards

def map_rule_to_action(card, rule):
    action = card[rule]
    return action

def map_action_to_rule(card, action):
    try:
        rule = card.index(action)
    except ValueError:
        rule = 3 #odd response

    return rule

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

# Create a convenient container for the SARS tuples required by NFQ.
Transitions = collections.namedtuple(
    'Transitions', ['state', 'action', 'reward', 'discount', 'next_state'])

# Simple replay buffer
class ReplayBuffer(object):
    """A simple Python replay buffer."""

    def __init__(self, capacity: int = None):
        self.buffer = collections.deque(maxlen=capacity)
        self._prev_state = None

    def add_first(self, obs):
        self._prev_state = obs

    def add(self, action: int, reward, obs, discount):
        transition = Transitions(
            state=self._prev_state,
            action=action,
            reward=reward,
            next_state=obs,
            discount=discount,
        )
        self.buffer.append(transition)
        self._prev_state = obs

    def sample(self, batch_size: int) -> Transitions:
        # Sample a random batch of Transitions as a list.
        batch_as_list = random.sample(self.buffer, batch_size)

        # Convert the list of `batch_size` Transitions into a single Transitions
        # object where each field has `batch_size` stacked fields.
        return tree_utils.stack_sequence_fields(batch_as_list)

    def flush(self) -> Transitions:
        entire_buffer = tree_utils.stack_sequence_fields(self.buffer)
        self.buffer.clear()
        return entire_buffer

    def is_ready(self, batch_size: int) -> bool:
        return batch_size <= len(self.buffer)
