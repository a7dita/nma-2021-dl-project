from itertools import permutations
import numpy as np
import collections
from acme.utils import loggers
from acme.utils import tree_utils
import time
import torch
import random

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

def nn_loop(environment,
             agent,
             num_episodes=None,
             num_steps=None,
             logger_time_delta=1.,
             label='training_loop',
             log_loss=False,
             ):
  """Perform the run loop.

  We are following the Acme run loop.

  Run the environment loop for `num_episodes` episodes. Each episode is itself
  a loop which interacts first with the environment to get an observation and
  then give that observation to the agent in order to retrieve an action. Upon
  termination of an episode a new episode will be started. If the number of
  episodes is not given then this will interact with the environment
  infinitely.

  Args:
    environment: dm_env used to generate trajectories.
    agent: acme.Actor for selecting actions in the run loop.
    num_steps: number of steps to run the loop for. If `None` (default), runs
      without limit.
    num_episodes: number of episodes to run the loop for. If `None` (default),
      runs without limit.
    logger_time_delta: time interval (in seconds) between consecutive logging
      steps.
    label: optional label used at logging steps.
  """
  logger = loggers.TerminalLogger(label=label, time_delta=logger_time_delta)
  iterator = range(num_episodes) if num_episodes else itertools.count()
  all_returns = []

  num_total_steps = 0
  for episode in iterator:
    # Reset any counts and start the environment.
    start_time = time.time()
    episode_steps = 0
    episode_return = 0
    episode_loss = 0

    environment.reset()
    observation = environment.card
    reward = 0
    discount = 0.9

    next_obs = None
    done = False

    # Make the first observation.
    agent.observe_first(observation)

    # Run an episode.
    while not done:

      # if episode_steps == 0:
      #   timestep = (episode_steps, reward, discount, observation)

      # Generate an action from the agent's policy and step the environment.
      action = agent.select_action(observation)
      reward, next_obs, done, _ = environment.step(action)

      # timestep = (episode_steps+1, reward, discount**episode_steps, observation)

      if done:
          break

      # Have the agent observe the timestep and let the agent update itself.
      # TODO how to implement discount???
      agent.observe(action, reward, next_obs, discount**episode_steps) #this discount will probably cause some weird behavior
      agent.update()

      # Book-keeping.
      episode_steps += 1
      num_total_steps += 1
      episode_return += reward

      if log_loss:
        episode_loss += agent.last_loss

      if num_steps is not None and num_total_steps >= num_steps:
        break

    # Collect the results and combine with counts.
    steps_per_second = episode_steps / (time.time() - start_time)
    result = {
        'episode': episode,
        'episode_length': episode_steps,
        'episode_return': episode_return,
    }
    if log_loss:
      result['loss_avg'] = episode_loss/episode_steps

    all_returns.append(episode_return)

    # Log the given results.
    logger.write(result)

    if num_steps is not None and num_total_steps >= num_steps:
      break
  return all_returns
