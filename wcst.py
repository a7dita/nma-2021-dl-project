import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
from itertools import permutations
from wcst_cards import card_generator

N_DISCRETE_ACTIONS = 4 # pick one of the four discrete cards

class WCST(gym.Env):

  """WCST environment that follows the OpenAI gym interface"""
  metadata = {'render.modes': ['human']}

    def __init__(self):
        """Initiate the env with action and observation space"""
        super(WCST, self).__init__()
        # Observations are discrete
        self.observation_space = card_generator()
        # TODO how to incorporate gym.space

        # Actions are discrete:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.card_deck = card_generator()
        self.correct_counter = 0 # number of correct responses in a row
        self.rule = random.randint(1,4)

    def _next_observation(self):
        """a card is shown with values of (colour, form, num of elements)"""
        card = np.random.choice(self.card_deck)
        return card

    def _take_action(self, obs, policy=None):
        """agent picks one of the four cards based on predefined policy"""

        if policy == None:
            action = self.action_space.sample()  # random policy, for now
        else:
            action = policy(obs)  # any other policy based on the observation

    def _calculate_reward(self, rule, obs, action):
        # the true rule is not part of the observation?

        right_action = obs[rule]
        reward = +1 if action == right_action else -1

        return reward

    def step(self, obs, policy=None):
        """Take one step in the environment"""

        rule = self.rule
        success_streak = random.randint(2,5)
        if self.success_counter > success_streak:
            rule = random.randint(1,4)
        else:
            pass

        action = self._take_action(obs, policy)
        reward = self._calculate_reward(rule, obs, action)

        self.current_step += 1
        if reward == 1:
            self.success_counter += 1 # count the number of correct moves in a row
        else:
            self.success_counter = 0 # reset after wrong move
        done = self.current_step >= 250  # the game is over after 250 steps

        obs = self._next_observation()
        return action, reward, obs, done, {}

    def reset(self):
        """reset the state of the environment to the initial state"""
        pass

    def render(self, mode="human", close=False):
        """render the environment to the screen"""
        print(f"Step: {self.current_step}")
        print(f"Action, reward, card, done : {self.step}")
        # TODO print more stuff here

    def close(self):
        pass
