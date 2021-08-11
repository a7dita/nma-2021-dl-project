import sys
import random
import numpy as np
import gym
from gym import spaces
from itertools import permutations
from wcst_cards import card_generator

# unused imports
# import json
# import pandas as pd

N_DISCRETE_ACTIONS = 4  # pick one of the four discrete cards


class WCST(gym.Env):

    """WCST environment that follows the OpenAI gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        """Initiate the env with action and observation space"""
        super(WCST, self).__init__()

        # Actions are discrete:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.card_deck = card_generator()
        # Observations are discrete
        self.observation_space = card_generator()
        # Initialize state
        self.success_counter = 0  # number of correct responses in a row
        self.current_step = 0

        self.rule = np.random.choice([0, 1, 2])

    def _next_observation(self):
        """a card is shown with values of (colour, form, num of elements)"""
        card = random.choice(self.card_deck)
        # NOTE do we discard used cards? -- no, we have 24 unique cards but 250 trials
        return card

    def _take_action(self, action):
        """update environment based on action given by agent"""
        # NOTE No effect of action on environment in WCST setting
        pass

    def _calculate_reward(self, action):
        # the true rule is not part of the observation?
        reward = +1 if action == self.rule else -1
        # FIXME reward = +1 if action = current_card[curret_rule]
        # [action = self.rule] is not the case
        # How to get the current_card?

        return reward

    def step(self, action):
        """Take one step in the environment"""

        success_streak = random.randint(2, 6)
        if self.success_counter > success_streak:

            available_rules = [x for x in [0, 1, 2] if x != self.rule]
            self.rule = np.random.choice(available_rules)
        else:
            pass

        self._take_action(action)

        obs = self._next_observation()
        reward = self._calculate_reward(action)

        self.current_step += 1
        if reward == 1:
            self.success_counter += 1  # count the number of correct moves in a row
        else:
            self.success_counter = 0  # reset after wrong move
        done = self.current_step >= 250  # the game is over after 250 steps

        return action, reward, obs, done

    def reset(self):
        """reset the state of the environment to the initial state"""
        self.success_counter = 0  # number of correct responses in a row
        self.current_step = 0
        return self._next_observation()

    def render(self, mode="human", close=False):
        """render the environment to the screen"""
        print("Step: {current_step}".format(current_step=self.current_step))
        # TODO print more stuff here

    def close(self):
        if self.env is not None:
            self.env.close()
        super().close()
