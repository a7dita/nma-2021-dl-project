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

        # Actions are discrete:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.card_deck = card_generator()
        self.current_step = 0
        # Observations are discrete
        self.observation_space = card_generator()
        self.rule = np.random.choice([0, 1, 2])
        self.success_counter = 0 # number of correct responses in a row


    def _next_observation(self):
        """a card is shown with values of (colour, form, num of elements)"""
        card = random.choice(self.card_deck) # do we discard used cards? -- no, we have 24 unique cards but at maximum 250 trials
        return card

    # def _take_action(self):
    #     """agent picks one of the four cards based on predefined policy"""
    #     # WIP

    def _calculate_reward(self, action):
        # the true rule is not part of the observation?

        right_action = self.rule
        reward = +1 if action == right_action else -1

        return reward

    def step(self, action):
        """Take one step in the environment"""

        success_streak = random.randint(2,5)
        if self.success_counter > success_streak:
            self.rule = np.random.choice([0, 1, 2])
        else:
            pass

        action = action
        reward = self._calculate_reward(action)

        self.current_step += 1

        if reward == 1:
            self.success_counter += 1 # count the number of correct moves in a row
        else:
            self.success_counter = 0 # reset after wrong move
        done = self.current_step >= 250  # the game is over after 250 steps

        obs = self._next_observation()
        return action, reward, obs, done

    def reset(self):
        """reset the state of the environment to the initial state"""
        self.success_counter = 0 # number of correct responses in a row
        return self._next_observation()

    def render(self, mode="human", close=False):
        """render the environment to the screen"""
        print('Step: {current_step}'.format(current_step=self.current_step))
        # TODO print more stuff here

    def close(self):
        if self.env is not None:
            self.env.close()
        super().close()
