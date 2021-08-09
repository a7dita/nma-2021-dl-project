import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

N_DISCRETE_ACTIONS = 4 # pick one of the four discrete cards

class WCST(gym.Env):
  """WCST Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(WCST, self).__init__()
    # define action and observation space
    # must be gym.spaces objects
    # Actions are discrete:
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)


  def _next_observation(self):
    # a card is shown with values of colour, shape, num of elements


  def _take_action(self, action):
    # pick one of the four cards


  def step(self, action):
    # make one step in the environment
    self._take_action(action)
    self.current_step += 1

    reward = # calculate the reward
    done = self.current_step >= 250 # the game is over after 250 steps
    obs = self._next_observation()

    return obs, reward, done, {}

  def reset(self):
    # reset the state of the environment to the initial state
    pass


  def render(self, mode='human', close=False):
    # render the environment to the screen
    print(f'Step: {self.current_step}')
    # print more stuff here

  def close(self):
      pass
