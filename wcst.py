import random
import numpy as np
import gym
from gym import spaces
from itertools import permutations
from wcst_cards import card_generator
from rule import map_rule_to_action

N_DISCRETE_ACTIONS = 4  # pick one of the four discrete cards
N_DISCRETE_CARDS = 24  # use a deck of 24 unique cards


class WCST(gym.Env):

    """WCST environment that follows the OpenAI gym interface"""

    def __init__(self):
        """Initiate the env with action and observation space"""
        super(WCST, self).__init__()

        # Actions are discrete:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Observations are discrete
        self.observation_space = spaces.Discrete(N_DISCRETE_CARDS)

        # initialise cards and rule

        self.card = None # the card to categorise
        self.card_deck = card_generator()
        # NOTE please do not delete the card deck! Not a duplicate variable - Now they are not. :P
        self.rule = np.random.choice([0, 1, 2])
        self.right_action = map_rule_to_action(self.rule) # Map rule {0,1,2} to action {1,2,3,4}

        #initialise counters
        self.current_step = 0
        self.success_counter = 0  # number of correct responses in a row
        self.switch_counter = 0 # keep track of rule switches; max should be 41

    def _next_observation(self):
        """a card is shown with values of (colour, form, num of elements)"""
        return random.choice(self.card_deck)  # please do not replace this w observation space
        # NOTE do we discard used cards? -- no, we have 24 unique cards but 250 trials

    def _calculate_reward(self, action):
        """Give reward of +1 if the action is correct
        or -1 otherwise"""
        reward = +1 if action == self.right_action else -1
        return reward

    def step(self, action):
        """Take one step in the environment"""
        self.current_step += 1

        if action == self.right_action: # correct move
            obs = self._next_observation() # show a new card
            self.card = obs
            self.success_counter += 1 # update success counter
            success_streak = random.randint(2, 5) # check if it's time to switch rule
            if self.success_counter > success_streak:
                self.rule = np.random.choice([0, 1, 2])
                self.right_action = map_rule_to_action(self.rule)

        else: # wrong move
            self.success_counter = 0
            obs = self.card
        reward = self._calculate_reward(action)
        done = self.current_step >= 250 or self.switch_counter >=41  # the game is over after 250 steps or 41 rule switches

        return action, reward, obs, done, {}

    def reset(self):
        """reset the state of the environment to the initial state"""
        self.card = self._next_observation()
        self.current_step = 0
        self.rule = np.random.choice([0, 1, 2])
        self.right_action = map_rule_to_action(self.rule)
        self.success_counter = 0  # reset success success_counter
        self.switch_counter = 0 # reset rule switch counter

    def render(self, mode="human", close=False):
        """Render environment to screen"""
        print("Step: {current_step}".format(current_step=self.current_step))
        # TODO print more stuff here

    def close(self):
        super().close()
