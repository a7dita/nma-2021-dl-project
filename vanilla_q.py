## TODO Create a vanilla Q-learning agent.
import typing
import numpy as np
from typing import Callable, Sequence
import pandas as pd
import helper_functions
from itertools import product


class Agent:
    def __init__(
        self,
        env,
        policy=None,
        step_size=0.5,
        discount_factor=0.9,
        epsilon=0.2,
    ):

        # Get size of state and action space from the environment
        self._num_obs = env.observation_space.n
        self._num_actions = env.action_space.n
        self._streak_memory = 1

        # Get list of possible states
        q_index = list(product(env.card_deck,
                               range(self._num_actions),
                               range(self._streak_memory+1)))

        self._num_states = len(q_index)

        # Create a table of Q-values with possible states as indexes.
        self._q = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(q_index),
            data=np.zeros((self._num_states, self._num_actions)),
        )

        # Store algorithm hyper-parameters.
        self._step_size = step_size
        self._discount_factor = discount_factor
        self._epsilon = epsilon

        # Store the environment
        self._env = env

        # Store behavior policy.
        self._behaviour_policy = policy

        # Initialize observation
        self._obs = env.card
        # Initialize action (which card is picked)
        self._action = None
        # Map action to rule (which category was picked on previous attempt)
        self._rule = 0
        # Get number of successive correct answers
        self._streak = 0

    def q_values(self, state):
        return self._q

    def get_state(self):
        state = (self._obs, self._rule, self._streak)
        return state

    def select_action(self, state):

        if self._behaviour_policy == None:
            # Default policy: random action
            # Exploration
            action = np.random.randint(low=0, high=self._num_actions)

        elif self._behaviour_policy == "value_max":
            # Select action by just looking at Q-table
            # Exploitation
            action = self._q.loc[[state]].idxmax(axis=1).values

        elif self._behaviour_policy == "epsilon_greedy":
            # Select action based on the epsilon-greedy policy
            # Finding out the exploration-exploitation balance
            if self._epsilon < np.random.random():
                action = self._q.loc[[state]].idxmax(axis=1).values
                # print(f"action set to idxmax: {action}")
                # q_value = self._q.loc[[state]].max(axis=1).values
                # print(f"q_value of action: {q_value}")
            else:
                action = np.random.randint(low=0, high=self._num_actions)
                # print(f"action set to random: {action}")

        # TODO implement other policies later
        return action

    def _td_error(self, s, a, r, g, next_s):
        # Compute the TD error.
        max_q = self._q.loc[[next_s]].max(axis=1).values
        cur_q = self._q.loc[[s], int(a)].values
        tde = r + g * max_q - cur_q
        return tde

    def render(self):
        self._env.render()
        # print(f"New obs: {self._obs}")
        # print(f"Prev. action: {self._action}")
        # print(f"Policy: {self._behaviour_policy}")

    def update(self):
        # Create complete state representation
        s = self.get_state()

        # Update current action based on policy
        a = self.select_action(s)

        # Update environment, get next card and reward
        r, next_obs, _, _ = self._env.step(a)

        # Update internal streak count
        if r == 1:
            self._streak = min(self._streak + 1, self._streak_memory)
        else:
            self._streak = 0

        # Update the current state.
        self._action = a
        self._rule = helper_functions.map_action_to_rule(self._obs, self._action)
        self._obs = next_obs

        # Get updated state vector
        next_s = self.get_state()

        # Get discount factor applied on future rewards
        g = self._discount_factor

        # Compute Temporal Difference error (TDE)
        tde = self._td_error(s, a, r, g, next_s)

        # Update the Q-value table value at (s, a).
        self._q.loc[[s], a] += self._step_size * tde

