## TODO Create a vanilla Q-learning agent.
import typing
import numpy as np
from typing import Callable, Sequence
import pandas as pd


class Agent:
    def __init__(
        self,
        env,
        policy=None,
        step_size=0.1,
        discount_factor=0.9,
        epsilon=0.1,
    ):

        # Get size of state and action space from the environment
        self._num_states = env.observation_space.n
        self._num_actions = env.action_space.n

        # Create a table of Q-values with card tuples as row indexes.
        self._q = pd.DataFrame(
            index=env.card_deck,
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

        # Initialize state
        self._state = env._next_observation()
        self._action = None

    def q_values(self, state):
        return self._q

    def select_action(self, state):

        if self._behaviour_policy == None:
            # Default policy: random action
            # Exploration
            action = np.random.randint(low=0, high=self._num_actions)

        elif self._behaviour_policy == "value_max":
            # Select action by just looking at Q-table
            # Exploitation
            action = self._q.loc[[state]].idxmax(axis=1)

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
        print(f"New state: {self._state}")
        print(f"Prev. action: {self._action}")
        print(f"Policy: {self._behaviour_policy}")

    def update(self):
        # Get action based on policy
        s = self._state
        a = self.select_action(s)

        # Update environment, get next_s and reward as observations
        a, r, next_s, _, _ = self._env.step(a)

        # Get discount factor applied on future rewards
        g = self._discount_factor

        # Compute Temporal Difference error (TDE)
        tde = self._td_error(s, a, r, g, next_s)

        # Update the Q-value table value at (s, a).
        self._q.loc[[s], a] += self._step_size * tde

        # Update the current state.
        self._state = next_s
        self._action = a
