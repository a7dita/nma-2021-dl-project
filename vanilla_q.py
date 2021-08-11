## TODO Create a vanilla Q-learning agent.
import typing
import numpy as np
from typing import Callable, Sequence
import pandas as pd


class VanillaQ:
    def __init__(
        self,
        env,
        policy=None,
        step_size=0.1,
        discount_factor=0.9,
        epsilon=0.1,
    ):

        # Get size of state and action space from the environment
        self._num_states = len(env.observation_space)
        self._num_actions = env.action_space.n

        # Create a table of Q-values with card tuples as row indexes.
        self._q = pd.DataFrame(index=env.observation_space,
                               data=np.zeros((self._num_states, self._num_actions)))

        # Store algorithm hyper-parameters.
        self._step_size = step_size
        self._discount_factor = discount_factor
        self._epsilon = epsilon

        # Store the environment
        self._env = env

        # Store behavior policy.
        self._behaviour_policy = behaviour_policy

        # Initialize state
        self._state = env._next_observation()
        self._action = None

    def q_values(self, state):
        return self._q

    def _td_error(self, s, a, r, g, next_s):
        # Compute the TD error.
        max_q = self._q.loc[[next_s]].max(axis=1).values
        cur_q = self._q.loc[[s],a].values
        tde = r + g * max_q - cur_q
        return tde

    def select_action(self, state, policy=None):
        if policy == 'epsilon-greedy':
            # Select epsilon-greedy action.
            # TODO implement later
            if self._epsilon < np.random.random():
                action = self._q[state].argmax() #wrong indexing
            else:
                action = np.random.randint(low=0, high=self._num_actions)
        elif policy == None:
            # Default policy: random action
            action = np.random.randint(low=0, high=self._num_actions)

        return action

    def update(self):
        # Get action based on policy
        s = self._state
        print(f"State: {s}")
        a = self.select_action(s)
        print(f"Action: {a}")

        # Update environment, get next_s and reward as observations
        a, r, next_s, _, _ = self._env.step(a)
        print(f"Reward: {r}")
        print(f"Next_s: {next_s}")

        # Get discount factor applied on future rewards
        g = self._discount_factor

        # Compute Temporal Difference error (TDE)
        tde = self._td_error(s, a, r, g, next_s)
        if r == 1:
            print(f"tde: {tde}")

        # Update the Q-value table value at (s, a).
        self._q.loc[[s], a] += self._step_size * tde
        # Update the current state.
        self._state = next_s
        self._action = a
