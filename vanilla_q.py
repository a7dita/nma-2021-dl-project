## Crete a vanilla Q-learning agent.

import typing
import numpy as np
from typing import Callable, Sequence


# A value-based policy takes the Q-values at a state and returns an action.
QValues = np.ndarray
Action = int
ValueBasedPolicy = Callable[[QValues], Action]


class VanillaQ:
    def __init__(
        self,
        env,
        behaviour_policy: ValueBasedPolicy = None,
        num_states=24,
        num_actions=4,
        step_size=0.1,
        discount_factor=0.9,
    ):

        # Create the table of Q-values, all initialized at zero.
        self._q = np.zeros((num_states, num_actions))

        # Store algorithm hyper-parameters.
        self._step_size = step_size
        self._discount_factor = discount_factor

        # Store the environment
        self._env = env

        # Store behavior policy.
        self._behaviour_policy = behaviour_policy

        self._state = None
        self._action = None
        self._next_state = None # is this needed?

    def q_values(self):
        return self._q

    def _td_error(self, s, a, r, g, next_s):
        # Compute the TD error.
        return r + g * np.max(self._q[next_s]) - self._q[s, a]

    def update(self):

        s = self._state
        a, r, next_s, _, _ = self.env.step(s, policy=self._behaviour_policy)
        g = self._discount_factor
        tde = self._td_error(s, a, r, g, next_s)

        # Update the Q-value table value at (s, a).
        self._q[s, a] += self._step_size * tde
        # Update the current state.
        self._state = next_s
