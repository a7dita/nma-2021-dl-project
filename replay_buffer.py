#!/usr/bin/env python3
import collections

# Create a convenient container for the SARS tuples required by NFQ.
Transitions = collections.namedtuple(
    'Transitions', ['state', 'action', 'reward', 'discount', 'next_state'])

# Simple replay buffer
class ReplayBuffer(object):
    """A simple Python replay buffer."""

    def __init__(self, capacity: int = None):
        self.buffer = collections.deque(maxlen=capacity)
        self._prev_state = None

    # TODO change timestep format to that of gym env
    # def add_first(self, initial_timestep: dm_env.TimeStep):
    #     self._prev_state = initial_timestep.observation

    # def add(self, action: int, timestep: dm_env.TimeStep):
    #     transition = Transitions(
    #         state=self._prev_state,
    #         action=action,
    #         reward=timestep.reward,
    #         discount=timestep.discount,
    #         next_state=timestep.observation,
    #     )
    #     self.buffer.append(transition)
    #     self._prev_state = timestep.observation

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
