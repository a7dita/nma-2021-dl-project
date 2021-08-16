## TODO Create a deep Q-learning agent.
import collections
import torch
import torch.nn as nn
import copy
from helper_functions import ReplayBuffer

# Create a convenient container for the SARS tuples required by NFQ.
Transitions = collections.namedtuple(
    'Transitions', ['state', 'action', 'reward', 'discount', 'next_state'])

class Agent():

  def __init__(self,
               environment,
               q_network: nn.Module,
               replay_capacity: int = 100_000,
               epsilon: float = 0.1,
               batch_size: int = 1,
               learning_rate: float = 3e-4):

    # Store agent hyperparameters and network.
    self._num_actions = environment.action_space.n
    self._epsilon = epsilon
    self._batch_size = batch_size
    self._q_network = q_network

    # create a second q net with the same structure and initial values, which
    # we'll be updating separately from the learned q-network.
    self._target_network = copy.deepcopy(self._q_network)

    # Container for the computed loss (see run_loop implementation above).
    self.last_loss = 0.0

    # Create the replay buffer.
    self._replay_buffer = ReplayBuffer(replay_capacity)

    # Keep an internal tracker of steps
    self._current_step = 0

    # Setup optimizer that will train the network to minimize the loss.
    self._optimizer = torch.optim.Adam(self._q_network.parameters(),lr = learning_rate)
    self._loss_fn = nn.MSELoss() # try different loss functions?

  def select_action(self, observation):
    # Compute Q-values.
    q_values = self._q_network(torch.tensor(observation).unsqueeze(0))  # Adds batch dimension.
    q_values = q_values.squeeze(0)   # Removes batch dimension

    # Select epsilon-greedy action.
    if self._epsilon < torch.rand(1):
      action = q_values.argmax(axis=-1)
    else:
      action = torch.randint(low=0, high=self._num_actions , size=(1,), dtype=torch.int64)
    return action

  def q_values(self, observation):
    q_values = self._q_network(torch.tensor(observation).unsqueeze(0))
    return q_values.squeeze(0).detach()

  def update(self):

    if not self._replay_buffer.is_ready(self._batch_size):
      # If the replay buffer is not ready to sample from, do nothing.
      return

    # Sample a minibatch of transitions from experience replay.
    transitions = self._replay_buffer.sample(self._batch_size)

    # Note: each of these tensors will be of shape [batch_size, ...].
    s = torch.tensor(transitions.state)
    a = torch.tensor(transitions.action,dtype=torch.int64)
    r = torch.tensor(transitions.reward)
    d = torch.tensor(transitions.discount)
    next_s = torch.tensor(transitions.next_state)

    # Compute the Q-values at next states in the transitions.
    with torch.no_grad():
      q_next_s = self._q_network(next_s)  # Shape [batch_size, num_actions].
      max_q_next_s = q_next_s.max(axis=-1)[0]
      # Compute the TD error and then the losses.
      target_q_value = r + d * max_q_next_s

    # Compute the Q-values at original state.
    q_s = self._q_network(s)

    # Gather the Q-value corresponding to each action in the batch.
    q_s_a = q_s.gather(1, a.view(-1,1)).squeeze(0)

    # TODO Average the squared TD errors over the entire batch using
    # self._loss_fn, which is defined above as nn.MSELoss()
    # HINT: Take a look at the reference for nn.MSELoss here:
    #  https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
    #  What should you put for the input and the target?
    loss = 0

    # Compute the gradients of the loss with respect to the q_network variables.
    self._optimizer.zero_grad()

    loss.backward()
    # Apply the gradient update.
    self._optimizer.step()

    # Store the loss for logging purposes (see run_loop implementation above).
    self.last_loss = loss.detach().numpy()

  # TODO timestep format needs to be changed for Gym
  # def observe_first(self, timestep: dm_env.TimeStep):
  #   self._replay_buffer.add_first(timestep)

  # def observe(self, action: int, next_timestep: dm_env.TimeStep):
  #   self._replay_buffer.add(action, next_timestep)

