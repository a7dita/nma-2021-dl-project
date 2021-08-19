## TODO Create a deep Q-learning agent.
import collections
import torch
import torch.nn as nn
import copy
from helper_functions import ReplayBuffer, map_action_to_rule
from tqdm import tqdm
import time
from itertools import count

# Create a convenient container for the SARS tuples required by NFQ.
Transitions = collections.namedtuple(
    "Transitions", ["state", "action", "reward", "discount", "next_state"]
)

class Agent():

  def __init__(self,
               env,
               q_network: nn.Module,
               policy: str = "epsilon_greedy", #switch not implemented, just set epsilon = 0
               replay_capacity: int = 100_000,
               epsilon: float = 0.1,
               batch_size: int = 10,
               learning_rate: float = 3e-4):

    # Store agent hyperparameters and network.
    self._num_actions = env.action_space.n
    self._epsilon = epsilon
    self._batch_size = batch_size
    self._learning_rate = learning_rate
    self._q_network = q_network

    self._streak_memory = 6
    self._discount = 0.9

    self._env = env

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

    # Initialize observation
    self._obs = env.card
    # Initialize action (which card is picked)
    self._action = 0
    # Map action to rule (which category was picked on previous attempt)
    self._rule = 0
    # Get number of successive correct answers
    self._streak = 0


  def select_action(self, observation):
    # Compute Q-values.
    q_values = self._q_network(torch.FloatTensor(observation).unsqueeze(0))  # Adds batch dimension.
    q_values = q_values.squeeze(0)   # Removes batch dimension

    # Select epsilon-greedy action.
    if self._epsilon < torch.rand(1):
      # TODO randomize initialization?
      action = q_values.argmax(axis=-1)
    else:
      action = torch.randint(low=0, high=self._num_actions , size=(1,), dtype=torch.int64)
    return action

  def q_values(self, observation):
    # q_values = self._q_network(torch.tensor(observation).unsqueeze(0))
    # return q_values.squeeze(0).detach()
    q_values = self._q_network(torch.FloatTensor(observation)).detach()

  def get_state(self):
    state = [x for x in self._obs]
    state.append(self._rule)
    state.append(self._streak)
    return state

  def update(self):

    if not self._replay_buffer.is_ready(self._batch_size):
      # If the replay buffer is not ready to sample from, do nothing.
      return

    # Sample a minibatch of transitions from experience replay.
    transitions = self._replay_buffer.sample(self._batch_size)

    # Note: each of these tensors will be of shape [batch_size, ...].
    s = torch.FloatTensor(transitions.state)
    a = torch.tensor(transitions.action,dtype=torch.int64)
    r = torch.FloatTensor(transitions.reward)
    d = torch.FloatTensor(transitions.discount)
    next_s = torch.FloatTensor(transitions.next_state)
    # print(f"next_s: {next_s}")
    # print(f"buffer: {self._replay_buffer.buffer}")

    # Compute the Q-values at next states in the transitions.
    with torch.no_grad():
      # NOTE: okay so here's the thing /blob404
      # - first nn layer should be (state_dimensions, hidden_size)
      # - for now i put only the card tuple as the state (i.e. 3 dimensions)
      # - it should work with both single states (3x1) and batches (e.g., 3x10)
      # - i can't figure out how to make it accept both shapes unless i transpose the batch vector like below
      # - (maybe there is some other way to do this)
      # - it's now running complete training cycles, but i have no idea if it's doing the right thing
      q_next_s = self._q_network(next_s.T)  # Shape [batch_size, num_actions].
      max_q_next_s = q_next_s.max(axis=-1)[0]
      # Compute the TD error and then the losses.
      target_q_value = r + d * max_q_next_s

    # Compute the Q-values at original state.
    q_s = self._q_network(s.T)

    # Gather the Q-value corresponding to each action in the batch.
    q_s_a = q_s.gather(1, a.view(-1,1)).squeeze(1)

    loss = self._loss_fn(target_q_value, q_s_a)

    # Compute the gradients of the loss with respect to the q_network variables.
    self._optimizer.zero_grad()

    loss.backward()
    # Apply the gradient update.
    self._optimizer.step()

    # Store the loss for logging purposes (see run_loop implementation above).
    self.last_loss = loss.detach().numpy()

  def observe_first(self, observation):
    self._replay_buffer.add_first(observation)

  def observe(self, action: int, reward, next_obs, discount):
    self._replay_buffer.add(action, reward, next_obs, discount)

  def run(self,
              num_episodes: int = 100,
              log_loss=False,
              logbook=None,
              ):
    """Perform the run loop.

    We are following the Acme run loop.

    Run the environment loop for `num_episodes` episodes. Each episode is itself
    a loop which interacts first with the environment to get an observation and
    then give that observation to the agent in order to retrieve an action. Upon
    termination of an episode a new episode will be started. If the number of
    episodes is not given then this will interact with the environment
    infinitely.

    Args:
      environment: dm_env used to generate trajectories.
      agent: acme.Actor for selecting actions in the run loop.
      num_training_steps: number of steps to run the loop for. If `None` (default), runs
        without limit.
      num_episodes: number of episodes to run the loop for. If `None` (default),
        runs without limit.
      label: optional label used at logging steps.
    """
    iterator = range(num_episodes) if num_episodes else itertools.count()
    all_returns = []

    for episode in tqdm(iterator):
      # Reset any counts and start the environment.
      start_time = time.time()
      episode_steps = 0
      cum_return = 0
      episode_loss = 0

      self._env.reset()
      self._obs = self._env.card

      state = self.get_state()
      done = False

      # Put first state into replay buffer
      self.observe_first(state)

      # Run an episode.
      while not done:

        # Generate an action from the agent's policy and step the environment.
        action = self._action = int(self.select_action(state))
        reward, next_obs, done, _ = self._env.step(action)

        if reward == 1:
            self._streak = min(self._streak+1, self._streak_memory)
        else:
            self._streak = 0

        self._rule = map_action_to_rule(self._obs, action)
        self._obs = next_obs

        next_state = self.get_state()

        # Have the agent observe the timestep and let the agent update itself.
        # TODO how to implement discount???
        discount = self._discount
        self.observe(action, reward, next_state, discount**episode_steps) #this discount will probably cause some weird behavior
        self.update()

        state = next_state

        # Book-keeping.
        episode_steps += 1
        cum_return += reward

        # if log_loss: # unused for now
        #   episode_loss += agent.last_loss

        if logbook:
          logbook.write_actions(episode, cum_return)

        if done:
            break

      if logbook:
        logbook.write_episodes(episode, episode_steps, cum_return)

      all_returns.append(cum_return)

    return all_returns
