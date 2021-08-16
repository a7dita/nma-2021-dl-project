# Project Abstract

The Wisconsin Card Sorting Test (WCST) is widely used to assess cognitive flexibility of human participants. We use two reinforcement learning (RL) algorithms to model cognitive processes involved in the WCST. First, we ask if the assumptions made by the RL model result in the agent's behaviour being comparable to that of human participants; and secondly, if we can modify the parameters of the model to approximate the behaviour of human participants with impaired cognitive function. We hypothesise that naive model-free Q-learning and deep Q networks (DQN) will produce different distributions of set-loss and perseverative errors. We expect that a DQN will most closely approximate the error distribution of healthy human individuals (e.g. Steinke et al., 2020). The error distribution in the simulated data will be further influenced by the memory capacity of the model. We expect that decreasing the memory capacity will shift the error distribution to more closely approximate that of individuals with impaired cognitive abilities, and that increasing the agent’s ‘memory’ will improve its absolute performance over longer runs and introduce anticipatory set-loss errors.

We start by using a simple Q-learning algorithm to perform the WCST following greedy or epsilon-greedy policies. Our basic Q-learning agent with 1-step memory is able to achieve a positive reward trajectory after 700-1100 steps using an epsilon-greedy policy with an epsilon of 0.05 and step size of 0.1.

During the final week of NMA, we intend to implement a deep Q-network to test if its performance is a better match to human data than the look-up table-based Q-learning implementation.


