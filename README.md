# Reinforced Foxes’ NMA 2021 DL Project 🦊
## How rewarding can sorting cards be? - Depends on how flexible you are


The Wisconsin Card Sorting Test (WCST) is a way to assess cognitive flexibility in human participants.

<p align="center">
<img src="sample/switch.jpg" width="40%"/>
<img src="sample/repeat.png" width="40%"/>
</p>

<!-- ![](sample/switch.jpg) -->
<!-- ![](sample/repeat.png) -->

On every trial, a card is shown that has to  be matched to one of the four key cards according to either colour, shape, or number of elements on the card. Three of the key cards match the card along one of the dimensions each, while the fourth card does not match at all—it is the odd one. The matching rule is drawn randomly and changes after several successful trials. The participant receives feedback about whether they should apply the same rule as on the last trial, or switch to a different rule.\
\
Common errors observed in people who are doing this task are not switching to a new rule after receiving “switch” feedback, and switching to a new rule after receiving “repeat” feedback.

- Perseveration errors - Erroneous category repetitions following negative feedback
- Set-loss errors - Erroneous category switches following positive feedback
/
We were interested in whether an RL agent will display a behaviour similar to that of human participants. Our hypothesis was that Q-learning algorithms will be able to approximate the conditional error distribution observed in humans. We expected that a Deep Q-network would produce the best fit to experimental data.


### We implemented the Wisconsin Card Sorting Task (WCST) environment using [OpenAI-Gym](https://gym.openai.com/).

- The cards are encoded as tuples of length 3 - (colour, shape, number) 

- Sorting rule changes randomly after 2-10 intervals (We draw these intervals from a geometric distribution with mean 3.5 so that the information remains non-uniformly distributed creating more scope of learning). 



### We implemented a vanilla Q agent

This Q-learning agebt with epsilon-greedy behavior policy, and off-policy learning, has an unique short-term memory of the last rule choice and that of a reward-streak of last n-steps.

### We implemented a DQN agent 

This agent has 2 hidden fully-connected linear layers - 50 nodes each, RELU activation function, and a replay buffer. The input here has 5 channels for capturing the relevant short-term memory information in similar fashion. 

