import random
import numpy as np
import gym
from gym import spaces
from gym.envs.classic_control import rendering
from PIL import Image
import pyglet
import time
from helper_functions import *

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

        # Card choices (action)

        self.choice_cards = [
            (1, 1, 1),  # ("red", "triangle" "one")
            (2, 2, 2),  # ("green", "star", "two")
            (3, 3, 3),  # ("yellow", "cross", "three")
            (4, 4, 4),  # ("blue", "circle", "four")
        ]

        # initialise cards and rule

        self.card_deck = card_generator()
        self.card = self._next_observation()  # the card to categorise
        # NOTE please do not delete the card deck! Not a duplicate variable - Now they are not. :P
        self.rule = np.random.choice([0, 1, 2])

        # initialise counters
        self.current_step = 0
        self.success_counter = 0  # number of correct responses in a row
        self.switch_counter = 0  # keep track of rule switches; max should be 41
        self.viewer = None

    def _next_observation(self):
        """a card is shown with values of (colour, form, num of elements)"""
        return random.choice(self.card_deck)
        # NOTE please do not replace this w observation space
        # NOTE do we discard used cards? -- no, we have 24 unique cards but 250 trials

    # def _calculate_reward(self, action):
    #     """Give reward of +1 if the action is correct
    #     or -1 otherwise"""
    #     reward = +1 if action == self.right_action else -1
    #     return reward

    def step(self, action):
        """Take one step in the environment"""

        self.current_step += 1

        # choice = self.choice_cards[int(action)] # Choice conversion
        # step_rule =  self.rule # Record rule
        # rule_feature = self.card[self.rule] # Record right feature

        # if choice[self.rule] == self.card[self.rule]: # correct move
        if action == map_rule_to_action(self.card, self.rule):
            self.success_counter += 1  # update success counter
            success_streak = min(
                np.random.geometric(0.4) + 1, 12
            )  # Randomize success threshold
            reward = 1  # Positive reward
            # NOTE (RE calculate_reward): Is this not enough? -- Probably enough, but having a separate function might be more readable?

            if self.success_counter >= success_streak:
                available_rules = [x for x in [0, 1, 2] if x != self.rule]
                self.rule = np.random.choice(available_rules)
                # NOTE We need to exclude the current rule
                # NOTE Don't need right_action
                self.switch_counter += 1  # Update switch counter

        else:  # wrong move
            self.success_counter = 0
            self.rule = self.rule  # Rule stays the same
            reward = -1  # Negative reward
            # NOTE (RE calculate_reward): Same as above

        self.card = self._next_observation()  # show a new card
        obs = self.card

        done = self.current_step >= 250 or self.switch_counter >= 41
        # game over after 250 steps or 41 rule switches

        return reward, obs, done, {}

    def reset(self):
        """reset the state of the environment to the initial state"""
        self.card = self._next_observation()
        self.current_step = 0
        self.rule = np.random.choice([0, 1, 2])
        # self.right_action = map_rule_to_action(self.card, self.rule)
        # NOTE Don't need right_action at initialization/reset.
        self.success_counter = 0  # reset success success_counter
        self.switch_counter = 0  # reset rule switch counter

    def render(
        self,
        action,
        reward,
        mode="human",
        close=False,
        graphical_render=False,
    ):
        """Render environment to screen"""

        print("Step: {current_step}".format(current_step=self.current_step))

        if graphical_render:

            if self.viewer is None:

                self.viewer = rendering.SimpleImageViewer()
                self.viewer.width = 960
                self.viewer.height = 540

                self.viewer.window = pyglet.window.Window(
                    width=self.viewer.width,
                    height=self.viewer.height,
                    caption="WCST session",
                    display=self.viewer.display,
                    vsync=False,
                    resizable=False,
                )

            ## Define images

            back = Image.open("stimuli/background.png")
            im1 = Image.open("stimuli/111.png")
            im2 = Image.open("stimuli/222.png")
            im3 = Image.open("stimuli/333.png")
            im4 = Image.open("stimuli/444.png")
            im5 = Image.open(
                f"stimuli/cards/{''.join(str(num + 1) for num in self.card)}.png"
            )
            im6 = Image.open("stimuli/frame.png")
            im7 = Image.open("stimuli/switch.jpg")
            im8 = Image.open("stimuli/repeat.png")

            back_im = back.copy()

            ## Call the first frame - 4 cards + 1 stimuli card

            back_im.paste(im1, (100, 50))
            back_im.paste(im2, (300, 50))
            back_im.paste(im3, (500, 50))
            back_im.paste(im4, (700, 50))
            back_im.paste(im5, (400, 300))

            self.viewer.imshow(np.array(back_im))

            ## Wait for a moment
            time.sleep(0.4)

            ## Call the second frame - a selection frame around the card of choice
            pile = action + 1

            if pile == 1:
                back_im.paste(im6, (90, 40))
                back_im.paste(im1, (100, 50))
            elif pile == 2:
                back_im.paste(im6, (290, 40))
                back_im.paste(im2, (300, 50))
            elif pile == 3:
                back_im.paste(im6, (490, 40))
                back_im.paste(im3, (500, 50))
            else:
                back_im.paste(im6, (690, 40))
                back_im.paste(im4, (700, 50))

            self.viewer.imshow(np.array(back_im))

            ## Wait for a moment again
            time.sleep(0.3)

            ## Call the third frame - "switch"/"repeat" instruction according to the reward
            if reward > 0:
                back_im.paste(im8, (170, 350))
            else:
                back_im.paste(im7, (170, 350))

            self.viewer.imshow(np.array(back_im))

            ## Freeze there for a moment
            time.sleep(0.5)

            return self.viewer.isopen

    def close(self):
        super().close()
