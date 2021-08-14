#!/usr/bin/env python3
# Run this to test the WCST environment:
# Take random actions in the env until the game is over
import wcst
import random

if __name__ == "__main__":
    env = wcst.WCST()
    env.reset()
    cum_reward = 0

    for i in range(250):
        action = random.randint(1, 4)  # taking random actions on every step
        reward, obs, done, _ = env.step(action)
        print("Step {}".format(i))
        print(
            "✨ Action taken: {}, reward: {}, card: {}, finished? {}".format(
                action, reward, obs, done
            )
        )
        cum_reward += reward
    env.close()
    print("Cumulative reward: {} 🪙".format(cum_reward))
