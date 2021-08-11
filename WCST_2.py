import numpy as np
import random

deck = [(1, 2, 3),
        (1, 2, 4),
        (1, 3, 2),
        (1, 3, 4),
        (1, 4, 2),
        (1, 4, 3),
        (2, 1, 3),
        (2, 1, 4),
        (2, 3, 1),
        (2, 3, 4),
        (2, 4, 1),
        (2, 4, 3),
        (3, 1, 2),
        (3, 1, 4),
        (3, 2, 1),
        (3, 2, 4),
        (3, 4, 1),
        (3, 4, 2),
        (4, 1, 2),
        (4, 1, 3),
        (4, 2, 1),
        (4, 2, 3),
        (4, 3, 1),
        (4, 3, 2)]

choices = [(1, 1, 1),
           (2, 2, 2),
           (3, 3, 3),
           (4, 4, 4)]

rule = [0, 1, 2]

success_count = 0
switch_count = 0

print(choices)

for i in range(1,251):

  card_idx = random.randint(0, 23)
  card = deck[card_idx]
  print("Card is: ", deck[card_idx])

  if success_count != 0 and success_count % 2 == 0:
    rule_idx = random.randint(0, 2)
    success_count = 0
    switch_count = switch_count + 1
  else:
    rule_idx = rule_idx

  print("Rule is: ", rule[rule_idx])

  response_idx = int(input("Card 1, 2, 3, 4? "))

  response_valid = card[rule_idx] == choices[response_idx - 1][rule_idx]

  if response_valid == True:
    success_count = success_count + 1
  else:
    success_count = 0

  if switch_count == 41:
    break

  print("Response was: ", response_valid)
  print("Consecutive successes: ", success_count)
