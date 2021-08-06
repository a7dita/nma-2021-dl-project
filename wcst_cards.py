# -*- coding: utf-8 -*-

def card_generator(li_values, length):
  "generate cards given a list of values and the length of tuple."

  from itertools import permutations
  cards = list(permutations(li_values, length))

  return cards

wcst_cards = card_generator([1,2,3,4], 3)

print(len(wcst_cards))
print(wcst_cards)
