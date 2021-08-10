# -*- coding: utf-8 -*-


def card_generator(li_values=[1, 2, 3, 4], length=3):
    "generate cards given a list of values and the length of tuple."

    cards = list(permutations(li_values, length))

    return cards
