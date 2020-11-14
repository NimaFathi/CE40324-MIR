import pandas
import xml.etree.ElementTree as xee

from .BigramIndex import BiGramIndex
from .PositionalIndex import PositionalIndex
from .QueryCorrection import *
from .infromation_retrieval_system import IRSystem


# print("pee pee")
op1 = int(input("Please select the language of the document:\n1. English\t2. Persian\n"))
while True:
    if op1 == 1:
        lang = 'english'
        break
    elif op1 == 2:
        lang = 'persian'
        break
    else:
        print("please enter a valid option (1 or 2)")
        op1 = int(input("1. English\t2. Persian\n"))

irs = IRSystem(lang)

k_w = input("Please enter k & w:\n").split()
k, w = int(k_w[0]), int(k_w[1])

query = input("Now enter your query:\n")

print("language: ", lang, "\nk: ", k, "\nw: ", w, "\nquery: ", query)
