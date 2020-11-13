import pandas
import xml.etree.ElementTree as xee

from .BigramIndex import BiGramIndex
from .PositionalIndex import PositionalIndex
from .QueryCorrection import *
from .infromation_retrieval_system import IRSystem


def load_en():
    document = pandas.read_csv('ted_talks.csv')
    return document


def load_pr():
    url = '{http://www.mediawiki.org/xml/export-0.10/}'
    root = xee.parse('./data/Persian.xml').getroot()
    # TODO: is this the suitable xml format?
    collection = [page.find(f'{url}revision/{url}text').text for page in root.findall(url + 'page')]
    titles = [page.find(f'{url}title').text for page in root.findall(url + 'page')]
    doc_ids = [int(page.find(f'{url}id').text) for page in root.findall(url + 'page')]

    return collection, titles, doc_ids


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
