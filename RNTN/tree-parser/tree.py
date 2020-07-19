import os
import re
import sys

sys.path.append('./tree-parser')

from collections import defaultdict
from nltk.parse import CoreNLPParser

import tree_parser as tp
import util

UNK = 'UNK'
DICTIONARY_FILENAME = 'models/dictionary.pickle'
TRAIN_CORPUS = 'trees/train.txt'

parser = tp.TreeParser()

def preprocess(ptb):
    transformed = re.sub(' +', ' ', ptb.replace('\n', '').replace('\t', ''))
    return re.sub('\(.*? ', ' (0 ', transformed)

def parse(text):
    parser = CoreNLPParser("http://localhost:9000")
    tree_parser = tp.TreeParser()
    result = parser.raw_parse(text.lower())
    trees = []
    for tree in [tree for tree in result]:
        tree.chomsky_normal_form()
        tree.collapse_unary(collapseRoot=True, collapsePOS=True)
        trees.append(tree_parser.create_tree_from_string(preprocess("%s" % tree)))

    return trees

def build_dictionary():
    print("Building dictionary...")

    with open(TRAIN_CORPUS, "r") as f:
        trees = [parser.create_tree_from_string(line.lower()) for line in f]

    print("Counting words...")
    words = defaultdict(int) # default value is 0
    for tree in trees:
        for leaf in tree.leaves():
            words[leaf] += 1

    dictionary[UNK] = 0
    dictionary = dict(zip(words.keys(), range(1, len(words) + 1)))
    util.save_to_file(dictionary, DICTIONARY_FILENAME)
    return dictionary

def load_dictionary():
    if not os.path.isfile(DICTIONARY_FILENAME):
        return build_dictionary()
    print("Loading dictionary...")
    return util.load_from_file(DICTIONARY_FILENAME)

def load_trees(dataset='train', binary=False):
    filename = "trees/%s.txt" % (dataset)
    with open(filename, 'r') as f:
        print("Reading %s..." % (filename))
        trees = [parser.create_tree_from_string(line.lower(), binary=binary) for line in f]
    return trees
