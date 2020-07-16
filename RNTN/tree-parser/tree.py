#!/bin/env python3

import os
from collections import defaultdict

from nltk.parse import CoreNLPParser
from nltk.tree import ParentedTree

import sys
sys.path.append('./tree-parser')
import tree_parser as tp


import util

UNK = 'UNK'

DICTIONARY_FILENAME = 'models/dictionary.pickle'
TRAIN_CORPUS = 'trees/train.txt'

parser = tp.TreeParser()

# WIP to parse a raw sentence into a semantic tree NOT YET DONE
def parse(text):
    parser = CoreNLPParser("http://localhost:9000")
    result = parser.raw_parse(text.lower())
    trees = [tree for tree in result]
    for tree in trees:
        print(tree)
        tree.chomsky_normal_form()
        tree.collapse_unary(collapseRoot=True, collapsePOS=True)
    trees = [ParentedTree.convert(tree) for tree in trees]
    return trees

def build_dictionary():
    print("Building dictionary...")

    with open(TRAIN_CORPUS, "r") as f:
        trees = [parser.create_tree_from_string(line.lower()) for line in f]

    print("Counting words...")
    words = defaultdict(int)
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

def load_trees(dataset='train'):
    filename = "trees/%s.txt" % (dataset)
    with open(filename, 'r') as f:
        print("Reading '%s'..." % (filename))
        trees = [parser.create_tree_from_string(line.lower()) for line in f]
    return trees

if __name__ == '__main__':
    word_map = load_word_map()
