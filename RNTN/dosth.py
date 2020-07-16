import sys
sys.path.append('./tree-parser')

import tree as tr
import tree_parser as tp

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def build_sentences_back_from_trees(filename_to_save, trees_to_build):
    trees = tr.load_trees(trees_to_build)
    max_number_of_tokens = 0
    with open(filename_to_save, 'w') as f:
        for tree in trees:
            label, sentence = tree.get_labeled_sentence()
            f.write("%s,%s\n" % (label, sentence))
            k = len(sentence.split(" "))
            if k > max_number_of_tokens:
                max_number_of_tokens = k

    return max_number_of_tokens



# Build train sentences from the tree PST dataset
max_number_of_tokens_train = build_sentences_back_from_trees("full_sentences/train.csv", "train")
max_number_of_tokens_dev = build_sentences_back_from_trees("full_sentences/dev.csv", "dev")
max_number_of_tokens_test = build_sentences_back_from_trees("full_sentences/test.csv", "test")

print("Max number of tokens for train: %d" % max_number_of_tokens_train)
print("Max number of tokens for dev: %d" % max_number_of_tokens_dev)
print("Max number of tokens for test: %d" % max_number_of_tokens_test)




#
#test = "(2 (3 (3 Effective) (2 but)) (1 (1 too-tepid) (2 biopic)))"

#parser = tp.TreeParser()
#tree = parser.create_tree_from_string(test)
#print(tree)
#print(1 + range(len([1, 2, 3, 4, 5])))
#print(tree.all_children())
#for child in tree.all_children():
#    print(child)
#tr.parse("What the fuck is this shit")









#tree = TreeParser()
#print(tree.create_tree_from_string("(2 (2 from) (2 innocence))").to_labeled_lines())
#print(tree.create_tree_from_string("(3 (2 It) (4 (4 (2 's) (4 (3 (2 a) (4 (3 lovely) (2 film))) (3 (2 with) (4 (3 (3 lovely) (2 performances)) (2 (2 by) (2 (2 (2 Buy) (2 and)) (2 Accorsi))))))) #(2 .)))").to_labeled_lines())
