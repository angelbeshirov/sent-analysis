from nltk.parse import CoreNLPParser
from nltk.tree import ParentedTree

def load_trees(dataset='train'):
    filename = "trees/{}.txt".format(dataset)
    with open(filename, 'r') as f:
        print("Reading '{}'...".format(filename))
        trees = [ParentedTree.fromstring(line.lower()) for line in f]
    return trees


trees = load_trees()
for tree in trees:
    tree.vector = [0, 1, 2, 3]

