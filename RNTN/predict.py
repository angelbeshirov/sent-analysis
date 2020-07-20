import sys
sys.path.append('./tree-parser')

import tree as tr

import argparse
import rntn

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--sentence", type=str, default="The weather is perfect for hiking today.", help="Sentence to classify")
args = parser.parse_args()

model = rntn.RNTN.load('models/dont-touch/best_model_retrain3733.pickle')

for tree in tr.parse(args.sentence):
    model.predict(tree)
    print(tree)
