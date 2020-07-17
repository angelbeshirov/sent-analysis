import sys
sys.path.append('../')

import tree as tr
import tree_parser as tp

def map_label_to_binary(label):
    if label == 0 or label == 1:
        return 0
    elif label == 3 or label == 4:
        return 1
    return -1

def build_sentences_back_from_trees(filename_to_save, trees_to_build, binary=False):
    trees = tr.load_trees(trees_to_build)
    max_number_of_tokens = 0
    with open(filename_to_save, 'w') as f:
        f.write("Label\tSentence\n")
        for tree in trees:
            label, sentence = tree.get_labeled_sentence()
            if binary:
                label = map_label_to_binary(label)
                if label == -1:
                    continue
            f.write("%s\t%s\n" % (label, sentence))
            k = len(sentence.split(" "))
            if k > max_number_of_tokens:
                max_number_of_tokens = k

    return max_number_of_tokens



# Build train sentences from the tree PST dataset
max_number_of_tokens_train = build_sentences_back_from_trees("full_sentences/train.tsv", "train", True)
max_number_of_tokens_dev = build_sentences_back_from_trees("full_sentences/dev.tsv", "dev", True)
max_number_of_tokens_test = build_sentences_back_from_trees("full_sentences/test.tsv", "test", True)

print("Max number of tokens for train: %d" % max_number_of_tokens_train)
print("Max number of tokens for dev: %d" % max_number_of_tokens_dev)
print("Max number of tokens for test: %d" % max_number_of_tokens_test)
