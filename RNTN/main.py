#!/bin/env python3

import sys
sys.path.append('./tree-parser')

import argparse
import rntn
import tree as tr

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--dim", type=int, default=25, help="Vector space dimension")
    parser.add_argument("-k", "--output-dim", type=int, default=5, help="Number of output classes")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("-f", "--dataset", type=str, default="train", choices=['train', 'dev', 'test'],
        help="Dataset to use")
    parser.add_argument("-l", "--learning-rate", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("-b", "--batch-size", type=int, default=30, help="Batch size")
    parser.add_argument("-r", "--reg", type=float, default=1e-6, help="Regularization")
    parser.add_argument("-t", "--test", action="store_true", help="Test a model")
    parser.add_argument("-m", "--model", type=str, default='models/RNTN.pickle', help="Model file")
    parser.add_argument("-rl", "--rootlevel", action="store_true", help="Only sentences or sentences and words")
    args = parser.parse_args()

    # Test
    if args.test:
        print("Testing...")
        model = rntn.RNTN.load(args.model)
        test_trees = tr.load_trees(args.dataset)
        loss, confusion_matrix = model.test(test_trees, rootlevel=args.rootlevel)
        accuracy = 100.0 * confusion_matrix.trace() / confusion_matrix.sum()
        print("Loss = %.3f, Correct = %d / %d, Accuracy = %.2f " % (loss, confusion_matrix.trace(), confusion_matrix.sum(), accuracy))
    else:
        print("Training...")
        # Initialize the model
        #print(args.rootlevel)
        model = rntn.RNTN(dim=args.dim, output_dim=args.output_dim, batch_size=args.batch_size,
            reg=args.reg, learning_rate=args.learning_rate, max_epochs=args.epochs)

        # Train
        train_trees = tr.load_trees(args.dataset)

        model.train(train_trees, model_filename=args.model, rootlevel=args.rootlevel)

if __name__ == '__main__':
    main()
