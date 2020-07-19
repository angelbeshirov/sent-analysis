import sys
sys.path.append('./tree-parser')

import argparse
import rntn
import tree as tr
import numpy as np

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
    parser.add_argument("-m", "--model", type=str, default='models/some_model.pickle', help="Model file")
    parser.add_argument("-rl", "--rootlevel", action="store_true", help="If true only sentences, otherwise sentences and words")
    parser.add_argument("-ft", "--finetune", action="store_true", help="Fine tune model from checkpoint")
    args = parser.parse_args()

    # Test
    if args.test:
        print("Testing...")
        model = rntn.RNTN.load(args.model)
        binary = True if model.output_dim == 2 else False
        test_trees = tr.load_trees(args.dataset, binary=binary)
        loss, confusion_matrix = model.test(test_trees, rootlevel=args.rootlevel)
        accuracy = 100.0 * confusion_matrix.trace() / confusion_matrix.sum()
        print("Loss = %.3f, Correct = %d / %d, Accuracy = %.4f " % (loss, confusion_matrix.trace(), confusion_matrix.sum(), accuracy))
        np.savetxt("output/test_results.txt", np.array(confusion_matrix), fmt="%s")
    else:
        model_filename=args.model
        output_dim=args.output_dim
        if args.finetune:
            print("Fine-tuning..." + args.model)
            model = rntn.RNTN.load(args.model)
            model_filename= args.model[:-7] + "_finetuned.pickle"
            output_dim=model.output_dim

            # Change max epoch and learning rate if needed
            model.learning_rate=args.learning_rate
            model.max_epochs=args.epochs
        else:
            print("Training..." + args.model)
            model = rntn.RNTN(dim=args.dim, output_dim=args.output_dim, batch_size=args.batch_size,
                reg=args.reg, learning_rate=args.learning_rate, max_epochs=args.epochs)

        # Train
        binary = True if output_dim == 2 else False
        train_trees = tr.load_trees(args.dataset, binary=binary)
        model.train(train_trees, model_filename=model_filename, rootlevel=args.rootlevel, finetune=args.finetune)

if __name__ == '__main__':
    main()
