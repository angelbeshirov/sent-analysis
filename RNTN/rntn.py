import sys
sys.path.append('./tree-parser')

import collections
import csv
import pickle
import time
from datetime import datetime

import numpy as np

import tree as tr
import tree_parser as tp

import util
import optimizer as op


class RNTN:

    def __init__(self, dim=10, output_dim=5, batch_size=30, reg=10, learning_rate=1e-2, max_epochs=2):
        self.dim = dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.reg = reg
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def train(self, trees, model_filename='models/best_model.pickle'):
        self.dictionary = tr.load_dictionary()
        #self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
        self.num_words = len(self.dictionary)
        self.init_params()
        self.optimizer = op.Optimizer(self, self.learning_rate, self.batch_size)

        test_trees = tr.load_trees('test')

        with open("training.csv", "a", newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            fieldnames = ["Timestamp", "Vector size", "Learning rate",
                          "Batch size", "Regularization",
                          "Train loss", "Train accuracy",
                          "Test loss", "Test accuracy", "Model filename"]
            if csvfile.tell() == 0:
                csvwriter.writerow(fieldnames)
            train_loss, train_accuracy, test_loss, test_accuracy = 0, 0, 0, 0

            for epoch in range(self.max_epochs):
                print("Running epoch %d/%d..." % (epoch + 1, self.max_epochs))
                start = time.time()
                self.optimizer.optimize(trees)

                # Test the model on train and test set
                train_loss, train_confusion_matrix = self.test(trees)
                train_accuracy = 100.0 * train_confusion_matrix.trace() / train_confusion_matrix.sum()
                test_loss, test_confusion_matrix = self.test(test_trees)
                test_accuracy = 100.0 * test_confusion_matrix.trace() / test_confusion_matrix.sum()

                end = time.time()
                print("\n    Train accuracy=%.2f, Test accuracy=%.2f, Time per epoch = %.4f" %
                    (train_accuracy, test_accuracy, end-start))


            # Append data to CSV file
            row = [datetime.now(), self.dim, self.learning_rate,
                   self.batch_size, self.reg,
                   train_loss, train_accuracy,
                   test_loss, test_accuracy, model_filename]
            csvwriter.writerow(row)

        self.save(model_filename)

    def test(self, trees):
        loss, confusin_matrix = 0.0, np.zeros((5,5))

        # forward pass
        for tree in trees:
            _loss, _confusin_matrix = self.forward_pass(tree)
            loss += _loss
            confusin_matrix += _confusin_matrix

        return loss / len(trees), confusin_matrix

    def predict(self, tree):
        if tree.isleaf():

            # check if the word map contains this word
            if tree.text in self.dictionary:
                tree.vector = self.embedding[:, self.dictionary[tree.text]]
            else:
                tree.vector = self.embedding[:, self.dictionary[tr.UNK]]
        else:
            self.predict(tree.children[0])
            self.predict(tree.children[1])

            # compute output
            h = np.hstack([tree.children[0].vector, tree.children[1].vector])
            tree.vector = np.tanh(
                np.tensordot(self.V, np.outer(h, h), axes=([1, 2], [0, 1])) + np.dot(self.W, h) + self.b) # np.outer is elementwise multiplication

        # softmax
        tree.output = util.softmax(np.dot(self.Ws, tree.vector) + self.bs)
        label = np.argmax(tree.output)
        tree.label = label
        return tree

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.dim, f)
            pickle.dump(self.output_dim, f)
            pickle.dump(self.batch_size, f)
            pickle.dump(self.reg, f)
            pickle.dump(self.learning_rate, f)
            pickle.dump(self.max_epochs, f)
            pickle.dump(self.stack, f)
            pickle.dump(self.dictionary, f)
            print("Model saved successfully to file %s" % (filename))

    def load(filename):
        with open(filename, 'rb') as f:
            dim = pickle.load(f)
            output_dim = pickle.load(f)
            batch_size = pickle.load(f)
            reg = pickle.load(f)
            learning_rate = pickle.load(f)
            max_epochs = pickle.load(f)
            stack = pickle.load(f)

            model = RNTN(dim=dim, output_dim=output_dim, batch_size=batch_size,
                         reg=reg, learning_rate=learning_rate, max_epochs=max_epochs)
            model.stack = stack
            model.embedding, model.V, model.W, model.b, model.Ws, model.bs = model.stack
            model.dictionary = pickle.load(f)
            print("Model loaded successfully from file %s" % (filename))
            return model

    def init_params(self):
        print("Initializing parameters...")
        #print(self.num_words)

        # word vectors
        self.embedding = 0.01 * np.random.randn(self.dim, self.num_words) # 10xN

        # RNTN parameters
        self.V = 0.01 * np.random.randn(self.dim, 2*self.dim, 2*self.dim) # 10x20x20
        self.W = 0.01 * np.random.randn(self.dim, 2*self.dim) # 10x20
        self.b = 0.01 * np.random.randn(self.dim) # 10

        # Softmax parameters
        self.Ws = 0.01 * np.random.randn(self.output_dim, self.dim) # 5x10
        self.bs = 0.01 * np.random.randn(self.output_dim) # 5

        self.stack = [self.embedding, self.V, self.W, self.b, self.Ws, self.bs]

        # Gradients
        self.dV = np.empty_like(self.V)
        self.dW = np.empty_like(self.W)
        self.db = np.empty_like(self.b)
        self.dWs = np.empty_like(self.Ws)
        self.dbs = np.empty_like(self.bs)

    def forward_pass(self, tree):
        cost = 0.0
        confusin_matrix = np.zeros((5,5)) # Confusion matrix

        if tree.isleaf():
            # output = word vector
            if tree.text in self.dictionary:
                tree.vector = self.embedding[:, self.dictionary[tree.text]]
            else:
                tree.vector = self.embedding[:, self.dictionary[tr.UNK]]
        else:
            # calculate output of child nodes
            lcost, lconfusion_matrix = self.forward_pass(tree.children[0])
            rcost, rconfusion_matrix = self.forward_pass(tree.children[1])
            cost += lcost + rcost
            confusin_matrix += lconfusion_matrix + rconfusion_matrix

            # compute output
            h = np.hstack([tree.children[0].vector, tree.children[1].vector])
            tree.vector = np.tanh(
                np.tensordot(self.V, np.outer(h, h), axes=([1, 2], [0, 1])) +
                np.dot(self.W, h) + self.b)

        tree.output = util.softmax(np.dot(self.Ws, tree.vector) + self.bs)

        # cost (error)
        cost -= np.log(tree.output[int(tree.label)]) # cross-entropy loss
        true_label = int(tree.label)
        predicted_label = np.argmax(tree.output)
        confusin_matrix[true_label, predicted_label] += 1 # 1 in the diagonal = correct prediction, incorrect otherwise

        return cost, confusin_matrix

    def compute_loss(self, trees):
        loss, gradient = 0.0, None
        self.embedding, self.V, self.W, self.b, self.Ws, self.bs = self.stack

        for tree in trees:
            _loss, _ = self.forward_pass(tree)
            loss += _loss

        # Initialize
        self.dEmbed = collections.defaultdict(lambda: np.zeros((self.dim,)))
        self.dV = np.zeros_like(self.V)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.dWs = np.zeros_like(self.Ws)
        self.dbs = np.zeros_like(self.bs)

        # Back propagattion
        for tree in trees:
            self.back_prop(tree)

        scale = 1.0 / self.batch_size
        for v in self.dEmbed.values():
            v *= scale

        loss += 0.5 * self.reg * np.sum(self.V ** 2)
        loss += 0.5 * self.reg * np.sum(self.W ** 2)
        loss += 0.5 * self.reg * np.sum(self.Ws ** 2)
        loss *= scale

        gradient = [self.dEmbed,
                   scale * (self.dV + (self.reg * self.V)),
                   scale * (self.dW + (self.reg * self.W)),
                   scale * self.db,
                   scale * (self.dWs + (self.reg * self.Ws)),
                   scale * self.dbs]

        return loss, gradient

    def back_prop(self, tree, error=None):
        # softmax grad
        deltas = tree.output # 5x1
        deltas[int(tree.label)] -= 1.0
        self.dWs += np.outer(deltas, tree.vector)
        self.dbs += deltas
        deltas = np.dot(self.Ws.T, deltas)
        if error is not None:
            deltas += error
        deltas *= (1 - tree.vector**2)

        # leaf node embedding gradient
        if tree.isleaf():
            if tree.text in self.dictionary:
                index = self.dictionary[tree.text]
            else:
                index = self.dictionary[tr.UNK]

            self.dEmbed[index] += deltas
            return
        # Hidden gradients
        else:
            h = np.hstack([tree.children[0].vector, tree.children[1].vector]) # 10x1 -> 10x2
            outer = np.outer(deltas, h)
            self.dV += (np.outer(h, h)[..., None] * deltas).T
            self.dW += outer
            self.db += deltas

            # Compute error for children
            deltas = np.dot(self.W.T, deltas)
            deltas += np.tensordot(self.V.transpose((0,2,1)) + self.V, outer.T,
                                   axes=([1,0], [0,1]))

            self.back_prop(tree.children[0], deltas[:self.dim])
            self.back_prop(tree.children[1], deltas[self.dim:])

    def update_params(self, scale, update):
        self.stack[1:] = [P+scale*dP for P, dP in zip(self.stack[1:], update[1:])]
        # Update embeddings
        dEmbed = update[0]
        for j in dEmbed.keys():
            self.embedding[:,j] += scale*dEmbed[j]



