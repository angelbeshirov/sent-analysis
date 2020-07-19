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

    def train(self, trees, model_filename='models/best_model.pickle', rootlevel=False, finetune=False):


        if finetune == False:
            self.dictionary = tr.load_dictionary()
            self.num_words = len(self.dictionary)
            self.init_params()
        self.optimizer = op.Optimizer(self, self.learning_rate, self.batch_size)

        binary = True if self.output_dim == 2 else False
        test_trees = tr.load_trees('test', binary=binary)

        accuracies=[0.0]

        if finetune:
            train_loss, train_confusion_matrix = self.test(trees, rootlevel)
            train_accuracy = 100.0 * train_confusion_matrix.trace() / train_confusion_matrix.sum()
            test_loss, test_confusion_matrix = self.test(test_trees, rootlevel)
            test_accuracy = 100.0 * test_confusion_matrix.trace() / test_confusion_matrix.sum()
            print("First accuracy before finetuning starts Train:%.3f, Test:%.3f" % (train_accuracy, test_accuracy))
            train_accuracies.append(train_accuracy)
            accuracies.append(test_accuracy)

        with open("training.csv", "a", newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            fieldnames = ["Timestamp", "Vector size", "Learning rate",
                          "Batch size", "Regularization",
                          "Train loss", "Train accuracy",
                          "Test loss", "Test accuracy", "Model filename"]
            if csvfile.tell() == 0:
                csvwriter.writerow(fieldnames)

            train_loss, train_accuracy, test_loss, test_accuracy = 0.0, 0.0, 0.0, 0.0

            for epoch in range(self.max_epochs):
                print("Running epoch %d/%d..." % (epoch + 1, self.max_epochs))
                start = time.time()
                self.optimizer.optimize(trees, rootlevel=rootlevel)

                # Test the model on train and test set
                train_loss, train_confusion_matrix = self.test(trees, rootlevel)
                train_accuracy = 100.0 * train_confusion_matrix.trace() / train_confusion_matrix.sum()
                test_loss, test_confusion_matrix = self.test(test_trees, rootlevel)
                test_accuracy = 100.0 * test_confusion_matrix.trace() / test_confusion_matrix.sum()

                end = time.time()
                print("\n    Train accuracy=%.2f, Test accuracy=%.2f, Time per epoch = %.4f" %
                    (train_accuracy, test_accuracy, end-start))

                if max(accuracies) < test_accuracy:
                    self.save(model_filename)
                    print("Saving new best model with test accuracy: %.3f" % test_accuracy)

                #if max(train_accuracies) < train_accuracy:
                #    acc_converted = int(train_accuracy*100)
                #    self.save(model_filename[:-7] + "_finetuned_train%d.pickle" % acc_converted)
                #    print("Saving new best model with train accuracy: %.3f" % train_accuracy)

                accuracies.append(test_accuracy)

                # Append data to CSV file
                row = [datetime.now(), self.dim, self.learning_rate,
                       self.batch_size, self.reg,
                       train_loss, train_accuracy,
                       test_loss, test_accuracy, model_filename]
                csvwriter.writerow(row)
            #self.save(model_filename)


    def test(self, trees, rootlevel=False):
        loss = 0.0
        confusion_matrix = np.zeros((self.output_dim,self.output_dim)) # Confusion matrix

        # forward pass
        for tree in trees:
            _loss, _confusion_matrix = self.forward_pass(tree, rootlevel)
            loss += _loss
            confusion_matrix += _confusion_matrix

        return loss / len(trees), confusion_matrix

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

            # Forward pass from the paper hT*V*h + Wh
            # h = 2d, V = d,2d,2d
            # sum-reduction on 0 and 1 -> 0 + d + 2d -> d,2d * 2d,1 -> d,          Wxh -> 10x20*20,1 -> 10,1
            tree.vector = np.tanh(np.dot(np.tensordot(h.T, self.V, axes=(0, 1)), h) + np.dot(self.W, h) + self.b)

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
            pickle.dump(self.tensors, f)
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
            tensors = pickle.load(f)

            model = RNTN(dim=dim, output_dim=output_dim, batch_size=batch_size,
                         reg=reg, learning_rate=learning_rate, max_epochs=max_epochs)
            model.tensors = tensors
            model.embedding, model.V, model.W, model.b, model.Ws, model.bs = model.tensors
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

        self.tensors = [self.embedding, self.V, self.W, self.b, self.Ws, self.bs]

        # Gradients
        self.dV = np.empty_like(self.V)
        self.dW = np.empty_like(self.W)
        self.db = np.empty_like(self.b)
        self.dWs = np.empty_like(self.Ws)
        self.dbs = np.empty_like(self.bs)

    def forward_pass(self, tree, rootlevel=False):
        loss = 0.0
        confusion_matrix = np.zeros((self.output_dim,self.output_dim)) # Confusion matrix

        if tree.isleaf():
            # output = word vector
            if tree.text in self.dictionary:
                tree.vector = self.embedding[:, self.dictionary[tree.text]]
            else:
                tree.vector = self.embedding[:, self.dictionary[tr.UNK]]
        else:
            # calculate output of child nodes
            lloss, lconfusion_matrix = self.forward_pass(tree.children[0], rootlevel=rootlevel)
            rloss, rconfusion_matrix = self.forward_pass(tree.children[1], rootlevel=rootlevel)
            loss += lloss + rloss
            confusion_matrix += lconfusion_matrix + rconfusion_matrix

            # compute output
            h = np.hstack([tree.children[0].vector, tree.children[1].vector])

            # Forward pass from the paper hT*V*h + Wh
            # h = 2d, V = d,2d,2d
            # sum-reduction on 0 and 1 -> 0 + d + 2d -> d,2d * 2d,1 -> d,          Wxh -> 10x20*20,1 -> 10,1
            tree.vector = np.tanh(np.dot(np.tensordot(h, self.V, axes=(0, 1)), h) + np.dot(self.W, h) + self.b) # 10x20 * 20,1

        tree.output = util.softmax(np.dot(self.Ws, tree.vector) + self.bs) # 5xd * d, -> 5,

        # loss
        loss -= np.log(tree.output[int(tree.label)]) # cross-entropy loss
        if rootlevel:
            if tree.parent == None:
                true_label = int(tree.label)
                predicted_label = np.argmax(tree.output)
                confusion_matrix[true_label, predicted_label] += 1
                #if true_label == predicted_label:
                    #print("Got correct with label %d" % predicted_label)
                    #print(tree)
        else:
            true_label = int(tree.label)
            predicted_label = np.argmax(tree.output)
            confusion_matrix[true_label, predicted_label] += 1 # 1 in the diagonal = correct prediction, incorrect otherwise

        return loss, confusion_matrix

    def compute_loss(self, trees, rootlevel=False):
        loss, gradient = 0.0, None
        self.embedding, self.V, self.W, self.b, self.Ws, self.bs = self.tensors

        for tree in trees:
            _loss, _ = self.forward_pass(tree, rootlevel=rootlevel)
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
        loss += 0.5 * self.reg * np.sum(self.embedding ** 2) # Changed
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
        deltas = np.dot(self.Ws.T, deltas) # 10x1
        if error is not None:
            deltas += error
        deltas *= (1 - tree.vector**2) # tanh derivative

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
            h = np.hstack([tree.children[0].vector, tree.children[1].vector]) # 20,
            self.dV += np.tensordot(deltas[..., None], np.outer(h, h)[None, ...], axes=((1), (0))) # Add 1 dimension to make it like dx2dx2d for V
            self.dW += np.outer(deltas, h) # 10x20
            self.db += deltas

            # Compute the error for children
            deltas_children = np.dot(self.W.T, deltas) # 10x20T*10x1 -> 20x1
            deltas_children += np.tensordot(np.tensordot(deltas, self.V.transpose((0,2,1)) + self.V, axes=((0),(0))), h, axes=(1, 0)) # outer.T 20x10
            self.back_prop(tree.children[0], deltas_children[:self.dim])
            self.back_prop(tree.children[1], deltas_children[self.dim:])

    def update_params(self, scale, update):
        self.tensors[1:] = [P+scale*dP for P, dP in zip(self.tensors[1:], update[1:])]

        # Update embeddings
        dEmbed = update[0]
        for j in dEmbed.keys():
            self.embedding[:,j] += scale*dEmbed[j]



