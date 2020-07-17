#!/bin/env python3

import random
import time

import numpy as np

class Optimizer:

    def __init__(self, model, learning_rate=1e-2, batch_size=30):
        self.model = model

        epsilon = 1e-8

        self.total_iter = 0
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.grads = [epsilon + np.zeros(m.shape) for m in self.model.stack]

        # initialize a variable to store all the losses
        self.losses = []
        self.exp_losses = []

    def optimize(self, trees, log_interval=1, rootlevel=False):
        m = len(trees)
        #print(rootlevel)
        # Randomly shuffle data
        random.shuffle(trees)

        it = 0
        its_per_optimization = np.floor((m - 1) / self.batch_size)
        for i in range(0, 1 + m - self.batch_size, self.batch_size):
            it += 1
            self.total_iter += 1

            batch = trees[i: i+self.batch_size]
            loss, grad = self.model.compute_loss(batch, rootlevel=rootlevel)

            self.losses.append(loss)

            # compute exponentially weighted loss
            if np.isfinite(loss):
                if self.total_iter > 1:
                    self.exp_losses.append(0.01*loss + 0.99*self.exp_losses[-1])
                else:
                    self.exp_losses.append(loss)

            # Perform one step of parameter update
            self.grads[1:] = [gt+g**2 for gt,g in zip(self.grads[1:], grad[1:])]
            update = [g*(1/np.sqrt(gt)) for gt,g in zip(self.grads[1:], grad[1:])]

            # update dictionary separately
            dEmbed = grad[0]
            dEmbedt = self.grads[0]
            for j in dEmbed.keys():
                dEmbedt[:,j] = dEmbedt[:,j] + dEmbed[j]**2
                dEmbed[j] = dEmbed[j] * (1/np.sqrt(dEmbedt[:,j]))
            update = [dEmbed] + update
            scale = -self.learning_rate

            self.model.update_params(scale=scale, update=update)

            # Logging
            if self.total_iter % log_interval == 0:
                print("\r    Iter = %d/%d, Total iterations: %d, Loss = %.4f, Expected = %.4f" %
                    (it, its_per_optimization, self.total_iter, loss, self.exp_losses[-1]), end=' ')




