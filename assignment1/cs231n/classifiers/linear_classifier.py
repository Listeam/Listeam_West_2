from __future__ import print_function

import os
from builtins import range
from builtins import object
import numpy as np
from .softmax import softmax_loss_vectorized
from past.builtins import xrange


class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(
        self,
        X,
        y,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        verbose=False,
    ):
        
        num_train, dim = X.shape
        num_classes = (
            np.max(y) + 1
        )
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            indices = np.random.choice(num_train,batch_size,replace=True)  #允许重复选择，速度更快
            X_batch = X[indices]
            y_batch = y[indices]


            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            self.W -= learning_rate * grad

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history

    def predict(self, X):

        y_pred = np.zeros(X.shape[0])

        scores = X.dot(self.W)
        scores -= np.max(scores,axis=1,keepdims=True)
        e_possibility = np.exp(scores)
        e_possibility /= np.sum(e_possibility,axis=1,keepdims=True)

        y_pred = np.argmax(e_possibility,axis=1)
        
        return y_pred
        

    def loss(self, X_batch, y_batch, reg):

        num_train = X_batch.shape[0]
        
        scores = X_batch.dot(self.W)
        scores -= np.max(scores,axis=1,keepdims=True)
        e_possibility = np.exp(scores)
        e_possibility /= np.sum(e_possibility,axis=1,keepdims=True)
        loss_total = np.sum(np.log(e_possibility[np.arange(num_train),y_batch]))
        loss = -loss_total/num_train + reg*np.sum(self.W**2)
        error = e_possibility.copy()
        error[np.arange(num_train),y_batch] -= 1
        grad = X_batch.T.dot(error)/num_train + 2*reg*self.W
        return loss, grad
        

    def save(self, fname):
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        params = {"W": self.W}
        np.save(fpath, params)
        print(fname, "saved.")

    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.W = params["W"]
        print(fname, "loaded.")
        return True


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
