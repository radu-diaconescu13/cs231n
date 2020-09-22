from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_training_examples = X.shape[0]
    num_classes = W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****s
    for i in range(num_training_examples):
        scores = X[i].dot(W)
        adjusted_scores = scores - np.max(scores)
        denom = np.sum(np.exp(adjusted_scores))
        nom = np.exp(adjusted_scores[y[i]])
        loss += -np.log(nom/denom)

        for j in range(num_classes):
            denom = np.sum(np.exp(adjusted_scores))
            nom = np.exp(adjusted_scores[j])
            if j != y[i]:
              dW[:, j] += X[i] * (nom/denom)
            else:
              dW[:, j] += X[i] * (nom/denom - 1)

    loss = loss/num_training_examples
    loss += reg * np.sum(W * W)

    dW = dW/num_training_examples + reg * 2 * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_classes = W.shape[1]

    scores = X.dot(W)
    adjusted_scores = scores - np.max(scores)
    exponential_scores = np.exp(adjusted_scores)
    sums_for_each_example = np.sum(np.exp(adjusted_scores), axis=1)

    logs = np.log(sums_for_each_example)
    correct_scores = adjusted_scores[range(num_train), list(y)]

    softmax_output = exponential_scores/(sums_for_each_example.reshape(-1, 1))

    softmax_output[range(num_train), list(y)] += - 1
    dW = (X.T).dot(softmax_output)
    dW = dW / num_train + 2 * reg * W

    losses = -correct_scores + logs
    loss = np.sum(losses)/num_train + reg * np.sum(W*W)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
