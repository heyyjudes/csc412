from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np
from autograd import grad
from autograd.util import quick_grad_check
from builtins import range
from scipy.misc import logsumexp
from scipy.special import logit
from kmeans import *
from bayes import *

def sigmoid(x):
    return 0.5*(np.tanh(x) + 1)

def log_bernoulli_prod(flat_data, theta, pi):
    '''
    return log(p(c|x)) = log(p(c,x)/sum_c(p(c,x))
    :param flat_data:
    :param theta:
    :param pi:
    :return:
    '''
    c, d = theta.shape
    P_n = []
    # for i in range(0, len(flat_data)):
    #     if flat_data[i] > 0.5:
    #         P_n.append(theta[:, i])
    #     else:
    #         P_n.append(1 - theta[:, i])
    P_n = np.where(flat_data > 0.5, theta, np.ones((c, d)) - theta)
    sum_c = np.sum(np.log(P_n), axis=1)
    sum_c += np.log(pi)
    #marginalizing all X's
    P_x = logsumexp(sum_c)
    sum_c -= P_x
    return sum_c

def Bernoulli(weights, inputs):
    # Outputs probability of a label being true according to logistic model.
    result = []
    K = 30
    pi = float(1/K)
    for i in range (0, len(inputs)):
        P = log_bernoulli_prod(inputs[i], weights, pi)
        result.append(np.sum(P))
    return np.asarray(result)

def training_loss(weights):
    # Training loss is the negative log-likelihood of the training labels.
    preds = Bernoulli(weights, inputs)
    label_probabilities = preds * targets + (1 - preds) * (1 - targets)
    return -np.sum(np.log(label_probabilities))

# Build a toy dataset.
inputs = np.load("toy_training_data.npy")
targets = np.load("toy_training_labels.npy")
targets = np.argmax(targets, axis=1)

weights = KMeans(np.transpose(inputs), 10, 1)
weights = np.transpose(weights)

# Build a function that returns gradients of training loss using autograd.
training_gradient_fun = grad(training_loss)

# Check the gradients numerically, just to be safe.
#weights = np.zeros((10, inputs.shape[1]))

#quick_grad_check(training_loss, weights)

# Optimize weights using gradient descent.
print("Initial loss:", training_loss(weights))
for i in range(100):
    weights -= training_gradient_fun(weights) * 0.01

print("Trained loss:", training_loss(weights))