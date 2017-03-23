import numpy as np
import test
import matplotlib.pyplot as plt

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

def softmax_loss(weights, inputs, labels):
    num = inputs.shape[0]
    scores = np.dot(inputs, weights)
    prob = softmax(scores)
    loss = (-1/num) * np.sum(labels*np.log(prob))
    grad = (-1 /num) * np.dot(inputs.T, (labels - prob))
    return loss, grad

def getPred(input, weights):
    prob = softmax(np.dot(input, weights))
    pred = np.argmax(prob, axis=1)
    return prob, pred

def getAccuracy(inputs,labels, weights):
    prob, pred = getPred(inputs, weights)
    target = labels.argmax(axis=1)
    accuracy = np.sum(np.where(pred == target, np.ones(labels.shape[0],), np.zeros(labels.shape[0],)))
    target_prob = []
    for i in range(0, len(labels)):
        target_prob.append(prob[i][target[i]])
    logliklihood = np.sum(np.log(target_prob))
    logliklihood /= inputs.shape[0]
    accuracy /= inputs.shape[0]
    return accuracy, logliklihood


if __name__ == "__main__":
    train_data = np.load("training_data.npy")
    train_labels = np.load("training_labels.npy")

    test_data = np.load("test_data.npy")
    test_labels = np.load("test_labels.npy")

    c = 10
    d = 784
    weights = np.zeros((d, c))
    iterations = 100
    lr = 0.001
    for i in range (0, iterations):
        loss, grad = softmax_loss(weights, train_data, train_labels)
        weights -= lr*grad
    #3 e
    print getAccuracy(train_data, train_labels, weights)
    print getAccuracy(test_data, test_labels, weights)

    #3 d
    fig, ax = plt.subplots()
    weights = weights.transpose()
    cax = test.plot_images(weights, ax)
    fig.colorbar(cax)
    plt.show()