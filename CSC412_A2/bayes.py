import test
import numpy as np
from scipy.misc import logsumexp
import matplotlib.pyplot as plt
#np.seterr(all='raise')

def fit_theta(train_data, train_labels):
    c = 10
    n, d = train_data.shape
    train_data = np.where(train_data>0.5, np.ones((n, d)), np.zeros((n, d)))
    pos = np.zeros((c, d))
    cl = np.sum(train_labels, axis=0)
    for i in range(0, n):
        for j in range(0, c):
            if train_labels[i][j]:
                non_zero = j
        pos[non_zero] += train_data[i]
    neg = cl[:, np.newaxis] - pos
    theta = (pos+1)/(pos+neg+2)
    #uncomment to generate photo
    test.plot_images(theta, plt)
    plt.show()
    return theta

def log_bernoulli_prod(flat_data, theta, pi):
    '''
    return log(p(c|x)) = log(p(c,x)/sum_c(p(c,x))
    :param flat_data:
    :param theta:
    :param pi:
    :return:
    '''
    c, d = theta.shape
    P_n = np.where(flat_data > 0.5, theta, np.ones((c, d)) - theta)
    # sum_c = np.product(P_n, axis=1)
    # sum_c = np.multiply(pi, sum_c)
    # P_x = np.sum(sum_c)
    # sum_c /= P_x
    sum_c = np.sum(np.log(P_n), axis=1)
    sum_c += np.log(pi)
    #marginalizing all X's
    P_x = logsumexp(sum_c)
    sum_c -= P_x
    return sum_c

def log_bernoulli_prod_top(flat_data, theta, pi):
    '''
    return log(p(c|x)) = log(p(c,x)/sum_c(p(c,x))
    :param flat_data:
    :param theta:
    :param pi:
    :return:
    '''
    c, d = theta.shape
    P_n = np.where(flat_data > 0.5, theta, np.ones((c, d)) - theta)
    P_n = P_n[:392]
    sum_c = np.sum(np.log(P_n), axis=1)
    sum_c += np.log(pi)
    #sum_c = np.multiply(pi, sum_c)
    # marginalizing all X's
    P_x = logsumexp(sum_c)
    sum_c -= P_x
    return sum_c

def avg_log_likelihood(data, labels, theta):
    '''
    1/N*sum(logP(c|x, theta, pi))

    :return:
    '''
    pi = 0.1
    c, d = theta.shape
    average = 0
    for i in range(0, len(data)):
        sum_c = log_bernoulli_prod(data[i], theta, pi)
        target = sum_c[np.argmax(labels[i])]

        average += target
    average /= len(data)
    return average

def calc_acc(data, labels, theta):
    '''
    not using pi should not matter in accuracy
    :param data:
    :param labels:
    :param theta:
    :return:
    '''
    pi = 0.1
    accuracy = 0
    for i in range(0, len(data)):
        prob_arr = log_bernoulli_prod(data[i], theta, pi)
        pred = np.argmax(prob_arr)
        target = np.argmax(labels[i])
        if pred == target:
            accuracy += 1
        # else:
        #     print prob_arr
        #     print 'pred', pred
        #     print 'target', target
    accuracy /= float(len(data))
    return accuracy

def class_liklihood(labels):
    n, m = labels.shape
    cl = np.sum(labels, axis=0)
    cl = cl.astype(float) / n
    return cl

def sample_marginal(data, labels, theta, size):
    c, d = theta.shape
    cl = class_liklihood(labels)
    classes = np.asarray(range(0, 10))
    result_arr = []
    c_sample = np.random.choice(classes, size, p=cl)
    print c_sample
    for num in c_sample:
        img_arr = []
        for n in range(d):
            P = [1- theta[num][n], theta[num][n]]
            pix = np.random.choice([0, 1], 1, p=P)
            img_arr.append(pix)
        result = np.round(img_arr)
        result_arr.append(result)
    #print c_sample
    test.plot_images(np.asarray(result_arr), plt)
    plt.show()

def sample_class_given_data(data, labels, theta, size):
    c, d = theta.shape
    cl = class_liklihood(labels)
    classes = np.asarray(range(0, size))
    c_sample = np.random.choice(classes, size, p=cl)
    for num in c_sample:
        result = np.random.binomial(size, theta[num])
        sum = np.sum(theta,)

def comb_marginal_pixels(data, labels, theta, size):
    result_arr = []
    theta_b = theta[:, 392:]
    c, d = theta_b.shape
    pi = 0.1
    for i in range(0, size):
        top_half = data[i, :392]
        bottom_half = data[i, 392:]
        #generating class probability given top data
        P_cx = np.exp(log_bernoulli_prod_top(data[i], theta, pi))

        img_arr = []
        for n in range(392, 784):
            total = 0
            for cl in range(0, c):
                prob = [1 - theta[cl][n], theta[cl][n]]
                choice = [0, 1]
                c_sample = np.random.choice(choice, 1, p=prob)
                total +=c_sample*P_cx[cl]
            img_arr.append(total)

        P_pixel = np.asarray(img_arr).reshape(392, )
        bottom_half = P_pixel
        # pre_sum = np.multiply(P_pixel, np.broadcast(P_cx, axis=0))
        #bottom_half = np.sum(pre_sum, axis=0)
        result = np.concatenate((top_half, bottom_half))
        result = np.round(result)
        result_arr.append(result)
    test.plot_images(np.asarray(result_arr), plt)
    plt.show()

def comb_marginal_pixels_gray(data, labels, theta, size):
    result_arr = []
    theta_b = theta[:, 392:]
    c, d = theta_b.shape
    pi = 0.1
    for i in range(0, size):
        top_half = data[i, :392]
        bottom_half = data[i, 392:]
        #generating class probability given top data
        P_cx = np.exp(log_bernoulli_prod_top(data[i], theta, pi))

        img_arr = []
        for n in range(392, 784):
            total = 0
            for cl in range(0, c):
                # prob = [1 - theta[cl][n], theta[cl][n]]
                # choice = [0, 1]
                # c_sample = np.random.choice(choice, 1, p=prob)
                total +=theta[cl][n]*P_cx[cl]
            img_arr.append(total)

        P_pixel = np.asarray(img_arr).reshape(392, )
        bottom_half = P_pixel
        # pre_sum = np.multiply(P_pixel, np.broadcast(P_cx, axis=0))
        #bottom_half = np.sum(pre_sum, axis=0)
        result = np.concatenate((top_half, bottom_half))
        result_arr.append(result)
    test.plot_images(np.asarray(result_arr), plt)
    plt.show()
    plt.savefig('pic_2.png')

if __name__ == "__main__":
    train_data = np.load("training_data.npy")
    train_labels = np.load("training_labels.npy")

    test_data = np.load("test_data.npy")
    test_labels = np.load("test_labels.npy")
    # 1 b)
    theta = fit_theta(train_data, train_labels)
    # 1 d)
    # print "train"
    # print avg_log_likelihood(train_data, train_labels, theta)
    # print calc_acc(train_data, train_labels, theta)
    #
    # print "test"
    print avg_log_likelihood(test_data, test_labels, theta)
    # print calc_acc(test_data, test_labels, theta)
    #2 c)
    #sample_marginal(train_data, train_labels, theta, 10)

    #2 f)
    #comb_marginal_pixels(test_data, test_labels, theta, 20)
    #comb_marginal_pixels_gray(test_data, test_labels, theta, 20)



