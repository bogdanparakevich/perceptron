import numpy as np
import time

def perceptron(x, y_true):

    def model(x, w0, w1):
        y_pred = w0 + w1 * x
        return y_pred

    n = y_true.shape[0]
    # count n for loss function

    w0 = 1
    w1 = 1
    # random weigth

    alfa = 0.0001
    # learning rate

    def loss_fun(n, y_true, model):
        return (1/n) * np.sum(np.square(y_true - model(x, w0, w1)))

    # loss_function = (1/n) * np.sum(np.square(y_true - y_pred))

    momentum = 0.9
    change_0 = 0.001
    change_1 = 0.001

    def nesterov(momentum, change):
        return momentum * change

    for i in range(100000):
        w0_pr = -(1 / n) * 2 * np.sum(y_true - model(x, w0, w1))
        w1_pr = (1 / n) * 2 * np.sum((y_true - model(x, w0, w1)) * (-x))
        change_new_w0 = nesterov(momentum, change_0) - (alfa * w0_pr)
        change_new_w1 = nesterov(momentum, change_1) - (alfa * w1_pr)
        w0_new = w0 + change_new_w0
        w1_new = w1 + change_new_w1
        loss = loss_fun(n, y_true, model)
        if i%10000 == 0:
            print(w0_new, w1_new, loss, time.perf_counter())
        if (abs(w0_new - w0) < 0.0000001) and (abs(w1_new - w1) < 0.0000001): 
            break
        w0 = w0_new
        w1 = w1_new
        change_0 = change_new_w0
        change_1 = change_new_w1



