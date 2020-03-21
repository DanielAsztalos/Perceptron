import numpy as np
import matplotlib.pyplot as plt
import time

train_data = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
train_labels = np.array([0, 0, 0, 1])

def hardlim(val):
    return (0 if val < 0 else 1)

def perceptron_learning(data, labels):
    N, n = data.shape
    lr = .1
    w = np.random.randn(n, 1)
    E = 1

    a = [0, -w[0] / w[2]]
    c = [-w[0]/w[1], 0]

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()
    plt.scatter(train_data[:, 1], train_data[:, 2])
    x = np.linspace(-5, 5, 50)
    m = (a[1] - a[0]) / (c[1] - c[0])
    b = a[1]
    line1, = ax.plot(x, x * m + b)
    
    while E != 0:
        E = 0
        
        for i in range(N):
            yi = hardlim(np.dot(data[i], w))
            ei = labels[i] - yi
            w += lr * ei * data[i].reshape(n, 1)
            E += ei ** 2

        a = [0, -w[0] / w[2]]
        c = [-w[0]/w[1], 0]
        m = (a[1] - a[0]) / (c[1] - c[0])
        b = a[1]
        plt.xlim((-0.1, 1.5))
        plt.ylim((-0.1, 1.5))
        # line1.set_xdata(a)
        line1.set_ydata(x * m + b)
        fig.canvas.draw()
        time.sleep(1)
        fig.canvas.flush_events()

    # print([hardlim(np.dot(x, w)) for x in data])

perceptron_learning(train_data, train_labels)