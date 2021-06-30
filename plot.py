import numpy as np
from matplotlib import pyplot as plt

def run_least_squares(gamma, n, snr, X):
    p = int(gamma * n)
    beta = np.random.default_rng().normal(0, 1, [p, 1])
    beta = beta * snr / np.linalg.norm(beta)
    epsilon = np.random.default_rng().normal(0, 1, [n, 1])

    Y = np.dot(X, beta) + epsilon

    least_squares = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)
    
    return np.linalg.norm(least_squares) ** 2 - 2 * np.asscalar(np.dot(np.transpose(least_squares), beta)) + np.linalg.norm(beta) ** 2

def generate_data(gamma, n, snr, num):
    tot = 0

    X = np.random.default_rng().normal(0, 1, [n, int(gamma * n)])

    for i in range(num):
        tot = tot + run_least_squares(gamma, n, snr, X)

    return tot / num

def R(gamma, variance, r):
    if gamma < 1:
        return variance * gamma / (1 - gamma)
    else:
        return r * (1 - 1/gamma) + variance * gamma / (gamma - 1)

def plot_risk(snr):
    gamma = np.arange(0, 3, 0.0001)
    y = list(map(R, gamma, [1] * len(gamma), [snr] * len(gamma)))
    plt.plot(gamma, y)
    plt.xlabel("$\gamma$")
    plt.ylabel("Risk")
    plt.ylim(0, 10)
    plt.xlim(0, 3)

plot_risk(5)

def plot_line(snr):
    plot_risk(snr)

    gamma = np.arange(0, 3, 0.1)
    risk = list(map(generate_data, gamma, [200] * len(gamma), [snr] * len(gamma), [5] * len(gamma)))
    plt.plot(gamma, risk, 'x')


plot_line(5)

plt.show()