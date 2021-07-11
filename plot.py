import numpy as np
from matplotlib import pyplot as plt

import sys

def generate_data(gamma, n, snr, num):
    tot = 0

    p = int(round(gamma * n))
    beta = np.random.normal(0, 1, [p, 1])
    beta = beta * snr ** (1/2) / np.linalg.norm(beta)
    X = np.random.normal(0, 1, [n, p])

    mp_inv = np.dot(np.linalg.pinv(np.dot(np.transpose(X), X)), np.transpose(X))

    for _ in range(num):
        epsilon = np.random.normal(0, 1, [n, 1])

        Y = np.dot(X, beta) + epsilon

        least_squares = np.dot(mp_inv, Y)

        x_0 = np.random.normal(0, 1, [p, 1])

        tot = tot + np.linalg.norm(least_squares - beta) ** 2

    return tot / num

def R(gamma, variance, snr):
    if gamma < 1:
        return variance * gamma / (1 - gamma)
    else:
        return snr * variance * (1 - 1/gamma) + variance * 1 / (gamma - 1)

def plot_risk(snr):
    gamma = np.arange(0, 10, 0.0001)
    y = list(map(R, gamma, [1] * len(gamma), [snr] * len(gamma)))
    plt.plot(gamma, y, color="blue")
    plt.xlabel("$\gamma$")
    plt.ylabel("Risk")
    plt.ylim(0, 10)
    plt.xlim(0, 10)

def plot_line(snr, num):
    plot_risk(snr)

    gamma = np.arange(0, 10, 0.25)
    risk = list(map(generate_data, gamma, [200] * len(gamma), [snr] * len(gamma), [num] * len(gamma)))
    print(risk)
    plt.plot(gamma, risk, 'x', color="red")


for snr in range(1, int(sys.argv[1]) + 1):
    plot_line(snr, int(sys.argv[2]))
    plt.savefig(f'./images/LR-SNR{snr}.png')
    plt.clf()