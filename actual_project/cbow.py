import numpy as np
from .global_functions import sigma


def updateweights(eta, Win, Wout, inwords, outword):
    # some notes on the input
    # eta is scalar
    # Win is a numpy array of V by N
    # Wout is a numpy array of N by V
    # inwords is a list (NOT numpy) of ints
    # outword is an int

    V = np.size(Win, 0)
    C = len(inwords)
    N = np.size(Win, 1)

    h = np.zeros(N)
    for i in range(C):
        h += Win[inwords[i]]

    h = np.dot(1.0/C, h).T
    softmax = 0
    for j in range(V):
        softmax += np.exp(np.dot(Wout[:, j].T, h))

    EH = np.zeros(N)
    for i in range(N):
        for j in range(V):
            gradout = np.exp(np.dot(Wout[:, j].T, h))/softmax - 1
            if i == 0:
                Wout[:, j] -= eta*gradout*h
            EH[i] += gradout*Wout[i, j]

    for i in range(C):
        Win[inwords[i]] -= 1/C*eta*EH

    return Win, Wout


def updateweights_negative(eta, Win, Wout, inwords, outword):
    # some notes on the input
    # eta is scalar
    # Win is a numpy array of V by N
    # Wout is a numpy array of N by V
    # inwords is a list (NOT numpy) of ints
    # outword is an int

    V = np.size(Win, 0)
    C = len(inwords)
    N = np.size(Win, 1)

    h = np.zeros(N)
    for i in range(C):
        h += Win[inwords[i]]

    h = np.dot(1.0/C, h).T
    softmax = 0
    for j in range(V):
        temp = Wout[:, j].T
        softmax += np.exp(np.dot(temp, h))

    EH = np.zeros(N)
    for i in range(N):
        for j in range(V):
            gradout = np.exp(np.dot(Wout[:, j].T, h))/softmax - 1
            if i == 0:
                Wout[:, j] -= eta*gradout*h

    K = 2   # I don't have any idea of what K is
    t = []  # t is an array of K size in which contains the values of negative and positive samples, whatever that means
    for j in range(K):
        EH = (sigma(Wout[outword].T*h)-t[j])*Wout[outword]
        Win[inwords[j]] -= 1/C*eta*EH

    return Win, Wout
