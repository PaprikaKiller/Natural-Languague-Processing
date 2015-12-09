import numpy as np
from .global_functions import sigma


def updateweights(eta, Win, Wout, inword, outwords):
    # some notes on the input
    # eta is scalar
    # Win is a numpy array of V by N
    # Wout is a numpy array of N by V
    # inwords is an int
    # outword is a list (NOT numpy) of ints

    V = np.size(Win, 0)
    C = len(outwords)

    h = Win[inword]
    h = h.T

    softmax = 0
    for j in range(V):
        softmax += np.exp(np.dot(Wout[:, j].T, h))

    gradin = 0
    for j in range(V):
        gradout = 0
        for c in range(C):
            gradout += np.exp(np.dot(Wout[:, j].T, h))/softmax - 1

        Wout[:, j] -= eta*gradout*h
        gradin += gradout*Wout[:, j]

    Win[inword] -= eta*gradin

    return Win, Wout


def updateweights_negative(eta, Win, Wout, inword, outwords):
    # some notes on the input
    # eta is scalar
    # Win is a numpy array of V by N
    # Wout is a numpy array of N by V
    # inwords is an int
    # outword is a list (NOT numpy) of ints

    V = np.size(Win, 0)
    C = len(outwords)

    h = Win[inword]
    h = h.T

    softmax = 0
    for j in range(V):
        softmax += np.exp(np.dot(Wout[:, j].T, h))

    gradin = 0
    for j in range(V):
        gradout = 0
        for c in range(C):
            gradout += np.exp(np.dot(Wout[:, j].T, h))/softmax - 1

        Wout[:, j] -= eta*gradout*h
        gradin += gradout*Wout[:, j]

    Win[inword] -= eta*gradin

    return Win, Wout
