import numpy as np


def updateweights(eta, Win, Wout, inwords, outword):

    V = np.size(Win, 0)
    C = len(inwords)
    N = np.size(Win, 0)

    h = np.zeros(N)
    for i in range(C):
        h += Win[inwords[i]]

    h = (1/C*h).T
    softmax = 0
    for j in range(V):
        softmax += np.exp(np.dot(Wout[:, j].T, h))

    EH = np.zeros(N)
    for i in range(N):
        for j in range(V):
            gradout = np.exp(np.dot(Wout[:, j].T, h))/softmax - 1
            if i == 0:
                Wout[:,j] -= eta*gradout*h
            EH[i] += gradout*Wout[i, j]

    for i in range(C):
        Win[inwords] -= 1/C*eta*EH

    return Win, Wout


def updateweights_negative(eta, Win, Wout, inwords, outword):

    # TODO: variable V and N are the same, something isn't right
    V = np.size(Win, 0)
    C = len(inwords)
    N = np.size(Win, 0)

    h = np.zeros(N)
    for i in range(C):
        h += Win[inwords[i]]

    # TODO: error on T. it says h is an int
    h = (1/C*h).T
    softmax = 0
    for j in range(V):
        softmax += np.exp(np.dot(Wout[:, j].T, h))

    EH = np.zeros(N)
    for i in range(N):
        for j in range(V):
            gradout = np.exp(np.dot(Wout[:, j].T, h))/softmax - 1
            if i == 0:
                Wout[:,j] -= eta*gradout*h
            EH[i] += gradout*Wout[i, j]

    for i in range(C):
        Win[inwords] -= 1/C*eta*EH

    return Win, Wout
