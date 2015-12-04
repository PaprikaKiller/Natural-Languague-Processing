import numpy as np


def updateweights(eta,Win,Wout,inword,outwords):
    #inword = text2num(inword)
    #outwords = text2num(outwords)
    V = np.size(Win,0)
    C = len(outwords)
    h = Win[inword]
    h = h.T
    softm = 0
    for j in range(V):
        softm += np.exp(np.dot(Wout[:,j].T,h))

    gradin = 0
    for j in range(V):
        gradout = 0
        for c in range(C):
            #t = np.zeros([V])
            #t[outwords[c]] = 1
            gradout += np.exp(np.dot(Wout[:,j].T,h))/softm - 1
        Wout[:,j] -= eta*gradout*h
        gradin += gradout*Wout[:,j]
    Win[inword] -= eta*gradin
    return Win, Wout
