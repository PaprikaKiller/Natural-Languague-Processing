import numpy as np
import re
import time
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
    print softm
    time.sleep(1)
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

def makevocab(path_to_file):
    with open(path_to_file,"r") as f:
        read_data = f.read()
        f.close()
    list_data = re.sub("[^\w]"," ",read_data).split()
    outding = []
    for i in range(len(list_data)):
        if not list_data[i] in outding:
            outding.append(list_data[i])
    return outding

def text2num(wordlist):
    global vocab
    outding = []
    for i in range(len(wordlist)):
        outding.append(vocab.index(wordlist[i]))
    return outding

if __name__ == "__main__":
    eta = 0.005
    vecsize = 10
    V = 100 #vocabulary size
    C = 3 #context before and after
    Win = np.random.rand(V,vecsize)
    Wout = np.random.rand(vecsize,V)
    vocab  = makevocab(path_to_file)
    bunchofnumbers = np.random.randint(0,V,1000)
    numiters = 100
    for i in range(C,numiters-C):
        inword = bunchofnumbers[i]
        outwords = np.append(bunchofnumbers[i-C:i],bunchofnumbers[i+1:i+C+1])
        Win,Wout = updateweights(eta,Win,Wout,inword,outwords)
