import re
import numpy as np
import skipgram
import cbow


def make_vocabulary(path_to_file, num_words=0):
    with open(path_to_file, "r") as f:
        read_data = f.read()
        f.close()
    list_words = re.sub("[^\w]", " ", read_data).split()

    if num_words == 0:
        num_words = len(list_words)

    unique_words = []
    for i in range(num_words):
        if not list_words[i] in unique_words:
            unique_words.append(list_words[i])

    list_nums = np.zeros(num_words)
    for i in range(num_words):
        list_nums[i] = unique_words.index(list_words[i])

    return list_nums


def train_model(num_iterations, path_to_file, eta=0.005, vec_size=10, context=3, model="skipgram", num_words=0):
    data = make_vocabulary(path_to_file, num_words)
    V = max(data)
    Win = np.random.rand(V, vec_size)
    Wout = np.random.rand(vec_size, V)

    for i in range(context, num_iterations-context):
        if model == "skipgram":
            inword = data[i]
            outwords = np.append(data[i-context:i],data[i+1:i+context+1])
            Win, Wout = skipgram.updateweights(eta,Win,Wout,inword,outwords)

        elif model == "skipgram_negative":
            inword = data[i]
            outwords = np.append(data[i-context:i],data[i+1:i+context+1])
            Win,Wout = skipgram.updateweights_negative(eta,Win,Wout,inword,outwords)

        elif model == "cbow":
            inword = np.append(data[i-context:i],data[i+1:i+context+1])
            outwords = data[i]
            Win,Wout = cbow.updateweights(eta,Win,Wout,inword,outwords)

        elif model == "cbow_negative":
            inword = np.append(data[i-context:i],data[i+1:i+context+1])
            outwords = data[i]
            Win,Wout = cbow.updateweights_negative(eta,Win,Wout,inword,outwords)
