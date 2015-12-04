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