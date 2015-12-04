import re
import numpy as np


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
