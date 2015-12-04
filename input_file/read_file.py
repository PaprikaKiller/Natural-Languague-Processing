import re


def read_file(path_file):
    with open(path_file, "r") as f:
        read_data = f.read()
        f.close()

    word_list = re.sub("[^\w]", " ", read_data).split()
    return word_list
