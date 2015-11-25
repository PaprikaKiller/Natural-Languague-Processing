def read_file(path_file):
    with open(path_file, "r") as f:
        read_data = f.read()

    f.close()
    return read_data
