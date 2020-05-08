import pickle


def save(f_name, data):
    with open(f_name, "wb") as f:
        pickle.dump(data, f)


def load(f_name):
    with open(f_name, "rb") as f:
        return pickle.load(f)
