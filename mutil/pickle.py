import pickle


def save(data, f_name):
    with open(f_name, "wb") as f:
        pickle.dump({'data': data}, f)


def load(f_name):
    with open(f_name, "rb") as f:
        return pickle.load(f)['data']
