class Empty:
    def __getattribute__(self, key):
        return empty


def empty(*args, **kwargs):
    return None
