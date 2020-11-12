class SublabelProvider(str):
    def __init__(self, l):
        self.l = iter(l)

    def __nonzero__(self):
        return True

    def format(self, *args, **kwargs):
        return next(self.l)
