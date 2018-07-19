class Reader:
    """
    Class that reads the data and restarts if it reaches the end.
    """

    def __init__(self, restart, filedir):
        self.filedir = filedir
        self.f = open(filedir, "r")
        self.restart = restart

    def next(self):
        line = self.f.readline()
        if not line:
            if self.restart:
                print("new epoch")
                self.f.close()
                self.f = open(self.filedir, "r")
                line = self.f.readline()
            else:
                return ''
        return (line)
