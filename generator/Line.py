from generator.Function import Function


class Line(Function):
    def __init__(self, offset=0.0, slope=1.0):
        self.offset = offset  # default offset 0
        self.slope = slope  # default slope 1

    def eval(self, x):
        return self.offset + self.slope * x