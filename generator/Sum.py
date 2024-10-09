from generator.Compound import Compound

class Sum(Compound):
    def eval(self, x: float) -> float:
        """Evaluate the sum of all summand functions at the given x value."""
        res = 0.0
        for f in self.sub_functions:
            res += f.eval(x)
        return res