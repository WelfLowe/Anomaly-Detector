from generator.Compound import Compound


class Product(Compound):
    def eval(self, x: float) -> float:
        """Evaluate the sum of all product functions at the given x value."""
        res = 1.0
        for f in self.sub_functions:
            res *= f.eval(x)
        return res