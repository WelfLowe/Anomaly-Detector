from typing import Optional
from generator.Compound import Compound


class FilterAmplitude(Compound):
    def __init__(self, lower_bound: Optional[float], upper_bound: Optional[float]):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        super().__init__()

    def eval(self, x: float) -> float:
        res = self.sub_functions[0].eval(x)
        if self.lower_bound is not None:
            res = max(res, self.lower_bound)
        if self.upper_bound is not None:
            res = min(res, self.upper_bound)
        return res
