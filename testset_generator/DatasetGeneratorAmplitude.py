import numpy as np
from generator.Line import Line
from generator.Period import Period
from generator.PulseTrain import PulseTrain
from generator.RandomNoise import RandomNoise
from generator.Sum import Sum
from testset_generator.DatasetGenerator import DatasetGenerator


class DatasetGeneratorAmplitude(DatasetGenerator):
    """normal aplitude noise ~ N(0, ampl) """
    """outlier aplitude noise ~ N(0, ampl * (1+s)) """
    def getFunction(self, severeness):
        s = Sum()
        s.add_function(Line(200, 0))  # Add a Line function with offset 200 and slope 0
        s.add_function(RandomNoise(0, 1))  # Add a Random function 
        
        # Monthly, T1=720
        p1 = 30 + np.random.normal(0, 2, 1)
        a1 = 15 + 2 * severeness + np.random.normal(0, 2, 1)
        s.add_function(Period(p1 * 24, 0, a1))  # Add a Period function for monthly period
        
        # Weekly, T2=168
        p2 = 7 + np.random.normal(0, 1, 1)
        a2 = 10 + severeness + np.random.normal(0, 1, 1)
        s.add_function(PulseTrain(p2 * 24, 2 * 24, 24, a2))  # Add a PulseTrain function for weekly period

        # Daily, T3=24
        p3 = 1 + np.random.normal(0, 0.1 , 1)
        a3 = 5 + 0.1 * severeness + np.random.normal(0, 0.1 , 1)
        s.add_function(Period(p3 * 24, -6, a3))  # Add a Period function for daily period
        return s