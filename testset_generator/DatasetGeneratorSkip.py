import numpy as np
from generator.Line import Line
from generator.Period import Period
from generator.PulseTrain import PulseTrain
from generator.RandomNoise import RandomNoise
from generator.Sum import Sum
from testset_generator.DatasetGenerator import DatasetGenerator


class DatasetGeneratorSkip(DatasetGenerator):
    """normal: noise and all overlaying sine curves are added """
    """outlier: noise and some overlaying sine curves are skipped; more skips with increasing s"""
    def getFunction(self, severeness):
        s = Sum()
        if severeness < 1: #included in severeness 0 (normal)
            s.add_function(RandomNoise(0, 1))  # Add a Random function 
        
        if severeness < 2: #included in severeness 0, 1
            # Daily, T3=24
            p3 = 1 + np.random.normal(0, 0.1 , 1)
            a3 = 5 + np.random.normal(0, 0.1, 1)
            s.add_function(Period(p3 * 24, -6, a3))  # Add a Period function for daily period

        if severeness < 3: #included in severeness 0, 1, 2
            # Weekly, T2=168
            p2 = 7 + np.random.normal(0, 1, 1)
            a2 = 10 + np.random.normal(0, 1, 1)
            s.add_function(PulseTrain(p2 * 24, 2 * 24, 24, a2))  # Add a PulseTrain function for weekly period

        if severeness < 4: #included in severeness 0, 1, 2, 3, 
            # Monthly, T1=720
            p1 = 30 + np.random.normal(0, 2, 1)
            a1 = 15 + np.random.normal(0, 2, 1)
            s.add_function(Period(p1 * 24, 0, a1))  # Add a Period function for monthly period

        #included in all severeness levels 
        s.add_function(Line(200, 0))  # Add a Line function with offset 200 and slope 0

        return s