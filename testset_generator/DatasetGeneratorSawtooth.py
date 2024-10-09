import numpy as np
from generator.Line import Line
from generator.Period import Period
from generator.PulseTrain import PulseTrain
from generator.RandomNoise import RandomNoise
from generator.Sawtooth import Sawtooth
from generator.Sum import Sum
from testset_generator.DatasetGenerator import DatasetGenerator

class DatasetGeneratorSawtooth(DatasetGenerator):
    """outlier: some overlaying sine curves are replaced with sawtooth curves; more with increasing s"""
    def getFunction(self, severeness):
        s = Sum()
        
        p3 = 1 + np.random.normal(0, 0.1 , 1)
        a3 = 5 + np.random.normal(0, 0.1, 1)
        p2 = 7 + np.random.normal(0, 1, 1)
        a2 = 10 + np.random.normal(0, 1, 1)
        p1 = 30 + np.random.normal(0, 2, 1)
        a1 = 15 + np.random.normal(0, 2, 1)

        if severeness == 0: 
            # Daily, T3=24
            s.add_function(Period(p3 * 24, -6, a3))  # Add a Period function for daily period
            # Weekly, T2=168
            s.add_function(PulseTrain(p2 * 24, 2 * 24, 24, a2))  # Add a PulseTrain function for weekly period
            # Monthly, T1=720
            s.add_function(Period(p1 * 24, 0, a1))  # Add a Period function for monthly period
        elif severeness == 1: 
            # Daily, T3=24
            s.add_function(Sawtooth(p3 * 24, -6, a3))  # Add a Sawtooth function for daily period
            # Weekly, T2=168
            s.add_function(PulseTrain(p2 * 24, 2 * 24, 24, a2))  # Add a PulseTrain function for weekly period
            # Monthly, T1=720
            s.add_function(Period(p1 * 24, 0, a1))  # Add a Period function for monthly period
        elif severeness == 2: 
            # Daily, T3=24
            s.add_function(Period(p3 * 24, -6, a3))  # Add a Period function for daily period
            # Weekly, T2=168
            s.add_function(Sawtooth(p2 * 24, -12, a2))  # Add a Sawtooth function for weekly period
            # Monthly, T1=720
            s.add_function(Period(p1 * 24, 0, a1))  # Add a Period function for monthly period
        if severeness == 3: 
            # Daily, T3=24
            s.add_function(Period(p3 * 24, -6, a3))  # Add a Period function for daily period
            # Weekly, T2=168
            s.add_function(PulseTrain(p2 * 24, 2 * 24, 24, a2))  # Add a PulseTrain function for weekly period
            # Monthly, T1=720
            s.add_function(Sawtooth(p1 * 24, 0, a1))  # Add a Sawtooth function for monthly period
        elif severeness == 4: 
            # Daily, T3=24
            s.add_function(Sawtooth(p3 * 24, -6, a3))  # Add a Sawtooth function for daily period
            # Weekly, T2=168
            s.add_function(Sawtooth(p2 * 24, -12, a2))  # Add a Sawtooth function for weekly period
            # Monthly, T1=720
            s.add_function(Period(p1 * 24, 0, a1))  # Add a Period function for monthly period
        elif severeness == 4: 
            # Daily, T3=24
            s.add_function(Period(p3 * 24, -6, a3))  # Add a Sawtooth function for daily period
            # Weekly, T2=168
            s.add_function(Sawtooth(p2 * 24, -12, a2))  # Add a Sawtooth function for weekly period
            # Monthly, T1=720
            s.add_function(Sawtooth(p1 * 24, 0, a1))  # Add a Sawtooth function for monthly period
        elif severeness == 5: 
            # Daily, T3=24
            s.add_function(Sawtooth(p3 * 24, -6, a3))  # Add a Sawtooth function for daily period
            # Weekly, T2=168
            s.add_function(Sawtooth(p2 * 24, -12, a2))  # Add a Sawtooth function for weekly period
            # Monthly, T1=720
            s.add_function(Sawtooth(p1 * 24, 0, a1))  # Add a Sawtooth function for monthly period

        #included in all severeness levels 
        s.add_function(RandomNoise(0, 1))  # Add a Random function 
        s.add_function(Line(200, 0))  # Add a Line function with offset 200 and slope 0

        return s