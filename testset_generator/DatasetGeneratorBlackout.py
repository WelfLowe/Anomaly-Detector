import numpy as np
from generator.Line import Line
from generator.Period import Period
from generator.Product import Product
from generator.PulseTrain import PulseTrain
from generator.RandomNoise import RandomNoise
from generator.Sum import Sum
from testset_generator.DatasetGenerator import DatasetGenerator

class DatasetGeneratorBlackout(DatasetGenerator):
    """Generates data series each with blackouts increasing in number and depth with the given severeness."""
    def getFunction(self, severeness):
        n=self.N
        # Orig function without offset
        s = Sum()
        s.add_function(RandomNoise(0, 1))  # Add a Random function 
        s.add_function(Line(50, 0))  # Add a Line function with offset 50 and slope 0
        
        # Monthly, T1=720
        p1 = 30 + np.random.normal(0, 2, 1)
        a1 = 15 + np.random.normal(0, 2, 1)
        s.add_function(Period(p1 * 24, 0, a1))  # Add a Period function for monthly period
        
        # Weekly, T2=168
        p2 = 7 + np.random.normal(0, 1, 1)
        a2 = 10 + np.random.normal(0, 1, 1)
        s.add_function(PulseTrain(p2 * 24, 2 * 24, 24, a2))  # Add a PulseTrain function for weekly period

        # Daily, T3=24
        p3 = 1 + np.random.normal(0, 0.1, 1)
        a3 = 5 + np.random.normal(0, 0.1, 1)
        s.add_function(Period(p3 * 24, -6, a3))  # Add a Period function for daily period
        
        #Blackout as a product function 
        p = Product()
        b = Sum()
        b.add_function(Line(1,0))  
        b.add_function(PulseTrain(T=n/(severeness+1), tau=10, phase=0, c=-0.1*severeness))  
        p.add_function(s)
        p.add_function(b)

        ss = Sum()
        ss.add_function(p)  # Add a Random function 
        ss.add_function(Line(150, 0))  # Add a Line function with offset 150 and slope 0

        return ss