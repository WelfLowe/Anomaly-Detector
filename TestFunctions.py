import matplotlib.pyplot as plt
import numpy as np

from generator.PulseTrain import PulseTrain
from generator.Sawtooth import Sawtooth
from generator.Period import Period
from generator.Line import Line
from generator.RandomNoise import RandomNoise
from generator.Sum import Sum
from generator.Product import Product
from generator.FilterAmplitude import FilterAmplitude

def plot_series(ys, color = "b"):
    xs = np.arange(ys.shape[0]) * np.ones_like(ys)

    plt.plot(xs, ys, color)
    plt.show()
    plt.close()
    
N = 1000
p = PulseTrain()  
yss = p.generateN(N)
#plot_series(yss, "r")

p = Sawtooth()  
yss = p.generateN(N)
#plot_series(yss, "r")

p = Period()  
yss = p.generateN(N)
#plot_series(yss, "r")

p = Line()  
yss = p.generateN(N)
#plot_series(yss, "r")

p = RandomNoise()  
yss = p.generateN(N)
#plot_series(yss, "r")

s = Sum()
#s.add_summand_function(Line(200, 0))  # Add a Line function with offset 200 and slope 0
s.add_function(RandomNoise(0, 1))  # Add a Random function 
# Monthly, T1=720
s.add_function(Period(30.0 * 24, 0, 15))  # Add a Period function for monthly period
# Weekly, T2=168
s.add_function(PulseTrain(7 * 24, 2 * 24, 24, 10))  # Add a PulseTrain function for weekly period
# Daily, T3=24
s.add_function(Period(24, -6, 5))  # Add a Period function for daily period

yys = s.generateN(N)
#plot_series(yys)

p = Product()
s = Sum()
s.add_function(Line(1,0))  
s.add_function(PulseTrain(T=250, tau=10, phase=0, c=-0.3))  
p.add_function(s)  
p.add_function(Period(30.0 * 24, 0, 15))  

yys = p.generateN(N)
#plot_series(yys)

filtered = FilterAmplitude(0, 0.9)
filtered.add_function(s)
yss = filtered.generateN(N)
plot_series(yss)