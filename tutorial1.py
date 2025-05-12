import numpy as np 
import matplotlib.pyplot as plt


def sphere_function(x, d):
    

    if (x<-5.12 | x > 5.12):
        raise ValueError("x must be in the range [-5.12, 5.12]")
        return None 
    s = 0

    for i in range(d):
        s += x[i]**2 
    return s

def h_algorithm(alpha, beta):


    x1 = np.random.uniform(-5.12, 5.12, 100)
    x2 = np.random.uniform(-5.12, 5.12, 100)

    
    activators = sphere_function(x1, x2)**beta
    inhibitors = sphere_function(x1, x2)**alpha
    gorwth_factor = activators/inhibitors
    median = np.median(gorwth_factor)



