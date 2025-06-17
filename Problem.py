import numpy as np

class ZDT:

    def __init__(self, n_obj=2, n_var=30, xu=1.0, xl=0):
        self.n_obj = n_obj
        self.n_var = n_var
        self.xu = xu
        self.xl = xl

class ZDT1(ZDT):
    
    def eval(self, x):
        f1 = x[0]
        g = 1 + 9 * np.sum(x[1:]) / (self.n_var - 1)
        h = 1 - np.sqrt(f1 / g)
        f2 = g * h

        return np.array([f1, f2])