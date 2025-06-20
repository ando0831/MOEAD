import numpy as np

class ZDT:

    def __init__(self, n_obj=2, n_var=30, xu=[1.0], xl=[0.0]):
        self.n_obj = n_obj
        self.n_var = n_var
        self.xu = xu * n_var
        self.xl = xl * n_var
        
    def get_xu(self):
        return self.xu
    
    def get_xl(self):
        return self.xl
    
    def generate_pop(self, pop_size):
        return np.random.uniform(low=self.xl, high=self.xu, size=(pop_size, self.n_var))

class ZDT1(ZDT):
    
    def eval(self, x):
        f1 = x[0]
        g = 1 + 9 * np.sum(x[1:]) / (self.n_var - 1)
        h = 1 - np.sqrt(f1 / g)
        f2 = g * h

        return np.array([f1, f2])

    
class ZDT2(ZDT):
    
    def eval(self, x):
        f1 = x[0]
        g = 1 + 9 * np.sum(x[1:]) / (self.n_var - 1)
        h = 1 - (f1 / g) ** 2
        f2 = g * h

        return np.array([f1, f2])
    
class ZDT3(ZDT):
    
    def eval(self, x):
        f1 = x[0]
        g = 1 + 9 * sum(x[1:]) / (self.n_var - 1)
        h = 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
        f2 = g * h
        return np.array([f1, f2])
    
class ZDT4(ZDT):
    def __init__(self, n_obj=2, n_var=10, xu=[5.0], xl=[-5.0]):
        super().__init__(n_obj=n_obj, n_var=n_var, xu=xu, xl=xl)
        self.xu[0] = 1.0
        self.xl[0] = 0.0

    def eval(self, x):
        f1 = x[0]
        #np.sum(x[1:] ** 2 - 10 * np.cos(4 * np.pi * x[1:]))
        g = 1.0 + 10 * (self.n_var - 1)
        for i in range(1, self.n_var):
            g += x[i] * x[i] - 10.0 * np.cos(4.0 * np.pi * x[i])

        if not np.isfinite(g) or g < 0:
            raise ValueError(f"g < 0: g = {g}")
        
        h = 1 - np.sqrt(f1 / g)
        f2 = g * h

        return np.array([f1, f2])
    
    def generate_pop(self, pop_size):
        population = super().generate_pop(pop_size)
        population[:, 0] = np.random.uniform(0, 1, size=pop_size)

        return population
    

class UF6(ZDT):
    def __init__(self, n_obj=2, n_var=30, xu=[1.0], xl=[0.0], N=2, epsilon=0.1):
        super().__init__(n_obj=n_obj, n_var=n_var, xu=xu, xl=xl)
        self.xl = -1 * np.ones(self.n_var)
        self.xl[0] = 0.0
        self.xu = 1 * np.ones(self.n_var)
        self.N = N
        self.epsilon = epsilon

    def eval(self, x):
        n = self.n_var
        N = self.N
        epsilon = self.epsilon
        x1 = x[0]
        #print("x1", x1)
        J1 = np.arange(2, n, 2)
        J2 = np.arange(1, n, 2) 

        y = np.zeros_like(x)
        for j in range(1, n):
            y[j] = x[j] - np.sin(6 * np.pi * x1 + (j * np.pi) / n)

        term1 = np.maximum(0, (2 * (1 / (2 * N) + epsilon) * np.sin(2 * N * np.pi * x1)))
        #print("term1", term1)
        term2 = (2 / len(J1)) * (4 * np.sum(y[J1] ** 2) - 
                                2 * np.prod(np.cos((20 * y[J1] * np.pi) / np.sqrt(J1))) + 2)
        term3 = (2 / len(J2)) * (4 * np.sum(y[J2] ** 2) - 
                                2 * np.prod(np.cos((20 * y[J2] * np.pi) / np.sqrt(J2))) + 2)

        f1 = x1 + term1 + term2
        f2 = 1 - x1 + term1 + term3
        #print("f1, f2", f1, f2)

        return np.array([f1, f2])