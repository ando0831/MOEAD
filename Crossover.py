import numpy as np
import random

class Crossover:

    def __init__(self, n_parents=2, n_offsprings=1):
        self.n_parents = n_parents
        self.n_offsprings = n_offsprings

# SBX交叉
class SBX(Crossover):

    def __init__(self, n_var=30, eta=20, n_parents=2, n_offsprings=1, xu=[1.0], xl=[0.0]):
        super().__init__(n_parents, n_offsprings)
        self.eta = eta
        self.xu = xu * n_var
        self.xl = xl * n_var

    def do(self, p1, p2):
        child1, child2 = np.copy(p1), np.copy(p2)
        for i in range(len(p1)):
            if random.random() <= 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    x1 = min(p1[i], p2[i])
                    x2 = max(p1[i], p2[i])
                    rand = random.random()
                    
                    beta1 = 1.0 + (2.0 * (x1 - self.xl[i]) / (x2 - x1))
                    alpha1 = 2.0 - pow(beta1, -(self.eta + 1.0))
                    if rand <= 1.0 / alpha1:
                        betaq1 = pow(rand * alpha1, 1.0 / (self.eta + 1.0))
                    else:
                        betaq1 = pow(1.0 / (2.0 - rand * alpha1), 1.0 / (self.eta + 1.0))
                    c1 = 0.5 * ((x1 + x2) - betaq1 * (x2 - x1))
                    child1[i] = np.clip(c1, self.xl[i], self.xu[i])

                    beta2 = 1.0 + (2.0 * (self.xu[i] - x2) / (x2 - x1))
                    alpha2 = 2.0 - pow(beta2, -(self.eta + 1.0))
                    if rand <= 1.0 / alpha2:
                        betaq2 = pow(rand * alpha2, 1.0 / (self.eta + 1.0))
                    else:
                        betaq2 = pow(1.0 / (2.0 - rand * alpha2), 1.0 / (self.eta + 1.0))
                    c2 = 0.5 * ((x1 + x2) + betaq2 * (x2 - x1))
                    child2[i] = np.clip(c1, self.xl[i], self.xu[i])

        if self.n_offsprings == 1:
            child = random.choice([child1, child2])
            return child
        else:
            return child1, child2
