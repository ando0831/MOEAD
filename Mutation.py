import numpy as np
import random

class Mutation:
    
    def __init__(self, mutation_prob=1/30):
        self.mutation_prob = mutation_prob


# PM突然変異
class PM(Mutation):
    
    def __init__(self, eta=20, prob=0.9, mutation_prob=1/30, xu=[1.0], xl=[0], n_var=30):
        super().__init__(mutation_prob)
        self.eta = eta
        self.prob = prob
        self.xu = xu * n_var
        self.xl = xl * n_var
        self.n_var = n_var
        self.mutation_prob = 1 / n_var

    def do(self, offspring):
        if random.random() < self.prob:
            for i in range(len(offspring)):
                if random.random() < self.mutation_prob:
                    delta1 = (offspring[i] - self.xl[i]) / (self.xu[i] - self.xl[i])
                    delta2 = (self.xu[i] - offspring[i]) / (self.xu[i] - self.xl[i])
                    U = random.random()
                    if U < 0.5:
                        delta = pow(2 * U + pow((1 - 2 * U) * (1 - delta1), self.eta + 1), 1 / (self.eta + 1)) - 1
                    else:
                        delta = 1 - pow(pow(2 * (1 - U) + 2 * (U - 0.5) * (1 - delta2), self.eta + 1), 1 / (self.eta + 1))
                    offspring[i] += delta * (self.xu[i] - self.xl[i])
                    offspring[i] = np.clip(offspring[i], self.xl[i], self.xu[i])

        return offspring