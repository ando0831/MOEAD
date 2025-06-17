import numpy as np
import random

class Mutation:
    def __init__(self, mutation_prob=1/30):
        self.mutation_prob = mutation_prob


# --- PM突然変異 ---
class PM(Mutation):

    def do(self, offspring, eta=20, n_var=30, xu=1.0, xl=0):

        self.mutation_prob = 1 / n_var

        for i in range(len(offspring)):
            if random.random() < self.mutation_prob:
                delta1 = offspring[i] - 0.0
                delta2 = 1.0 - offspring[i]
                U = random.random()
                if U < 0.5:
                    delta = pow(2 * U + pow((1 - 2 * U) * (1 - delta1), eta + 1), 1 / (eta + 1)) - 1
                else:
                    delta = 1 - pow(pow(2 * (1 - U) + 2 * (U - 0.5) * (1 - delta2), eta + 1), 1 / (eta + 1))
                offspring[i] += delta
                offspring[i] = np.clip(offspring[i], xl, xu)

        return offspring