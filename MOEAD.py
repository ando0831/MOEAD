import numpy as np
import random
from Decomposition import Tchebicheff
from Crossover import SBX
from Mutation import PM
from HyperVolume import HV
from Problem import ZDT1

class MOEAD:
    
    def __init__(self, n_parents=2, n_offsprings=1, pop_size=100, n_neighbors=10):
        self.n_parents = n_parents
        self.n_offsprings = n_offsprings
        self.pop_size = pop_size
        self.n_neighbors = n_neighbors
        self.weights = np.array([[i / (pop_size - 1), 1 - i / (pop_size - 1)]  for i in range(pop_size)])
        self.neighbors = np.argsort([[np.linalg.norm(w - wi) for w in self.weights] for wi in self.weights], axis=1)[:, :self.n_neighbors]
        
    # 重みベクトルを返す
    def get_weight_vector(self):
        return self.weights
    
    # T近傍を返す
    def get_neighbors(self):
        return self.neighbors
    
    # HVを返す
    def get_HV(self):
        return self.HVlist
    
    # 関数値を返す
    def get_F(self):
        return self.f_values
    
    # 変数値を返す
    def get_X(self):
        return self.pop
    
    # 最適化
    def optimize(self, Problem, population, n_gen,
                 Decomposition=Tchebicheff(),
                 Crossover=SBX(),
                 Mutation=PM(),
                 save_HV=True):
        #xuとxlを定義
        Crossover.xu = Problem.xu
        Crossover.xl = Problem.xl
        Mutation.xu = Problem.xu
        Mutation.xl = Problem.xl
        # HVのインスタンス化
        hypervolume = HV()
        # 初期解
        self.pop = population
        # 初期値
        self.f_values = np.array([Problem.eval(F) for F in self.pop])
        # 初期理想点
        ideal = np.min(self.f_values, axis=0)
        # save_HV=TrueならHVを保存
        if save_HV == True:
            self.HV_past = [0] * n_gen
            self.HV_past[0] = hypervolume.eval(self.f_values)
        
        for n in range(1, n_gen):
            order = np.random.permutation(self.pop_size)
            for i in order:
                # 交叉
                k, l = np.random.choice(self.neighbors[i], 2, replace=False)
                offspring = Crossover.do(self.pop[k], self.pop[l])
                # 突然変異
                offspring = Mutation.do(offspring)
                # 関数値
                f_offspring = Problem.eval(offspring)
                # 理想点
                ideal = np.minimum(ideal, f_offspring)
                # 置換判定
                for j in self.neighbors[i]:
                    if Decomposition.do(f_offspring, self.weights[j], ideal) < Decomposition.do(self.f_values[j], self.weights[j], ideal):
                        self.pop[j] = offspring
                        self.f_values[j] = f_offspring
            # HVを保存
            if save_HV == True:
                self.HV_past[n] += (hypervolume.eval(self.f_values))