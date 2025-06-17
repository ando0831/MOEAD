import numpy as np

class HV:
    
    def __init__(self, r_point=[1.0, 1.0]):
        self.r_point = r_point
        
    # --qはpに支配されているか(最小化問題想定)--
    def is_dominated(self, p, q):
        return np.all(p <= q) and np.any(p < q)

    # --非劣解の集合(パレートフロント)を抽出--
    def get_pareto_front(self, F):
        pareto_points = []
        # 非劣解を抽出
        for i, p in enumerate(F):
            dominated = False
            for j, q in enumerate(F):
                if i != j and self.is_dominated(p, q):
                    dominated = True
                    break
            if not dominated:
                pareto_points.append(p)
        pareto_points = np.array(pareto_points)
        #f1で昇順ソート
        pareto_points = pareto_points[pareto_points[:, 0].argsort()]

        return pareto_points
    
    def eval(self, F):
        HyperVolume = 0
        A = self.get_pareto_front(F)
        for i in range(len(A)-1):
            HyperVolume += (A[i+1][0] - A[i][0]) * (self.r_point[1] - A[i][1])
        HyperVolume += (self.r_point[0] - A[len(A)-1][0]) * (self.r_point[1] - A[len(A)-1][1])

        return HyperVolume