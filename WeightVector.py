import numpy as np

def simple(pop_size):
    return np.array([[i / (pop_size - 1), 1 - i / (pop_size - 1)]  for i in range(pop_size)])

def das_dennis_weights(n_obj=2, H=99):
    from itertools import combinations_with_replacement
    import numpy as np

    def gen_points(n, h):
        from math import comb
        result = []
        for c in combinations_with_replacement(range(h + n), n - 1):
            point = [c[0]] + [c[i] - c[i-1] for i in range(1, n - 1)] + [h + n - 1 - c[-1]]
            result.append([v / h for v in point])
        return np.array(result)

    return gen_points(n_obj, H)