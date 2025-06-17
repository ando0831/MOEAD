import numpy as np

class Decomposition:
    
    def __init__(self):
        pass
        
class Tchebicheff(Decomposition):
    
    def do(self, f, weight, ideal):
        return max(weight * np.abs(f - ideal))