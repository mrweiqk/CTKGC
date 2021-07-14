import numpy as np
import scipy.sparse as sp
import torch




class Max:
    def __init__(self):

        self.hit1 = 0
        self.hit3=0
        self.hit10 = 0
        self.MR = 0
        self.MRR = 0