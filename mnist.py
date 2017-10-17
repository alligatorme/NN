import numpy as np


class layer:
    def __init__(self,n):
        self.wt=np.random.randn(n,1)
        self.bs=np.random.randn(1)

class cycle:
    def __init__(self):
        pass

    def fwd(self,inp):
        for lyr in self.cascade:
            inp=sigmoid(np.dot(lyr.wt,inp)+lyr.bs)
        return inp

    def bwd(self,inp,otp):
        for lyr in self.cascade:
            lyr.z=np.dot(w,inp)+b
            inp=lyr.act=sigmoid(lyr.z)
        lyr=self.cacade[-1]
        inp=self.cost(lyr.z,lyr.act,otp)
        for lyr in self.cascade[:-1:-1]:
          np.dot(lyr.wt.T,inp)*sigmoid_prime(lyr.z)

