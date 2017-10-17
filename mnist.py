import numpy as np


class layer:
    def __init__(self,n):
        self.wt=np.random.randn(n,1)
        self.bs=np.random.randn(1)

class cycle:
    def __init__(self):
        pass

    def fwd(self,inp):
        for lyr in self.csd:
            inp=sigmoid(np.dot(lyr.wt,inp)+lyr.bs)
        return inp

    def bkp(self,inp,otp):
        for lyr in self.csd:
            lyr.z=np.dot(w,inp)+b
            inp=lyr.act=sigmoid(lyr.z)
        lyr=self.csd[-1]
        inp=self.cost(lyr.z,lyr.act,otp)
        delta=None
        for hd,lyr,tl in zip(self.csd[:-2:-1],self.csd[1:-1:-1],self.csd[2::-1]):
            if delta is None:
                cost=self.cost(lyr.act,otp)
            else:
                cost=np.dot(tl.wt.T,delta)    
            delta=cost*sigmoid_prime(lyr.z)
            np.dot(delta,hd.act.T)

    
