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
		reduce(forward,self.csd)
		lyr=self.csd[-1]
		lyr.nb=(self.cost).delta(lyr.z,lyr.act,otp)
		reduce(backward,self.csd[::-1])

def forward(hd,lyr):
	lyr.z=np.dot(lyr.w,hd.act)+lyr.b
	lyr.act=sigmoid(lyr.z)
	return lyr

def backward(hd,lyr):
	if hd.nb!=None:
		hd.nb=np.dot(lyr.wt.transpose(),lyr.nb)*sigmoid_prime(hd.z) 
	lyr.nw=np.dot(lyr.nb,hd.act.transpose())


		# for hd,lyr in zip(self.csd[:-1],self.csd[1:]):
		#     lyr.z=np.dot(w,hd.act)+b
		#     lyr.act=sigmoid(lyr.z)


		# lyr=self.csd[-1]
		# inp=self.cost(lyr.z,lyr.act,otp)
		# delta=None
		# for hd,lyr,tl in zip(self.csd[:-2:-1],self.csd[1:-1:-1],self.csd[2::-1]):
		# 	if delta is None:
		# 		cost=self.cost(lyr.act,otp)
		# 	else:
		# 		cost=np.dot(tl.wt.transpose(),delta)    
		# 	delta=cost*sigmoid_prime(lyr.z)
		# 	np.dot(delta,hd.act.transpose())
