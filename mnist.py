import numpy as np
from functools import reduce
import pickle
import gzip

def load_data():
	f = gzip.open('mnist.pkl.gz', 'rb')
	td, validation_data, test_data = pickle.load(f,encoding='bytes')
	f.close()
	# print(td[0][0],td[1][0])
	return (td[0][0],td[1][0])
	# return (training_data, validation_data, test_data)

class layer:
	def __init__(self,n,k):
		self.nb=[]
		self.nw=[]
		if n:
			self.wt=np.random.randn(k,n)
			self.bs=np.random.randn(k)
		else:
			self.nb=None

class cycle:
	def __init__(self,fbr):		
		self.csd=[]
		self.csd.append(layer(0,0))
		for i,j in zip(fbr[:-1],fbr[1:]):
			self.csd.append(layer(i,j))	

	def load(self,inp):
		self.csd[0].act=inp

	def fwd(self,inp):
		for lyr in self.csd[1:]:
			inp=sigmoid(np.dot(lyr.wt,inp)+lyr.bs)
		return inp

	def bkp(self,otp):
		reduce(forward,self.csd)
		lyr=self.csd[-1]
		lyr.nb=lyr.act-otp
		reduce(backward,self.csd[::-1])

def forward(hd,lyr):
	lyr.z=np.dot(lyr.wt,hd.act)+lyr.bs
	lyr.act=sigmoid(lyr.z)
	return lyr

def backward(lyr,hd):
	if hd.nb!=None:
		hd.nb=np.dot(lyr.wt.transpose(),lyr.nb)*sigmoid_prime(hd.z) 
	lyr.nw=np.dot(lyr.nb,hd.act.transpose())
	return hd

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


if __name__=="__main__":
	x,y=load_data()
	fbr=[784,10,11,12,1]
	mrk=cycle(fbr)
	mrk.load(x)
	mrk.bkp(y)
	print(y,mrk.csd[-1].act)



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
