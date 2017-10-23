import numpy as np
from functools import reduce
import pickle
import gzip

def load_data():
	f = gzip.open('mnist.pkl.gz', 'rb')
	td, validation_data, test_data = pickle.load(f,encoding='bytes')
	f.close()
	return (td[0][0],td[1][0])
	# return (training_data, validation_data, test_data)

class layer:
	def __init__(self,n,k):
		self.nb=[]
		self.nw=[]
		if n:
			self.wt=np.random.randn(n,k)
			self.bs=np.random.randn(1,k)
			self.pwt=np.zeros(self.wt.shape)
			self.pbs=np.zeros(self.bs.shape)
		else:
			self.nb=None
		

	def pst_wt(self):
		print(self.pwt.shape,self.nw.shape)
		self.pwt+=crs(src=self.nw)
		self.nw=[]
	def pst_bs(self):
		self.pbs+=crs(src=self.nb.T)
		self.nb=[]

	def update(self):
		self.wt+=self.pwt
		self.bs+=self.pbs

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
			inp=sigmoid(np.dot(inp,lyr.wt)+lyr.bs)
		return inp

	def bkp(self,inp,otp):
		self.load(inp)
		reduce(forward,self.csd)
		lyr=self.csd[-1]
		lyr.nb=lyr.act-otp
		if lyr.nb.shape==(1,): lyr.nb=lyr.nb[0]
		reduce(backward,self.csd[::-1])


def forward(hd,lyr):
	lyr.z=np.dot(hd.act,lyr.wt)+lyr.bs
	# print('origin=',lyr.wt.shape)
	lyr.act=sigmoid(lyr.z)
	# print('fwd=',lyr.act.shape)
	return lyr

def backward(lyr,hd):
	if hd.nb!=None:
		hd.nb=np.dot(lyr.wt,lyr.nb)*sigmoid_prime(hd.z.T) 
		# print(lyr.wt.shape,lyr.nb.shape,hd.z.T.shape)
		# print(hd.nb.shape)
	lyr.nw=np.dot(hd.act.T,lyr.nb.T)
	print(lyr.nw.shape)
	# lyr.pst_bs()
	# lyr.pst_wt()
	
	return hd

def crs(src,np=10,eta=0.1):
	return -eta/np*src 

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def refresh(patch,eta):
	for x,y in patch:
		self.bkp(x,y)
	for lyr in self.csd[1:]:
		lyr.update()

if __name__=="__main__":
	x,y=load_data()
	x.shape=(1,x.shape[0])
	print(x.shape)
	fbr=[784,10,11,12,1]
	mrk=cycle(fbr)
	mrk.bkp(x,y)
	print(mrk.fwd(x))
	print(y,mrk.csd[-1].act)

