import numpy as np
from functools import reduce,partial
import pickle
import gzip
from attach import elapse

class layer:
	def __init__(self,n,k):
		self.nb=[]
		self.nw=[]
		if n:
			self.wt=np.random.randn(1,k,n)
			self.bs=np.random.randn(1,k,1)
		else:
			self.nb=None

	def update(self,func):
		# print('upt=',self.wt.shape,self.bs.shape)
		self.wt+=func(self.nw)
		self.bs+=func(self.nb)
		self.nb=[]
		self.nw=[]


class cycle:
	def __init__(self,fbr,ptc):		
		self.csd=[]
		self.csd.append(layer(0,0))
		for i,j in zip(fbr[:-1],fbr[1:]):
			self.csd.append(layer(i,j))	

	def load(self,inp):
		self.csd[0].act=inp

	def fwd(self,inp):
		for lyr in self.csd[1:]:
			# print(np.dot(lyr.wt[0],inp).shape,inp.shape,lyr.bs.shape)
			inp=sigmoid(np.dot(lyr.wt[0],inp)+lyr.bs[0][:][0])
		return inp
	
	def bkp(self,inp,otp):
		self.load(inp)
		reduce(forward,self.csd)
		init_cost(self.csd[-1],otp)		
		reduce(backward,self.csd[::-1])

def init_cost(lyr,otp):
	lyr.nb=(lyr.act-otp)*sigmoid_prime(lyr.z)
	# print(lyr.act.shape,otp.shape,lyr.z.shape)

def forward(hd,lyr):
	# print(lyr.wt.shape,hd.act.shape,lyr.bs.shape)
	lyr.z=np.matmul(lyr.wt,hd.act)+lyr.bs
	# print(lyr.z.shape)
	lyr.act=sigmoid(lyr.z)
	return lyr

def backward(lyr,hd):
	if hd.nb!=None:
		# print(lyr.wt.transpose((0,2,1)).shape,lyr.nb.shape,hd.z.shape)
		hd.nb=np.matmul(lyr.wt.transpose((0,2,1)),lyr.nb)*sigmoid_prime(hd.z) 
		# print(hd.nb.shape)	
	lyr.nw=np.matmul(lyr.nb,hd.act.transpose((0,2,1)))
	return hd

def crs(npc,eta,src):
	# print('crs=',np.sum(src,axis=0).shape)
	return -eta/npc*np.sum(src,axis=0) 

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

# @elapse
def evl(mrk,tsd):
	tsr=np.zeros((2,10))
	ptg=0
	for x,y in tsd:
		if np.argmax(mrk.fwd(x))==y:
			tsr[0][y]+=1
			ptg+=1
		tsr[1][y]+=1
	print(tsr)	
	return np.around(100*tsr[0]/tsr[1],decimals=2),ptg

def sgd(fbr,trd,npc,eta,epk=1,tsd=None):
	mrk=cycle(fbr,epk)
	for k in range(epk):
		for x,y in pack(trd,epk):
			mrk.bkp(x,y)
			for lyr in mrk.csd[1:]:
				lyr.update(partial(crs,npc,eta))
		if tsd:
			print ("Epoch {0}: {1}".format(k, evl(mrk,tsd)))
		else:
			print ("Epoch complete")

def load_data1():
	f = gzip.open('mnist.pkl.gz', 'rb')
	td, vd, tsd = pickle.load(f,encoding='bytes')
	f.close()
	return td,vd,tsd

def vect(j):
	e=np.zeros(10)
	e[j]=1
	return e

def pack(td,epk):
	n=len(td[1])
	size=len(td[0][0])
	idx=np.arange(0,n,epk)
	np.random.shuffle(idx)
	for i in idx:
		x=np.concatenate(td[0][i:i+epk])
		x=np.reshape(x,(epk,size,1))
		y=np.concatenate(list(map(vect, td[1][i:i+epk])))
		y=np.reshape(y,(epk,10,1))
		yield x,y
	# return [(np.reshape(np.concatenate(td[0][i:i+epk]),(epk,size,1)),np.reshape(np.concatenate(list(map(vect, td[1][i:i+epk]))),(epk,10,1))) for i in range(0,n,epk)]


if __name__=="__main__":
	fbr=[784,30,10]
	# mrk=cycle(fbr,10)
	
	td,_,tsd=load_data1()
	sgd(fbr,td,10,3,epk=50,tsd=list(zip(tsd[0],tsd[1])))
	
	# x,y=pack(td,12)
	# mrk.bkp(x,y)
	# for lyr in mrk.csd[1:]:
	# 	lyr.update(partial(crs,12,0.5))
 



	
	

