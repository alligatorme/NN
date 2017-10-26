import numpy as np
from functools import reduce,partial
import pickle
import gzip
from attach import elapse
# @elapse
def load_data():
	f = gzip.open('mnist.pkl.gz', 'rb')
	td, vd, tsd = pickle.load(f,encoding='bytes')
	f.close()
	# return (td[0][0],td[1][0])
	return list(zip(td[0],td[1])),list(zip(tsd[0],tsd[1]))


class layer:
	def __init__(self,n,k,j):
		self.nb=[]
		self.nw=[]
		if n:
			self.wt=np.random.randn(j,k,n)
			self.bs=np.random.randn(j,k,1)
			self.pwt=np.zeros(self.wt.shape)
			self.pbs=np.zeros(self.bs.shape)
		else:
			self.nb=None
		

	def pst_wt(self):
		self.pwt+=self.nw
		self.nw=[]
	def pst_bs(self):
		self.pbs+=self.nb
		self.nb=[]

	def update(self,func):
		self.wt+=func(self.pwt)
		self.bs+=func(self.pbs)
		self.pwt=np.zeros(self.wt.shape)
		self.pbs=np.zeros(self.bs.shape)



class cycle:
	def __init__(self,fbr,ptc):		
		self.csd=[]
		self.csd.append(layer(0,0,ptc))
		for i,j in zip(fbr[:-1],fbr[1:]):
			self.csd.append(layer(i,j,ptc))	

	def load(self,inp):
		self.csd[0].act=inp

	def fwd(self,inp):
		for lyr in self.csd[1:]:
			inp=sigmoid(np.dot(lyr.wt,inp)+lyr.bs)
		return inp
	
	def bkp(self,inp,otp):
		self.load(inp)
		# for i,j in zip(self.csd[:-1],self.csd[1:]):
		# 	forward(i,j)
		# init_cost(self.csd[-1],otp)
		# for i,j in zip(self.csd[-1:0:-1],self.csd[-2::-1]):
		# 	backward(i,j)	
		reduce(forward,self.csd)
		init_cost(self.csd[-1],otp)		
		reduce(backward,self.csd[::-1])

def init_cost(lyr,otp):
	lyr.nb=(lyr.act-otp)*sigmoid_prime(lyr.z)
	# print(lyr.act.shape,otp.shape,lyr.z.shape)

def forward(hd,lyr):
	lyr.z=np.matmul(lyr.wt,hd.act)+lyr.bs
	# print(hd.act.shape,lyr.wt.shape,lyr.bs.shape,np.matmul(lyr.wt,hd.act).shape)
	lyr.act=sigmoid(lyr.z)
	return lyr

def backward(lyr,hd):
	if hd.nb!=None:
		print(lyr.wt.transpose((0,2,1)).shape,lyr.nb.shape,hd.z.shape)
		hd.nb=np.matmul(lyr.wt.transpose((0,2,1)),lyr.nb)*sigmoid_prime(hd.z) 
		print(hd.nb.shape)	
	lyr.nw=np.matmul(lyr.nb,hd.act.transpose((0,2,1)))
	lyr.pst_bs()
	lyr.pst_wt()	
	return hd

def crs(npc,eta,src):
	return -eta/npc*src 

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

# @elapse
def evl(tsd):
	tsr=np.zeros((2,10))
	ptg=0
	for x,y in tsd:
		if np.argmax(mrk.fwd(x))==y:
			tsr[0][y]+=1
			ptg+=1
		tsr[1][y]+=1
	return np.around(100*tsr[0]/tsr[1],decimals=2),ptg

def sgd(trd,npc,eta,epk=1,tsd=None):
	mrk=cycle(fbr,epk)
	for k in range(epk):
		np.random.shuffle(trd)
		for patch in (trd[k:k+npc] for k in range(0,len(trd),npc)):
			for x,y in patch:
				# for i in range(50):
				# if x.shape[0]!=1:x.shape=(1,x.shape[0])
				mrk.bkp(x,y)
				for lyr in mrk.csd[1:]:
					lyr.update(partial(crs,npc,eta))

		if tsd:
			print ("Epoch {0}: {1}".format(k, evl(tsd)))
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
	for i in range(0,epk-1,epk):
		x=np.concatenate(td[0][i:i+epk])
		x=np.reshape(x,(epk,size,1))

		y=np.concatenate(list(map(vect, td[1][i:i+epk])))
		y=np.reshape(y,(epk,10,1))
	return x,y


if __name__=="__main__":
	fbr=[784,30,10]
	mrk=cycle(fbr,12)
	
	td,_,_=load_data1()
	x,y=pack(td,12)
	mrk.bkp(x,y)
	# sgd(trd,10,3,epk=2,tsd=tsd)
	# for x,y in load_data1():
	# 	if x.shape[0]!=1:x.shape=(1,x.shape[0])
	# 	for i in range(1):
	# 		print(y)
	# 		mrk.bkp(x,y)
	# 		for lyr in mrk.csd[1:]:
	# 			lyr.update()
	# 			print('wt=',lyr.wt.T)
	# 			print('bs=',lyr.bs)
	# 		vd=np.argmax(mrk.csd[-1].act)		
			
	# 		# print(mrk.csd[2].z)
	# 		if y==vd: break

	
	

