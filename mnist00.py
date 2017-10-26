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
	def __init__(self,n,k):
		self.nb=[]
		self.nw=[]
		if n:
			self.wt=np.random.randn(k,n)
			self.bs=np.random.randn(k)
			self.pwt=np.zeros(self.wt.shape)
			self.pbs=np.zeros(self.bs.shape)
		else:
			self.nb=None
		

	def pst_wt(self):
		self.pwt+=self.nw
		self.nw=[]
	def pst_bs(self):
		self.pbs+=self.nb.T
		self.nb=[]

	def update(self,func):
		self.wt+=func(self.pwt)
		self.bs+=func(self.pbs)
		self.pwt=np.zeros(self.wt.shape)
		self.pbs=np.zeros(self.bs.shape)



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
		# for i,j in zip(self.csd[:-1],self.csd[1:]):
		# 	forward(i,j)
		# init_cost(self.csd[-1],otp)
		# for i,j in zip(self.csd[-1:0:-1],self.csd[-2::-1]):
		# 	backward(i,j)	
		reduce(forward,self.csd)
		init_cost(self.csd[-1],otp)		
		reduce(backward,self.csd[::-1])

def init_cost(lyr,otp):
	dmp=np.zeros((10,1))
	dmp[otp]=1
	lyr.nb=(lyr.act.T-dmp)*sigmoid_prime(lyr.z.T)

def forward(hd,lyr):
	lyr.z=np.dot(hd.act,lyr.wt)+lyr.bs
	lyr.act=sigmoid(lyr.z)
	return lyr

def backward(lyr,hd):
	if hd.nb!=None:
		hd.nb=np.dot(lyr.wt,lyr.nb)*sigmoid_prime(hd.z.T) 	
	lyr.nw=np.dot(hd.act.T,lyr.nb.T)
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
	for k in range(epk):
		np.random.shuffle(trd)
		for patch in (trd[k:k+npc] for k in range(0,len(trd),npc)):
			for x,y in patch:
				# for i in range(50):
				if x.shape[0]!=1:x.shape=(1,x.shape[0])
				mrk.bkp(x,y)
				for lyr in mrk.csd[1:]:
					lyr.update(partial(crs,npc,eta))

		if tsd:
			print ("Epoch {0}: {1}".format(k, evl(tsd)))
		else:
			print ("Epoch complete")

def load_data1():
	f = gzip.open('mnist.pkl.gz', 'rb')
	td, validation_data, tsd = pickle.load(f,encoding='bytes')
	f.close()
	n=2
	return zip(td[0][:n],td[1][:n])

if __name__=="__main__":
	fbr=[784,30,10]
	mrk=cycle(fbr)

	trd,tsd=load_data()
	sgd(trd,10,3,epk=2,tsd=tsd)


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

	
	

