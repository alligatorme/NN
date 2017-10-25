import numpy as np
from functools import reduce
import pickle
import gzip

def load_data():
	f = gzip.open('mnist.pkl.gz', 'rb')
	td, validation_data, tsd = pickle.load(f,encoding='bytes')
	f.close()
	# return (td[0][0],td[1][0])
	return list(zip(td[0][:],td[1][:])),list(zip(tsd[0],tsd[1]))


class layer:
	def __init__(self,n,k):
		self.nb=[]
		self.nw=[]
		if n:
			# self.wt=np.random.randn(n,k)
			# self.bs=np.random.randn(1,k)
			self.wt=0.01*np.ones((n,k))
			self.bs=0.01*np.ones((1,k))
			self.pwt=np.zeros(self.wt.shape)
			self.pbs=np.zeros(self.bs.shape)
		else:
			self.nb=None
		

	def pst_wt(self):
		# print(self.pwt.shape,self.nw.shape)
		self.pwt+=self.nw
		# print(self.nw)
		self.nw=[]
	def pst_bs(self):
		self.pbs+=self.nb.T
		self.nb=[]

	def update(self):
		self.wt+=crs(self.pwt)
		self.bs+=crs(self.pbs)


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
		init_cost(self.csd[-1],otp)		
		reduce(backward,self.csd[::-1])
		# print(otp,':',np.argmax(mrk.csd[-1].act))

def init_cost(lyr,otp):
	dmp=np.zeros((10,1))
	dmp[otp]=1
	lyr.nb=(lyr.act.T-dmp)*sigmoid_prime(lyr.z.T)
	# print(dmp.T,lyr.act.T,lyr.z)

def forward(hd,lyr):
	# print(hd.act.shape,lyr.wt.shape)
	# print(np.dot(hd.act,lyr.wt))
	# print(lyr.bs.shape)
	# print(lyr.bs)
	lyr.z=np.dot(hd.act,lyr.wt)+lyr.bs
	lyr.act=sigmoid(lyr.z)
	return lyr

def backward(lyr,hd):
	if hd.nb!=None:
		hd.nb=np.dot(lyr.wt,lyr.nb)*sigmoid_prime(hd.z.T) 	
	lyr.nw=np.dot(hd.act.T,lyr.nb.T)
	print('nb=',lyr.nb)#,'act=',hd.act,'nw=',lyr.nw)
	# print(lyr.nw)
	lyr.pst_bs()
	lyr.pst_wt()	
	return hd

def crs(src,npc=1,eta=3.0):
	# print(-eta/npc*src)
	return -eta/npc*src 

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	# ap=sigmoid(z)*(1-sigmoid(z))
	# print(ap)
	# return ap
	return sigmoid(z)*(1-sigmoid(z))

def evl(tsd):
	# tsr= [(np.argmax(mrk.fwd(x)), y) for (x, y) in tsd]
	tsr=np.zeros((2,10))
	for x,y in tsd:
		if np.argmax(mrk.fwd(x))==y:
			tsr[0][y]+=1
		tsr[1][y]+=1
	return 100*tsr[0]/tsr[1]

def sgd(trd,npc,eta,epk=1,tsd=None):
	for k in range(epk):
		np.random.shuffle(trd)
		for patch in [trd[k:k+npc] for k in range(0,len(trd),npc)]:			
			for x,y in patch:
				# for i in range(50):
				if x.shape[0]!=1:x.shape=(1,x.shape[0])
				mrk.bkp(x,y)
				for lyr in mrk.csd[1:]:
					lyr.update()
			# vd=np.argmax(mrk.csd[-1].act)
			# print(y,'=',vd)


		if tsd:
			print ("Epoch {0}: {1}".format(k, evl(tsd)))
		else:
			print ("Epoch complete")

def load_data1():
	f = gzip.open('mnist.pkl.gz', 'rb')
	td, validation_data, tsd = pickle.load(f,encoding='bytes')
	f.close()
	n=4
	return zip(td[0][:n],td[1][:n])

if __name__=="__main__":
	fbr=[784,30,10]
	mrk=cycle(fbr)

	# trd,tsd=load_data()
	# sgd(trd,10,1,epk=20,tsd=tsd)


	for x,y in load_data1():
		if x.shape[0]!=1:x.shape=(1,x.shape[0])
		for i in range(1):
			mrk.bkp(x,y)
			for lyr in mrk.csd[1:]:
				lyr.update()
			vd=np.argmax(mrk.csd[-1].act)		
			print(y,':',vd)
			# print(mrk.csd[2].z)
			if y==vd: break

	
	

