import numpy as np
from scipy.linalg import norm, inv
from math import sqrt

def standardize(D):
	S = D-np.tile(D.mean(axis=0),(D.shape[0],1))
	return (S.transpose()@S) / (S.shape[0] - 1)

def sthresh(x,t):
	"""x and t are double"""
	return np.sign(x) * max(abs(x)-t, 0.0)

def sthreshmat(x,tau,t):
	"""x and t are matrices, soft thresholding"""
	assert x.shape == t.shape, 'matrix shape mismatch'
	return np.sign(x) * np.maximum(abs(x)-tau*t, 0.0)

def pseudol(X,W):
	return -np.log(X.diagonal()).sum() + 0.5*(X.transpose()*W).sum()

class cc_fista(object):
	"""concord fista"""
	def __init__(self, D, lam, pMat=None, 
				DisS=0, penalize_diag=0, s_f = False,
				tol=1e-5, maxit=100, steptype=1):
		super(cc_fista, self).__init__()
		
		p = D.shape[1]
		if DisS:
			self.S = D
		else:
			self.S = standardize(D)

		if s_f:
			d = int(p/2)
			# penalty
			ff_lambdamat = lam*100*np.ones((d,d))
			np.fill_diagonal(ff_lambdamat,0)
			fs_lambdamat = lam*np.ones((d,d))
			self.LambdaMat = np.concatenate((ff_lambdamat,fs_lambdamat),axis=1)
			self.X0 = np.concatenate((np.identity(d),np.zeros((d,d))),axis=1)
		else:
			# penalty
			self.LambdaMat = lam*np.ones((p,p))
			if not penalize_diag:
				np.fill_diagonal(self.LambdaMat, 0)
			if pMat is not None:
				self.LambdaMat[pMat==0] *= 100
			self.X0 = np.identity(p)
		print('LambdaMat shape: ',self.LambdaMat.shape)
		print('Init shape: ',self.X0.shape)
		self.tol = tol
		self.maxit = maxit
		self.steptype = steptype

	def infer(self):
		# mat/obj init
		X = self.X0.copy()
		Theta = self.X0.copy()
		W = self.S@X
		WTh = self.S@Theta
		h = pseudol(X,W)
		# const init
		hn = hTh = Qn = f = 0.0
		taun = alpha = 1.0
		c = 0.9
		itr = diagitr = backitr = 0
		loop = True

		G = 0.5 * (WTh + WTh.transpose())
		G += - np.diag(1.0/Theta.diagonal())

		while loop:
			print(itr)
			tau = taun
			diagitr = backitr = 0

			while True:
				print('.')
				if diagitr != 0 or backitr != 0: 
					tau = tau * c

				Xn = sthreshmat(Theta-tau*G, tau, self.LambdaMat);
				if Xn.diagonal().min()<1e-8 and diagitr<50:
					diagitr += 1
					continue

				Step = Xn - Theta
				hTh = pseudol(Theta,WTh)
				Qn = hTh + (Step*G).sum() + (1/(2*tau))*(norm(Step)**2)
				Wn = self.S @ Xn
				hn = pseudol(Xn,Wn)
				if hn > Qn:
					backitr += 1
				else:
					break

			alphan = (1 + sqrt(1+4*(alpha**2)))/2;
			Theta = Xn + ((alpha-1)/alphan) * (Xn-X)
			WTh = self.S@Theta
			Gn = 0.5 * (WTh + WTh.transpose())
			Gn += - np.diag(1.0/Theta.diagonal())

			if self.steptype == 0:
				taun = 1
			elif self.steptype == 1:
				taun = tau
			elif self.steptype == 2:
				# taun = (Step.transpose()@Step).trace() / (Step.transpose()@(Gn-G)).trace()
				# using *.sum() is much faster:
				taun = (Step*Step).sum() / (Step*(Gn-G)).sum()
				if taun < 0.0:
					taun = tau
			# compute subg
			tmp = Gn + np.sign(Xn)*self.LambdaMat
			subg = sthreshmat(Gn, 1.0, self.LambdaMat)
			subg[Xn!=0] = tmp[Xn!=0]
			subgnorm = norm(subg)
			Xnnorm = norm(Xn)
			# iteration update
			alpha = alphan
			X = Xn
			h = hn
			G = Gn
			f = h + (abs(Xn)*self.LambdaMat).sum()
			itr += 1
			loop = itr<self.maxit and subgnorm/Xnnorm>self.tol
		return Xn

	def infer_s_f(self):
		# mat/obj init
		d = self.X0.shape[0]
		X = self.X0.copy()
		Theta = self.X0.copy()
		W = self.S@X.transpose()
		WTh = self.S@Theta.transpose()
		h = pseudol(X,W)
		# const init
		hn = hTh = Qn = f = 0.0
		taun = alpha = 1.0
		c = 0.9
		itr = diagitr = backitr = 0
		loop = True

		G = 0.5 * (Theta@self.S.transpose() + Theta@self.S)
		G += np.concatenate((-np.diag(1.0/Theta.diagonal()),np.zeros((d,d))), axis=1)

		inner_ctr = 0
		while loop:
			print(itr)
			tau = taun
			diagitr = backitr = 0

			while True:
				if diagitr != 0 or backitr != 0: 
					tau = tau * c
					print('tau',tau)
				Xn = sthreshmat(Theta-tau*G, tau, self.LambdaMat)
				if Xn.diagonal().min()<1e-8 and diagitr<50:
					print('diagitr',diagitr)
					diagitr += 1
					continue

				Step = Xn - Theta
				Qn = pseudol(Theta,WTh) + (Step*G).sum() + (1/(2*tau))*(norm(Step)**2)
				hn = pseudol(Xn,self.S@Xn.transpose())
				if hn > Qn:
					backitr += 1
					print('backitr',backitr)
				else:
					break
				if inner_ctr > 3:
					break
				else:
					inner_ctr += 1

			print('tau selected: ', tau)
			alphan = (1 + sqrt(1+4*(alpha**2)))/2
			Theta = Xn + ((alpha-1)/alphan) * (Xn-X)
			Gn = 0.5 * (Theta@self.S.transpose() + Theta@self.S)
			Gn += np.concatenate((-np.diag(1.0/Theta.diagonal()),np.zeros((d,d))), axis=1)

			if self.steptype == 0:
				taun = 1
			elif self.steptype == 1:
				taun = tau
			elif self.steptype == 2:
				taun = (Step*Step).sum() / (Step*(Gn-G)).sum()
				if taun < 0.0:
					taun = tau
			# compute subg
			tmp = Gn + np.sign(Xn)*self.LambdaMat
			subg = sthreshmat(Gn, 1.0, self.LambdaMat)
			subg[Xn!=0] = tmp[Xn!=0]
			subgnorm = norm(subg)
			Xnnorm = norm(Xn)
			# iteration update
			alpha = alphan
			X = Xn
			h = hn
			G = Gn
			f = h + (abs(Xn)*self.LambdaMat).sum()
			itr += 1
			print(subgnorm/Xnnorm)
			loop = itr<self.maxit and subgnorm/Xnnorm>self.tol
		return Xn

def test():
	# data_prep
	p = 10
	omega = np.identity(p)
	omega[1,5] = omega[5,1] = 0.99
	omega[2,6] = omega[6,2] = 0.99
	sigma = inv(omega)
	vectors = np.random.multivariate_normal(np.zeros(p),sigma,200)
	# infer
	fi = cc_fista(vectors,0.5)
	invcov = fi.infer()
	# info
	print(np.count_nonzero(omega))
	print(np.count_nonzero(invcov))
	import ipdb; ipdb.set_trace()
	print('omega:\n',omega)
	print('inferred invcov:\n',invcov)
	# check if nonzero entries align
	omega[omega!=0] = 1
	invcov[invcov!=0] = 1
	print('non-overlap nonzero entry count: ', np.count_nonzero(omega-invcov))

if __name__ == '__main__':
	test()