import numpy as np
import scipy.linalg as LA

def standardize(D):
	S = D-np.tile(D.mean(axis=0),(D.shape[0],1))
	return (S.transpose()@S) / (S.shape[0] - 1)

def sgn():
	pass

def sthresh():
	pass
	
def sthreshmat():
	pass

class cc_fista(object):
	"""concord fista"""
	def __init__(self, D, lam, pMat=None, 
				DisS=0, penalize_diag=0,
				tol=1e-5, maxit=300, steptype=1):
		super(cc_fista, self).__init__()
		
		p = D.shape[1]
		if DisS:
			self.S = D
		else:
			self.S = standardize(D)
		# penalty
		self.lamdaMat = lam*np.ones((p,p))
		if not penalize_diag:
			np.fill_diagonal(self.lamdaMat, 0)
		if pMat is not None:
			self.lamdaMat[pMat==0] *= 100
		# other init
		self.X0 = np.identity(p)
		self.tol = tol
		self.maxit = maxit
		self.steptype = steptype

	def infer():
		# mat/obj init
		X = self.X0.copy()
		Theta = self.X0.copy()
		W = self.S@X
		WTh = self.S@Theta
		h = -np.log(X.diagonal()).sum() + 0.5*(X@W).trace()
		# const init
		hn = hTh = Qn = f = 0.0
		taun = alpha = 1.0
		c = 0.9
		itr = diagitr = backitr = 0
		loop = 1

		G = 0.5 * (WTh + WTh.transpose())
  		G += - np.diag(1.0/Theta.diagonal())

  		while loop != 0:
  			tau = taun
  			diagitr = backitr = 0

  			while True:
  				if diagitr != 0 or backitr != 0: 
  					tau = tau * c
  				tmp = Theta - tau*G