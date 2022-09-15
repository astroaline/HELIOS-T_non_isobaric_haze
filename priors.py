class Priors2(object):

    def __init__(self):
        pass

    def DeltaFunctionPrior(self,r,x1,x2):
        """Uniform[0:1]  ->  Delta[x1]"""
        return x1

    def UniformPrior(self,r,x1,x2):
        """Uniform[0:1]  ->  Uniform[x1:x2]"""
        return x1+r*(x2-x1)

    def LogPrior(self,r,x1,x2):
        """Uniform[0:1]  ->  LogUniform[x1:x2]"""
        if (r <= 0.0):
                return -1.0e32
        else:
            from math import log10
            lx1=log10(x1); lx2=log10(x2)
            return 10.0**(lx1+r*(lx2-lx1))

    def GaussianPrior(self,r,mu,sigma):
        """Uniform[0:1]  ->  Gaussian[mean=mu,variance=sigma**2]"""
        from math import sqrt
        from scipy.special import erfcinv
        if (r <= 1.0e-16 or (1.0-r) <= 1.0e-16):
                return -1.0e32
        else:
                return mu+sigma*sqrt(2.0)*erfcinv(2.0*(1.0-r))
