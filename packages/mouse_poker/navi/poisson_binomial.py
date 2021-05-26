from math import atan2, cos, exp, log, pi, sin, sqrt

import numpy as np
from scipy.fft import fft


class PoissonBinomial:
    """
    The poisson binomial distribution is essentially a binomial
    distribution where the probability of success on each trial
    is not necessarily identically distributed. This implementation
    and results are checked against https://github.com/tsakim/poibin/blob/master/poibin.py
    If all ps in prob_array are equal, this distribution reverts to the
    binomial distribution. This resultant p-values have also been checked 
    against the scipy implementation
    
    from https://pypi.org/project/poisson-binomial/#files
    """
    def __init__(self,prob_array):
        self.p = np.array(prob_array)
        self.pmf = self.get_poisson_binomial()
        self.cdf = np.cumsum(self.pmf)
        
    def x_or_less(self,x):
        return self.cdf[x]
    def x_or_more(self,x):
        return 1-self.cdf[x]+self.pmf[x]

    def get_poisson_binomial(self):

        """This version of the poisson_binomial is implemented 
        from the fast fourier transform method described in 
        'On computing the distribution function for the 
        Poisson binomial distribution'by Yili Hong 2013."""

        real = np.vectorize(lambda x: x.real)

        def x(w,l):
            v_atan2 = np.vectorize(atan2)
            v_sqrt = np.vectorize(sqrt)
            v_log = np.vectorize(log)

            if l==0:
                return complex(1,0)
            else:

                wl = w*l
                real = 1+self.p*(cos(wl)-1)
                imag = self.p*sin(wl)
                mod = v_sqrt(imag**2+real**2)
                arg = v_atan2(imag,real)
                d = exp((v_log(mod)).sum())
                arg_sum = arg.sum()
                a = d*cos(arg_sum)
                b = d*sin(arg_sum)
                return complex(a,b)

        n = self.p.size 
        w = 2*pi/(1+n)

        xs = [x(w,i) for i in range((n+1)//2+1)]
        for i in range((n+1)//2+1,n+1):
            c = xs[n+1-i]
            xs.append(c.conjugate())

        return real(fft(xs))/(n+1)
