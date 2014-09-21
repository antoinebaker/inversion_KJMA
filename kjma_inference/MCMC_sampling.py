"""MCMC sampling algorithms"""


import numpy as np
from random import random as runif

def metropolis_mcmc(x0,findE,epsilon,N): 
    d=len(x0)
    x = x0
    xs = np.zeros((N,d))
    accept = 0

    for t in range(N):     # loop N times
        xs[t,] = x         # store all x's
        xnew = x + np.random.normal(scale=epsilon,size=d)
        dE = findE(xnew) - findE(x)
        a = np.exp(-dE)    # acceptance ratio
        if runif()<a:      # accept 
            x = xnew
            accept += 1
    
    freq = 100*accept/float(N)
    print "acceptance freq of {}%".format(freq)
    return xs

def acf(x, length=20):
    a = np.zeros(length)
    a[0] = 1
    y = x - x.mean()
    xvar = x.var(ddof=1)
    for i in range(1,length):
        a[i] = np.mean(y[i:]*y[:-i])/xvar
    return a

def hamiltonian_mcmc(x0,findE,gradE,epsilon,Tau,N):
    d=len(x0)
    xs = np.zeros((N,d))
    g = gradE(x0); x = x0; E = findE(x0)
    accept = 0

    for t in range(N):                  # loop N times
        xs[t,] = x
        p =  np.random.normal(size=d)   # initial momentum is Normal(0,1)
        H = 0.5*np.sum(p**2) + E        # evaluate H(x,p)

        xnew = x; gnew = g
        for tau in range(Tau):          # make Tau leapfrog steps
            p = p - 0.5*epsilon*gnew    # make half step in p
            xnew = xnew + epsilon*p     # make step in x
            gnew = gradE(xnew)          # find new gradient
            p = p - 0.5*epsilon*gnew    # make half step in p
        
        Enew = findE(xnew)
        Hnew = 0.5*np.sum(p**2) + Enew  # find new value of H
        dH = Hnew - H                   # acceptance ratio is exp(-dH)

        if runif()<np.exp(-dH):            # accept
            accept += 1
            g = gnew; x = xnew; E = Enew
        
    freq = 100*accept/float(N)
    print "acceptance freq of {}%".format(freq)
    return xs

def hamiltonian_mcmc_mass_matrix(x0,M_inv,M_sqrt,findE,gradE,epsilon,Tau,N):
    d=len(x0)
    xs = np.zeros((N,d))
    g = gradE(x0); x = x0; E = findE(x0)
    accept = 0

    for t in range(N):                            # loop N times
        xs[t,] = x
        q = np.random.normal(size=d)                  # q ~ N(0,1)
        p = np.dot(M_sqrt,q)                          # p ~ N(0,M)
        H = 0.5*np.sum(p*np.dot(M_inv,p)) + E         # evaluate H(x,p)

        xnew = x; gnew = g
        for tau in range(Tau):                        # make Tau leapfrog steps
            p = p - 0.5*epsilon*gnew                     # make half step in p
            xnew = xnew + epsilon*np.dot(M_inv,p)        # make step in x
            gnew = gradE(xnew)                           # find new gradient
            p = p - 0.5*epsilon*gnew                     # make half step in p
        
        Enew = findE(xnew)
        Hnew = 0.5*np.sum(p*np.dot(M_inv,p)) + Enew   # find new value of H
        dH = Hnew - H                                 # acceptance ratio is exp(-dH)

        if runif()<np.exp(-dH):                       # accept
            accept += 1
            g = gnew; x = xnew; E = Enew
        
    freq = 100*accept/float(N)
    print "acceptance freq of {}%".format(freq)
    return xs



