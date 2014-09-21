"""
For generating artificial initiation rate and artificial noisy data

"""

import numpy as np
from past_light_cone import past_light_cone_matrix_area

def gaussian_blob(x,t,mean_x,mean_t,sigma_x,sigma_t):
    rx = ((x-mean_x)/float(sigma_x))**2
    rt = ((t-mean_t)/float(sigma_t))**2
    rx = rx[np.newaxis].T
    g = np.exp(-rx-rt)/(2*np.pi*sigma_x*sigma_t)
    return g

def generate_artificial_data(xdata,tdata,x,t,I_xt,v,sigma_d):
    lim = {'x':xdata,'y':x,'t':tdata,'u':t}
    for s in lim.keys():
        lim['N'+s] = len(lim[s])
        lim['d'+s] = lim[s][1]- lim[s][0]
    lim['v'] = v

    R = past_light_cone_matrix_area(lim=lim)
    s = np.exp(-np.dot(R,I_xt.ravel()))
    # s reshaped as a Nx*Nt matrix:
    true_s = s.reshape(len(xdata),len(tdata))
    # add gaussian noise
    data = true_s + np.random.normal(scale=sigma_d, size = true_s.shape)
    return true_s, data

def get_I_xt_from_I_it(I_it,x,t,x_oris):
    # let's check that x_oris is indeed a subset of x
    x_in_x_oris = 1*(x[:,np.newaxis]==x_oris)
    # true if each x_ori appears once and only once in x
    x_oris_subset_x = np.all( x_in_x_oris.sum(0) == 1 ) 
    if not x_oris_subset_x:
        raise ValueError("x_oris must be a subset of x")
    # we convert I_it (ori i, time t, unit ini/min) into I_xt (position x, time t, unit ini/kb/min)
    dx = x[1]-x[0]
    I_xt = np.dot(x_in_x_oris, I_it)/float(dx)
    return I_xt
