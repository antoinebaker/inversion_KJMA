"""Old version of module compute_MAP_KJMA.

Notes
-----
In this version, the product of C0 with theta was defined
through a  product of sparse matrices :
C0*theta = (Id_y_C0_u) * (C0_y_Id_u * theta),
where Id_y_C0_u and C0_y_Id_u are sparse (Ny*Nu,Ny*Nu) 
matrices.

In the new version, faster and using less memory, 
the product of C0 with theta uses np.tensordot and no longer 
uses sparse matrices, only C0_u and C0_y. 
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import toeplitz, cholesky, eigh, inv
from numpy.linalg import slogdet
import scipy.sparse as sps

# WARNING 
# param['C0'] sparse matrix so matrix-vector product is C0v = param['C0']*v
# for numpy array the matrix-vector product is Av = np.dot(A,v)

# TODO : check input arguments (eg param, theta_MAP, etc...) ?


def objective_MAP_KJMA(theta,param):
    """Objective function E(theta) for the MAP optimization.
    
    E(theta) = 0.5*(d-s)/(sigma_d**2) +0.5*theta**2"""
    C0_theta = param['Id_y_C0_u']*(param['C0_y_Id_u']*theta)
    m = param['m0'] + C0_theta
    I = 10**m
    A = np.dot(param['R'],I)
    s = np.exp(-A)
    d = param['d']
    E = 0.5*np.sum(theta**2) + 0.5*np.sum((d-s)**2)/(param['sigma_d']**2) 
    return E

# E'(theta)
def gradient_MAP_KJMA(theta,param):
    """Gradient of the objective function E(theta) for the MAP optimization"""
    C0_theta = param['Id_y_C0_u']*(param['C0_y_Id_u']*theta)
    m = param['m0'] + C0_theta
    I = 10**m
    A = np.dot(param['R'],I)
    s = np.exp(-A)
    d = param['d']
    u = np.dot(s*(d-s),param['R'])*I
    u_C0 = (u*param['Id_y_C0_u'])*param['C0_y_Id_u']
    grad_E = theta + np.log(10)*u_C0/(param['sigma_d']**2)
    return grad_E

# E''(theta) p for any vector p
def hessp_MAP_KJMA(theta,p,param):
    """Product H(theta) times p, for any vector p, where H(theta) is the Hessian of the objective function E(theta) for the MAP optimization"""
    C0_theta = param['Id_y_C0_u']*(param['C0_y_Id_u']*theta)
    m = param['m0'] + C0_theta
    I = 10**m
    A = np.dot(param['R'],I)
    s = np.exp(-A)
    d = param['d']
    u = np.dot(s*(d-s),param['R'])*I
    C0_p = param['Id_y_C0_u']*(param['C0_y_Id_u']*p)
    w = np.dot(param['R'],I*C0_p)
    y = np.dot(s*(2*s-d)*w,param['R'])*I + u*C0_p
    y_C0 = (y*param['Id_y_C0_u'])*param['C0_y_Id_u']
    Hp = p + (np.log(10)**2)*y_C0/(param['sigma_d']**2)
    return Hp

#E''(theta)
def hessian_MAP_KJMA(theta,param):
    """Hessian of the objective function E(theta) or the MAP optimization"""
    C0_theta = param['Id_y_C0_u']*(param['C0_y_Id_u']*theta)
    m = param['m0'] + C0_theta
    I = 10**m
    A = np.dot(param['R'],I)
    s = np.exp(-A)
    r = param['d']-s
    s_I = np.outer(s,I)
    r_I = np.outer(r,I)
    J_m = -np.log(10)*(s_I*param['R'])
    K_m = -np.log(10)*(r_I*param['R'])
    J_theta =  (J_m*param['Id_y_C0_u'])*param['C0_y_Id_u']
    K_theta =  (K_m*param['Id_y_C0_u'])*param['C0_y_Id_u']
    rJ_vec = np.dot(r,J_m)
    Nyu = param['Ny']*param['Nu']
    assert len(rJ_vec)==Nyu
    D = sps.dia_matrix((rJ_vec,0), shape=(Nyu,Nyu))
    D_C0 = (D*param['Id_y_C0_u'])*param['C0_y_Id_u']
    C0T_D_C0 = param['C0_y_Id_u'].T*(param['Id_y_C0_u'].T*D_C0)
    H1_theta = np.dot(J_theta.T,J_theta)
    H2_theta = np.dot(K_theta.T,J_theta) + np.log(10)*C0T_D_C0
    H = np.identity(param['N_theta']) + (H1_theta - H2_theta)/(param['sigma_d']**2)
    return H

#pseudo E''(theta) p for any vector p 
def pseudo_hessp_MAP_KJMA(theta,p,param):
    "Product H(theta) p, for any vector p, where H(theta) is the pseudo-hessian of the objective function E(theta) for the MAP optimization. The pseudo-Hessian is guaranteed to be positive-definite for all values of theta."""
    C0_theta = param['Id_y_C0_u']*(param['C0_y_Id_u']*theta)
    m = param['m0'] + C0_theta
    I = 10**m
    A = np.dot(param['R'],I)
    s = np.exp(-A)
    C0_p = param['Id_y_C0_u']*(param['C0_y_Id_u']*p)
    w = np.dot(param['R'],I*C0_p)
    y = np.dot(s*s*w,param['R'])*I
    y_C0 = (y*param['Id_y_C0_u'])*param['C0_y_Id_u']
    Hp = p + (np.log(10)**2)*y_C0/(param['sigma_d']**2)
    return Hp

#pseudo E''(theta)
def pseudo_hessian_MAP_KJMA(theta,param):
    """Pseudo-hessian of the objective function E(theta) for the MAP optimization. The pseudo-Hessian is guaranteed to be positive-definite for all values of theta."""
    C0_theta = param['Id_y_C0_u']*(param['C0_y_Id_u']*theta)
    m = param['m0'] + C0_theta
    I = 10**m
    A = np.dot(param['R'],I)
    s = np.exp(-A)
    s_I = np.outer(s,I)
    J_m = -np.log(10)*(s_I*param['R'])
    J_theta =  (J_m*param['Id_y_C0_u'])*param['C0_y_Id_u']
    H1_theta = np.dot(J_theta.T,J_theta)
    H = np.identity(param['N_theta']) + H1_theta/(param['sigma_d']**2)
    return H

# compute C0
def compute_C0_y_C0_u_for_param(param):
    zapsmall = 1e-15
    
    if param['u0']==0:
        range_u = param['Nu']
        C0_u = np.identity(param['Nu'])
    else:
        #Sigma_u = R D R.T, C0 = R sqrt(D), Sigma_u = C0 C0.T
        Delta_u = (param['u']-param['u'][0])/param['u0']
        Sigma_u = toeplitz(np.exp(-0.5*Delta_u**2))
        D_u, R_u = eigh(Sigma_u)
        index_u, = np.where(D_u > zapsmall)
        range_u = len(index_u)
        sigma_u = np.sqrt(D_u[index_u])
        C0_u = np.dot(R_u[:,index_u],np.diag(sigma_u))
    Id_y = sps.identity(param['Ny'])
    Id_y_C0_u = param['sigma0']*sps.kron(Id_y,C0_u)
    
    if param['y0']==0:
        range_y = param['Ny']
        C0_y = np.identity(param['Ny'])
    else:
        #Sigma_y = R D R.T, C0 = R sqrt(D), Sigma_y = C0 C0.T
        Delta_y = (param['y']-param['y'][0])/param['y0']
        Sigma_y = toeplitz(np.exp(-0.5*Delta_y**2))
        D_y, R_y = eigh(Sigma_y)
        index_y, = np.where(D_y > zapsmall)
        range_y = len(index_y) 
        sigma_y = np.sqrt(D_y[index_y])
        C0_y = np.dot(R_y[:,index_y],np.diag(sigma_y))
    Id_u = sps.identity(range_u)   
    C0_y_Id_u = sps.kron(C0_y,Id_u)
    param['Id_y_C0_u'] = Id_y_C0_u
    param['C0_y_Id_u'] = C0_y_Id_u
    param['range_y'] = range_y
    param['range_u'] = range_u
    param['N_theta'] = range_y*range_u

def compute_theta_MAP_estimate(param):
    """Perfom the minimization of the objective function E(theta)"""
    # optimization 
    options={'xtol': 1e-8, 'disp': False, 'maxiter':200}
    theta0 = np.zeros(param['N_theta'],float)
    res = minimize(objective_MAP_KJMA, theta0, 
                   args=(param,), method='Newton-CG', 
                   jac=gradient_MAP_KJMA, hessp=pseudo_hessp_MAP_KJMA, 
                   options=options)
    theta_MAP = res.x
    return theta_MAP

def compute_I_MAP_estimate(param, theta_MAP):
    """Compute the MAP estimate of I.""" 
    # theta_MAP vector N_theta, m_MAP vector Nyu
    m_MAP = param['m0'] + param['Id_y_C0_u']*(param['C0_y_Id_u']*theta_MAP)
    # I_MAP matrix Ny*Nu
    I_MAP = 10**m_MAP.reshape(param['Ny'],param['Nu'])
    return I_MAP

# Laplace approximation theta ~ N(theta_MAP,Sigma_MAP)
# evidence Z(d) = P(d)
# log Z = - E_MAP - (N_d/2)*log(2*pi*sigma_d**2) - 0.5*log(det(H_MAP)) 
def compute_logZ(param, theta_MAP, H_MAP):
    """Compute the log evidence, using the laplace approximation"""
    
    E_MAP = objective_MAP_KJMA(theta_MAP,param)
    N_d = len(param['d'])
    sigma_d = param['sigma_d']
    (sign, logdet) = np.linalg.slogdet(H_MAP)
    assert sign==1
    log_Z = - E_MAP - 0.5*N_d*np.log(2*np.pi*sigma_d**2) - 0.5*logdet
    return log_Z

def KJMA_samples(thetas, param):
    """ Gives the corresponding KJMA kinetics for all samples given by the MCMC routine (in 'theta' coordinate)""" 
    N_sample = thetas.shape[0]
    keys_yu = {'I','s','rho_ini','rho_ter'}
    keys_y = {'d_ini','d_ter','T','p'}
    samples = {}
    
    for key in keys_yu: 
        samples[key] = np.zeros((param['Ny'],param['Nu'],N_sample))
    for key in keys_y: 
        samples[key] = np.zeros((param['Ny'],N_sample))
    samples['theta'] = np.zeros((param['range_y'],param['range_u'],N_sample))
    
    for s in range(N_sample):
        theta = thetas[s,:]
        C0_theta = param['Id_y_C0_u']*(param['C0_y_Id_u']*theta)
        m = param['m0'] + C0_theta
        I = 10**m
        I = I.reshape(param['Ny'],param['Nu'])
        KJMA = compute_KJMA_kinetics_all(I,param)        
        KJMA['I'] = I
        
        for key in keys_yu: samples[key][:,:,s] = KJMA[key]
        for key in keys_y: samples[key][:,s] = KJMA[key]
        samples['theta'][:,:,s] = theta.reshape(param['range_y'],param['range_u'])
    
    return samples
