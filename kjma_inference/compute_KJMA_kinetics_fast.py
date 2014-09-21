"""Computes the replication kinetics from the initiation rate,
using the KJMA analytical formulas.

Notes
-----
   In the current implementation, it only works if dy = v du, 
   where dy is the spatial resolution of the initiation rate, du
   the temporal resolution, and v the replication fork velocity.

"""


import numpy as np

#I_yu matrix (Ny,Nu)
def compute_integrals_of_I(I_yu,lim):
    """Computes the integral of I over the past-cone 
    and the boundaries of the past-cone.
    
    Parameters
    ----------
    I_yu : array
        Initiation rate. Array of shape (Ny, Nu)
    lim : dict
        lim['v'] : float, fork velocity (in kb/min)
        lim['dy'] : float, spatial resolution (in kb)
        lim['du'] : float, temporal resolution (in min)
    
    Returns
    -------
    [A_yu, L_r_yu, L_l_yu] : list of arrays.
        A_yu : integral of I over V_yu
        L_r_yu = integral of I over L_r_yu
        L_l_yu = integral of I over L_l_yu
                
    Notes
    ------
    Assumes dy=v*du.
    
    """
    
    Ny, Nu = I_yu.shape
    assert lim['dy']==lim['v']*lim['du']
    area = lim['dy']*lim['du']    

    #init arrays
    #on elargit les matrices pour les conditions aux bord
    I = np.zeros((Ny+2,Nu+2),float)
    I[1:(Ny+1),1:(Nu+1)] = I_yu.copy()
    A = np.zeros((Ny+2,Nu+2),float)
    A_r = np.zeros((Ny+2,Nu+2),float)
    A_l = np.zeros((Ny+2,Nu+2),float)
    L_r = np.zeros((Ny+2,Nu+2),float)
    L_l = np.zeros((Ny+2,Nu+2),float)

    #cacul des integrales par recurrence
    for u in range(1,Nu+1):        
        A[1:(Ny+1),u] = A_r[0:Ny,u-1] + A[1:(Ny+1),u-1] + A_l[2:(Ny+2),u-1]\
            + area*(0.25*I[0:Ny,u-1] + 0.75*I[1:(Ny+1),u-1] + 0.25*I[2:(Ny+2),u-1])\
            + area*0.25*I[1:(Ny+1),u]
 
        A_r[1:(Ny+1),u] = A_r[0:Ny,u-1] + area*0.25*I[0:Ny,u-1]\
            + area*(0.5*I[1:(Ny+1),u-1] + 0.25*I[1:(Ny+1),u])
        A_l[1:(Ny+1),u] = A_l[2:(Ny+2),u-1] + area*0.25*I[2:(Ny+2),u-1]\
            + area*(0.5*I[1:(Ny+1),u-1] + 0.25*I[1:(Ny+1),u])

        L_r[1:(Ny+1),u] = L_r[0:Ny,u-1]\
            + lim['dy']*0.5*(I[0:Ny,u-1] + I[1:(Ny+1),u])
        L_l[1:(Ny+1),u] = L_l[2:(Ny+2),u-1]\
            + lim['dy']*0.5*(I[2:(Ny+2),u-1] + I[1:(Ny+1),u])
    
    A_yu = A[1:(Ny+1),1:(Nu+1)]
    L_r_yu = L_r[1:(Ny+1),1:(Nu+1)]
    L_l_yu = L_l[1:(Ny+1),1:(Nu+1)]

    return [A_yu, L_r_yu, L_l_yu]


#FIX ME
#I_yu matrix (Ny,Nu)
def compute_integrals_of_I_any_v(I_yu,lim):
    """Doesn't work!!!!!!"""
    
    raise NameError('This function does not work.')
    Ny, Nu = I_yu.shape

    #init arrays
    #on elargit les matrices pour les conditions aux bord
    I = np.zeros((Ny+2,Nu+2),float)
    I[1:(Ny+1),1:(Nu+1)] = I_yu.copy()
    A = np.zeros((Ny+2,Nu+2),float)
    L_r = np.zeros((Ny+2,Nu+2),float)
    L_l = np.zeros((Ny+2,Nu+2),float)
    
    #cacul des integrales par recurrence
    for u in range(1,Nu+1):        
        A[1:(Ny+1),u] = A[1:(Ny+1),u-1] + lim['du']*(L_r[1:(Ny+1),u-1]+L_l[1:(Ny+1),u-1])
        
        L_r[1:(Ny+1),u] = L_r[1:(Ny+1),u-1] + lim['v']*lim['du']*I[1:(Ny+1),u-1]\
            - lim['v']*lim['du']*(L_r[1:(Ny+1),u-1] - L_r[0:(Ny+0),u-1])/(1.*lim['dy'])
        L_l[1:(Ny+1),u] = L_l[1:(Ny+1),u-1] + lim['v']*lim['du']*I[1:(Ny+1),u-1]\
            + lim['v']*lim['du']*(L_l[2:(Ny+2),u-1] - L_l[1:(Ny+1),u-1])/(1.*lim['dy'])
        
    A_yu = A[1:(Ny+1),1:(Nu+1)]
    L_r_yu = L_r[1:(Ny+1),1:(Nu+1)]
    L_l_yu = L_l[1:(Ny+1),1:(Nu+1)]

    return [A_yu, L_r_yu, L_l_yu]


def compute_KJMA_kinetics_of_I(I,lim):
    """Compute s, rho_r, rho_l, rho_ini, rho_ter, P, pol 
    from the initiation rate I.""" 
    
    assert lim['dy']==lim['v']*lim['du']   
    A, L_r, L_l =  compute_integrals_of_I(I,lim)
    
    s = np.exp(-A)
    rho_r = L_r*s
    rho_l = L_l*s
    rho_ini = I*s
    rho_ter = 2*L_r*L_l*s/float(lim['v']) 
    P = rho_r+rho_l
    pol = rho_r-rho_l
    return [s, rho_r, rho_l, rho_ini, rho_ter, P, pol]


def compute_KJMA_kinetics_all(I,lim):
    """Computes the replication kinetics from the initation rate.
    
    Parameters
    ----------
    I : array
        Initation rate. Array of shape (Ny, Nu).
    lim : dict
        lim['v'] : float, fork velocity.
        lim['dy'] : float, spatial resolution.
        lim['du'] : float, temporal resolution.
    
    Returns
    -------
    KJMA : dict of arrays.
    Arrays of shape (Ny, Nu)
        KJMA['I'] : initiation rate 
        KJMA['s'] : unreplicated fraction
        KJMA['P'] : prob distr of the replication timing
        KJMA['rho_r'] : density of right-moving forks
        KJMA['rho_l'] : density of left-moving forks
        KJMA['pol'] : rho_r - rho_l
        KJMA['rho_ini'] : density of initiations 
        KJMA['rho_ter'] : density of terminations
    Arrays of shape (Ny,)
        KJMA['T'] : mean replication timing
        KJMA['p'] : replication fork polarity
        KJMA['d_ini'] : spatial density of initiations
        KJMA['d_ter'] : spatial density of terminations
    Arrays of shape (Nu,)
        KJMA['f_L'] : length of replicated DNA at time u
        KJMA['r_L'] : rate of DNA synthesis at time u 
        KJMA['I_L'] : number of initiations per length of unreplicated DNA at time u 
        KJMA['n_ter'] : temporal density of initiations
        KJMA['n_ini'] : temporal density of terminations 
                
    Notes
    ------
    Assumes dy=v*du
    
    """    
    
    assert lim['dy']==lim['v']*lim['du']   
    A, L_r, L_l =  compute_integrals_of_I(I,lim)
    
    s = np.exp(-A)
    rho_r = L_r*s
    rho_l = L_l*s
    rho_ini = I*s
    rho_ter = 2*L_r*L_l*s/float(lim['v']) 
    P = rho_r+rho_l
    pol = rho_r-rho_l
    p = np.sum(pol,axis=1)*lim['du']
    T = np.sum(s,axis=1)*lim['du']
    f_L = np.sum(1-s,axis=0)*lim['dy']  #longueur d'ADN repliquee au temps t
    r_L = np.sum(P,axis=0)*lim['dy']    #taux de synthese d'ADN au temps t
    I_L = np.sum(rho_ini,axis=0)/np.sum(s,axis=0)
    d_ini = np.sum(rho_ini,axis=1)*lim['du']
    d_ter = np.sum(rho_ter,axis=1)*lim['du']
    n_ini = np.sum(rho_ini,axis=0)*lim['dy']
    n_ter = np.sum(rho_ter,axis=0)*lim['dy']
    N_ini = np.sum(rho_ini)*lim['du']*lim['dy']
    N_ter = np.sum(rho_ter)*lim['du']*lim['dy']
    KJMA = {'I':I, 's':s, 'rho_r':rho_r, 'rho_l':rho_l, 
            'rho_ini':rho_ini, 'rho_ter':rho_ter, 'P':P, 'pol':pol, 
            'd_ini':d_ini, 'd_ter':d_ter, 'T':T, 'p':p, 
            'n_ini':n_ini, 'n_ter':n_ter, 'N_ini':N_ini, 'N_ter':N_ter, 
            'I_L':I_L, 'f_L':f_L, 'r_L':r_L}
    return KJMA
