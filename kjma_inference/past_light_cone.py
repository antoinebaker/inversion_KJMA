"""For computing the past-light-cone matrix"""



import numpy as np

def past_light_cone_matrix_area_special_case(lim):
    """Compute the past-light-cone matrix R, but only in a special case 
    (cf Notes).
    
    The integral of I (given as an array at positions y and times u)
    over V_xt, the past-light-cone at postion x and time t, is given by
    sum_{yu} R[xt,yu] I[yu].

    Parameters
    ----------
    lim : dict
        lim['v'] : float, fork velocity
        lim['x'] : array, x-positions  
        lim['t'] : array, t-times
        lim['y'] : array, y-positions
        lim['u'] : array, u-times
        lim['dy'] : float, spatial resolution of y
        lim['du'] : float, temporal resolution of u
    
    Returns
    -------
    R : array
        Past-light-cone matrix, array of shape (Nx*Nt,Ny*Nu)
   
    Notes
    ------
    Assumes dy=v*du, and (x,t) subset of (y,u).
    Faster that past_light_cone_matrix_area.
    """ 
    # matrix (Nx*Nt, Ny*Nu)     
    xx, tt = np.meshgrid(lim['x'], lim['t'], indexing='ij')
    yy, uu = np.meshgrid(lim['y'], lim['u'], indexing='ij')
    mx, my = np.meshgrid(xx.ravel(), yy.ravel(), indexing='ij')
    mt, mu = np.meshgrid(tt.ravel(), uu.ravel(), indexing='ij')
    
    a = lim['du']*lim['dy']
    R = a*(abs(mx-my) < lim['v']*(mt-mu))\
        + 0.5*a*(abs(mx-my)==lim['v']*(mt-mu))\
        - 0.25*a*((mx==my) & (mt==mu))
    return R

def past_light_cone_matrix_area(lim):
    """Compute the past-light-cone matrix R.
    
    The integral of I (given as an array at positions y and times u)
    over V_xt, the past-light-cone at position x and time t, is given by
    sum_{yu} R[xt,yu] I[yu].

    Parameters
    ----------
    lim : dict
        lim['v'] : float, fork velocity
        lim['x'] : array, x-positions  
        lim['t'] : array, t-times
        lim['y'] : array, y-positions
        lim['u'] : array, u-times
        lim['dy'] : float, spatial resolution of y
        lim['du'] : float, temporal resolution of u
     
    Returns
    -------
    R : array
        Past-light-cone matrix, array of shape (Nx*Nt,Ny*Nu)
    
    """
    # matrix (Nx*Nt, Ny*Nu)     
    xx, tt = np.meshgrid(lim['x'], lim['t'], indexing='ij')
    yy, uu = np.meshgrid(lim['y'], lim['u'], indexing='ij')
    mx, my = np.meshgrid(xx.ravel(), yy.ravel(), indexing='ij')
    mt, mu = np.meshgrid(tt.ravel(), uu.ravel(), indexing='ij')
     
    def area(ta,tb,t0,t1):
	max_ta_t0 = np.maximum(ta,t0)
        min_tb_t1 = np.minimum(tb,t1)
        max_ta_t1 = np.maximum(ta,t1)
	i1 = (max_ta_t0<min_tb_t1)*0.5*((min_tb_t1-t0)**2-(max_ta_t0-t0)**2)
	i2 = (max_ta_t1<tb)*(t1-t0)*(tb-max_ta_t1)
        return i1+i2
    
    tX_ym = mt-abs(my-0.5*lim['dy']-mx)/float(lim['v'])
    tX_yp = mt-abs(my+0.5*lim['dy']-mx)/float(lim['v'])
    
    t0 = mu-0.5*lim['du']
    t1 = mu+0.5*lim['du']

    R1 = (my+0.5*lim['dy']<=mx)*lim['v']*area(tX_ym,tX_yp,t0,t1)
    R2 = ((my-0.5*lim['dy']<mx) & (mx<my+0.5*lim['dy']))*lim['v']\
        *(area(tX_ym,mt,t0,t1)+area(tX_yp,mt,t0,t1))
    R3 = (mx<my-0.5*lim['dy'])*lim['v']*area(tX_yp,tX_ym,t0,t1)
    R = R1+R2+R3
    return R 
