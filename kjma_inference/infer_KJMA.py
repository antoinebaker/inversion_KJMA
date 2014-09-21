"""The class InferKJMA is defined here."""

import numpy as np
import time
from scipy.linalg import eigh
from compute_KJMA_kinetics_fast import compute_KJMA_kinetics_all
from past_light_cone import past_light_cone_matrix_area
# Maybe use available MCMC packages such as emcee, pyStan, or pyMC, instead of current scripts ?
from MCMC_sampling import metropolis_mcmc, hamiltonian_mcmc, hamiltonian_mcmc_mass_matrix
from compute_MAP_KJMA import compute_theta_MAP_estimate, compute_I_MAP_estimate
from compute_MAP_KJMA import compute_C0_y_C0_u_for_param, compute_logZ
from compute_MAP_KJMA import product_C0_vec
from compute_MAP_KJMA import objective_MAP_KJMA, gradient_MAP_KJMA, hessian_MAP_KJMA


# TODO : overwrite special method __setattr__ instead of update_param ?
# TODO : change parameter names in param, compute_MAP_KJMA.py and past_light_cone.py (xdata,tdata,x,t) instead of (x,t,y,u)

# this function is called in the method grid_search_v_ell_0() of the class InferKJMA
def log_exp_trick(logZs):
    """Computes the proba from the unnormalized log-proba
    using the log-exp trick.

    Parameters
    ----------
    logZs : array
        The unnormalized log-proba.
    
    Returns
    -------
    Ps : array, same shape as logZs
        The corresponding proba.

    Notes
    -----
    The log-exp trick uses the identity:
    P[j] = exp(logZ[j]) /  ( sum_{i} exp(logZ[i]) )
         = 1 / (sum_{i} exp(logZ[i] - logZ[j]) ) 
    
    """
    a = logZs.ravel()
    b = np.exp( a[:,np.newaxis] - a )
    c = 1/b.sum(axis=0)
    P = c.reshape(logZs.shape)
    return P

# Definition of the class InferKJMA
class InferKJMA:
    """The class InferKJMA is used to infer from noisy data the underlying 
    replication program, and predict the corresponding replication kinetics.

    Parameters
    ----------
    x, t : arrays
        Times and positions at which the initiation rate should be inferred. 
        Note that the arrays have to be evenly spaced (`dx`= constant and 
        `dt`= constant)
    xdata, tdata : arrays
        Times and positions corresponding to `data`. 
    data : array
        Noisy unreplicated fractions, array of shape (len(xdata), len(tdata)). 
    v : float
        Replication fork velocity.
    sigma_d : float
        Noise level.
    I_0, sigma_0 : floats
        Set the prior dynamic range of the initiation rate.
    tau_0 : floats
        Sets the prior temporal smoothness scales of the initiation 
        rate.
    FixedOris : bool, default False
        Whether we have fixed well-positioned oris or a spatially smooth initiation rate.
        If `FixedOris=True`, the positions `x_oris` of the oris must be given.
        If `FixedOris=False`, the spatial smoothness `ell_0` must be given.
    x_oris : array (None by default)
        If FixedOris = True, sets the oris positions.
    ell_0 : float (None by default)
        If FixedOris = False, sets the spatial smoothness scale. 
    """
    
    def __init__(self, x, t, 
                 data, xdata, tdata, 
                 v, sigma_d, 
                 I_0, sigma_0, tau_0, ell_0 = None, 
                 FixedOris = False, x_oris = None):
        assert data.shape == (len(xdata), len(tdata))
        self.x = x
        self.t = t
        self.data = data
        self.xdata = xdata
        self.tdata = tdata
        self.v = v
        self.sigma_d = sigma_d
        self.sigma_0 = sigma_0
        self.I_0 = I_0
        self.tau_0 = tau_0
        self.FixedOris = FixedOris
        if FixedOris: 
            if x_oris is None:
                raise ValueError("If FixedOris = True, you must specify x_oris")
            # let's check that x_oris is indeed a subset of x
            x_in_x_oris = 1*(x[:,np.newaxis]==x_oris)
            # true if each x_ori appears once and only once in x
            x_oris_subset_x = np.all( x_in_x_oris.sum(0) == 1 ) 
            if not x_oris_subset_x:
                raise ValueError("x_oris must be a subset of x")
            self.x_oris = x_oris
            self.ell_0 = 0
        else:
            if ell_0 is None:
                raise ValueError("If FixedOris = False, you must specify ell_0")
            self.ell_0 = ell_0
        self._init_param()

    def _init_param(self):
        """Utility function. 
        Init the `param` dictionary used as input for most functions 
        in the compute_MAP_KJMA module
        """ 
        
        param={}
        param['y'] = self.x_oris if self.FixedOris else self.x 
        param['u'] = self.t
        for s in ['y','u']: param['N'+s] = len(param[s])
        param['sigma_d'] = self.sigma_d
        param['sigma_0'] = self.sigma_0
        param['m_0'] = np.log10(self.I_0)
        param['ell_0'] = self.ell_0
        param['tau_0'] = self.tau_0
        param['d'] = self.data.ravel()   # d vector Nxt
        self._param = param
        # we set non-computed attributes to None
        # that way, we know if these attributess are already computed or need to be computed
        keys = ['MAP','samples','samples_stats', 'I_MAP', 'logZ', 
                '_theta_MAP','_H_MAP','_H_MAP_D','_H_MAP_R', '_thetas']
        for key in keys: setattr(self, key, None)

    def update_param(self, **new_param):
        """The safe way to change parameters (erase the 
        computations done for the previous set of parameters).
        """
        
        for key, val in new_param.iteritems(): setattr(self, key, val)
        self._init_param()    

    def _compute_theta_MAP(self, verbose=False):
        """Utility function. 
        Compute the hidden attribute theta_MAP (from which 
        I_MAP can be computed).
        """
        
        tic = time.clock()
        lim={'v':self.v, 'x':self.xdata, 't':self.tdata, 'u':self.t}
        lim['du'] = lim['u'][1]-lim['u'][0]
        # if FixedOris, I_iu (ori i, time u) is in ini/min, RI = integral of I*(dy=1)*du 
        lim['y'] = self.x_oris if self.FixedOris else self.x
        lim['dy'] = 1 if self.FixedOris else lim['y'][1]-lim['y'][0]
        self._param['R'] = past_light_cone_matrix_area(lim)
        if verbose: print "R computed in {}s".format(time.clock()-tic)
        
        tic = time.clock()
        compute_C0_y_C0_u_for_param(param = self._param)
        if verbose: print "C0 computed in {}s".format(time.clock()-tic)
        
        tic = time.clock()
        self._theta_MAP = compute_theta_MAP_estimate(param = self._param)
        if verbose: print "theta_MAP found in {}s".format(time.clock()-tic)
        
    def _compute_H_MAP(self, verbose=False):
        """Utility function. 
        Compute the hidden attribute H_MAP (Hessian of the 
        objective function at the MAP, useful for MCMC sampling, 
        and for computing the log evience)
        """
        
        # compute _theta_MAP if necessary
        if self._theta_MAP is None: self._compute_theta_MAP(verbose)
        
        tic = time.clock()
        self._H_MAP = hessian_MAP_KJMA(theta = self._theta_MAP, param = self._param)
        if verbose: print "H_MAP computed in {}s".format(time.clock()-tic)
    
    def _compute_H_MAP_DR(self, verbose=False):
        """Utility function. 
        Compute the eigendecomposition H_MAP = R D R.T, 
        where H_MAP is the Hessian of the 
        objective function at the MAP 
        (useful for MCMC sampling).
        """
        
        # compute _H_MAP if necessary
        if self._H_MAP is None: self._compute_H_MAP(verbose)
        
        tic = time.clock()
        self._H_MAP_D, self._H_MAP_R = eigh(self._H_MAP)
        if verbose: print "Eigendecompostion of H_MAP computed in {}s".format(time.clock()-tic)
        
    def compute_I_MAP(self, verbose=False):
        """Computes the MAP estimate of I only.

        Notes
        -----
        If self.FixedOris = True, I = I_it (ori i, time t, unit ini/min)
        If self.FixedOris = False, I = I_xt  (position x, time t, unit ini/kb/min)
        """
        
        if self._theta_MAP is None: self._compute_theta_MAP(verbose)
        self.I_MAP = compute_I_MAP_estimate(param = self._param, theta_MAP = self._theta_MAP)
        
    def _get_I_xt_from_I_it(self,I_it):
        """Utility function. Converts I_it to I_xt.

        Parameters
        ----------
        I_it (ori i, time t, unit ini/min)

        Returns
        -------
         I_xt  (position x, time t, unit ini/kb/min)
        """
        assert self.FixedOris
        # we convert I_it (ori i, time t, unit ini/min) into I_xt (position x, time t, unit ini/kb/min)
        dx = self.x[1]-self.x[0]
        x_in_x_oris = 1*(self.x[:,np.newaxis]==self.x_oris)
        I_xt = np.dot(x_in_x_oris, I_it)/float(dx)
        return I_xt

    def compute_MAP(self, verbose=False):
        """Computes the MAP estimates of I and all replication kinetics 
        quantities, such as the replication timing distribution P, 
        the right-moving fork density rho_r, etc.
        """
        
        # compute I_MAP if necessary
        if self.I_MAP is None: self.compute_I_MAP(verbose)
        tic = time.clock()
        # I matrix I_xt (position x, time t, unit ini/kb/min)
        I = self._get_I_xt_from_I_it(I_it=self.I_MAP) if self.FixedOris else self.I_MAP
        dx, dt = self.x[1]-self.x[0], self.t[1]-self.t[0]
        lim = {'v':self.v, 'dy':dx,'du':dt}
        KJMA = compute_KJMA_kinetics_all(I=I, lim=lim) 
        if verbose: print "KJMA kinetics computed in {}s".format(time.clock()-tic)
        # theta reshaped as matrix range_y * range_u
        range_y, range_u = self._param['range_y'],self._param['range_u']
        KJMA['theta'] = self._theta_MAP.reshape(range_y, range_u)
        self.MAP = KJMA
        
    def compute_logZ(self, verbose=False):
        """Computes the log evidence"""
        
        if self._theta_MAP is None: self._compute_theta_MAP(verbose)
        if self._H_MAP is None: self._compute_H_MAP(verbose)
        self.logZ = compute_logZ(param = self._param, 
                                 theta_MAP = self._theta_MAP, H_MAP = self._H_MAP)
    
    def compute_MCMC_samples(self, method="hmc_mass", verbose=False):
        """MCMC sampling using the method given in argument

        Parameter
        ---------
        method : str, default to "hmc_mass"
            MCMC algorithm used to sample the posterior.
        
            If `method`= "hmc_mass", the Hamiltonian MCMC algorithm 
            is used, with the Hessian at the MAP as a mass matrix 
            for the momentum
        
            If `method`="laplace_approx", the Laplace approximation
            is used (works poorly!)
        
            If `method`= "vanilla_hmc", the vanilla Hamiltonian MCMC 
            algorithm is used, with an identity mass matrix (slow).
        
            Finally, if `method`="metropolis", the classic Metropolis 
            Hastings algorithm is used (very slow).
        
        """
        
        self.mcmc_method = method
        # get the MCMC samples in theta coordinate
        self._compute_thetas(verbose)
        # compute KJMA kinetics for each sample
        self._compute_samples()
        # and the associated stats
        self._compute_samples_stats()
      
    def _compute_thetas(self, verbose):
        """Utilitity function. Compute the MCMC samples in theta coordinates 
        using the method given in argument
        """
        
        if self._theta_MAP is None: self._compute_theta_MAP(verbose)

        # functions passed to the MCMC routine
        def findE(x): return objective_MAP_KJMA(x,self._param)
        def gradE(x): return gradient_MAP_KJMA(x,self._param)
        #MCMC initialized at the MAP
        x0 = self._theta_MAP 

        # We need the eigendecompostion of H_MAP for methods hmc_mass and laplace_approx
        if self.mcmc_method=="hmc_mass" or self.mcmc_method=="laplace_approx":
            if self._H_MAP_D is None: self._compute_H_MAP_DR(verbose)
            D, R = self._H_MAP_D, self._H_MAP_R         # H_MAP = R D R.T
            C_MAP = np.dot(R,np.diag(1/np.sqrt(D)))     # C_MAP = R 1/sqrt(D)
            H_MAP_inv = np.dot(C_MAP,C_MAP.T)           # H_MAP_inv = C_MAP C_MAP.T
            H_MAP_sqrt =  np.dot(R,np.diag(np.sqrt(D))) # H_MAP = H_MAP_sqrt H_MAP_sqrt.T
        
        # MCMC
        tic = time.clock()
        if self.mcmc_method=="metropolis":
            thetas = metropolis_mcmc(x0,findE,0.002,500)
        if self.mcmc_method=="vanilla_hmc":
            thetas = hamiltonian_mcmc(x0,findE,gradE,0.002,10,500)
        if self.mcmc_method=="hmc_mass":
            thetas = hamiltonian_mcmc_mass_matrix(x0,H_MAP_inv,H_MAP_sqrt,findE,gradE,0.1,10,500)
        if self.mcmc_method=="laplace_approx":
            # Laplace approximation theta ~ N(theta_MAP,Sigma_MAP)
            # theta = theta_MAP + C_MAP N(0,Id)
            kappas = np.random.normal(0,1,(500, self._param['N_theta']))
            thetas = theta_MAP + np.dot(kappas,C_MAP.T)
        if verbose: print "MCMC run in {}s".format(time.clock()-tic)
        self._thetas = thetas
        
    def _compute_samples(self):
        """Gives the corresponding KJMA kinetics for all samples 
        given by the MCMC routine.
        
        Parameters
        ----------
        thetas : array
            Array of shape (N_sample,N_theta).
        
        param : dict 
            Dict of keys {'y', 'u', ...}
        
        Returns
        -------
        samples : dict of arrays
            For key in {'I','s','rho_ini','rho_ter'}, 
               `samples[key]` array of shape (N_y, N_u, N_sample)
            For key in {'T','p','d_ini','d_ter'}, 
               `samples[key]` array of shape (N_y, N_sample)
            For key = 'theta',
               `samples[key]` array of shape (range_y, range_u, N_sample)
        
        """ 
        thetas, param = self._thetas, self._param
        
        Nx, Nt, N_sample = len(self.x), len(self.t), thetas.shape[0]
        keys_yu = {'I','s','rho_ini','rho_ter'}
        keys_y = {'d_ini','d_ter','T','p'}
        samples = {}
        for key in keys_yu: 
            samples[key] = np.zeros((Nx,Nt,N_sample))
        for key in keys_y: 
            samples[key] = np.zeros((Nx,N_sample))
        samples['theta'] = np.zeros((param['range_y'],param['range_u'],N_sample))
        
        dx, dt = self.x[1]-self.x[0], self.t[1]-self.t[0]
        lim = {'v':self.v, 'dy':dx,'du':dt}
        for s in range(N_sample):
            theta = thetas[s,:]
            C0_theta = product_C0_vec(param, theta)
            m = param['m_0'] + C0_theta
            I = 10**m
            I = I.reshape(param['Ny'],param['Nu'])
            # if FixedOris, we need to convert I to I_xt (postion x, time t, unit ini/kb/min)
            if self.FixedOris: I = self._get_I_xt_from_I_it(I_it=I)
            KJMA = compute_KJMA_kinetics_all(I=I, lim=lim) 
            for key in keys_yu: samples[key][:,:,s] = KJMA[key]
            for key in keys_y: samples[key][:,s] = KJMA[key]
            samples['theta'][:,:,s] = theta.reshape(param['range_y'],param['range_u'])
        
        self.samples = samples

    def _compute_samples_stats(self):
        """Gives the corresponding statistics (mean, median, etc.) of the samples.

        Parameters
        ----------
        samples : dict of arrays
            For key in {'I','s','rho_ini','rho_ter'}, 
                `samples[key]` array of shape (N_y, N_u, N_sample)
            For key in {'T','p','d_ini','d_ter'}, 
                `samples[key]` array of shape (N_y, N_sample)
            For key = 'theta',
                `samples[key]` array of shape (range_y, range_u, N_sample)
    
        Returns
        -------
        samples_stats : dict of dict of arrays
            For key in samples.keys() :
                `samples_stats['mean'][key]` : mean of `samples[key]` 
                `samples_stats['mean'][key]` : mean of `samples[key]` 
                `samples_stats['sd'][key]` : stddev of `samples[key]` 
                `samples_stats['5'][key]` : 5% percentile of `samples[key]` 
                `samples_stats['50'][key]` : 50% percentile (median) of `samples[key]` 
                `samples_stats['95'][key]` : 95% percentile of `samples[key]` 
        
        """ 
        samples = self.samples
        keys_yu = {'I','s','rho_ini','rho_ter','theta'}
        keys_y = {'d_ini','d_ter','T','p'}
        samples_stats= {}
        for key in {'mean','sd','5','50','95'}: samples_stats[key]={}
        for key in keys_yu: 
            samples_stats['mean'][key]=np.mean(samples[key],axis=2)
            samples_stats['sd'][key]=np.sqrt(np.var(samples[key],axis=2))
            samples_stats['5'][key]=np.percentile(samples[key],q=5,axis=2)
            samples_stats['50'][key]=np.percentile(samples[key],q=50,axis=2)
            samples_stats['95'][key]=np.percentile(samples[key],q=95,axis=2)
        for key in keys_y:
            samples_stats['mean'][key]=np.mean(samples[key],axis=1)
            samples_stats['sd'][key]=np.sqrt(np.var(samples[key],axis=1))
            samples_stats['5'][key]=np.percentile(samples[key],q=5,axis=1)
            samples_stats['50'][key]=np.percentile(samples[key],q=50,axis=1)
            samples_stats['95'][key]=np.percentile(samples[key],q=95,axis=1)
        self.samples_stats = samples_stats
    
    def grid_search_v_ell_0(self, vs, ell_0s, verbose=False):
        """Grid search for the fork velocity v and 
        the spatial scale ell_0. Return the posterior 
        probabilty P(v,ell_0).

        Parameters
        ----------
        vs : array
            Replication fork velocity
        ell_0s : array
           Spatial smoothness scale


        Return
        ------
        Ps : array
           Posterior probabilty P(v,ell_0), array of 
           shape (len(vs), len(ell_0s)).
        
        """
        if self.FixedOris: 
            raise ValueError("You cannot perform a grid search for ell_0 for FixedOris!")
        logZs = np.zeros( (len(vs), len(ell_0s)) )
        for i, v in enumerate(vs):
            for j, ell_0 in enumerate(ell_0s):
                self.update_param(v=v, ell_0=ell_0)
                tic = time.clock()
                self.compute_logZ()
                if verbose: print "logZ for v={}, ell_0={} computed in {}s".format(v,ell_0,time.clock()-tic)
                logZs[i,j] = self.logZ
        # finding the v and ell_0 that maximize the evidence
        index_max = np.argmax(logZs)
        (i_max, j_max) = np.unravel_index(index_max, dims=logZs.shape)
        v_max, ell_0_max = vs[i_max], ell_0s[j_max]
        if verbose: print "Evidence maximized at v={}, ell_0 = {}".format(v_max,ell_0_max)
        # the posterior probability for v and ell_0
        P = log_exp_trick(logZs)
        P_max = P[i_max,j_max]
        if verbose: print "At the resolution considered, P(v,ell_0) = {0:.3f}".format(P_max)
        if verbose: print "We set v, ell_0 to v={}, ell_0 = {}".format(v_max,ell_0_max)
        self.update_param(v=v_max, ell_0=ell_0_max)
        return P
        
