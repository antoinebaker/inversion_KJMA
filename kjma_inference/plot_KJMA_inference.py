"""Plotting functions and better defaults for matplotlib""" 

from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np

def set_matplotlib_defaults():
    #colorbrewer2 Dark2 qualitative color table
    dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
                    (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                    (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                    (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                    (0.4, 0.6509803921568628, 0.11764705882352941),
                    (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                    (0.6509803921568628, 0.4627450980392157, 0.11372549019607843)]
    
    rcParams['patch.facecolor'] = dark2_colors[0]
    rcParams['axes.color_cycle'] = dark2_colors
    rcParams['font.size'] = 18
    rcParams['font.family'] = 'STIXGeneral'
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['image.origin']='innner'
    rcParams['image.aspect']='auto'
    rcParams['image.cmap']='jet'     # colormap
    rcParams['axes.labelcolor']='k'
    rcParams['axes.edgecolor']='gray'
    rcParams['axes.titlesize']='medium'
    rcParams['xtick.color']='gray'
    rcParams['xtick.direction']='out'
    rcParams['xtick.labelsize']='small'
    rcParams['ytick.color']='gray'
    rcParams['ytick.direction']='out'
    rcParams['ytick.labelsize']='small'

#utility function for adding a colorbar
def add_colorbar(ax, im):
    """ Add a colobar on top of an image"""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    cbar.locator = MaxNLocator(3)
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize='x-small')
    
def remove_junk_xyplot(ax):
    """Minimize chartjunk by stripping out unnecesasry plot borders and axis ticks"""
    ax.spines['top'].set_visible(False) # no top or right spine
    ax.spines['right'].set_visible(False)
    ax.locator_params(nbins=3)    # 3 ticks at most
    ax.yaxis.set_ticks_position('left')  # ticks only on left and bottom spines
    ax.xaxis.set_ticks_position('bottom')
    # x and y axis are slightly moved outward
    ax.spines['bottom'].set_position(('outward', 10))  
    ax.spines['left'].set_position(('outward', 10))

def remove_junk_imshow(ax):
    """Minimize chartjunk by stripping out unnecesasry plot borders and axis ticks"""
    ax.locator_params(nbins=3)    # 3 ticks at most
    ax.yaxis.set_ticks_position('left')  # ticks only on left and bottom spines
    ax.xaxis.set_ticks_position('bottom')


# for displaying math titles
math_titles = {'I': r'$I(x,t)$',
               's': r'$s(x,t)$',
               'P': r'$P(x,t)$',
               'pol': r'$p(x,t)$',
               'rho_r': r'$\rho_{(+)}(x,t)$',
               'rho_l': r'$\rho_{(-)}(x,t)$',
               'rho_ini': r'$\rho_{ini}(x,t)$',
               'rho_ter': r'$\rho_{ter}(x,t)$',
               'T':r'$T(x)$',
               'p':r'$p(x)$',
               'd_ini': r'$\rho_{ini}(x)$',
               'd_ter': r'$\rho_{ter}(x)$'}
   
math_titles_MAP = {'I': r'$I_{MAP}(x,t)$',
                   's': r'$s_{MAP}(x,t)$',
                   'P': r'$P_{MAP}(x,t)$',
                   'pol': r'$p_{MAP}(x,t)$',
                   'rho_r': r'$\rho_{(+),MAP}(x,t)$',
                   'rho_l': r'$\rho_{(-),MAP}(x,t)$',
                   'rho_ini': r'$\rho_{ini,MAP}(x,t)$',
                   'rho_ter': r'$\rho_{ter,MAP}(x,t)$',
                   'T':r'$T_{MAP}(x)$',
                   'p':r'$p_{MAP}(x)$',
                   'd_ini': r'$\rho_{ini,MAP}(x)$',
                   'd_ter': r'$\rho_{ter,MAP}(x)$'}



def plot_initiation_rate(I_xt,extent):
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(I_xt.T, extent=extent)
    add_colorbar(ax, im)
    remove_junk_imshow(ax)
    ax.set_xlabel('x (kb)')
    ax.set_ylabel('t (min)')
    ax.text(0.9, 0.9, r'$I(x,t)$',
            verticalalignment='top', horizontalalignment='right',
            transform=ax.transAxes,
            color='white')


def plot_KJMA_kinetics(KJMA, extent):
    fig, axs = plt.subplots(2,3,sharex=True, sharey=True, figsize=(10,8))
    keys = ['s', 'P', 'rho_r', 'rho_l', 'rho_ini', 'rho_ter']
    for (i,j), ax in np.ndenumerate(axs):
        key = keys[2*j+i]
        val, valname = KJMA[key], math_titles[key]
        im = ax.imshow(val.T, extent=extent)
        if key=='s': im.set_clim(0,1)
        add_colorbar(ax, im)
        remove_junk_imshow(ax)
        if j==0: ax.set_ylabel('t (min)')
        if i==1: ax.set_xlabel('x (kb)')
        ax.text(0.9, 0.9, valname,
                verticalalignment='top', horizontalalignment='right',
                transform=ax.transAxes,
                color='white')
    fig.tight_layout()

def plot_check_true_s(x, true_s, KJMA):
    fig, ax = plt.subplots()
    ax.plot(x, KJMA['s'], 'gray')
    ax.plot(x, true_s)
    remove_junk_xyplot(ax)
    ax.set_xlabel('$x$  (kb)')
    ax.set_ylabel(r'$s(x,t)$')

def plot_waterfall_true_s(xdata, tdata, true_s, x, t, KJMA):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in range(len(t)):
        ax.plot(x, t[i]+0*x, KJMA['s'][:,i],'k',alpha=0.1)
    for i in range(len(tdata)):
        ax.plot(xdata, tdata[i]+0*xdata, true_s[:,i])
    ax.set_ylim(50,0)
    ax.set_zlim(-0.2,1.2)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('s')
    ax.locator_params(nbins=3)

def plot_waterfall_data(xdata, tdata, data):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in range(len(tdata)):
        ax.plot(xdata, tdata[i]+0*xdata, data[:,i], '.')
    ax.set_ylim(50,0)
    ax.set_zlim(-0.2,1.2)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('data')
    ax.locator_params(nbins=3)

def plot_compare_xt_images(KJMA,KJMA_MAP,key,clim,extent):
    fig, axs = plt.subplots(1,2,sharey=True, figsize=(6,4))
    for (i,), ax in np.ndenumerate(axs):
        if i==0: val, valname = KJMA[key], math_titles[key]
        if i==1: val, valname = KJMA_MAP[key], math_titles_MAP[key]   
        im = ax.imshow(val.T, extent=extent)
        im.set_clim(clim)
        add_colorbar(ax, im)
        remove_junk_imshow(ax)
        if i==0: ax.set_ylabel('t (min)')
        ax.set_xlabel('x (kb)')
        ax.text(0.9, 0.9, valname,
                verticalalignment='top', horizontalalignment='right',
                transform=ax.transAxes,
                color='white')
    fig.tight_layout()

def plot_compare_x_profiles(x,KJMA,KJMA_MAP):
    fig, axs = plt.subplots(2,2,sharex=True, sharey=False, figsize=(8,6))
    keys = ['T', 'p', 'd_ini', 'd_ter']
    for (i,j), ax in np.ndenumerate(axs):
        key = keys[2*j+i]
        ax.plot(x,KJMA[key])
        ax.plot(x,KJMA_MAP[key],'--')
        remove_junk_xyplot(ax)
        ax.set_ylabel(math_titles[key])
        if i==1: ax.set_xlabel('x (kb)')
        if key=='T': ax.set_ylim([0,50])
        if key=='p': ax.set_ylim([-1,1])
        if key=='d_ini': ax.set_ylim([-0.002,0.2])
        if key=='d_ter': ax.set_ylim([-0.001,0.1])
        ax.locator_params(nbins=3)
    fig.tight_layout()

def plot_x_profiles_MCMC_samples(x,samples):
    fig, axs = plt.subplots(2,2,sharex=True, sharey=False, figsize=(8,6))
    keys = ['T', 'p', 'd_ini', 'd_ter']
    for (i,j), ax in np.ndenumerate(axs):
        key = keys[2*j+i]
        for sample in [10,20,30,40,50]:
            ax.plot(x,samples[key][:,sample],color='blue',alpha=0.3)
        remove_junk_xyplot(ax)
        if i==1: ax.set_xlabel('$x$  (kb)')
        ax.set_ylabel(math_titles[key])    
        if key=='T': ax.set_ylim([0,50])
        if key=='p': ax.set_ylim([-1,1])
        if key=='d_ini': ax.set_ylim([-0.002,0.2])
        if key=='d_ter': ax.set_ylim([-0.001,0.1])
        ax.locator_params(nbins=3)
    fig.tight_layout()

def plot_compare_x_profiles_credible_interval(x,KJMA,samples_stats):
    fig, axs = plt.subplots(2,2,sharex=True, sharey=False, figsize=(8,6))
    keys = ['T', 'p', 'd_ini', 'd_ter']
    for (i,j), ax in np.ndenumerate(axs):
        key = keys[2*j+i]
        y1 = samples_stats['5'][key]
        y2 = samples_stats['95'][key]
        ax.fill_between(x,y1,y2,color='blue',alpha=0.3)
        ax.plot(x,KJMA[key],color='k', alpha=0.5, linewidth=2)
        remove_junk_xyplot(ax)
        ax.set_ylabel(math_titles[key])
        if i==1: ax.set_xlabel('x (kb)')
        if key=='T': ax.set_ylim([0,50])
        if key=='p': ax.set_ylim([-1,1])
        if key=='d_ini': ax.set_ylim([-0.002,0.2])
        if key=='d_ter': ax.set_ylim([-0.001,0.1])
        ax.locator_params(nbins=3)
    fig.tight_layout()

def compare_s_true_data_s_MAP(x, true_s, data, KJMA, KJMA_MAP):
    fig, axs = plt.subplots(1,3, sharex=True, sharey=True, figsize=(15,5))
    
    ax = axs[0]
    ax.plot(x, true_s)
    remove_junk_xyplot(ax)
    ax.set_xlabel('$x$  (kb)')
    ax.set_ylabel(r'$s(x,t)$')
    ax.set_title('true s at 5 min resolution')
    
    ax = axs[1]
    ax.plot(x, data)
    remove_junk_xyplot(ax)
    ax.set_xlabel('$x$  (kb)')
    ax.set_title('noisy data at 5 min resolution')
    
    ax = axs[2]
    ax.plot(x, KJMA['s'],'gray', alpha=0.5)
    ax.plot(x, KJMA_MAP['s'], 'r', alpha=0.5)
    remove_junk_xyplot(ax)
    ax.set_xlabel('$x$  (kb)')
    ax.set_title('s recovered at 0.5 min resolution')
    ax.locator_params(nbins=4)
    ax.set_ylim(-0.1,1.1)
    fig.tight_layout()

def plot_replication_program():
    from sampling_KJMA_kinetics import get_oriter_from_oris
    from sampling_KJMA_kinetics import get_origins_from_phantoms
    from sampling_KJMA_kinetics import get_timing_from_oris
    lim = {'y':np.arange(100),'u':np.arange(100),'v':1,'dy':1,'Ny':100,'du':1,'Nu':100}
    phantoms = [(10,20), (20,25), (60,30)]
    phantoms = np.array(phantoms, dtype=[('y',float),('u',float)]) 
    oris = get_origins_from_phantoms(phantoms, lim)
    oriter = get_oriter_from_oris(oris, lim)
    timing = get_timing_from_oris(oris, lim)
    fig, ax = plt.subplots(figsize=(5,5))
    remove_junk_xyplot(ax)
    ax.set_ylim([0, 100])
    ax.set_xlim([0, 100])
    ax.set_yticks([0,100])
    ax.set_xticks([0,100])
    ax.plot(oriter['y'],oriter['u'],color='k', linewidth=2)
    ax.set_xlabel('position')
    ax.set_ylabel('time')
    delta_y = 1
    ax.text(x=10,y=20-delta_y,s=r'$O_1$', color='gray',
            verticalalignment='top', horizontalalignment='center')
    ax.text(x=20,y=25-delta_y,s=r'$O_2$', color='gray',
            verticalalignment='top', horizontalalignment='center')
    ax.text(x=60,y=30-delta_y,s=r'$O_3$', color='gray',
            verticalalignment='top', horizontalalignment='center')
    ax.text(x=17.5,y=27.5+delta_y,s=r'$T_1$', color='gray',
            verticalalignment='bottom', horizontalalignment='center')
    ax.text(x=42.5,y=47.5+delta_y,s=r'$T_2$', color='gray',
        verticalalignment='bottom', horizontalalignment='center')
  

def animation_replication_program():
    from sampling_KJMA_kinetics import get_oriter_from_oris
    from sampling_KJMA_kinetics import get_origins_from_phantoms
    from sampling_KJMA_kinetics import get_timing_from_oris
    lim = {'y':np.arange(100),'u':np.arange(100),'v':1,'dy':1,'Ny':100,'du':1,'Nu':100}
    phantoms = [(10,20), (20,25), (60,30)]
    phantoms = np.array(phantoms, dtype=[('y',float),('u',float)]) 
    oris = get_origins_from_phantoms(phantoms, lim)
    oriter = get_oriter_from_oris(oris, lim)
    timing = get_timing_from_oris(oris, lim)
    
    dark_color = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
                  (0.8509803921568627, 0.37254901960784315, 0.00784313725490196)]
    
    from matplotlib import animation
    tmin, tmax, x = 0, 100, np.arange(100)
    t = np.linspace(tmin,tmax,200) # time coordinates
    replicated = np.zeros((len(t), len(x)), dtype=int)
    for i, t_i in enumerate(t):
        replicated[i,:] = (timing['u'] < t_i)
    
    fig, ax = plt.subplots(figsize=(5,5))
    ax.spines['top'].set_visible(False) # no top or right spine
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False) # no top or right spine
    ax.spines['bottom'].set_visible(False)
    ax.set_ylim([tmin, tmax])
    ax.set_xlim([0, 100])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('position')
    ax.set_ylabel('time')
    ax.plot(oriter['y'],oriter['u'],color='k', linewidth=2)
    timeline, = ax.plot([],[],color='gray',alpha=0.5, linewidth=2)
    DNA, = ax.plot([],[],color=dark_color[0])
    ax.hlines(y=tmin+2,xmin=x.min(),xmax=x.max(),color=dark_color[1])
    
    def init():
        timeline.set_data([0,99], [0,0])
        DNA.set_data([], [])
        
    def update(n): 
        # n = frame counter
        timeline.set_data([0,100],[t[n],t[n]])
        DNA.set_data(x, tmin+2+2*replicated[n,:])
    
    anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(t), interval=40, blit=True)
    return anim

def plot_posterior_v_ell_0(grid_search):
    fig, ax = plt.subplots(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    xpos, ypos = np.meshgrid(grid_search['v'],grid_search['ell_0'])
    
    xpos = xpos.flatten()-0.05
    ypos = ypos.flatten()-2.5
    zpos = np.zeros(len(grid_search['v']) * len(grid_search['ell_0']))
    dx = 0.1 * np.ones_like(zpos)
    dy = 5 * np.ones_like(zpos)
    dz = grid_search['P'].flatten()
    
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
    ax.set_xticks(grid_search['v'])
    ax.set_yticks(grid_search['ell_0'])
    ax.set_zticks([0,0.5,1])
    ax.set_xlabel(r'$v$')
    ax.set_ylabel(r'$\ell_0$')
    ax.set_zlabel(r'$P(v,\ell_0)$')


def plot_initiation_rate_with_sample(I_xt, extent, x, t, v, N_cell_cycle=3):
    from sampling_KJMA_kinetics import get_one_cell_cycle_realisation
    #plot the initation rate
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(I_xt.T, extent=extent)
    add_colorbar(ax, im)
    remove_junk_imshow(ax)
    ax.set_xlabel('x (kb)')
    ax.set_ylabel('t (min)')
    ax.text(0.9, 0.9, r'$I(x,t)$',
            verticalalignment='top', horizontalalignment='right',
            transform=ax.transAxes,
            color='white')
    # sample and plot one-cell-cyle realisations of the replication program
    lim = {'y':x,'u':t,'v':v}
    for key in ['y','u']: lim['d'+key]=lim[key][1]-lim[key][0]
    for cell_cycle in range(N_cell_cycle):
        oriter = get_one_cell_cycle_realisation(I_xt,lim)
        ax.plot(oriter['y'],oriter['u'],linewidth=3,alpha=0.7)
    ax.set_ylim(0,50)
    ax.set_xlim(0,100)

def plot_observed_efficiency(x, KJMA):
    fig, ax = plt.subplots(figsize=(5,5))
    E = 100*KJMA['d_ini']*(x[1]-x[0])
    ax.bar(x-2.5, E, width=5, edgecolor='')
    ax.set_xlabel('$x$  (kb)')
    ax.set_ylabel('observed efficiency (%)')
    ax.set_ylim(0,100)
    ax.set_xlim(0,100)
    ax.locator_params(nbins=5)

def plot_compare_x_profiles_wellpos_oris(x,KJMA,KJMA_MAP):
    fig, axs = plt.subplots(2,2,sharex=True, sharey=False, figsize=(8,6))
    keys = ['T', 'p', 'd_ini', 'd_ter']
    for (i,j), ax in np.ndenumerate(axs):
        key = keys[2*j+i]
        ax.plot(x,KJMA[key]) 
        ax.plot(x,KJMA_MAP[key],'--')
        if key=='d_ini': ax.plot(x,KJMA[key])
        if key=='d_ini':ax.plot(x,KJMA_MAP[key],'--')
        remove_junk_xyplot(ax)
        ax.set_ylabel(math_titles[key])
        if i==1: ax.set_xlabel('x (kb)')
        if key=='T': ax.set_ylim([0,50])
        if key=='p': ax.set_ylim([-1,1])
        if key=='d_ini': ax.set_ylim([-0.002,1])
        if key=='d_ter': ax.set_ylim([-0.001,0.05])
        ax.locator_params(nbins=3)
    fig.tight_layout()

def plot_x_profiles_MCMC_samples_wellpos_oris(x,samples):
    fig, axs = plt.subplots(2,2,sharex=True, sharey=False, figsize=(8,6))
    keys = ['T', 'p', 'd_ini', 'd_ter']
    for (i,j), ax in np.ndenumerate(axs):
        key = keys[2*j+i]
        for sample in [10,20,30,40,50]:
            ax.plot(x,samples[key][:,sample],color='blue',alpha=0.3)
        remove_junk_xyplot(ax)
        if i==1: ax.set_xlabel('$x$  (kb)')
        ax.set_ylabel(math_titles[key])    
        if key=='T': ax.set_ylim([0,50])
        if key=='p': ax.set_ylim([-1,1])
        if key=='d_ini': ax.set_ylim([-0.002,1])
        if key=='d_ter': ax.set_ylim([-0.001,0.05])
        ax.locator_params(nbins=3)
    fig.tight_layout()

def plot_compare_x_profiles_credible_interval_wellpos_oris(x,KJMA,samples_stats):
    fig, axs = plt.subplots(2,2,sharex=True, sharey=False, figsize=(8,6))
    keys = ['T', 'p', 'd_ini', 'd_ter']
    for (i,j), ax in np.ndenumerate(axs):
        key = keys[2*j+i]
        y1 = samples_stats['5'][key]
        y2 = samples_stats['95'][key]
        ax.fill_between(x,y1,y2,color='blue',alpha=0.3)
        ax.plot(x,KJMA[key],color='k', alpha=0.5, linewidth=2)
        remove_junk_xyplot(ax)
        ax.set_ylabel(math_titles[key])
        if i==1: ax.set_xlabel('x (kb)')
        if key=='T': ax.set_ylim([0,50])
        if key=='p': ax.set_ylim([-1,1])
        if key=='d_ini': ax.set_ylim([-0.002,1])
        if key=='d_ter': ax.set_ylim([-0.001,0.05])
        ax.locator_params(nbins=3)
    fig.tight_layout()

def plot_compare_I_t_profiles(t,I_it,I_it_MAP):
    fig, axs = plt.subplots(1,2,sharex=True, sharey=True, figsize=(10,5))
    for (i,), ax in np.ndenumerate(axs):
        ax.plot(t,I_it[i,:], label='true')
        ax.plot(t,I_it_MAP[i,:],'--',label='MAP')
        remove_junk_xyplot(ax)
        ax.set_ylabel('I(x,t)')
        ax.set_xlabel('x (kb)')
        ax.locator_params(nbins=3)
        if i==1: ax.legend(fontsize='small')
    fig.tight_layout()
