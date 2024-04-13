import jax
import jax.numpy as jnp
import numpy as np
import jax_cfd.base.grids as grids
from jax_cfd.spectral import utils as spectral_utils
import copy
import math


class fourierGrid():
    """ Need an object to store the Fourier grid for a given gridsize, for calculating
        the isotropically averaged 1d spectra """
    def __init__(self,nx,dk=1,dl=1):
        self.nx=nx
        self.dk=dk
        self.dl=dl

        self.nk = int(self.nx/2 + 1)
        
        self.ll = self.dl*jnp.concatenate((jnp.arange(0.,nx/2),
                    jnp.arange(-self.nx/2,0.)))
        self.kk = self.dk*jnp.arange(0.,self.nk)
        
        ## Get k1d
        self.kmax = jnp.minimum(jnp.max(abs(self.ll)), jnp.max(abs(self.kk)))
        self.dkr = jnp.sqrt(self.dk**2 + self.dl**2)
        self.k1d=jnp.arange(0, self.kmax, self.dkr)
        self.k1d_plot=self.k1d+self.dkr/2
        
        ## Get kappas
        self.k, self.l = jnp.meshgrid(self.kk, self.ll)
        self.kappa2=(self.l**2+self.k**2)
        self.kappa=jnp.sqrt(self.kappa2)

    def get_ispec(self,field):
        ## Array to output isotropically averaged wavenumbers
        phr = np.zeros((self.k1d.size))
    
        ispec=copy.copy(np.array(field))
    
        ## Account for complex conjugate
        ispec[:,0] /= 2
        ispec[:,-1] /= 2
    
        ## Loop over wavenumbers. Average all modes within a given |k| range
        for i in range(self.k1d.size):
            if i == self.k1d.size-1:
                fkr = (self.kappa>=self.k1d[i]) & (self.kappa<=self.k1d[i]+self.dkr)
            else:
                fkr = (self.kappa>=self.k1d[i]) & (self.kappa<self.k1d[i+1])
            phr[i] = ispec[fkr].mean(axis=-1) * (self.k1d[i]+self.dkr/2) * math.pi / (self.dk * self.dl)
    
            phr[i] *= 2 # include full circle
            
        return phr

def get_ke(omega,fourier_grid):
    """ For a voriticity field and fourier grid, calculate isotropically averaged
        KE spectra.
        omega:        2D tensor of vorticity in real space
        fourier_grid: Fourier grid object corresponding to the input vorticity field
    returns:
        k1d_plot: 1d wavenumber bins (centered)
        kespec:   KE spectrum in each wavenumber bin
    """
    omegah=np.fft.rfftn(omega)
    grid = grids.Grid((omega.shape[0], omega.shape[1]), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    velocity_solve = spectral_utils.vorticity_to_velocity(grid)
    vxhat, vyhat = velocity_solve(omegah)
    KEh=abs(vxhat**2)+abs(vyhat**2)
    kespec=fourier_grid.get_ispec(KEh)
    return fourier_grid.k1d_plot,kespec
