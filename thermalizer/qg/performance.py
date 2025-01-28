import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.animation as animation
from IPython.display import HTML
import thermalizer.kolmogorov.util as util
from tqdm import tqdm


####
### Functions to run emulator and thermalize trajectories
###
def run_emu(ics,emu,therm=None,n_steps=1000,silent=False):
    """ Run an emuluator on some ICs for 2 layer QG
        ics:     initial conditions for emulator
        emu:     torch emulator model
        therm:   diffusion model object - pass this if we want to
                 also classify the noise level during rollout
        n_steps: how many emulator steps to run
        silent:  silence tqdm progress bar (for slurm scripts) """
    ## Set up state and diagnostic tensors
    state_vector=torch.zeros((len(ics),n_steps,2,64,64),device="cuda")
    ## Set ICs
    state_vector[:,0]=ics
    state_vector=state_vector.to("cuda")
    noise_classes=None
    if therm:
        noise_classes=torch.zeros(len(state_vector),len(state_vector[1]))

    with torch.no_grad(): 
        for aa in tqdm(range(1,len(state_vector[1])),disable=silent):
            state_vector[:,aa]=emu(state_vector[:,aa-1])+state_vector[:,aa-1]
            if therm:
                preds=therm.model.noise_class(state_vector[:,aa])
                noise_classes[:,aa]=preds.cpu()
    enstrophies=(abs(state_vector**2).sum(axis=(2,3)))
    return state_vector, enstrophies, noise_classes

def therm_algo_batc(ics,emu,therm,n_steps=-1,start=10,stop=4,forward=True,silent=False):
    """ Thermalization algorithm 1 - what I originally tested, passing whole batch to
        thermalizer model:
        ics:     initial conditions for emulator
        emu:     torch emulator model
        therm:   diffusion model object
        n_steps: how many emulator steps to run
        start:   noise level to start thermalizing
        stop:    noise level to stop thermalizing
        forward: Add forward diffusion noise when thermalizing
        silent:  silence tqdm progress bar (for slurm scripts) """
    ## Set up state and diagnostic tensors
    state_vector=torch.zeros((len(ics),n_steps,2,64,64),device="cuda")
    ## Set ICs
    state_vector[:,0]=ics
    state_vector=state_vector.to("cuda")
    noise_classes=torch.zeros(len(state_vector),len(state_vector[1]))
    therming_counts=torch.zeros_like(noise_classes)

    with torch.no_grad(): 
        for aa in tqdm(range(1,len(state_vector[1])),disable=silent):
            state_vector[:,aa]=emu(state_vector[:,aa-1]).squeeze()+state_vector[:,aa-1]
            preds=therm.model.noise_class(state_vector[:,aa])
            noise_classes[:,aa]=preds.cpu()
            if max(preds)>start:
                thermed,therming_counts[:,aa]=therm.denoise_heterogen(state_vector[:,aa],preds,stop=stop,forward_diff=forward)
                state_vector[:,aa]=thermed.squeeze()
    enstrophies=(abs(state_vector**2).sum(axis=(2,3)))
    return state_vector, enstrophies, noise_classes, therming_counts


def therm_algo(ics,emu,therm,n_steps=-1,start=10,stop=4,forward=True,silent=False,noise_limit=100):
    """ Thermalization algorithm - we only thermalize elements over the
        initialisation threshold:
        ics:     initial conditions for emulator
        emu:     torch emulator model
        therm:   diffusion model object
        n_steps: how many emulator steps to run
        start:   noise level to start thermalizing
        stop:    noise level to stop thermalizing
        forward: Add forward diffusion noise when thermalizing
        silent:  silence tqdm progress bar (for slurm scripts)
        noise_limit: if predicted noise level exceeds this threshold, cut the run

        returns: state_vector, enstrophies, noise_classes, therming_counts """

    ## Set up state and diagnostic tensors
    state_vector=torch.zeros((len(ics),n_steps,2,64,64),device="cuda")
    ## Set ICs
    state_vector[:,0]=ics
    noise_classes=torch.zeros(len(state_vector),len(state_vector[1]))
    therming_counts=torch.zeros_like(noise_classes)
    
    ## Run
    with torch.no_grad(): 
        for aa in tqdm(range(1,len(state_vector[1])),disable=silent):
            ## Emulator step
            state_vector[:,aa]=emu(state_vector[:,aa-1]).squeeze()+state_vector[:,aa-1]
            ## Noise level check
            preds=therm.model.noise_class(state_vector[:,aa])
            noise_classes[:,aa]=preds.cpu() ## Store noise levels
            ## Indices of thermalized fields
            therm_select=preds>(start)
            therming=state_vector[therm_select,aa]
            ## If we have any fields over threshold, run therm
            if len(therming)>0:
                thermed,counts=therm.denoise_heterogen(therming,preds[therm_select],stop=stop,forward_diff=forward)
                for bb,idx in enumerate(torch.argwhere(therm_select).flatten()):
                    state_vector[idx,aa]=thermed[bb].squeeze()
                    therming_counts[idx,aa]=counts[bb]
            if noise_limit:
                if preds.max()>noise_limit:
                    print("breaking due to noise limit")
                    break
    state_vector=state_vector.to("cpu")
    enstrophies=(abs(state_vector**2).sum(axis=(2,3)))
    return state_vector, enstrophies, noise_classes, therming_counts


def therm_algo_free(ics,emu,therm,n_steps=100,start=10,stop=4,forward=True,silent=False,noise_limit=100):
    """ Thermalization algorithm  - correct algo where we only thermalize elements over the
        initialisation threshold. Here we don't store the whole trajectory, so we can run for longer
        than memory requirement normally allows.
        
        Input parameters:
        ics:         initial conditions for emulator
        emu:         torch emulator model
        therm:       diffusion model object
        n_steps:     how many emulator steps to run
        start:       noise level to start thermalizing
        stop:        noise level to stop thermalizing
        forward:     Add forward diffusion noise when thermalizing
        silent:      silence tqdm progress bar (for slurm scripts)
        noise_limit: if predicted noise level exceeds this threshold, cut the run

        returns: state_vector"""

    ## Set ICs
    state_vector=ics
    
    ## Run
    with torch.no_grad(): 
        for aa in tqdm(range(1,n_steps),disable=silent):
            ## Emulator step
            state_vector=emu(state_vector)+state_vector
            ## Noise level check
            preds=therm.model.noise_class(state_vector)
            ## Indices of thermalized fields
            therm_select=preds>(start)
            therming=state_vector[therm_select]
            ## If we have any fields over threshold, run therm
            if len(therming)>0:
                thermed,counts=therm.denoise_heterogen(therming,preds[therm_select],stop=stop,forward_diff=forward)
                for bb,idx in enumerate(torch.argwhere(therm_select).flatten()):
                    state_vector[idx]=thermed[bb].squeeze()
            if noise_limit:
                if preds.max()>noise_limit:
                    print("breaking due to noise limit at point %d" % aa)
                    break
    return state_vector, aa


class QGAnimation():
    def __init__(self,ds,emu,fps=10,nSteps=1000,skip=1,savestring=None):
        """ Animate a simulated and emulated trajectory side by side
            """
        self.ds=ds.numpy()
        self.emu=emu.numpy()
        self.fps=fps
        self.nFrames=int(nSteps)
        self.step_counter=0
        self.times=np.arange(0,self.nFrames+0.001,1)
        self.skip=skip
        self.savestring=savestring
        
    def _push_forward(self):
        """ Here we just update the step counter - we are not pushing the emulator forward
            unlike in original iterations. Also we have no spectral stuff yet """

        #_,self.kesim=util.get_ke(self.ds[self.step_counter],self.grid)
        #_,self.kepred=util.get_ke(self.emu[self.step_counter],self.grid)

        self.step_counter+=1
        
        return
    
    def animate(self):
        fig, axs = plt.subplots(2, 3,figsize=(12,6))
        self.title = plt.suptitle(t='2 layer QG, eddy config. Emulator steps=%d, Numerical timesteps=%d, Physical time=%.1f (s)' % (0,0,0))
        self.ax1=axs[0][0].imshow(self.ds[0][0], cmap=sns.cm.icefire,interpolation='none')
        fig.colorbar(self.ax1, ax=axs[0][0])
        axs[0][0].set_xticks([]); axs[0][0].set_yticks([])
        axs[0][0].set_title("Simulation")

        self.ax2=axs[1][0].imshow(self.ds[0][1], cmap=sns.cm.icefire,interpolation='none')
        fig.colorbar(self.ax2, ax=axs[1][0])
        axs[1][0].set_xticks([]); axs[1][0].set_yticks([])

        self.ax3=axs[0][1].imshow(self.emu[0][0], cmap=sns.cm.icefire,interpolation='none')
        fig.colorbar(self.ax3, ax=axs[0][1])
        axs[0][1].set_xticks([]); axs[0][1].set_yticks([])
        axs[0][1].set_title("Emulator")

        self.ax4=axs[1][1].imshow(self.emu[0][1], cmap=sns.cm.icefire,interpolation='none')
        fig.colorbar(self.ax4, ax=axs[1][1])
        axs[1][1].set_xticks([]); axs[1][1].set_yticks([])

        self.ax5=axs[0][2].imshow(self.ds[0][0]-self.emu[0][0], cmap=sns.cm.icefire,interpolation='none')
        fig.colorbar(self.ax3, ax=axs[0][2])
        axs[0][2].set_xticks([]); axs[0][2].set_yticks([])
        axs[0][2].set_title("Diff")

        self.ax6=axs[1][2].imshow(self.ds[0][1]-self.emu[0][1], cmap=sns.cm.icefire,interpolation='none')
        fig.colorbar(self.ax4, ax=axs[1][2])
        axs[1][2].set_xticks([]); axs[1][2].set_yticks([])
        
        
        fig.tight_layout()
        
        anim = animation.FuncAnimation(
                                       fig, 
                                       self.animate_func, 
                                       frames = self.nFrames,
                                       interval = 1000 / self.fps, # in ms
                                       )
        plt.close()
        
        if self.savestring:
            print("saving")
            # saving to m4 using ffmpeg writer 
            writervideo = animation.FFMpegWriter(fps=self.fps) 
            anim.save('%s.mp4' % self.savestring, writer=writervideo) 
            plt.close()
        else:
            return HTML(anim.to_html5_video())
        
    def animate_func(self,i):
        if i % self.fps == 0:
            print( '.', end ='' )
            
        self.i=i

        self.title.set_text('2 layer QG, eddy config. Emulator steps=%d, Numerical timesteps=%d, Physical time=%.1f (s)' % (self.step_counter,self.step_counter*10,self.step_counter*10*3600))
        ## Set image and colorbar for each panel
        image=self.ds[self.step_counter,0]
        self.ax1.set_array(image)
        self.ax1.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))

        image=self.ds[self.step_counter,1]
        self.ax2.set_array(image)
        self.ax2.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))

        image=self.emu[self.step_counter,0]
        self.ax3.set_array(image)
        self.ax3.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))

        image=self.emu[self.step_counter,1]
        self.ax4.set_array(image)
        self.ax4.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))

        image=self.ds[self.step_counter,0]-self.emu[self.step_counter,0]
        self.ax5.set_array(image)
        self.ax5.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))

        image=self.ds[self.step_counter,1]-self.emu[self.step_counter,1]
        self.ax6.set_array(image)
        self.ax6.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
    
        
        for aa in range(self.skip):
            self._push_forward()
        
        return 

