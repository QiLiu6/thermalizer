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

class QGAnimation():
    def __init__(self,ds,emu,fps=10,nSteps=1000,skip=1,savestring=None):
        """ Animate a simulated and emulated trajectory
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
        """ Update predicted q by one emulator pass """

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

