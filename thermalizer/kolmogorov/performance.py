import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.animation as animation
from IPython.display import HTML

import thermalizer.kolmogorov.util as util


class KolmogorovAnimation():
    def __init__(self,ds,model,fps=10,nSteps=1000,normalise=True,cache_residuals=False,savestring=None):
        """ Run an emulator alongside a simulated trajectory, for a RESIDUAL emulator.
            Animate the evolution of the simulated
            vs emulated vorticity fields. Have an option to cache the residuals, such that we can estimate the
            covariance of the noise in fields 
            ds contains the true simulation data. Assume that it is an unnormalised torch tensor.
            """
        self.ds=ds/model.config["field_std"]
        self.model=model
        self.fps=fps
        self.nSteps=nSteps
        self.nFrames=int(self.nSteps)
        self.pred=self.ds[0]
        self.normalise=normalise
        self.cache_residuals=cache_residuals
        
        self.mse=[]
        self.correlation=[]
        self.autocorrelation=[]
        self.criterion=nn.MSELoss()
        self.times=np.arange(0,self.nFrames+0.001,1)

        self.residuals=None
        self.savestring=savestring
        self.grid=grid=util.fourierGrid(64)
        self.k1d_plot,self.ketrue=util.get_ke(self.ds[0],self.grid)
        self.k1d_plot,self.kepred=util.get_ke(self.pred,self.grid)
        
        if self.cache_residuals:
            self.residuals=np.empty(ds.shape)
        
    def _push_forward(self):
        """ Update predicted q by one emulator pass """
        
        ## Convert q to standardised q
        x=torch.tensor(self.pred).float().unsqueeze(0).unsqueeze(0)

        x=self.model(x)
        
        if self.normalise==True:
            x=x-torch.mean(x)
                    
        self.pred=(self.pred+x.squeeze()).detach()
        
        self.correlation.append(pearsonr(self.pred.flatten(),self.ds[self.i].numpy().flatten())[0])
        self.autocorrelation.append(pearsonr(self.ds[0].numpy().flatten(),self.ds[self.i].numpy().flatten())[0])
        self.mse.append(self.criterion(torch.tensor(self.pred),torch.tensor(self.ds[self.i].numpy())))

        if self.cache_residuals:
            self.residuals[self.i]=self.s[self.i]-self.q_i_pred

        _,self.ketrue=util.get_ke(self.ds[self.i],self.grid)
        _,self.kepred=util.get_ke(self.pred,self.grid)
        
        return
    
    def animate(self):
        fig, axs = plt.subplots(1, 5,figsize=(17,3))
        self.ax1=axs[0].imshow(self.ds[0], cmap=sns.cm.icefire,interpolation='none')
        fig.colorbar(self.ax1, ax=axs[0])
        axs[0].set_xticks([]); axs[0].set_yticks([])
        axs[0].set_title("Simulation")

        self.ax2=axs[1].imshow(self.ds[0], cmap=sns.cm.icefire,interpolation='none')
        fig.colorbar(self.ax2, ax=axs[1])
        axs[1].set_xticks([]); axs[1].set_yticks([])
        axs[1].set_title("Emulator")

        self.ax3=axs[2].imshow(self.ds[0], cmap=sns.cm.icefire,interpolation='none')
        fig.colorbar(self.ax3, ax=axs[2])
        axs[2].set_xticks([]); axs[2].set_yticks([])
        axs[2].set_title("Residuals")

        fig.tight_layout()
        
        ## Time evol metrics
        axs[3].set_title("Correlation")
        self.ax4=[axs[3].plot(-1),axs[3].plot(-1)]
        axs[3].set_ylim(0,1)
        axs[3].set_xlim(0,self.times[-1])

        axs[4].set_title("KE spectra")
        axs[4].set_ylim(1e1,1e7)
        axs[4].set_xlim(6e-1,3.5e1)
        self.ax5=[axs[4].loglog(-1),axs[4].loglog(-1)]
        self.ax5[0][0].set_xdata(self.k1d_plot)
        self.ax5[0][0].set_ydata(self.ketrue)
        
        self.ax5[1][0].set_xdata(self.k1d_plot)
        self.ax5[1][0].set_ydata(self.kepred)
        
        self.time_text=axs[2].text(-20,-20,"")
        
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
        self.time_text.set_text("%d timesteps" % (i))
    
        ## Set image and colorbar for each panel
        image=self.ds[self.i].numpy()
        self.ax1.set_array(image)
        self.ax1.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        
        image=self.pred.numpy()
        self.ax2.set_array(image)
        self.ax2.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        
        image=(self.pred-self.ds[self.i]).numpy()
        self.ax3.set_array(image)
        self.ax3.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
 
        self.ax4[0][0].set_xdata(np.array(self.times[0:len(self.correlation)]))
        self.ax4[0][0].set_ydata(np.array(self.correlation))
        
        self.ax4[1][0].set_xdata(np.array(self.times[0:len(self.autocorrelation)]))
        self.ax4[1][0].set_ydata(np.array(self.autocorrelation))

        self.ax5[0][0].set_xdata(self.k1d_plot)
        self.ax5[0][0].set_ydata(self.ketrue)
        
        self.ax5[1][0].set_xdata(self.k1d_plot)
        self.ax5[1][0].set_ydata(self.kepred)
        
        self._push_forward()
        
        return 
        