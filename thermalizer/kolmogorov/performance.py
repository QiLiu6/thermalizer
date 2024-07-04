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


class EmulatorRollout():
    """ Run a batch of emulator rollouts along test trajectories """
    def __init__(self,test_suite,model_emu):
        """ test_suite: torch tensor of test data with shape [batch, snapshot, Nx, Ny], where
                        snapshots are taken 10 numerical timesteps apart
            model_emu:  a trained pytorch CNN emulator of the residuals, over 10 numerical timesteps apart
        """
        self.test_suite=test_suite
        self.test_suite/=model_emu.config["field_std"]
        self.model_emu=model_emu

        ## Set up field tensors
        self.emu=torch.zeros(self.test_suite.shape,dtype=torch.float32)
        ## Set t=0 to be the same
        self.emu[:,0,:,:]=self.test_suite[:,0,:,:]

        ## Ensure models are in eval
        self.model_emu.eval()
        
        if torch.cuda.is_available():
            #self.device=torch.device('cuda')
            self.device="cuda"
            ## Put models on GPU
            self.model_emu.to(self.device)
            ## Put tensors on GPU
            #self.test_suite=self.test_suite.to(self.device)
            #self.emu=self.emu.to(self.device)
        else:
            self.device="cpu"

        self._init_metrics()

    def _init_metrics(self):
        self.mseloss=torch.nn.MSELoss(reduction="none")
        ## Set up metric tensors
        self.mse_auto=torch.zeros(self.test_suite.shape[0],self.test_suite.shape[1])
        self.mse_emu=torch.zeros(self.test_suite.shape[0],self.test_suite.shape[1])

        self.autocorr=[]
        self.corr_emu=[]

        self.grid=util.fourierGrid(64)
        self.ke_true=torch.zeros(self.test_suite.shape[0],self.test_suite.shape[1],len(self.grid.k1d_plot))
        self.ke_emu=torch.zeros(self.ke_true.shape)

        return

    @torch.no_grad()
    def _evolve(self):
        for aa in tqdm(range(1,len(self.test_suite[1]))):
            ## Step fields forward
            emu_unsq=self.emu[:,aa-1,:,:].unsqueeze(1).to(self.device)
            preds=self.model_emu(emu_unsq)
            means=torch.mean(preds,axis=(-1,-2))
            self.emu[:,aa,:,:]=(preds-means.unsqueeze(1).unsqueeze(1)+emu_unsq).squeeze().cpu()

            ## MSE metrics
            loss=self.mseloss(self.test_suite[:,0],self.test_suite[:,aa])
            self.mse_auto[:,aa]=torch.mean(loss,dim=(1,2))
            loss=self.mseloss(self.test_suite[:,aa],self.emu[:,aa])
            self.mse_emu[:,aa]=torch.mean(loss,dim=(1,2))

    def _KE_spectra(self):
        ## Move to cpu for KE spectra calculation
        self.test_suite=self.test_suite.to("cpu")
        self.emu=self.emu.to("cpu")
        self.therm=self.therm.to("cpu")
        for aa in tqdm(range(1,len(self.test_suite[1]))):
            for bb in range(0,len(self.test_suite[0])):
                _,ke=util.get_ke(self.test_suite[aa,bb],self.grid)
                self.ke_true[aa,bb]=torch.tensor(ke)
                _,ke=util.get_ke(self.emu[aa,bb],self.grid)
                self.ke_emu[aa,bb]=torch.tensor(ke)
                _,ke=util.get_ke(self.therm[aa,bb],self.grid)
                self.ke_therm[aa,bb]=torch.tensor(ke)
  
        ## Move to back to gpu
        self.test_suite=self.test_suite.to(self.device)
        self.emu=self.emu.to(self.device)
        self.therm=self.therm.to(self.device)


## NB can prob unifiy the score and DDPM classes with a common parent
## but if we end up ditching DDPM, we can just delete that. Lets see how
## score-matched tests go
class ThermalizeKolmogorovScore():
    def __init__(self,test_suite,model_emu,model_therm,thermalize_delay=100,thermalize_interval=5,thermalize_timesteps=1,thermalize_h=0.01):
        self.test_suite=test_suite/model_emu.config["field_std"]
        self.model_emu=model_emu
        self.model_therm=model_therm
        self.thermalize_delay=thermalize_delay
        self.thermalize_interval=thermalize_interval
        self.thermalize_timesteps=thermalize_timesteps
        self.thermalize_h=thermalize_h

        ## Set up field tensors
        self.emu=torch.zeros(self.test_suite.shape)
        self.therm=torch.zeros(self.test_suite.shape)

        ## Ensure models are in eval
        self.model_emu.eval()
        self.model_therm.eval()
        
        if torch.cuda.is_available():
            self.device=torch.device('cuda')
            ## Put models on GPU
            self.model_emu=self.model_emu.to(self.device)
            self.model_therm=self.model_therm.to(self.device)
            ## Put tensors on GPU
            self.test_suite=self.test_suite.to(self.device)
            self.emu=self.emu.to(self.device)
            self.therm=self.therm.to(self.device)

        ## Set t=0 to be the same
        self.emu[:,0,:,:]=self.test_suite[:,0,:,:]
        self.therm[:,0,:,:]=self.test_suite[:,0,:,:]

        self._init_metrics()

    def _init_metrics(self):
        self.mseloss=torch.nn.MSELoss(reduction="none")
        ## Set up metric tensors
        self.mse_auto=torch.zeros(self.test_suite.shape[0],self.test_suite.shape[1])
        self.mse_emu=torch.zeros(self.test_suite.shape[0],self.test_suite.shape[1])
        self.mse_therm=torch.zeros(self.test_suite.shape[0],self.test_suite.shape[1])

        self.autocorr=[]
        self.corr_emu=[]
        self.corr_therm=[]

        self.grid=util.fourierGrid(64)
        self.ke_true=torch.zeros(self.test_suite.shape[0],self.test_suite.shape[1],len(self.grid.k1d_plot))
        self.ke_emu=torch.zeros(self.ke_true.shape)
        self.ke_therm=torch.zeros(self.ke_true.shape)

        return

    @torch.no_grad()
    def _evolve(self):
        for aa in tqdm(range(1,len(self.test_suite[1]))):
            ## Step fields forward
            emu_unsq=self.emu[:,aa-1,:,:].unsqueeze(1)
            self.emu[:,aa,:,:]=(self.model_emu(emu_unsq)+emu_unsq).squeeze()

            therm_unsq=self.therm[:,aa-1,:,:].unsqueeze(1)
            self.therm[:,aa,:,:]=(self.model_emu(therm_unsq)+therm_unsq).squeeze()

            should_thermalize = (aa % self.thermalize_interval == 0) and (aa>self.thermalize_delay)
            if should_thermalize:
                for step in range(self.thermalize_timesteps):
                    z=torch.randn_like(self.therm[:,aa,:,:],device=self.device)
                    thermed=self.model_therm(self.therm[:,aa,:,:].unsqueeze(1))
                    self.therm[:,aa,:,:]+=thermed.squeeze()*0.5*self.thermalize_h+math.sqrt(self.thermalize_h)*z

            ## MSE metrics
            loss=self.mseloss(self.test_suite[:,0],self.test_suite[:,aa])
            self.mse_auto[:,aa]=torch.mean(loss,dim=(1,2))
            loss=self.mseloss(self.test_suite[:,aa],self.emu[:,aa])
            self.mse_emu[:,aa]=torch.mean(loss,dim=(1,2))
            loss=self.mseloss(self.test_suite[:,aa],self.therm[:,aa])
            self.mse_therm[:,aa]=torch.mean(loss,dim=(1,2))

    def _KE_spectra(self):
        ## Move to cpu for KE spectra calculation
        self.test_suite=self.test_suite.to("cpu")
        self.emu=self.emu.to("cpu")
        self.therm=self.therm.to("cpu")
        for aa in tqdm(range(1,len(self.test_suite[1]))):
            for bb in range(0,len(self.test_suite[0])):
                _,ke=util.get_ke(self.test_suite[aa,bb],self.grid)
                self.ke_true[aa,bb]=torch.tensor(ke)
                _,ke=util.get_ke(self.emu[aa,bb],self.grid)
                self.ke_emu[aa,bb]=torch.tensor(ke)
                _,ke=util.get_ke(self.therm[aa,bb],self.grid)
                self.ke_therm[aa,bb]=torch.tensor(ke)
  
        ## Move to back to gpu
        self.test_suite=self.test_suite.to(self.device)
        self.emu=self.emu.to(self.device)
        self.therm=self.therm.to(self.device)


class ThermalizeKolmogorovDDPM():
    def __init__(self,test_suite,model_emu,model_therm,thermalize_delay=100,thermalize_interval=5,thermalize_timesteps=2):
        self.test_suite=test_suite/model_emu.config["field_std"]
        self.model_emu=model_emu
        self.model_therm=model_therm
        self.thermalize_delay=thermalize_delay
        self.thermalize_interval=thermalize_interval
        self.thermalize_timesteps=thermalize_timesteps

        ## Set up field tensors
        self.emu=torch.zeros(self.test_suite.shape)
        self.therm=torch.zeros(self.test_suite.shape)

        ## Ensure models are in eval
        self.model_emu.eval()
        self.model_therm.eval()
        
        if torch.cuda.is_available():
            self.device=torch.device('cuda')
            ## Put models on GPU
            self.model_emu=self.model_emu.to(self.device)
            self.model_therm=self.model_therm.to(self.device)
            ## Put tensors on GPU
            self.test_suite=self.test_suite.to(self.device)
            self.emu=self.emu.to(self.device)
            self.therm=self.therm.to(self.device)

        ## Set t=0 to be the same
        self.emu[:,0,:,:]=self.test_suite[:,0,:,:]
        self.therm[:,0,:,:]=self.test_suite[:,0,:,:]

        self._init_metrics()

    def _init_metrics(self):
        self.mseloss=torch.nn.MSELoss(reduction="none")
        ## Set up metric tensors
        self.mse_auto=torch.zeros(self.test_suite.shape[0],self.test_suite.shape[1])
        self.mse_emu=torch.zeros(self.test_suite.shape[0],self.test_suite.shape[1])
        self.mse_therm=torch.zeros(self.test_suite.shape[0],self.test_suite.shape[1])

        self.autocorr=[]
        self.corr_emu=[]
        self.corr_therm=[]

        self.grid=util.fourierGrid(64)
        self.ke_true=torch.zeros(self.test_suite.shape[0],self.test_suite.shape[1],len(self.grid.k1d_plot))
        self.ke_emu=torch.zeros(self.ke_true.shape)
        self.ke_therm=torch.zeros(self.ke_true.shape)

        return

    @torch.no_grad()
    def _evolve(self):
        for aa in tqdm(range(1,len(self.test_suite[1]))):
            ## Step fields forward
            emu_unsq=self.emu[:,aa-1,:,:].unsqueeze(1)
            preds=self.model_emu(emu_unsq)
            means=torch.mean(preds,axis=(-1,-2))
            self.emu[:,aa,:,:]=(preds-means.unsqueeze(1).unsqueeze(1)+emu_unsq).squeeze()

            therm_unsq=self.therm[:,aa-1,:,:].unsqueeze(1)
            preds=self.model_emu(therm_unsq)
            means=torch.mean(preds,axis=(-1,-2))
            self.therm[:,aa,:,:]=(preds-means.unsqueeze(1).unsqueeze(1)+therm_unsq).squeeze()

            should_thermalize = (aa % self.thermalize_interval == 0) and (aa>self.thermalize_delay)
            if should_thermalize:
                thermed=self.model_therm.denoising(self.therm[:,aa,:,:].unsqueeze(1),self.thermalize_timesteps)
                self.therm[:,aa,:,:]=thermed.squeeze()

            ## MSE metrics
            loss=self.mseloss(self.test_suite[:,0],self.test_suite[:,aa])
            self.mse_auto[:,aa]=torch.mean(loss,dim=(1,2))
            loss=self.mseloss(self.test_suite[:,aa],self.emu[:,aa])
            self.mse_emu[:,aa]=torch.mean(loss,dim=(1,2))
            loss=self.mseloss(self.test_suite[:,aa],self.therm[:,aa])
            self.mse_therm[:,aa]=torch.mean(loss,dim=(1,2))

    def _KE_spectra(self):
        ## Move to cpu for KE spectra calculation
        self.test_suite=self.test_suite.to("cpu")
        self.emu=self.emu.to("cpu")
        self.therm=self.therm.to("cpu")
        for aa in tqdm(range(1,len(self.test_suite[1]))):
            for bb in range(0,len(self.test_suite[0])):
                _,ke=util.get_ke(self.test_suite[aa,bb],self.grid)
                self.ke_true[aa,bb]=torch.tensor(ke)
                _,ke=util.get_ke(self.emu[aa,bb],self.grid)
                self.ke_emu[aa,bb]=torch.tensor(ke)
                _,ke=util.get_ke(self.therm[aa,bb],self.grid)
                self.ke_therm[aa,bb]=torch.tensor(ke)
  
        ## Move to back to gpu
        self.test_suite=self.test_suite.to(self.device)
        self.emu=self.emu.to(self.device)
        self.therm=self.therm.to(self.device)


class KolmogorovAnimation():
    def __init__(self,ds,model,fps=10,nSteps=1000,normalise=True,cache_residuals=False,savestring=None):
        """ Run an emulator alongside a simulated trajectory, for a RESIDUAL emulator.
            Animate the evolution of the simulated
            vs emulated vorticity fields. Have an option to cache the residuals, such that we can estimate the
            covariance of the noise in fields 
            ds contains the true simulation data. Assume that it is a NORMALISED torch tensor.
            """
        self.ds=ds
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
        

class ScoreAnimation():
    """ Animation to plot the score as we add incrementally increasing Gaussian noise
        to a held-out sample. """
    def __init__(self,img,thermalizer,noise_coeff=1,fps=10,nSteps=1000,savestring=None,skip=1):
        """ img=torch tensor of a single kolmogorov field
            thermalizer=trained thermalizer model
            noise_coeff=coefficient of the most amount of noise we want to add. We will incrementally
            move to this amount of noise in a linear fashion.
            """
        self.img=img ## torch tensor
        self.img_plot=self.img
        
        self.thermalizer=thermalizer
        self.therm_img=img
        self.noise_coeff=noise_coeff
        self.fps=fps
        self.nSteps=nSteps
        self.skip=skip
        self.nFrames=int(self.nSteps)
        self.noise=np.linspace(0,self.noise_coeff,self.nFrames)
        self.savestring=None
        self.noise_add=torch.rand(self.img.shape)
        self.norms=torch.zeros(self.nFrames)

    def _inc_noise(self):
        self.img_plot=self.img+self.noise[self.i]*self.noise_add
        self.therm_img=self.thermalizer(self.img_plot.unsqueeze(0).unsqueeze(0)).detach()
        self.norms[self.i]=torch.linalg.norm(self.therm_img.flatten())
        
    
    def animate(self):
        fig, axs = plt.subplots(1, 2,figsize=(14,6))
        ## Upper layer
        self.ax1=axs[0].imshow(self.img, cmap=sns.cm.icefire,interpolation='none')
        fig.colorbar(self.ax1, ax=axs[0])
        axs[0].set_xticks([]); axs[0].set_yticks([])
        axs[0].set_title("Input image")

        self.ax2=axs[1].imshow(self.therm_img, cmap=sns.cm.icefire,interpolation='none')
        fig.colorbar(self.ax2, ax=axs[1])
        axs[1].set_xticks([]); axs[1].set_yticks([])
        axs[1].set_title("Thermalizer output")

        self.time_text=axs[0].text(-2,-2,"")
        
        
        #fig.tight_layout()
        
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
            
        self.i=self.skip*i
        self._inc_noise()
        self.time_text.set_text("%.2f Noised amount" % (self.noise[self.i]))
    
        ## Set image and colorbar for each panel
        image=self.img_plot.cpu().numpy()
        lim=np.max(np.abs(image))
        self.ax1.set_array(image)
        self.ax1.set_clim(-lim,lim)
        
        image=self.therm_img.squeeze().numpy()
        lim=np.max(np.abs(image))
        self.ax2.set_array(image)
        self.ax2.set_clim(-lim,lim)
        return 

def long_run_figures(model,emu_run,steps=int(1e6)):
    """ For a given emulator model and set of test ICs, run for `steps` iterations
        We will plot enstrophy over time, and plot the fields at the end of the rollout
        Here we are testing long term stability - on timescales longer than we can store
        test data for.
        
        Test set will be just the initial states, i.e. [batch size, nx, ny]
        Also assuming model is already on GPU"""

    assert 0.9<emu_run.std().item()<1.1, "Fields are not normalised"

    ## Just going to assume we have a GPU available, as we are not doing long
    ## rollouts on CPU
    emu_run=emu_run.to("cuda")
    model=model.to("cuda")
    
    enstrophies=torch.zeros((len(emu_run),steps))
    with torch.no_grad():
        for aa in range(steps):
            emu_run=model(emu_run.unsqueeze(1)).squeeze()+emu_run
            enstrophies[:,aa]=abs(emu_run**2).sum(axis=(1,2))

    enstrophy_figure=plt.figure()
    plt.title("Enstrophy from long emulator rollout")
    for aa in range(len(enstrophies)):
        plt.plot(enstrophies[aa],color="gray",alpha=0.4)
    plt.ylim(1000,10000)
    plt.xlabel("Emulator timestep")
    plt.ylabel("Enstrophy")
   
    field_figure=plt.figure(figsize=(14,6))
    plt.suptitle("Model predictions after %d emulator steps" % steps)
    for aa in range(1,9):
        plt.subplot(2,4,aa)
        plt.imshow(emu_run[aa].cpu().squeeze(),cmap=sns.cm.icefire)
        plt.colorbar()
    return enstrophy_figure, field_figure


def plot_thermalizer_norms(test_snap,thermalizer,num_trials=50,noise_steps=200,noise_coeff=1):
    """ For a given test image, add incremental amounts of noise to it. Pass this image
        to the thermalizer, and then plot the L2 norm of the score field that the thermalizer
        returns """
    thermalizer=thermalizer.to("cuda")
    thermalizer.eval()
    num_trials=50
    imgs=torch.tensor(test_snap.unsqueeze(0).repeat(num_trials,1,1),device="cuda")
    noise_levels=np.logspace(-2,2,noise_steps)
    norms=torch.zeros(num_trials,noise_steps)
    noises=torch.randn(size=(num_trials,test_snap.shape[0],test_snap.shape[1]),device="cuda")
    
    with torch.no_grad():
        for aa in range(noise_steps):
            therm_out=thermalizer((imgs+noise_levels[aa]*noises).unsqueeze(1)).squeeze()
            norms_level=torch.einsum("ijk,ijk->i",therm_out,therm_out)
            norms[:,aa]=norms_level
    
    norms2=norms.detach().cpu()
    fig=plt.figure()
    for aa in range(len(norms)):
        plt.loglog(noise_levels,norms[aa],color="gray",alpha=0.4)
    plt.xlabel("noise coeff")
    plt.ylabel("L2 norm of score field")
    plt.title(r"$\parallel s_\theta(\mathbf{x})\parallel^2$ as we apply noise to a true image")
    return fig
