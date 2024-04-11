import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import pearsonr
import thermalizer.ks.solver as ks_solver
from tqdm import tqdm


def plot_training_sim(config):
    ## First test a training sim
    ks=ks_solver.KS(L=config["ks"]["L"],N=config["ks"]["N"],nsteps=config["ks"]["nsteps"],dt=config["ks"]["dt"])
    ks.simulate()
    ks_evol=np.fft.ifft(ks.vv,axis=-1).real
    fig=plt.figure(figsize=(12,2))
    plt.imshow(ks_evol.T[:,::config["ks"]["ratio"]],cmap="inferno")
    plt.title("Training sim")
    plt.axvline(config["ks"]["spinup"])
    for aa in range(1,config["ks"]["trajectories"]+1):
        plt.axvline(config["ks"]["decorr_steps"]*aa)
    plt.colorbar()
    return fig

def prepare_test_set(config):
    fig1=plt.figure()
    kespec=np.zeros(config["ks"]["N"]//2+1)
    test_data=torch.empty(config["ks"]["test_sims"],
                        config["ks"]["test_window"],
                        config["ks"]["N"])

    for aa in tqdm(range(config["ks"]["test_sims"])):
        ks=ks_solver.KS(L=config["ks"]["L"],N=config["ks"]["N"],nsteps=config["ks"]["ratio"]*(config["ks"]["spinup"]+config["ks"]["test_window"]),dt=config["ks"]["dt"])
        ks.simulate()
        ks_evol=np.fft.ifft(ks.vv,axis=-1).real

        ## Get kespec of last snapshot to average
        kespec+=abs(np.fft.rfftn(ks_evol[-1])**2)

        ## Drop numerical timesteps, keep only physical timestep intervals
        ks_evol=ks_evol[::config["ks"]["ratio"]]
        
        corrs=[]
        ## Plot correlation between timesteps
        for bb in range(config["ks"]["spinup"],len(ks_evol)-1):
            corrs.append(pearsonr(ks_evol.T[:,config["ks"]["spinup"]].flatten(),ks_evol.T[:,bb].flatten())[0])
        plt.title("Autocorrelation after spinup - full decorrelation window")
        plt.plot(corrs,color="black",alpha=0.3)
        plt.xlabel("Timestep # after spinup phase ended")
        test_data[aa]=torch.tensor(ks_evol[config["ks"]["spinup"]:-1])
        
        
    plt.axvspan(0,config["ks"]["increment"]*config["ks"]["rollout"],color="pink",alpha=0.3)
    plt.xlim(0,config["ks"]["decorr_steps"])


    ## Plot a single validation sim
    fig2=plt.figure(figsize=(12,1))
    plt.title("Full test window")
    plt.imshow(test_data[0].T,cmap="inferno")
    for aa in range(1,config["ks"]["rollout"]+1):
        plt.axvline(config["ks"]["increment"]*aa,color="black",linestyle="dashed",alpha=0.6)
    plt.colorbar()

    fig3=plt.figure()
    plt.title("KE spectra of test samples")
    plt.semilogy(kespec/config["ks"]["test_sims"])

    return test_data, fig1, fig2, fig3

def build_train_dataset(config):
    train_data=torch.empty(config["ks"]["num_sims"]*config["ks"]["trajectories"],
                        config["ks"]["rollout"],
                        config["ks"]["N"])

    for aa in tqdm(range(config["ks"]["num_sims"])):
        ks=ks_solver.KS(L=config["ks"]["L"],N=config["ks"]["N"],nsteps=config["ks"]["nsteps"],dt=config["ks"]["dt"])
        ks.simulate()
        ks_evol=np.fft.ifft(ks.vv,axis=-1).real
        ## Drop numerical timesteps, keep only physical timestep intervals
        ks_evol=ks_evol[::config["ks"]["ratio"]]
        for bb in range(config["ks"]["trajectories"]):
            for cc in range(config["ks"]["rollout"]):
                train_data[config["ks"]["trajectories"]*aa+bb,cc]=torch.tensor(ks_evol[config["ks"]["spinup"]+config["ks"]["decorr_steps"]*bb+cc*config["ks"]["increment"]])
    return train_data


class KSPerformance():
    def __init__(self,config,model,test_data):
        self.config=config
        self.model=model
        self.test_data=test_data

        # use GPUs if available
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model=self.model.eval()
        self.model=self.model.to(self.device)
        self.emu_cached=False
        self.sim_cached=False

    def _cache_emulator_MSE(self):
        self.losses=[]
        criterion=nn.MSELoss()
        test_state=self.test_data[:,0,:].unsqueeze(1).to(self.device)
        pred_states=test_state
        for aa in range(1,test_data.shape[1]):
            test_state=model(test_state)+test_state
            pred_states=torch.cat((pred_states,test_state),axis=1)
            loss=criterion(test_state,self.test_data[:,0,:].unsqueeze(1).to(self.device))
            self.losses.append(loss.detach().cpu().numpy())
        self.pred_states=pred_states

    def _cache_sim_MSE(self):
        ## Get divergence losses for real simulations
        ks=ks_solver.KS(L=config["ks"]["L"],N=config["ks"]["N"],nsteps=config["ks"]["ratio"]*config["ks"]["test_window"],dt=config["ks"]["dt"])
        ks.IC(u0=test_data[0][0]+np.random.normal(0,0.00001,128))
        ks.simulate()
        ks_evol=np.fft.ifft(ks.vv,axis=-1).real
        ## Drop numerical timesteps, keep only physical timestep intervals
        ks_evol=ks_evol[::config["ks"]["ratio"]]
        
        for aa in range(1):
            ks=ks_solver.KS(L=config["ks"]["L"],N=config["ks"]["N"],nsteps=config["ks"]["ratio"]*config["ks"]["test_window"],dt=config["ks"]["dt"])
            ks.IC(u0=test_data[0][0]+np.random.normal(0,0.00001,128))
            ks.simulate()
            ks_evol2=np.fft.ifft(ks.vv,axis=-1).real
            ## Drop numerical timesteps, keep only physical timestep intervals
            ks_evol2=ks_evol2[::config["ks"]["ratio"]]
        
        self.div_loss=[]
        sim1=torch.tensor(ks_evol)
        sim2=torch.tensor(ks_evol2)
        for aa in range(1,len(ks_evol2)):
            loss=criterion(sim1[aa],sim2[aa])
            self.div_loss.append(loss.numpy())
        return

    def plot_emu_mse(self):
        if self.emu_cached==False:
            self._cache_emulator_MSE()
        fig=plt.figure()
        plt.semilogy(losses,label="Mse between truth and emulator predictions")
        plt.xlabel("timestep (#)")
        plt.ylabel("MSE")
        plt.legend()
        return fig
    
    def plot_emu_mse_withsim(self):
        if self.emu_cached==False:
            self._cache_emulator_MSE()
        if self.sim_cached==False:
            self._cache_sim_MSE()
        fig=plt.figure()
        plt.semilogy(losses,label="Mse between truth and emulator predictions")
        plt.semilogy(div_loss,label="Mse between sims with same IC + tiny noise")
        plt.xlabel("timestep (#)")
        plt.ylabel("MSE")
        plt.legend()
        return fig

    def plot_emu_fields(self,index=1,cutoff=-1):
        cutoff=900
        test_sim_num=2
        fig=plt.figure(figsize=(12,6))
        plt.subplot(3,1,1)
        plt.title("True")
        plt.imshow(self.test_data[index][:cutoff].T.detach().cpu().numpy(),cmap="inferno")
        plt.colorbar()
        
        
        plt.subplot(3,1,2)
        plt.title("Emulator")
        plt.imshow(self.pred_states[index][:cutoff].T.detach().cpu().numpy(),cmap="inferno")
        plt.colorbar()
        
        
        plt.subplot(3,1,3)
        plt.title("Residual")
        plt.imshow(self.pred_states[index][:cutoff].T.detach().cpu().numpy()-self.test_data[index][:cutoff].T.detach().cpu().numpy(),cmap="inferno")
        plt.colorbar()
        
        return fig
        