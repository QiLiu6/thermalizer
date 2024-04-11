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