import numpy as np
import yaml
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

import thermalizer.dataset.datasets as datasets
import thermalizer.models.cnn as cnn
import thermalizer.models.unet_modern as munet
import thermalizer.models.drn as drn
import thermalizer.systems.regression_systems as systems
import thermalizer.kolmogorov.util as util
import thermalizer.kolmogorov.simulate as simulate
import thermalizer.kolmogorov.performance as performance


import os
## Stop jax hoovering up GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

config={}
config["input_channels"]=1
config["output_channels"]=1
config["conv_layers"]=8
config["activation"]="ReLU"
config["batch_norm"]=False
config["file_path"]="/scratch/cp3759/thermalizer_data/kolmogorov/reynolds10k/emu.p"
config["subsample"]=None
config["save_name"]="model_weights.pt"
#config["add_noise"]=1e-4
config["optimization"]={}
config["optimization"]["epochs"]=120
config["optimization"]["lr"]=0.001
config["optimization"]["wd"]=0.05
config["optimization"]["batch_size"]=64
config["optimization"]["scheduler"]=True


train_data,valid_data,config=datasets.parse_data_file(config)

ds_train=datasets.FluidDataset(train_data)
ds_valid=datasets.FluidDataset(valid_data)

train_loader = DataLoader(
    ds_train,
    num_workers=10,
    batch_size=config["optimization"]["batch_size"],
    sampler=RandomSampler(ds_train),
)
valid_loader = DataLoader(
    ds_valid,
    num_workers=10,
    batch_size=config["optimization"]["batch_size"],
    sampler=RandomSampler(ds_valid),
)

wandb.init(project="kolmogorov", entity="m2lines",config=config,dir="/scratch/cp3759/thermalizer_data/wandb_data")
wandb.config["save_path"]=wandb.run.dir
config["save_path"]=wandb.run.dir
config["wandb_url"]=wandb.run.get_url()

model=cnn.FCNN(config)

wandb.config["cnn learnable parameters"]=sum(p.numel() for p in model.parameters())
wandb.watch(model, log_freq=1)


#system=RolloutSystem1D(model,config)
system=systems.RolloutResidualSystem(model,config)

logger = WandbLogger()
lr_monitor=LearningRateMonitor(logging_interval='epoch')

trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=config["optimization"]["epochs"],
    logger=logger,
    enable_progress_bar=False,
    callbacks=[lr_monitor],
    enable_checkpointing=False
    )

trainer.fit(system, train_loader, valid_loader)

model.save_model()

## Load test data
with open("/scratch/cp3759/thermalizer_data/kolmogorov/reynolds10k/test40.p", 'rb') as fp:
    test_suite = pickle.load(fp)

## Make sure train and test increments are the same
assert test_suite["increment"]==config["increment"]


fig_ens,fig_field=performance.long_run_figures(model,test_suite["data"][:,0,:,:].to("cuda")/model.config["field_std"])
wandb.log({"Long Ens": wandb.Image(fig_ens)})
wandb.log({"Long field": wandb.Image(fig_field)})

## Run rollout against test sims, plot MSE
emu_rollout=performance.EmulatorRollout(test_suite["data"],model)
emu_rollout._evolve()
fig_mse=plt.title("MSE wrt. true trajectory, emu step=%.2f" % test_suite["increment"])
plt.loglog(emu_rollout.mse_emu[0],color="blue",alpha=0.1,label="Emulator")
for aa in range(1,len(emu_rollout.mse_auto)):
    #plt.loglog(therm_rollout.mse_auto[aa],color="gray",alpha=0.4)
    plt.loglog(emu_rollout.mse_emu[aa],color="blue",alpha=0.1)
plt.ylim(1e-3,1e5)
plt.legend()
plt.ylabel("MSE")
plt.xlabel("# of steps")
wandb.log({"Rollout MSE": wandb.Image(fig_mse)})
plt.close()

## Plot random samples
samps=6
indices=np.random.randint(0,len(emu_rollout.test_suite),size=samps)
time_snaps=np.random.randint(0,len(emu_rollout.test_suite[0]),size=samps)
fig_samps=plt.figure(figsize=(20,6))
plt.suptitle("Sim (top) and emu (bottom) at random time samples")
for aa in range(samps):
    plt.subplot(2,samps,aa+1)
    plt.title("emu step # %d" % time_snaps[aa])
    plt.imshow(emu_rollout.test_suite[indices[aa],time_snaps[aa]],cmap=sns.cm.icefire,interpolation='none')
    plt.colorbar()

    plt.subplot(2,samps,aa+1+samps)
    plt.imshow(emu_rollout.emu[indices[aa],time_snaps[aa]],cmap=sns.cm.icefire,interpolation='none')
    plt.colorbar()
plt.tight_layout()
wandb.log({"Random samples": wandb.Image(fig_samps)})
plt.close()

## Enstrophy animation, along true trajectory
ens_fig=plt.figure(figsize=(12,5))
plt.suptitle("Enstrophy over time")
for aa in range(len(emu_rollout.test_suite)):
    plt.subplot(1,2,1)
    ens_sim=torch.sum(torch.abs(emu_rollout.test_suite[aa])**2,axis=(-1,-2))
    ens_emu=torch.sum(torch.abs(emu_rollout.emu[aa])**2,axis=(-1,-2))
    plt.plot(ens_sim,color="blue",alpha=0.2)
    plt.plot(ens_emu,color="red",alpha=0.2)
    plt.xlabel("Emulator passes")
    plt.ylim(0,15000)
    
    plt.subplot(1,2,2)
    plt.plot(ens_sim,color="blue",alpha=0.2)
    plt.plot(ens_emu,color="red",alpha=0.2)
    plt.xscale("log")
    plt.xlabel("Emulator passes")
    plt.ylim(0,15000)
wandb.log({"Enstrophy": wandb.Image(ens_fig)})

## Run and save animation
steps=10000
model.to("cpu")
emu_rollout.test_suite[0]=emu_rollout.test_suite[0].to("cpu")
anim=performance.KolmogorovAnimation(emu_rollout.test_suite[0],model,fps=90,nSteps=steps,savestring=config["save_path"]+"/anim")
anim.animate()

## Metrics figure
fig_metrics=plt.figure(figsize=(14,13))
plt.subplot(3,3,1)
plt.title("MSE")
plt.loglog(anim.mse)
plt.xlabel("# passes")

plt.subplot(3,3,2)
plt.title("Correlation")
plt.plot(anim.correlation,label="Corr(sim, emu)")
plt.plot(anim.autocorrelation,label="Sim Corr(t0,t)")
plt.xlabel("# passes")
plt.legend()

plt.subplot(3,3,3)
plt.title("KE spectra after %d timesteps (normalised)" % steps)
k1d,ke=util.get_ke(emu_rollout.test_suite[0][-1],anim.grid)
plt.loglog(k1d,ke,label="Simulation")
k1d,ke=util.get_ke(anim.pred,anim.grid)
plt.loglog(k1d,ke,label="Emulator")
plt.xlabel("wavenumber")
plt.legend()

plt.subplot(3,3,4)
plt.title("Sim")
plt.imshow(emu_rollout.test_suite[0][-1],cmap=sns.cm.icefire,interpolation='none')
plt.colorbar()

plt.subplot(3,3,5)
plt.title("Emulated")
plt.imshow(anim.pred,cmap=sns.cm.icefire,interpolation='none')
plt.colorbar()

plt.subplot(3,3,6)
plt.title("Residuals")
plt.imshow(emu_rollout.test_suite[0][-1]-anim.pred,cmap=sns.cm.icefire,interpolation='none')
plt.colorbar()

wandb.log({"Metrics": wandb.Image(fig_metrics)})
plt.close()
wandb.finish()
