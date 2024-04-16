import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import yaml
from pathlib import Path
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

import seaborn as sns

import thermalizer.dataset.datasets as datasets
import thermalizer.models.cnn as cnn
import thermalizer.systems.regression_systems as systems
import thermalizer.kolmogorov.util as util
import thermalizer.kolmogorov.simulate as simulate
import thermalizer.kolmogorov.performance as performance



config={}
config["optimization"]={}
config["input_channels"]=1
config["output_channels"]=1
config["activation"]="ReLU"
config["batch_norm"]=True
config["conv_layers"]=5
config["file_path"]="/scratch/cp3759/thermalizer_data/kolmogorov/ratio10_inc1/all.pt"
config["subsample"]=None
config["save_name"]="model_weights.pt"

config["optimization"]["epochs"]=120
config["optimization"]["lr"]=0.0005
config["optimization"]["wd"]=0.05
config["optimization"]["scheduler"]=None
config["optimization"]["batch_size"]=64


ds=datasets.KolmogorovDataset(config["file_path"],subsample=config["subsample"])


train_loader = DataLoader(
    ds,
    num_workers=10,
    batch_size=config["optimization"]["batch_size"],
    sampler=SubsetRandomSampler(ds.train_idx),
)
valid_loader = DataLoader(
    ds,
    num_workers=10,
    batch_size=config["optimization"]["batch_size"],
    sampler=SubsetRandomSampler(ds.valid_idx),
)

config["train_fields"]=len(ds.train_idx)
config["valid_fields"]=len(ds.valid_idx)
config["field_std"]=ds.x_std

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

## Compare sim to emulator rollout for N passes.
N_passes=3000
dt=0.001
Dt=0.01
spinup=5000
steps=int(Dt/dt)*N_passes

ds_64=simulate.run_kolmogorov_sim(dt,Dt,steps,spinup=spinup,downsample=4)
ds_64=torch.tensor(ds_64.to_numpy())

anim=performance.KolmogorovAnimation(ds_64,model,fps=90,nSteps=N_passes,savestring=config["save_path"]+"/anim")
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
k1d,ke=util.get_ke(ds_64[-1]/ds.x_std,anim.grid)
plt.loglog(k1d,ke,label="Simulation")
k1d,ke=util.get_ke(anim.pred,anim.grid)
plt.loglog(k1d,ke,label="Emulator")
plt.xlabel("wavenumber")
plt.legend()

plt.subplot(3,3,4)
plt.title("Sim")
plt.imshow(ds_64[-1]/ds.x_std,cmap=sns.cm.icefire,interpolation='none')
plt.colorbar()

plt.subplot(3,3,5)
plt.title("Emulated")
plt.imshow(anim.pred,cmap=sns.cm.icefire,interpolation='none')
plt.colorbar()

plt.subplot(3,3,6)
plt.title("Residuals")
plt.imshow(ds_64[-1]/ds.x_std-anim.pred,cmap=sns.cm.icefire,interpolation='none')
plt.colorbar()

figure_metrics=wandb.Image(fig_metrics)
wandb.log({"Power spectrum": figure_metrics})

model.save_model()
wandb.finish()
print("done")
