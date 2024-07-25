import wandb
import math
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

import thermalizer.models.diffusion as diffusion
import thermalizer.models.unet as unet
import thermalizer.dataset.datasets as datasets
import thermalizer.kolmogorov.util as util

config={}
config["input_channels"]=1
config["output_channels"]=1
config["num_blocks"]=[2,2,2]
config["hidden_channels"]=32
config["time_embedding_dim"]=128
config["image_size"]=64
config["noise_sampling_coeff"]=0.35
config["denoise_time"]=400
config["activation"]="gelu"
config["norm"]=False
config["file_path"]="/scratch/cp3759/thermalizer_data/kolmogorov/reynolds10k/diff.p"
config["whitening"]="/scratch/cp3759/thermalizer_data/kolmogorov/reynolds10k/white_10k.pt"
config["input_channels"]=1
config["output_channels"]=1
config["subsample"]=50000
config["save_name"]="model_weights.pt"
config["dim_mults"]=[2,4]
config["base_dim"]=32
config["timesteps"]=1000
#config["add_noise"]=1e-4
config["optimization"]={}
config["optimization"]["epochs"]=30
config["optimization"]["lr"]=0.001
config["optimization"]["wd"]=0.05
config["optimization"]["batch_size"]=64
config["optimization"]["scheduler"]=True

print(config)

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

device="cuda"

## Set up fourier grid for calculating KE of validation imgs
grid=util.fourierGrid(64)

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

device="cuda"

## Set up fourier grid for calculating KE of validation imgs
grid=util.fourierGrid(64)

wandb.init(project="kol_diffusion",entity="chris-pedersen",config=config,dir="/scratch/cp3759/pyqg_data/wandb_runs")

## Make sure save path, train set size, wandb url are passed to config before model is initialised!
## otherwise these important things aren't part of the model config property
config["save_path"]=wandb.run.dir
config["wandb_url"]=wandb.run.get_url()
config["train_set_size"]=len(train_loader.dataset)

model_cnn=unet.Unet(config)
model=diffusion.Diffusion(config, model=model_cnn).to(device)
config["num_params"]=sum(p.numel() for p in model.parameters() if p.requires_grad)
wandb.config.update(config)
wandb.watch(model, log_freq=1)

optimizer=AdamW(model.parameters(),lr=config["optimization"]["lr"])
scheduler=OneCycleLR(optimizer,config["optimization"]["lr"],total_steps=config["optimization"]["epochs"]*len(train_loader),pct_start=0.25,anneal_strategy='cos')
loss_fn=nn.MSELoss(reduction='mean')

global_steps=0
for i in range(1,config["optimization"]["epochs"]):
    model.train()
    ## Loop over batches
    for j,image in enumerate(train_loader):
        image=image.to(device)
        noise=torch.randn_like(image).to(device)
        pred,_,_=model(image,noise)
        loss=loss_fn(pred,noise)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        global_steps+=1

    ## Push loss values for each epoch to wandb
    log_dic={}
    log_dic["epoch"]=i
    log_dic["training_loss"]=(loss.detach().cpu().item()/(len(image)))

    valid_imgs=next(iter(valid_loader))
    valid_imgs=valid_imgs.to(device)
    denoised,noised=model.denoising(valid_imgs,config["denoise_time"])

    ## Get losses of noise level and denoised images
    noise_loss=loss_fn(valid_imgs,noised)
    valid_denoise_loss=loss_fn(valid_imgs,denoised)

    fig_fields=plt.figure(figsize=(18,9))
    plt.suptitle("Original (top), Noised (middle), Denoised (lower)")
    for aa in range(5):
        plt.subplot(3,5,aa+1)
        plt.imshow(valid_imgs[aa].squeeze().cpu(),cmap=sns.cm.icefire)
        plt.colorbar()
        plt.subplot(3,5,aa+6)
        plt.imshow(noised[aa].squeeze().cpu(),cmap=sns.cm.icefire)
        plt.colorbar()
        plt.subplot(3,5,aa+11)
        plt.imshow(denoised[aa].squeeze().cpu(),cmap=sns.cm.icefire)
        plt.colorbar()
    plt.tight_layout()
    wandb.log({"Fields":wandb.Image(fig_fields)})
    plt.close()

    fig_ke=plt.figure()
    plt.title("KE spectra: black=true, red=noised, blue=denoised")
    for aa in range(len(valid_imgs)):
        k1d_plot,ke=util.get_ke(valid_imgs[aa].squeeze().cpu(),grid)
        plt.loglog(k1d_plot,ke,color="black",alpha=0.1)
        k1d_plot,ke=util.get_ke(noised[aa].squeeze().cpu(),grid)
        plt.loglog(k1d_plot,ke,color="red",alpha=0.1)
        k1d_plot,ke=util.get_ke(denoised[aa].squeeze().cpu(),grid)
        plt.loglog(k1d_plot,ke,color="blue",alpha=0.1)
    wandb.log({"Spectra":wandb.Image(fig_ke)})
    plt.close()
    
    log_dic["denoise_loss"]=(valid_denoise_loss.detach().cpu().item()/(len(valid_imgs)))
    log_dic["noise_loss"]=(noise_loss.detach().cpu().item()/(len(valid_imgs)))
    wandb.log(log_dic)

model.model.save_model()
wandb.finish()
print("done")
