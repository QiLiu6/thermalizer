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
from torch.utils.data.sampler import SubsetRandomSampler

import thermalizer.models.diffusion as diffusion
import thermalizer.models.unet as unet
import thermalizer.dataset.datasets as datasets
import thermalizer.kolmogorov.util as util

import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", required=True, type=str, help="name of config file"
)
args = parser.parse_args()

config = yaml.safe_load(Path(args.config).read_text())
print(config)

ds=datasets.KolmogorovDataset(config["data_path"],subsample=config["subsample"])

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

device="cuda"

## Set up fourier grid for calculating KE of validation imgs
grid=util.fourierGrid(64)

wandb.init(project="kol_diffusion",entity="m2lines",config=config,dir="/scratch/cp3759/pyqg_data/wandb_runs")

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
for i in range(1,config["optimization"]["epochs"]+1):
    train_running_loss = 0.0
    model.train()
    ## Loop over batches
    for j,image in enumerate(train_loader):
        image=image[:,0,:,:].unsqueeze(1).to(device)
        noise=torch.randn_like(image).to(device)
        pred=model(image,noise)
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

    ## Validation stuff
    model.eval()
    valid_imgs=next(iter(valid_loader))
    valid_imgs=valid_imgs[:,0,:,:].unsqueeze(1).to(device)
    noise=torch.randn_like(valid_imgs).to(device)
    t=(torch.ones(len(valid_imgs),dtype=torch.int64)*config["denoise_time"]).to(device)
    
    noised=model._forward_diffusion(valid_imgs,t,noise)
    denoised=model.denoising(noised,config["denoise_time"])
    
    ## Get losses of noise level and denoised images
    noise_loss=loss_fn(valid_imgs,noised)
    valid_denoise_loss=loss_fn(valid_imgs,denoised)
    
    valid_imgs=valid_imgs.to("cpu")
    noised=noised.to("cpu")
    denoised=denoised.to("cpu")
    
    index=np.random.randint(config["optimization"]["batch_size"])
    fig_plot=plt.figure(figsize=(14,3))
    plt.suptitle("Denoising held-outs at epoch %d" % i)
    plt.subplot(1,4,1)
    plt.title("True")
    plt.imshow(valid_imgs[index].squeeze().cpu(),cmap=sns.cm.icefire)
    plt.colorbar()
    
    plt.subplot(1,4,2)
    plt.title("Noised")
    plt.imshow(noised[index].squeeze().cpu(),cmap=sns.cm.icefire)
    plt.colorbar()
    
    plt.subplot(1,4,3)
    plt.title("Denoised")
    plt.imshow(denoised[index].squeeze().cpu(),cmap=sns.cm.icefire)
    plt.colorbar()
    
    plt.subplot(1,4,4)
    plt.title("Residual")
    plt.imshow(valid_imgs[index].squeeze().cpu()-denoised[index].squeeze().cpu(),cmap=sns.cm.icefire)
    plt.colorbar()

    denoise_fig=wandb.Image(fig_plot)
    wandb.log({"Denoise":denoise_fig})
    
    fig_ke=plt.figure()
    plt.title("KE spectra: black=true, red=noised, blue=denoised")
    for aa in range(len(valid_imgs)):
        k1d_plot,ke=util.get_ke(valid_imgs[aa].squeeze(),grid)
        plt.loglog(k1d_plot,ke,color="black",alpha=0.1)
        k1d_plot,ke=util.get_ke(noised[aa].squeeze(),grid)
        plt.loglog(k1d_plot,ke,color="red",alpha=0.1)
        k1d_plot,ke=util.get_ke(denoised[aa].squeeze(),grid)
        plt.loglog(k1d_plot,ke,color="blue",alpha=0.1)

    spectra_fig=wandb.Image(fig_ke)
    wandb.log({"Spectra":spectra_fig})

    log_dic["denoise_loss"]=(valid_denoise_loss.detach().cpu().item()/(len(valid_imgs)))
    log_dic["noise_loss"]=(noise_loss.detach().cpu().item()/(len(valid_imgs)))
    wandb.log(log_dic)
    plt.close()

model.model.save_model()
wandb.finish()
print("done")
