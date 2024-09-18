import wandb
import math
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

import thermalizer.models.diffusion as diffusion
import thermalizer.models.unet as unet
import thermalizer.models.unet_modern as munet
import thermalizer.models.aunet as aunet
import thermalizer.models.misc as misc
import thermalizer.dataset.datasets as datasets
import thermalizer.kolmogorov.util as util



config={}
config["image_size"]=64
config["time_embedding_dim"]=256
config["noise_sampling_coeff"]=None
config["timesteps"]=1000
config["denoise_time"]=400
config["input_channels"]=1
config["output_channels"]=1
config["save_name"]="model_weights.pt"
config["activation"]="gelu"
config["norm"]=False
config["dim_mults"]=[2,2] 
config["base_dim"]=32
config["hidden_channels"]=32

## Dataset
config["file_path"]="/scratch/cp3759/thermalizer_data/kolmogorov/reynolds10k/diff.p"
config["subsample"]=None
config["train_ratio"]=0.95

## Optimisation
config["optimization"]={}
config["optimization"]["epochs"]=15
config["optimization"]["lr"]=2e-5
config["optimization"]["wd"]=0.05
config["optimization"]["batch_size"]=64

print(config)

train_data,valid_data,config=datasets.parse_data_file(config)

ds_train=datasets.FluidDataset(train_data)
ds_train.x_data=ds_train.x_data.to("cuda")

## Batch lists
idx=torch.randperm(len(ds_train))
batches=[]
for aa in range(0,len(ds_train),config["optimization"]["batch_size"]):
    batches.append(idx[aa:aa+config["optimization"]["batch_size"]])

device="cuda"

wandb.init(project="diffusion",entity="chris-pedersen",config=config,dir="/scratch/cp3759/pyqg_data/wandb_runs")
config["save_path"]=wandb.run.dir
config["wandb_url"]=wandb.run.get_url()

model_unet=munet.ModernUnet(config)

model=diffusion.Diffusion(config, model=model_unet).to(device)
#model=diffusion.Diffusion(config, model=model_cnn).to(device)

config["num_params"]=sum(p.numel() for p in model.parameters() if p.requires_grad)
wandb.config.update(config)
wandb.watch(model, log_freq=1)

optimizer=AdamW(model.parameters(),lr=config["optimization"]["lr"])
loss_fn=nn.MSELoss(reduction='mean')


for aa in range(1,config["optimization"]["epochs"]+1):
    model.train()
    epoch_train_loss=0
    train_sample_counter=0
    ## Loop over batches
    for j,batch_idx in enumerate(batches):
        idx=samples=torch.randint(0,len(ds_train),(config["optimization"]["batch_size"],))
        image=ds_train[batch_idx]
        
        noise=torch.randn_like(image).to(device)
        pred,_,_=model(image,noise)
        loss=loss_fn(pred,noise)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #scheduler.step()
        epoch_train_loss+=((loss.detach(). cpu().item())*len(image))
        train_sample_counter+=len(image)

    log_dic={}
    log_dic["epoch"]=aa
    log_dic["training_loss"]=epoch_train_loss/train_sample_counter

    samples=model.sampling(40)
    samples_fig=plt.figure(figsize=(18, 9))
    plt.suptitle("Samples after %d epochs" % aa)
    for i in range(40):
        plt.subplot(5, 8, 1 + i)
        plt.axis('off')
        plt.imshow(samples[i].squeeze(0).data.cpu().numpy(),
                cmap=sns.cm.icefire)
        plt.colorbar()
    plt.tight_layout()
    wandb.log({"Samples":wandb.Image(samples_fig)})
    plt.close()

    hist_figure=plt.figure()
    plt.suptitle("Sampled distribution after %d epochs" % aa)
    plt.hist(samples.flatten().cpu(),bins=1000)
    wandb.log({"Hist":wandb.Image(hist_figure)})
    plt.close()

    wandb.log(log_dic)

print("training done")

model.model.save_model()
wandb.finish()
print("done")
