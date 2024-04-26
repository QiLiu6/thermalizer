import torch.func as func
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

import thermalizer.models.cnn as cnn
import thermalizer.systems.regression_systems as systems
import thermalizer.dataset.datasets as datasets
import thermalizer.kolmogorov.performance as perf

import argparse

parser = argparse.ArgumentParser(description="Diffusion model denoiser")
parser.add_argument("--sigma", required=True, type=float, help="noise level for denoiser training")

config={}
config["input_channels"]=1
config["output_channels"]=1
config["batch_norm"]=True
config["activation"]="ReLU"
config["conv_layers"]=8
config["data_path"]="/scratch/cp3759/thermalizer_data/kolmogorov/ratio10_inc1_diffusion/all.pt"
config["subsample"]=None
config["sigma"]=args.sigma
config["save_name"]="model_weights.pt"
config["optimization"]={}
config["optimization"]["batch_size"]=10
config["optimization"]["epochs"]=120
config["optimization"]["lr"]=0.0005
config["optimization"]["wd"]=0.05
config["optimization"]["scheduler"]=True

model=cnn.FCNN(config)
model=model.to("cuda")

ds=datasets.KolmogorovDataset(config["data_path"],subsample=config["subsample"])

train_loader = DataLoader(
    ds,
    num_workers=4,
    batch_size=config["optimization"]["batch_size"],
    sampler=SubsetRandomSampler(ds.train_idx),
)

valid_loader = DataLoader(
    ds,
    num_workers=4,
    batch_size=config["optimization"]["batch_size"],
    sampler=SubsetRandomSampler(ds.valid_idx),
)

config["train_fields"]=len(ds.train_idx)
config["valid_fields"]=len(ds.valid_idx)
config["field_std"]=ds.x_std

device="cuda"

wandb.init(project="score_denoising", entity="m2lines",config=config,dir="/scratch/cp3759/thermalizer_data/wandb_data")
wandb.config["save_path"]=wandb.run.dir
config["save_path"]=wandb.run.dir
config["wandb_url"]=wandb.run.get_url()

model=cnn.FCNN(config)

wandb.config["cnn learnable parameters"]=sum(p.numel() for p in model.parameters())
wandb.watch(model, log_freq=1)


#system=RolloutSystem1D(model,config)
#system=SlicedScoreSystem(model,config)
system=systems.DenoisingScoreSystem(model,config)

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

## Plot score as we move away from the data manifold
valid=next(iter(valid_loader))
valid=valid.to("cuda")
model.eval()
fig_norm=perf.plot_thermalizer_norms(valid[0,0],model)
norm_fig=wandb.Image(fig_norm)
wandb.log({"Score norm":norm_fig})

wandb.finish()
print("DONE")
