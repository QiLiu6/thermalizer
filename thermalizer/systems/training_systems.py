import os
import pickle
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import thermalizer.dataset.datasets as datasets
import thermalizer.models.misc as misc
import thermalizer.models.diffusion as diffusion

"""
Things to check:
1. Are we uploading model weights, dataset sizes correctly
2. Are the train, valid datasets sized properly
3. Is the checkpointing working properly

Once these are established, test scaling of runs as we go to more GPU

Then start 'production' jobs for big DRN, MUnet

"""


def setup():
    """Sets up the process group for distributed training.
       We are using torchrun so not using rank and world size arguments """
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")

def cleanup():
    """Cleans up the process group."""
    dist.destroy_process_group()


class Trainer:
    """ Base trainer class """
    def __init__(self,config):
        self.config=config
        
        if self.config["ddp"]:
            setup()
            self.gpu_id=int(os.environ["LOCAL_RANK"])
            self.ddp=True
            self.world_size=dist.get_world_size()
            if self.gpu_id==0:
                self.logging=True
            else:
                self.logging=False
        else:
            self.gpu_id="cuda"
            self.ddp=False
            self.logging=True

        self._prep_data()
        self._prep_model()
        self._prep_optimizer()
        if self.logging:
            self.init_wandb()
            ## Sync all configs
            wandb.config=self.config
            self.model.config=self.config

    def init_wandb(self):
        ## Set up wandb stuff
        wandb.init(entity="chris-pedersen",project=self.config["project"],
                        dir="/scratch/cp3759/thermalizer_data/wandb_data",config=self.config)
        self.config["save_path"]=wandb.run.dir
        self.config["wandb_url"]=wandb.run.get_url()

    def _prep_data(self):
        train_data,valid_data,config=datasets.parse_data_file(self.config)
        ds_train=datasets.FluidDataset(train_data)
        ds_valid=datasets.FluidDataset(valid_data)
        self.config=config ## Update config dict

        if self.ddp:
            train_sampler=DistributedSampler(ds_train)
            valid_sampler=DistributedSampler(ds_valid)
        else:
            train_sampler=RandomSampler(ds_train)
            valid_sampler=RandomSampler(ds_valid)

        self.train_loader=DataLoader(
                ds_train,
                num_workers=self.config["loader_workers"],
                batch_size=self.config["optimization"]["batch_size"],
                sampler=train_sampler,
            )

        self.valid_loader=DataLoader(
                ds_valid,
                num_workers=self.config["loader_workers"],
                batch_size=self.config["optimization"]["batch_size"],
                sampler=valid_sampler,
            )

    def _prep_model(self):
        self.model=misc.model_factory(self.config).to(self.gpu_id)
        self.config["cnn learnable parameters"]=sum(p.numel() for p in self.model.parameters())

        if self.ddp:
            self.model = DDP(self.model,device_ids=[self.gpu_id])

    def _prep_optimizer(self):
        self.criterion=nn.MSELoss()
        self.optimizer=torch.optim.AdamW(self.model.parameters(),
                            lr=self.config["optimization"]["lr"],
                            weight_decay=self.config["optimization"]["wd"])

    def training_loop(self):
        raise NotImplementedError("Implemented by subclass")

    def valid_loop(self):
        raise NotImplementedError("Implemented by subclass")

    def run(self):
        raise NotImplementedError("Implemented by subclass")


class ResidualEmulatorTrainer(Trainer):
    def __init__(self,config):
        super().__init__(config)
        self.val_loss=0
        self.val_loss_check=100

    def training_loop(self):
        """ Training loop for residual emulator """
        self.model.train()
        
        nsamp=0
        epoch_loss=0
        for x_data in self.train_loader:
            x_data=x_data.to(self.gpu_id)
            nsamp+=x_data.shape[0]
            self.optimizer.zero_grad()
            loss=0
            if self.config["input_channels"]==1:
                x_data=x_data.unsqueeze(2)
            for aa in range(0,x_data.shape[1]-1):
                if aa==0:
                    x_t=x_data[:,0]
                else:
                    x_t=x_dt+x_t
                x_dt=self.model(x_t)
                loss_dt=self.criterion(x_dt,x_data[:,aa+1]-x_data[:,aa,:])
                loss+=loss_dt
            loss.backward()
            self.optimizer.step()
            epoch_loss+=loss.item()
        if self.logging:
            log_dic={}
            log_dic["train_loss"]=epoch_loss/nsamp ## Average over full epoch
            log_dic["epoch"]=self.epoch
            wandb.log(log_dic)
        return loss

    def valid_loop(self):
        """ Training loop for residual emulator """
        log_dic={}
        self.model.eval()
        epoch_loss=0
        nsamp=0
        with torch.no_grad():
            for x_data in self.valid_loader:
                x_data=x_data.to(self.gpu_id)
                nsamp+=x_data.shape[0]
                loss=0
                if self.config["input_channels"]==1:
                    x_data=x_data.unsqueeze(2)
                for aa in range(0,x_data.shape[1]-1):
                    if aa==0:
                        x_t=x_data[:,0]
                    else:
                        x_t=x_dt+x_t
                    x_dt=self.model(x_t)
                    loss_dt=self.criterion(x_dt,x_data[:,aa+1]-x_data[:,aa,:])
                    loss+=loss_dt
                epoch_loss+=loss.detach()
        epoch_loss/=nsamp ## Average over full epoch
        ## Now we want to allreduce loss over all processes
        if self.ddp:
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            ## Acerage across all processes
            self.val_loss = epoch_loss.item()/self.world_size
        else:
            self.val_loss = epoch_loss.item()
        if self.logging:
            log_dic={}
            log_dic["valid_loss"]=self.val_loss ## Average over full epoch
            log_dic["epoch"]=self.epoch
            wandb.log(log_dic)
        return loss

    def checkpointing(self):
        """ Checkpointing performs two actions:
            1. Save last checkpoint.
            2. Overwrite lowest validation checkpoint if val loss is lower
        """
        self.save_checkpoint(self.config["save_path"]+"/checkpoint_last.p")
        if (self.epoch>2) and (self.val_loss<self.val_loss_check):
            print("Saving new checkpoint with improved validation loss at %s" % self.config["save_path"]+"/checkpoint_best.p")
            self.val_loss_check=self.val_loss ## Update checkpointed validation loss
            self.save_checkpoint(self.config["save_path"]+"/checkpoint_best.p")
        
    def save_checkpoint(self, checkpoint_string):
        """ Checkpoint model and optimizer """
        save_dict={
                    'epoch': self.epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': self.val_loss,
                    'config':self.config,
                    }
        with open(checkpoint_string, 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def load_checkpoint(self,file_string):
        """ Load checkpoint from saved file """
        with open(file_string, 'rb') as fp:
            model_dict = pickle.load(fp)
        assert model_dict["config"]==self.config, "Configs not the same"
        self.model=misc.load_model(file_string).to(self.gpu_id)
        self._prep_optimizer()
        self.optimizer.load_state_dict(model_dict['optimizer_state_dict'])
        return

    def run(self):
        for epoch in range(1,self.config["optimization"]["epochs"]+1):
            self.epoch=epoch
            if self.ddp:
                self.train_loader.sampler.set_epoch(epoch)
            self.training_loop()
            self.valid_loop()
            if self.logging: ## Only run checkpointing on logging node
                self.checkpointing()
        if self.ddp:
            cleanup()
        print("DONE on rank %d" % self.gpu_id)
        return


class DDPMClassifierTrainer(Trainer):
    def __init__(self,config):
        super().__init__(config)
        self.lambda_c=config["regression_loss_weight"]

    def _prep_model(self):
        model_unet=misc.model_factory(config).to(self.gpu_id)
        self.model=diffusion.Diffusion(config, model=model_unet).to(self.gpu_id)
        if self.ddp:
            self.model = DDP(self.model,device_ids=[self.gpu_id])

    def load_checkpoint(self,file_string):
        """ Load checkpoint from saved file """
        with open(file_string, 'rb') as fp:
            model_dict = pickle.load(fp)
        assert model_dict["config"]==self.config, "Configs not the same"
        self.model=misc.load_diffusion_model(file_string).to(self.gpu_id)
        self._prep_optimizer()
        self.optimizer.load_state_dict(model_dict['optimizer_state_dict'])
        return

    def training_loop(self):
        """ Training loop for residual emulator """
        log_dic={}
        for j,image in enumerate(self.train_loader):
            image=image.to(self.gpu_id)
            #image=ds_train[batch_idx]
            #tot_samps+=image.shape[0]
            self.optimizer.zero_grad()
            noise=torch.randn_like(image).to(self.gpu_id)
            pred,_,t,pred_level=self.model(image,noise,True)
            loss_score=self.criterion(pred,noise)
            loss_classifier=F.cross_entropy(pred_level,t)
            loss=loss_score+self.lambda_c*loss_classifier
            loss.backward()
            self.optimizer.step()
        return loss

    def valid_loop(self):
        raise NotImplementedError("Not yet implemented")

    def run(self):
        for epoch in range(1,self.config["optimization"]["epochs"]+1):
            if self.ddp:
                self.train_loader.sampler.set_epoch(epoch)
            self.model.train()
            self.training_loop()
            #valid_loop(model, valid_loader)
        if self.ddp:
            cleanup()
        print("DONE on rank %d" % self.gpu_id)
        return
