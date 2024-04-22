from pytorch_lightning import LightningModule
import torch
import torch.nn as nn


class BaseRegSytem(LightningModule):
    """ Base class to implement common methods. We leave the definition of the step method to child classes """
    def __init__(self,network,config:dict):
        super().__init__()
        self.config=config
        self.criterion=nn.MSELoss()
        self.network=network

    def forward(self,x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer=torch.optim.AdamW(self.parameters(),lr=self.config["optimization"]["lr"],weight_decay=self.config["optimization"]["wd"])
        if self.config["optimization"]["scheduler"]:
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.2)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}
        else:
            return {"optimizer": optimizer}
        
    def step(self,batch,kind):
        raise NotImplementedError("To be defined by child class")

    def training_step(self, batch, batch_idx):
        return self.step(batch,"train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch,"valid")


class RolloutSystem(BaseRegSytem):
    """ Regress over multiple timesteps """
    def __init__(self,network,config:dict):
        super().__init__(network,config)

    def step(self,batch,kind):
        """ Evaluate loss function """
        x_data=batch
        loss=0
        for aa in range(0,x_data.shape[1]-1):
            if aa==0:
                ## Make sure to unsqueeze - conv1d takes [N_batch, n_channels, seq_length]
                ## And for KS we have just 1 feature :))
                x_pred=self(x_data[:,0,:].unsqueeze(1))
            else:
                x_pred=self(x_pred)
            loss_dt=self.criterion(x_pred,x_data[:,aa+1,:].unsqueeze(1))
            self.log(f"{kind}_loss_%d" % aa, loss_dt, on_step=False, on_epoch=True)
            loss+=loss_dt
            
        self.log(f"{kind}_loss", loss, on_step=False, on_epoch=True)     
        return loss


class RolloutResidualSystem(BaseRegSytem):
    """ Regress over multiple timesteps """
    def __init__(self,network,config:dict):
        super().__init__(network,config)

    def step(self,batch,kind):
        """ Evaluate loss function """
        x_data=batch
        loss=0
        for aa in range(0,x_data.shape[1]-1):
            if aa==0:
                x_t=x_data[:,0,:].unsqueeze(1)
            else:
                x_t=x_dt+x_t
            x_dt=self(x_t)
            loss_dt=self.criterion(x_dt,x_data[:,aa+1,:].unsqueeze(1)-x_data[:,aa,:].unsqueeze(1))
            self.log(f"{kind}_loss_%d" % aa, loss_dt, on_step=False, on_epoch=True)
            loss+=loss_dt
            
        self.log(f"{kind}_loss", loss, on_step=False, on_epoch=True)     
        return loss


class SlicedScoreSystem(BaseRegSytem):
    """ Sliced score matching loss with variance reduction
        eq 8 from https://arxiv.org/abs/1905.07088 """
    def __init__(self,network,config:dict):
        super().__init__(network,config)

    def validation_step(self, batch, batch_idx):
        """ Enable gradient tracking on validation step
            -- disabled by lightning by default """
        torch.set_grad_enabled(True)
        return self.step(batch,"valid")

    def step(self,batch,kind):
        """ Evaluate loss function """
        ## Our CCNs work with arbitrary numbers of input/output channels
        ## so our tensor shape has to be [batch_size,num channels,Nx,Ny]
        ## but our dataset is structured [batch_size,rollout number, Nx, Ny]
        batch_size=batch.shape[0]
        x=batch[:,0,:,:].unsqueeze(1)
        x=x.requires_grad_()

        ## Draw random vector for slicing
        random_vec=torch.rand(batch_size,x.shape[-1]**2,device="cuda",requires_grad=True)

        y=self(x)
        y=y.view(batch_size,x.shape[-1]**2)
        prod=torch.einsum("ij,ij->i",y,random_vec)
        prod=torch.sum(prod)

        ## \nabla_\mathbf{x}\mathbf{s}_\theta(\mathbf{x}_i)\mathbf{v}_{ij}
        grads=torch.autograd.grad(prod, x,retain_graph=True)[0]
        ## Reshape to vector
        grads=grads.view(batch_size,x.shape[-1]**2)

        loss1=torch.mean(torch.einsum("ij,ij->i",random_vec,grads))
        loss2=torch.mean(0.5*(torch.sum(y,axis=1)**2))
        loss=loss1+loss2

        self.log(f"{kind}_loss1", loss1, on_step=False, on_epoch=True)  
        self.log(f"{kind}_loss2", loss2, on_step=False, on_epoch=True) 
        self.log(f"{kind}_loss", loss, on_step=False, on_epoch=True) 
        
        return loss
        
        
class DenoisingScoreSystem(BaseRegSytem):
    """ Denoising score matching loss
        eq 7 from https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf """
    def __init__(self,network,config:dict):
        super().__init__(network,config)
        self.sigma=config["sigma"]

    def validation_step(self, batch, batch_idx):
        """ Enable gradient tracking on validation step
            -- disabled by lightning by default """
        torch.set_grad_enabled(True)
        return self.step(batch,"valid")

    def step(self,batch,kind):
        """ Evaluate loss function """
        ## Our CCNs work with arbitrary numbers of input/output channels
        ## so our tensor shape has to be [batch_size,num channels,Nx,Ny]
        ## but our dataset is structured [batch_size,rollout number, Nx, Ny]
        batch_size=batch.shape[0]
        x=batch[:,0,:,:].unsqueeze(1)
        noise=torch.rand(x.shape,device="cuda")*self.sigma**2
        pred=self(x+noise)

        loss=self.criterion(pred,noise/self.sigma**2)
        self.log(f"{kind}_loss", loss, on_step=False, on_epoch=True)
        return loss
