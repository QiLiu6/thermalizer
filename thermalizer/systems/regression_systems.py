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