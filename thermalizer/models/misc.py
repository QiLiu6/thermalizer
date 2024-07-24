import numpy as np
import io
import pickle
import tqdm
import xarray as xr
import math
import torch
from torch import nn

import thermalizer.models.diffusion as diffusion
import thermalizer.models.cnn as cnn
import thermalizer.models.unet as unet
import thermalizer.models.unet_modern as munet
import thermalizer.models.drn as drn


""" Store some miscellaneous helper methods that are frequently used """

## Activation registry for resnet modules
ACTIVATION_REGISTRY = {
    "relu": nn.ReLU(),
    "silu": nn.SiLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
}


########################## Loading models ################################
## Torch models trained using cuda and then pickled cannot be loaded
## onto cpu using the normal pickle methods: https://github.com/pytorch/pytorch/issues/16797
## This method replaces the pickle.load(input_file), using the same syntax
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def load_model(file_string):
    """ Load a pickled model, either on gpu or cpu """
    with open(file_string, 'rb') as fp:
        if torch.cuda.is_available():
            model_dict = pickle.load(fp)
        else:
            model_dict = CPU_Unpickler(fp).load()

    ## Check if unet, modern unet, or drn. Otherwise load default CNN
    try:
        if model_dict["config"]["model_type"]=="Unet":
            model=unet.Unet(model_dict["config"])
        elif model_dict["config"]["model_type"]=="ModernUnet":
            model=munet.ModernUnet(model_dict["config"])
        elif model_dict["config"]["model_type"]=="DRN":
            model=drn.ResNet(model_dict["config"])
        else:
            model=cnn.FCNN(model_dict["config"])
    except:
        model=cnn.FCNN(model_dict["config"])

    ## Load state_dict
    model.load_state_dict(model_dict["state_dict"])
    return model


def load_diffusion_model(file_string):
    """ Load a diffusion model. Read config file from the pickle
        Reconstruct the CNN, then use same config file to create
        a diffusion model with the loaded CNN """

    with open(file_string, 'rb') as fp:
        if torch.cuda.is_available():
            model_dict = pickle.load(fp)
        else:
            model_dict = CPU_Unpickler(fp).load()
    model_cnn=unet.Unet(model_dict["config"])
    model_cnn.load_state_dict(model_dict["state_dict"])
    diffusion_model=diffusion.Diffusion(model_dict["config"], model=model_cnn)
    return diffusion_model


def estimate_covmat(field_tensor,nsamp=None):
    """ Estimate covariance matrix from some tensor of fields. Can either be
        flattened to batched 1D tensors, or batched 2D fields 
        use nsamp to estimate covmat from a subsample of the data """

    ## If nsamp is not provided, use every sample in field_tensor
    if nsamp==None:
        nsamp=len(field_tensor)

    ## If the field tensor isn't flattened, flatten
    if len(field_tensor.shape)>2:
        field_tensor=field_tensor.reshape((len(field_tensor),64*64))

    ## Initialise covariance matrix
    cov=torch.zeros((64**2,64**2))

    for aa in tqdm(range(nsamp)):
        cov+=torch.outer(test_suite[aa][0].flatten(),test_suite[aa][0].flatten())
    cov/=(nsamp-1)
    return cov


class FieldNoiser():
    """ Forward diffusion module for various different noise schedulers """
    def __init__(self,timesteps,scheduler):
        self.timesteps=timesteps
        self.scheduler=scheduler
        #print(self.timesteps)

        if self.scheduler=="cosine":
            self.betas=self._cosine_variance_schedule(self.timesteps)
        elif self.scheduler=="linear":
            self.betas=self._linear_variance_schedule(self.timesteps)
        elif self.scheduler=="sigmoid":
            self.betas=self._sigmoid_variance_schedule(self.timesteps)


        self.alphas=1.-self.betas
        self.alphas_cumprod=torch.cumprod(self.alphas,dim=-1)
        self.sqrt_alphas_cumprod=torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod=torch.sqrt(1.-self.alphas_cumprod)

        if torch.cuda.is_available():
            self.device=torch.device('cuda')
            self.betas=self.betas.to(self.device)
            self.alphas=self.alphas.to(self.device)
            self.alphas_cumprod=self.alphas_cumprod.to(self.device)
            self.sqrt_alphas_cumprod=self.sqrt_alphas_cumprod.to(self.device)
            self.sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod.to(self.device)

    def _cosine_variance_schedule(self,timesteps,epsilon= 0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)
        return betas

    def __linear_variance_schedule(self,timesteps):
        betas=torch.linspace(0,1,steps=timesteps+1,dtype=torch.float32)
        return betas

    def _linear_variance_schedule(self,timesteps):
        """
        linear schedule, proposed in original ddpm paper
        """
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)
        
    def _sigmoid_variance_schedule(self,timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
        """
        sigmoid schedule
        proposed in https://arxiv.org/abs/2212.11972 - Figure 8
        better for images > 64x64, when used during training
        """
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
        v_start = torch.tensor(start / tau).sigmoid()
        v_end = torch.tensor(end / tau).sigmoid()
        alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def forward_diffusion(self,x_0,t,noise):
        """ Add noise to a clean field for t noise timesteps """
        assert x_0.shape==noise.shape, "Noise and fields have different shapes"
        #q(x_{t}|x_{0})
        return self.sqrt_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*x_0+ \
                self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*noise
