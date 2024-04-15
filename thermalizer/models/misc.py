import numpy as np
import io
import torch
import pickle
import pyqg_explorer.models.diffusion as diffusion
import pyqg_explorer.models.fcnn as fcnn
import pyqg_explorer.models.unet as unet
import xarray as xr

""" Store some miscellaneous helper methods that are frequently used """


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

    ## Check if unet, otherwise assume FCNN
    try:
        if model_dict["config"]["arch"]=="unet":
            model=unet.U_net(model_dict["config"])
        else:
            model=fcnn.FCNN(model_dict["config"])
    except:
        model=fcnn.FCNN(model_dict["config"])

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

