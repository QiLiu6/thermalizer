import torch.nn as nn
import numpy as np
import os
import pickle

class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv1d(1,64,kernel_size=3,padding='same', padding_mode='circular')
        self.act1=nn.ReLU()
        self.conv2=nn.Conv1d(64,64,kernel_size=3,padding='same', padding_mode='circular')
        self.act2=nn.ReLU()
        self.conv3=nn.Conv1d(64,1,kernel_size=3,padding='same', padding_mode='circular')

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        return x


def make_block(in_channels: int, out_channels: int, kernel_size: int, 
        ReLU = 'ReLU', batch_norm = True) -> list:
    '''
    Packs convolutional layer and optionally ReLU/BatchNorm2d
    layers in a list
    '''
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
        padding='same', padding_mode='circular')
    block = [conv]
    if ReLU == 'ReLU':
        block.append(nn.ReLU())
    elif ReLU == 'SiLU':
        block.append(nn.SiLU())
    elif ReLU == 'LeakyReLU':
        block.append(nn.LeakyReLU(0.2))
    elif ReLU == 'False':
        pass
    else:
        print('Error: wrong ReLU parameter:')
    if batch_norm==True:
        block.append(nn.BatchNorm2d(out_channels))
    elif batch_norm=="GroupNorm":
        block.append(nn.GroupNorm(4,out_channels))
    
    return block


class FCNN(nn.Module):
    def __init__(self,config):
        '''
        Packs sequence of n_conv=config["conv_layers"] convolutional layers in a list.
        First layer has config["input_channels"] input channels, and last layer has
        config["output_channels"] output channels
        '''
        super().__init__()
        self.config=config

        blocks = []
        ## If the conv_layers key is missing, we are running
        ## with an 8 layer CNN
        if ("conv_layer" in self.config) == False:
            self.config["conv_layers"]=8
        ## Back batch_norm toggle backwards compatible - models trained pre-3rd April 2024
        ## won't have a batch_norm entry - they will all just have batch_norm though
        if ("batch_norm" in self.config) == False:
            self.config["batch_norm"]=True
        blocks.extend(make_block(self.config["input_channels"],128,5,self.config["activation"],batch_norm=self.config["batch_norm"])) #1
        blocks.extend(make_block(128,64,5,self.config["activation"],batch_norm=self.config["batch_norm"]))                            #2
        if self.config["conv_layers"]==3:
            blocks.extend(make_block(64,self.config["output_channels"],3,'False',False))
        elif self.config["conv_layers"]==4:
            blocks.extend(make_block(64,32,3,self.config["activation"],batch_norm=self.config["batch_norm"]))                            
            blocks.extend(make_block(32,self.config["output_channels"],3,'False',False))
        else: ## 5 layers or more
            blocks.extend(make_block(64,32,3,self.config["activation"],batch_norm=self.config["batch_norm"])) ## 3rd layer
            for aa in range(4,config["conv_layers"]):
                ## 4th and above layer
                blocks.extend(make_block(32,32,3,self.config["activation"],batch_norm=self.config["batch_norm"]))
            ## Output layer
            blocks.extend(make_block(32,self.config["output_channels"],3,'False',False))
        self.conv = nn.Sequential(*blocks)

    def forward(self, x):
        if len(x.shape)==3:
            x=x.unsqueeze(0)
        x = self.conv(x)
        return x

    def save_model(self):
        """ Save the model config, and optimised weights and biases. We create a dictionary
        to hold these two sub-dictionaries, and save it as a pickle file """
        if self.config["save_path"] is None:
            print("No save path provided, not saving")
            return
        save_dict={}
        save_dict["state_dict"]=self.state_dict() ## Dict containing optimised weights and biases
        save_dict["config"]=self.config           ## Dict containing config for the dataset and model
        save_string=os.path.join(self.config["save_path"],self.config["save_name"])
        with open(save_string, 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model saved as %s" % save_string)
        return
