from torch.utils.data import Dataset
import numpy as np
import torch
import math
import pickle


def parse_data_file(config):
    field_std=4.8 ## We are just using this for all Kolmogorov flows..
    with open(config["file_path"], "rb") as input_file:
        data = pickle.load(input_file)
    data_config=data["data_config"]

    ## Subsample data if requested
    if "subsample" in config.keys():
        if config["subsample"] is not None:
            data=data["data"][:config["subsample"]]

    ## Set seed for train/valid splits
    if "seed" in config.keys():
        seed=config["seed"]
    else:
        seed=42
        
    ## Get train/valid splits & cut data
    train_idx,valid_idx=get_split_indices(len(data))
    train_dataset=data[train_idx]/field_std
    valid_dataset=data[valid_idx]/field_std

    ## Update config dict with data config
    for key in data_config.keys():
        config[key]=data_config[key]
    config["field_std"]=field_std
    config["train_fields"]=len(train_dataset)
    config["valid_fields"]=len(valid_dataset)
    
    return train_dataset, valid_dataset, config

def get_split_indices(set_size,seed=42,train_ratio=0.75,valid_ratio=0.25):
    """ Get indices for train, valid and test splits """
    
    rng = np.random.default_rng(seed)
    ## Randomly shuffle indices of entire dataset
    rand_indices=rng.permutation(np.arange(set_size))

    ## Set number of train, valid and test points
    num_train=math.floor(set_size*train_ratio)
    num_valid=math.floor(set_size*valid_ratio)
    
    ## Make sure we aren't overcounting
    assert (num_train+num_valid) <= set_size
    
    ## Pick train, test and valid indices from shuffled list
    train_idx=rand_indices[0:num_train]
    valid_idx=rand_indices[num_train+1:num_train+num_valid]
    
    ## Make sure there's no overlap between train, valid and test data
    assert len(set(train_idx) & set(valid_idx))==0, (
            "Common elements in train, valid or test set")
    return train_idx, valid_idx


class BaseDataset(Dataset):
    """ Base object to store core dataset methods """
    def __init__(self,seed=42,subsample=None,drop_spin_up=True,train_ratio=0.75,valid_ratio=0.25,test_ratio=0.0):
        super().__init__()
        self.train_ratio=train_ratio
        self.valid_ratio=valid_ratio
        self.test_ratio=test_ratio
        self.subsample=subsample
        self.seed=seed
        self.rng = np.random.default_rng(self.seed)

    def _get_split_indices(self):
        """ Set indices for train, valid and test splits """

        ## Randomly shuffle indices of entire dataset
        rand_indices=self.rng.permutation(np.arange(self.len))

        ## Set number of train, valid and test points
        num_train=math.floor(self.len*self.train_ratio)
        num_valid=math.floor(self.len*self.valid_ratio)
        num_test=math.floor(self.len*self.test_ratio)
        
        ## Make sure we aren't overcounting
        assert (num_train+num_valid+num_test) <= self.len
        
        ## Pick train, test and valid indices from shuffled list
        self.train_idx=rand_indices[0:num_train]
        self.valid_idx=rand_indices[num_train+1:num_train+num_valid]
        self.test_idx=rand_indices[len(self.valid_idx)+1:]
        
        ## Make sure there's no overlap between train, valid and test data
        assert len(set(self.train_idx) & set(self.valid_idx) & set(self.test_idx))==0, (
                "Common elements in train, valid or test set")
        return

    def _subsample(self):
        """ Take a subsample of the loaded data. Update len """
        self.x_data=self.x_data[:self.subsample]
        self.len=len(self.x_data)
        return

    def __len__(self):
        return self.len


class KSDataset(BaseDataset):
    """
    Dataset for Kuramoto-Sivashinky solutions
    """
    def __init__(self,file_path,seed=42,subsample=None,train_ratio=0.75,valid_ratio=0.25,test_ratio=0.0):
        """
        file_path:     path to data
        seed:          random seed used to create train/valid/test splits
        subsample:     None or int: if int, subsample the dataset to a total of N=subsample maps
        train_ratio:
        valid_ratio:
        test_ratio:
        """
        super().__init__(subsample=subsample,seed=seed,train_ratio=train_ratio,valid_ratio=valid_ratio,test_ratio=test_ratio)
        self.file_path=file_path
        if isinstance(self.file_path,str):
            self.x_data=torch.load(self.file_path)
        else:
            self.x_data=self.file_path
        self.len=len(self.x_data)

        if self.subsample:
            self._subsample()
        self._get_split_indices()
            
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x_data[idx]


class KolmogorovDataset(Dataset):
    """
    Dataset for Kolmogorov flow trajectories
    """
    def __init__(self,data_tensor):
        """
        file_path:     path to data
        seed:          random seed used to create train/valid/test splits
        subsample:     None or int: if int, subsample the dataset to a total of N=subsample maps
        train_ratio:
        valid_ratio:
        test_ratio:
        """
        
        super().__init__()
        self.x_data=data_tensor
        self.len=len(self.x_data)

    def __len__(self):
        return self.len
            
    def __getitem__(self, idx):
        """ Return elements at each index specified by idx. Will rescale to unit
            variance using the std of the full dataset. NB that we are not rescaling
            the mean, as we are assuming these fields are already 0 mean """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x_data[idx]
