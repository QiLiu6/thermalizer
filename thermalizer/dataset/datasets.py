from torch.utils.data import Dataset
import numpy as np
import torch
import math
import pickle


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


class KolmogorovDataset(BaseDataset):
    """
    Dataset for Kolmogorov flow trajectories
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
            with open(self.file_path, "rb") as input_file:
                data = pickle.load(input_file)
                self.x_data=data["data"]
                self.data_config=data["data_config"]
        else:
            self.x_data=self.file_path
        self.len=len(self.x_data)
        self.x_std=4.8

        if self.subsample:
            self._subsample()
        self._get_split_indices()
            
    def __getitem__(self, idx):
        """ Return elements at each index specified by idx. Will rescale to unit
            variance using the std of the full dataset. NB that we are not rescaling
            the mean, as we are assuming these fields are already 0 mean """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x_data[idx]/self.x_std

    def update_config_dict(self,config):
        """ Pass a config dict, and add data config elements to it """
        for key in self.data_config.keys():
            config[key]=self.data_config[key]
        config["field_std"]=self.x_std
        config["train_fields"]=len(self.train_idx)
        config["valid_fields"]=len(self.valid_idx)
        return config
