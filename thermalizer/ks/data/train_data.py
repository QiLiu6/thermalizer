from tqdm import tqdm
import yaml
from pathlib import Path
import pickle
import torch
import thermalizer.ks.util as ks_util

config = yaml.safe_load(Path("data_config.yml").read_text())
print(config)

## Determine number of sims to run to generate the training set size
## Lets say trajectories per IC
config["ks"]["ratio"]=int(config["ks"]["Dt"]/config["ks"]["dt"])
config["ks"]["num_sims"]=config["ks"]["dataset_size"]//config["ks"]["trajectories"]
config["ks"]["nsteps"]=config["ks"]["ratio"]*(config["ks"]["spinup"]+config["ks"]["trajectories"]*config["ks"]["decorr_steps"]+config["ks"]["increment"]*config["ks"]["rollout"])

train_data=ks_util.build_train_dataset(config)
config["data"]=train_data
with open(config["ks"]["save_name"], 'wb') as handle:
    pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)
