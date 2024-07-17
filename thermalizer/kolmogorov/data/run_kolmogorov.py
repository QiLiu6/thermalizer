import thermalizer.kolmogorov.simulate as simulate
import thermalizer.kolmogorov.util as util
import torch
import argparse
import pickle
import yaml
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str, help="name of config file")
parser.add_argument("--save_path", required=True, type=str, help="save_path")

args = parser.parse_args()

config = yaml.safe_load(Path(args.config).read_text())
config["save_path"]=args.save_path
config["viscosity"]=1/config["reynolds"]
print(config)

## From this we can generate a list of indices to apply cuts.
## These cuts split the simulation into short training trajectories
cuts=[]
for aa in range(config["trajectories"]):
    for bb in range(config["rollout"]):
        cuts.append(config["spinup"]+aa*config["decorr_steps"]+bb*config["increment"])

sim_stack=torch.tensor([],dtype=torch.float32)
for aa in tqdm(range(config["n_sims"])):
    sim=simulate.get_sim_batch(config["gridsize"],config["dt"],config["viscosity"],cuts,config["downsample"])
    ## Reshape to [batch idx, rollout idx, nx, ny]
    sim=sim.reshape(config["trajectories"],config["rollout"],sim.shape[-1],sim.shape[-1])
    sim_stack=torch.cat((sim_stack,torch.tensor(sim,dtype=torch.float32)))

save_dict={"data_config":config,
           "data":sim_stack}
           
with open(config["save_path"], 'wb') as f:
    pickle.dump(save_dict, f)
