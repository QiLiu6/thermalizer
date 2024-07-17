import thermalizer.kolmogorov.simulate as simulate
import thermalizer.kolmogorov.util as util
import jax_cfd.base.grids as grids
import torch
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy
import math
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--run_number', type=int, default=0)
args, extra = parser.parse_known_args()
save_string="/scratch/cp3759/thermalizer_data/kolmogorov/test_suite/test%d.pt" % args.run_number

passes=10000
spinup=5000
dt=0.001
Dt=0.01
ratio=int(Dt/dt)
n_sims=1
steps=passes*ratio

print("Numerical timesteps=",steps)
print("Recurrent passes of NN=",passes)

ds_64=simulate.run_kolmogorov_sim(dt,Dt,steps,spinup=spinup,downsample=4)
ds_tensor=torch.tensor(ds_64.to_numpy())
torch.save(ds_tensor, save_string)
print("DONE")
