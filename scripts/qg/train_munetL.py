import wandb
import os
from thermalizer.systems import training_systems

## Stop jax hoovering up GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

config={}
config["input_channels"]=2
config["output_channels"]=2
config["model_type"]="ModernUnet"
config["dim_mults"]=[2,2,2]
config["hidden_channels"]=64
config["activation"]="gelu"
config["loader_workers"]=3
config["ddp"]=False
config["project"]="icml_emu_qg"
config["rollout_scheduler"]=20000
config["max_rollout"]=2
config["norm"]=False
config["sigma"]=1e-4
#config["PDE"]="Kolmogorov"
#config["file_path"]="/scratch/cp3759/thermalizer_data/kolmogorov/reynolds10k/emu_big.p"
config["PDE"]="QG"
config["file_path"]="/scratch/cp3759/thermalizer_data/qg/dt5/eddy_emu_500.pt"
config["subsample"]=None
config["train_ratio"]=0.95
config["save_name"]="model_weights.pt"
#config["short_rollout"]=1
#config["add_noise"]=1e-4
config["optimization"]={}
config["optimization"]["epochs"]=12
config["optimization"]["lr"]=0.00005
config["optimization"]["wd"]=0.05
config["optimization"]["batch_size"]=24
config["optimization"]["gradient_clipping"]=1.
#config["optimization"]["scheduler_step"]=100000
#config["optimization"]["scheduler_gamma"]=0.5

trainer=training_systems.ResidualEmulatorTrainer(config)
trainer.run()
trainer.performance()
wandb.finish()
