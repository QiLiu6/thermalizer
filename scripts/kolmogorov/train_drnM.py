import os
from thermalizer.systems import training_systems
import wandb

## Stop jax hoovering up GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

config={}
config["input_channels"]=1
config["output_channels"]=1
config["model_type"]="DRN"
config["num_blocks"]=[2,2,2]
config["hidden_channels"]=32
config["activation"]="gelu"
config["loader_workers"]=3
config["ddp"]=False
config["project"]="icml_emu_kol"
config["rollout_scheduler"]=20000
#config["ema_decay"]=0.995
config["max_rollout"]=2
config["norm"]=False
config["sigma"]=1e-5
config["PDE"]="Kolmogorov"
config["file_path"]="/scratch/cp3759/thermalizer_data/kolmogorov/reynolds10k/emu_big.p"
config["test_data"]="/scratch/cp3759/thermalizer_data/kolmogorov/reynolds10k/test40.p"
config["subsample"]=None
config["train_ratio"]=0.95
config["save_name"]="model_weights.pt"
config["residual_loss"]="Residual"
#config["short_rollout"]=1
#config["add_noise"]=1e-4
config["optimization"]={}
config["optimization"]["epochs"]=12
config["optimization"]["lr"]=0.00005
config["optimization"]["wd"]=0.05
config["optimization"]["batch_size"]=32
config["optimization"]["gradient_clipping"]=1.
#config["optimization"]["scheduler_step"]=100000
#config["optimization"]["scheduler_gamma"]=0.5

trainer=training_systems.ResidualEmulatorTrainer(config)
trainer.run()
trainer.performance()
wandb.finish()
