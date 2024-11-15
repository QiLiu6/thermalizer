import thermalizer.kolmogorov.inference as inference
import wandb

project_string="therm_sweep"
steps=10000

def sweep_wrapper():
    wandb.init(entity="chris-pedersen",project=project_string,dir="/scratch/cp3759/thermalizer_data/wandb_data")
    print("sweeping")
    inference.therm_inference("sweep",wandb.config.start,wandb.config.stop,steps,True,project_string)

# 2: Define the search space
sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "minimize", "name": "spectral similarity thermalized"},
    "parameters": {
        "start": {"values": [10,9,8,7,6]},
        "stop": {"values": [5,4,3,2]},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_string,entity="chris-pedersen")

wandb.agent(sweep_id, function=sweep_wrapper, count=20)