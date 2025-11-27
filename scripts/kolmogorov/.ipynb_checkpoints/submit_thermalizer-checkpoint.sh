#!/bin/bash

#SBATCH --job-name=kol_thermqlizer
#SBATCH --time=42:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128GB
#SBATCH --output=slurm_%j.out

# Begin execution
module purge

singularity exec --nv \
    --overlay /scratch/ql2221/torch_sing/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; python3 /home/cp3759/Projects/thermalizer/scripts/kolmogorov/train_thermalizer.py"

