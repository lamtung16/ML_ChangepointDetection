import pandas as pd
import os
import shutil
from datetime import datetime
from itertools import product

# CREATE PARAMETERS CSV
# Define hyperparameters
dataset = ['cancer']
num_layers = [1, 2, 3, 4]
layer_size = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
test_fold = [1, 2]
test_ratio = [0.1, 0.2, 0.3, 0.4, 0.5]

# Create parameter grid
param_combinations = list(product(dataset, num_layers, layer_size, test_fold, test_ratio))

# Create DataFrame and save it into csv
params_df = pd.DataFrame(param_combinations, columns=['dataset', 'num_layers', 'layer_size', 'test_fold', 'test_ratio'])
params_df.to_csv("params.csv", index=False)

# CREATE RUN_ONE.SH
# Define job parameters
n_tasks, ncol = params_df.shape

# Create SLURM script
run_one_contents = f"""#!/bin/bash
#SBATCH --array=0-{n_tasks-1}
#SBATCH --time=12:00:00
#SBATCH --mem=2GB
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm-out/slurm-%A_%a.out
#SBATCH --error=slurm-out/slurm-%A_%a.out
#SBATCH --job-name=mlp

python run_one.py $SLURM_ARRAY_TASK_ID
"""

# Write the SLURM script to a file
run_one_sh = os.path.join("run_one.sh")
with open(run_one_sh, "w") as run_one_f:
    run_one_f.write(run_one_contents)