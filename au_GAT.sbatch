#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --workdir=/home/nfs/tsipko/edgenets

module use /opt/insy/modulefiles
module load cuda/10.1 cudnn/10.1-7.6.0.64

srun -u --output="author_attr_GAT.out" python3 autorship_attribution_GAT.py
