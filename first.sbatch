#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --workdir=/home/nfs/tsipko/edgenets

srun -u --output="simple_arch.out" python3.4 simple_architecture.py
