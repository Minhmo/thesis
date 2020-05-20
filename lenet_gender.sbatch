#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --workdir=/home/nfs/tsipko/edgenets

srun -u --output="out_lenet_gender.out" python3.4 lenet_gender.py
