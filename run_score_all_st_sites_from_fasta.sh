#!/bin/bash
#SBATCH --job-name=score_st_sites_full
#SBATCH --account=luvul0
#SBATCH --mail-user=luvul@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=180g
#SBATCH --time=02-00:00:00
#SBATCH --output=/home/luvul/log/%u/%x-%j.log
#SBATCH --error=/home/luvul/log/%u/error-%x-%j.log

module load mamba
source activate 1433predictor2026

export LD_PRELOAD=/home/luvul/.conda/envs/1433predictor2026/lib/libstdc++.so.6

which python
python -c "import sys; print(sys.executable)"

cd /home/luvul/1433predictor

# python score_all_st_sites_from_fasta.py

/home/luvul/.conda/envs/1433predictor2026/bin/python score_all_st_sites_from_fasta.py

