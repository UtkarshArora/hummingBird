#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=15:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu
#SBATCH --job-name=trainHbird
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --mail-user=ua2152@nyu.edu
#SBATCH --output=train_latest.out
#SBATCH --error=train_latest.err

module purge

singularity exec --nv \
	    --overlay /scratch/ua2152/hBird/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif\
	    /bin/bash -c "source /ext3/env.sh; python train.py"
