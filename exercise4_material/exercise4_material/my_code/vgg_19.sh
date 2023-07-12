#!/bin/bash
#SBATCH --clusters=tinygpu
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=24:00:00

cd $HOME
# module load python/3.10-anaconda
source miniconda/bin/activate
srun python /home/woody/iwso/iwso092h/dl/vgg_19.py 