#!/bin/bash

#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"

#BSUB -J 3dcnn_ep200_drop04_xtra_lay
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"

#BSUB -o logs/xtra_drop04_3dcnn_%J.out
#BSUB -e logs/xtra_drop04_3dcnn_%J.err
#BSUB -W 02:00

module load python3/3.11.13
source "$HOME/.venvs/idlcv-p2/bin/activate"

cd "$HOME/idlcv/project2"

echo "Host: $(hostname)"
nvidia-smi

python -u -m src.train_3dcnn
