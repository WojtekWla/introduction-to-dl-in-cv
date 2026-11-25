#!/bin/bash
# Selective Search proposals for all images + proposal recall on the train split.
# CPU only, so it runs on the general "hpc" queue instead of the course GPU queue.
# Submit from the project folder on the HPC:  mkdir -p logs && bsub < jobs/proposals.sh

#BSUB -q hpc
#BSUB -J p4_proposals
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 01:00
#BSUB -o logs/proposals_%J.out
#BSUB -e logs/proposals_%J.err

module load python3/3.11.13
source "$HOME/.venvs/idlcv-p2/bin/activate"

cd "$HOME/idlcv/project4"

echo "Host: $(hostname)"
time python -u proposals.py
python -u evaluate_proposals.py
