#!/bin/bash
# Selective Search proposals for all images + proposal recall on the train split.
# CPU only, so it runs on the general "hpc" queue instead of the course GPU queue.
#
# Submit from the project folder on the HPC (mkdir -p logs first):
#   bsub < jobs/proposals.sh                                   # baseline: fast, 300px -> data/potholes/proposals
#   MODE=quality MAX_DIM=2000 OUT_NAME=proposals_quality_full bsub < jobs/proposals.sh
# The variables are copied into the job because bsub passes on the submitting shell's environment.

#BSUB -q hpc
#BSUB -J p4_proposals
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 03:00
#BSUB -o logs/proposals_%J.out
#BSUB -e logs/proposals_%J.err

MODE="${MODE:-fast}"
MAX_DIM="${MAX_DIM:-300}"
OUT_DIR="$HOME/idlcv/data/potholes/${OUT_NAME:-proposals}"

module load python3/3.11.13
source "$HOME/.venvs/idlcv-p2/bin/activate"

cd "$HOME/idlcv/project4"

echo "Host: $(hostname)  mode=$MODE  max_dim=$MAX_DIM  out=$OUT_DIR"
time python -u proposals.py --mode "$MODE" --max-dim "$MAX_DIM" --out-dir "$OUT_DIR"
python -u evaluate_proposals.py --proposals-dir "$OUT_DIR"
