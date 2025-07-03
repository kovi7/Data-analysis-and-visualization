#!/bin/bash -l
#SBATCH --job-name=protbert
#SBATCH --partition=common
#SBATCH --qos=jk439876_common
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --mem=12000
#SBATCH --time=0-3:00:00
#SBATCH --output=protbert_data/logs/protbert_ss3_%A_%a.out  # %A=jobID, %a=arrayIndex
#SBATCH --error=protbert_data/logs/protbert_ss3_%A_%a.err

# Input file pattern
DEV="gpu"
INPUT_BASE="uniprot_sprot_flat"
INPUT_FILE="split_data/${INPUT_BASE}_${SLURM_ARRAY_TASK_ID}k.fasta"
OUTPUT_FILE="protbert_data/${INPUT_BASE}_${SLURM_ARRAY_TASK_ID}k_protbert.fasta"

# Load modules and activate environment
source venv_ProtBert/bin/activate  # Update with your venv path

# Run prediction on GPU
echo "Processing ${INPUT_FILE} (Task ${SLURM_ARRAY_TASK_ID}) on ${DEV}"
srun python3 python_scripts/protbert_runner.py ${INPUT_FILE} ${OUTPUT_FILE} ${DEV}

cat protbert_data/${INPUT_BASE}_*k_protbert.fasta > protbert_data/uniprot_sprot_flat_protbert.fasta

deactivate
