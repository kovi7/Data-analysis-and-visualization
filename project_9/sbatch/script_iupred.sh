#!/bin/bash -l
#SBATCH --job-name=iupred     # Some descriptive name for a job
#SBATCH --qos=jk439876_common         # Queue you belong
#SBATCH --partition=common     # Most likely you do not need to change it, all students belong to 'common'           # Only if you need GPU
#SBATCH --mem=12000            # This is RAM (not vRAM)
#SBATCH --cpus-per-task=1
#SBATCH --time=0-3:30:10
#SBATCH --output="iupred_data/logs/iupred.%A_%a.out"       # it is usefull to capture the stdout 
#SBATCH --error="iupred_data/logs/iupred.%A_%a.err"        # it is even more usefull to cature the stderr 
#SBATCH --array=0-4

#cd /home/jk439876/hpc/lab1_3/

FILES=($(ls split_data/UP000000625_83333_flat_*k.fasta))
INPUT_FILE=${FILES[$SLURM_ARRAY_TASK_ID]}
#start_time=$(date +%s)

time python3 python_scripts/iupred_runner.py $INPUT_FILE

cat iupred_data/UP000000625_83333_flat_*k_iupred_long.fasta > iupred_data/UP000000625_83333_flat_iupred_long.fasta

#end_time=$(date +%s)
#echo "Total runtime: $((end_time - start_time)) seconds" 
