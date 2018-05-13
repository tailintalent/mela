#!/bin/bash
#SBATCH -J varen      # job name
#SBATCH -o /om/user/tailin/bin/varre/syn2.11/var_%j_%a.out  # output and error file name (%j expands to jobID)
#SBATCH -e /om/user/tailin/bin/varre/syn2.11/var_%j_%a.err  # output and error file name (%j expands to jobID)
#SBATCH -N 1        # Number of nodes
#SBATCH -c 1               # total number of mpi tasks requested
#SBATCH --mem=10240     # memory (MB)
#SBATCH --array=0-15
#SBATCH --qos=normal
#SBATCH -t 168:00:00        # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=tailin@mit.edu
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes

source activate /om/user/tailin/python3
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
python run_variational_bounce_states_finetune.py $SLURM_ARRAY_TASK_ID
