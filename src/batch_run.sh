#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --job-name=hetfl
#SBATCH --array=0-1
#SBATCH --error=out.err
#SBATCH --cpus-per-task=10

case $SLURM_ARRAY_TASK_ID in
   0)  REG=0 ;;
   1)  REG=0.5 ;;
esac

module load mamba
source activate hetFL
srun python run.py --reg_term=$REG --p_in 1 --p_out 0 --n_iters 10 
