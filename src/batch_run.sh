#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --mem=4G
#SBATCH --job-name=cfl
#SBATCH --array=0-3
#SBATCH --error=array.err
#SBATCH --cpus-per-task=10

case $SLURM_ARRAY_TASK_ID in
   0)  REG=0 ;;
   1)  REG=0.1 ;;
   2)  REG=0.5 ;;
   3)  REG=1 ;;
esac

module load miniconda
source activate hetFL
srun python run.py --reg_term=$REG