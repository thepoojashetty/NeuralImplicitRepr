#!/bin/bash -l
#SBATCH --job-name=NIR
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH -o /home/hpc/rzku/hpcv720h/NIR/NIR.out
#SBATCH -e /home/hpc/rzku/hpcv720h/NIR/NIR.err
#Timelimit format: "hours:minutes:seconds" -- max is 24h
#SBATCH --time=24:00:00
### Choose a specific GPU: #SBATCH --gres=gpu:q5000:1
### Run `sinfo -h -o "%n %G"` for GPU types

# cp -r /cluster/ix87iquc/data/NIR /scratch/$SLURM_JOB_ID/NIR
# unzip /home/hpc/rzku/hpcv720h/NIR/synthetic_curated_image_dump.zip -d /home/hpc/rzku/hpcv720h/NIR/
# cp -r /cluster/ix87iquc/data/Testdata/. /scratch/$SLURM_JOB_ID/
# cp /cluster/ix87iquc/data/synthetic_curated_image_dump.zip /scratch/$SLURM_JOB_ID/
# unzip /scratch/$SLURM_JOB_ID/synthetic_curated_image_dump.zip -d /scratch/$SLURM_JOB_ID/

module add python
conda activate nirenv

cd /home/hpc/rzku/hpcv720h/NIR/
srun python main.py
