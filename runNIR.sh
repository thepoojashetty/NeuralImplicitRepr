#!/bin/bash
#SBATCH --job-name=NIR
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12000
#SBATCH --gres=gpu:1
#SBATCH -o /cluster/ix87iquc/NIR.out
#SBATCH -e /cluster/ix87iquc/NIR.err
#SBATCH --mail-user=pooja.shetty@fau.de
#SBATCH --mail-type=ALL
#Timelimit format: "hours:minutes:seconds" -- max is 24h
#SBATCH --time=24:00:00
### Choose a specific GPU: #SBATCH --gres=gpu:q5000:1
### Run `sinfo -h -o "%n %G"` for GPU types

cp -r /cluster/ix87iquc/data/NIR /scratch/$SLURM_JOB_ID/NIR
# unzip /cluster/ix87iquc/data/synthetic_curated_image_dump.zip -d /scratch/$SLURM_JOB_ID/
# cp -r /cluster/ix87iquc/data/Testdata/. /scratch/$SLURM_JOB_ID/
cp /cluster/ix87iquc/data/synthetic_curated_image_dump.zip /scratch/$SLURM_JOB_ID/
unzip /scratch/$SLURM_JOB_ID/synthetic_curated_image_dump.zip -d /scratch/$SLURM_JOB_ID/

#export PATH="/cluster/ix87iquc/miniconda/bin:$PATH"
source /cluster/ix87iquc/miniconda/etc/profile.d/conda.sh
conda activate nirenv

cd /scratch/$SLURM_JOB_ID/NIR
python main.py
