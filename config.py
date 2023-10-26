import os

#Hyperparams
NUM_OF_COORD=64
LEARNING_RATE=0.001
BATCH_SIZE=32
NUM_EPOCHS=200

#Dataset
DATA_DIR="/scratch/"+os.environ["SLURM_JOB_ID"]+"/"
NUM_WORKERS=4

#Model
CKPT_DIR_PATH="/cluster/ix87iquc/data/NIR/Model/"

#logging
LOG_PATH="/cluster/ix87iquc/data/NIR/tb_logs"

#Compute related
ACCELERATOR="gpu"
PRECISION=16

