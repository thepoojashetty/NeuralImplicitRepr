import os

#Hyperparams
#NUM_OF_COORD=4096
LEARNING_RATE=0.0001
BATCH_SIZE=128
NUM_EPOCHS=100

#Dataset
DATA_DIR="/scratch/"+os.environ["SLURM_JOB_ID"]+"/"
NUM_WORKERS=4

#Model
CKPT_DIR_PATH="/cluster/ix87iquc/data/NIR/Model/"

#logging
LOG_PATH="/cluster/ix87iquc/data/NIR/tb_logs"

#Inference
TEST_DATA="/cluster/ix87iquc/data/Testdata/img/Fust & Schoeffer Durandus Gotico-Antiqua 118G_A_8.png"
GENERATED_SKEL="/cluster/ix87iquc/data/NIR/generated_skel.png"

#Compute related
ACCELERATOR="gpu"
PRECISION=16

