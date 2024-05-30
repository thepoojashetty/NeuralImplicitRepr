import os

#Hyperparams
#NUM_OF_COORD=4096
LEARNING_RATE=0.0001
BATCH_SIZE=128
NUM_EPOCHS=500

#Dataset
DATA_DIR="/home/hpc/rzku/hpcv720h/NIR/synthetic_curated_image_dump/"
NUM_WORKERS=4

#Model
CKPT_DIR_PATH="/home/hpc/rzku/hpcv720h/NIR/AutoEncoder/Model/"

#logging
LOG_PATH="/home/hpc/rzku/hpcv720h/NIR/AutoEncoder/tb_logs"

#Inference
TEST_DATA="/home/hpc/rzku/hpcv720h/NIR/Testdata/img/Fust & Schoeffer Durandus Gotico-Antiqua 118G_A_8.png"
GENERATED_SKEL="/home/hpc/rzku/hpcv720h/NIR/generated_skel.png"

#Compute related
ACCELERATOR="auto"
STRATEGY="ddp"
DEVICES=-1
PRECISION=16

