import os

#Hyperparams
#NUM_OF_COORD=4096
LEARNING_RATE=0.0001
BATCH_SIZE=256
NUM_EPOCHS=500

#Dataset
DATA_DIR="/home/hpc/iwi5/iwi5192h/NIR/synthetic_curated_image_dump/"
NUM_WORKERS=4

#Model
CKPT_DIR_PATH="/home/hpc/iwi5/iwi5192h/NIR/Model/"
AE_CKPT_DIR_PATH="/home/hpc/iwi5/iwi5192h/NIR/AutoEncoder/Model/"

#logging
LOG_PATH="/home/hpc/iwi5/iwi5192h/NIR/tb_logs"

#Inference
TEST_DATA="/home/hpc/iwi5/iwi5192h/NIR/Testdata/img/Fust & Schoeffer Durandus Gotico-Antiqua 118G_A_8.png"
GENERATED_SKEL="/home/hpc/iwi5/iwi5192h/NIR/generated_skel.png"

#Compute related
ACCELERATOR="auto"
STRATEGY="ddp"
DEVICES=-1
PRECISION=16

