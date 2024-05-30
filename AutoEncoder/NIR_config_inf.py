#Hyperparams
LEARNING_RATE=0.0001
BATCH_SIZE=64
NUM_EPOCHS=1

#Dataset
# DATA_DIR="/Users/poojashetty/Documents/AI_SS23/Project/NeuralImplicitRepr/NIR/Testdata"
DATA_DIR="/Users/poojashetty/Documents/AI_SS23/Project/NeuralImplicitRepr/NIR/synthetic_curated_image_dump"
NUM_WORKERS=4

#Model
CKPT_DIR_PATH="/Users/poojashetty/Documents/AI_SS23/Project/NeuralImplicitRepr/NIR/Model/"
AE_CKPT_DIR_PATH="/Users/poojashetty/Documents/AI_SS23/Project/NeuralImplicitRepr/NIR/AutoEncoder/Model/"

#logging
LOG_PATH="./tb_logs"

#Inference
# TEST_DATA="/Users/poojashetty/Documents/AI_SS23/Project/NeuralImplicitRepr/NIR/Testdata/img/Fust & Schoeffer Durandus Gotico-Antiqua 118G_a_44.png"
TEST_DATA="/Users/poojashetty/Documents/AI_SS23/Project/NeuralImplicitRepr/NIR/Testdata/img/Fust & Schoeffer Durandus Gotico-Antiqua 118G_E_14.png"
# TEST_DATA="/Users/poojashetty/Documents/AI_SS23/Project/NeuralImplicitRepr/NIR/Testdata/img/Fust & Schoeffer Durandus Gotico-Antiqua 118G_A_8.png"
# TEST_DATA = "/Users/poojashetty/Documents/AI_SS23/Project/NeuralImplicitRepr/NIR/Testdata/img/Fust & Schoeffer Durandus Gotico-Antiqua 118G_D_13.png"
# TEST_DATA = "/Users/poojashetty/Documents/AI_SS23/Project/NeuralImplicitRepr/NIR/Testdata/img/Fust & Schoeffer Durandus Gotico-Antiqua 118G_M_22.png"
GENERATED_SKEL="/Users/poojashetty/Documents/AI_SS23/Project/NeuralImplicitRepr/NIR/generated_skel.png"

