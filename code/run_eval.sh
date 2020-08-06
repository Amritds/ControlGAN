working_dir=$(pwd)

cd $HOME
#export CUDA_VISIBLE_DEVICES=0
# Add the Nvidia drivers to the path
export PATH="/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"

# Tools config for CUDA, Anaconda installed in the common /tools directory
source .bashrc

# Set pytorch cache
export TORCH_MODEL_ZOO="/data2/adsue/pretrained"

# Activate your environment
source activate adsue

cd $working_dir

# Run the code. The -u option is used here to use unbuffered writes
# so that output is piped to the file as and when it is produced.
python -W ignore main.py --cfg cfg/eval_coco.yml --gpu 3&> ./out_ControlGANtrain &

