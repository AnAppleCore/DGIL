# check the config file to see the exact number of GPUs used.
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main.py \
    --config ./configs/DGIL/minidomainnet/der.json