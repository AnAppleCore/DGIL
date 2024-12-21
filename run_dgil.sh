# check the config file to see the exact number of GPUs used.
CUDA_VISIBLE_DEVICES=1 \
python main.py \
    --config ./configs/DGIL/officecaltech/l2p_dgil.json