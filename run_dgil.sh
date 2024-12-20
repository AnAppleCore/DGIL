# check the config file to see the exact number of GPUs used.
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main.py \
    --config ./configs/DGIL/imageclef/replay_dgil.json

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main.py \
    --config ./configs/DGIL/officecaltech/replay_dgil.json

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main.py \
    --config ./configs/DGIL/office31/replay_dgil.json

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main.py \
    --config ./configs/DGIL/officehome/replay_dgil.json


# check the config file to see the exact number of GPUs used.
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main.py \
    --config ./configs/DGIL/imageclef/der_dgil.json

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main.py \
    --config ./configs/DGIL/officecaltech/der_dgil.json

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main.py \
    --config ./configs/DGIL/office31/der_dgil.json

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main.py \
    --config ./configs/DGIL/officehome/der_dgil.json


# check the config file to see the exact number of GPUs used.
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main.py \
    --config ./configs/DGIL/imageclef/finetune_dgil.json

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main.py \
    --config ./configs/DGIL/officecaltech/finetune_dgil.json

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main.py \
    --config ./configs/DGIL/office31/finetune_dgil.json

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main.py \
    --config ./configs/DGIL/officehome/finetune_dgil.json




# miniDomainNet
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main.py \
    --config ./configs/DGIL/minidomainnet/replay_dgil.json

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main.py \
    --config ./configs/DGIL/minidomainnet/der_dgil.json

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main.py \
    --config ./configs/DGIL/minidomainnet/finetune_dgil.json