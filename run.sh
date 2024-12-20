# check the config file to see the exact number of GPUs used.
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main.py \
    --config ./configs/DGIL/imageclef/replay.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main.py \
    --config ./configs/DGIL/officecaltech/replay.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main.py \
    --config ./configs/DGIL/office31/replay.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main.py \
    --config ./configs/DGIL/officehome/replay.json


# check the config file to see the exact number of GPUs used.
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main.py \
    --config ./configs/DGIL/imageclef/der.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main.py \
    --config ./configs/DGIL/officecaltech/der.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main.py \
    --config ./configs/DGIL/office31/der.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main.py \
    --config ./configs/DGIL/officehome/der.json


# check the config file to see the exact number of GPUs used.
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main.py \
    --config ./configs/DGIL/imageclef/finetune.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main.py \
    --config ./configs/DGIL/officecaltech/finetune.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main.py \
    --config ./configs/DGIL/office31/finetune.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main.py \
    --config ./configs/DGIL/officehome/finetune.json




# miniDomainNet
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main.py \
    --config ./configs/DGIL/minidomainnet/replay.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main.py \
    --config ./configs/DGIL/minidomainnet/der.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main.py \
    --config ./configs/DGIL/minidomainnet/finetune.json