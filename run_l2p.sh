# check the config file to see the exact number of GPUs used.
for dataset in "core50" "digitsdg" "digitsfive" "minidomainnet" "officehome" "office31" "officecaltech" "imageclef"; do
    for log_file_name in "l2p" "l2p_dgil"; do
        CUDA_VISIBLE_DEVICES=0 \
        python main.py \
            --config ./configs/DGIL/${dataset}/${log_file_name}.json
    done
done