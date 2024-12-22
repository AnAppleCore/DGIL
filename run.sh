# check the config file to see the exact number of GPUs used.
for log_file_name in "l2p" "l2p_dgil" "l2p_dgil_v2"; do
    for dataset in "imageclef" "office31" "officehome" "minidomainnet"; do # "officecaltech" 
        CUDA_VISIBLE_DEVICES=0 \
        python main.py \
            --config ./configs/DGIL/${dataset}/${log_file_name}.json
    done
done