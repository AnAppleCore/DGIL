# check the config file to see the exact number of GPUs used.
for log_file_name in "dualprompt" "dualprompt_dgil" "dualprompt_dgil_v2"; do
    for dataset in "core50" "digitsdg" "digitsfive" "minidomainnet" "officehome" "office31" "officecaltech" "imageclef"; do
        CUDA_VISIBLE_DEVICES=1 \
        python main.py \
            --config ./configs/DGIL/${dataset}/${log_file_name}.json
    done
done