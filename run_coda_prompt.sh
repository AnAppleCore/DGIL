# check the config file to see the exact number of GPUs used.
for dataset in "core50" "digitsdg" "digitsfive" "minidomainnet" "officehome" "office31" "officecaltech" "imageclef"; do
    for log_file_name in "coda_prompt" "coda_prompt_dgil"; do
        CUDA_VISIBLE_DEVICES=3 \
        python main.py \
            --config ./configs/DGIL/${dataset}/${log_file_name}.json
    done
done