# check the config file to see the exact number of GPUs used.
for dataset in "imageclef"
do
    for log_file_name in "dot_dgil" "dot"
    do
        CUDA_VISIBLE_DEVICES=0 \
        python main.py \
            --config ./configs/DGIL/${dataset}/${log_file_name}.json
    done
done