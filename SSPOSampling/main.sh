#!/bin/bash

if [ -d "temp/" ] && [ "$(ls -A temp/)" ]; then
    echo "Cleaning temp/ directory..."
    rm -rf temp/*
fi


# 八卡并行，每卡建议2条流程，若四卡，则把num_iterations换成8，即num_iterations=2*total_gpu_gum
k_num=0
processes_per_gpu=$(python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['dpo_sampling']['processes_per_gpu'])")
total_gpu_num=$(python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['dpo_sampling']['total_gpu_num'])")
first_gpu=$(python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['dpo_sampling']['first_gpu'])")
num_iterations=$((total_gpu_num*processes_per_gpu))
echo "First available GPU: $first_gpu"
echo "num_iterations: $num_iterations"
echo "processes_per_gpu: $processes_per_gpu"


mkdir -p log/test/

pids=()

for ((i=0; i<num_iterations; i++))
do
    cuda_num=$(((k_num / processes_per_gpu)+first_gpu))
    nohup python -u sample.py --k_num $k_num --cuda $cuda_num --all_k $num_iterations > log/test/log_k_$k_num.log 2>&1 &

    pids+=($!)
    k_num=$((k_num + 1))
done

for pid in "${pids[@]}"
do
    wait $pid
done

python process_data.py
