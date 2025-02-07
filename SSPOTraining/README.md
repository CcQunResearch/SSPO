# SSPO

Source code for SSPO training in paper: 

**Fine-grained Video Dubbing Duration Alignment with Segment Supervised Preference Optimization**

## Run

The code can be run in the following command:

```shell script
cd Main
deepspeed --include localhost:0,1,2,3,4,5,6,7 deepspeed_dpo.py \
 --data_file_name dpo_en_demo.json \
 --model_path /path_2_models \
 --base_model /sft_model_name \
 --template default \
 --max_samples 600 \
 --train_batch_size 64 \
 --train_micro_batch_size_per_gpu 2 \
 --num_epochs 4 \
 --lr 4e-6 \
 --beta 0.5 \
 --tag $task_name \
 --lora --save_merged_model
```

## Dependencies

- [pytorch](https://github.com/pytorch/pytorch) == 2.2.2

- [transformers](https://github.com/huggingface/transformers) == 4.43.4