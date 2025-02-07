import argparse


def pargs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local rank for distributed training on GPUs(deepspeed required)")

    parser.add_argument('--data_file_name', type=str, default='sft.json')
    parser.add_argument('--val_size', type=float, default=0.1)

    parser.add_argument('--ds_config', type=str, default='deepspeed_zero3_config.json')
    parser.add_argument('--initial_scale_power', type=int, default=16)

    # gradient_accumulation_steps is dynamically adjusted based on train_batch_size, train_micro_batch_size_per_gpu, and the number of Gpus used for training
    parser.add_argument('--template', type=str, default="default")
    parser.add_argument('--max_samples', type=int, default=1e9)
    parser.add_argument('--train_batch_size', type=int, default=72, help='total batch size')
    parser.add_argument('--train_micro_batch_size_per_gpu', type=int, default=3, help='batch size per gpu')
    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)

    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--validate_interval', type=int, default=10)

    parser.add_argument('--model_path', type=str, default='/data/jovyan/workspace/yezang/DurationAlignment/Models')
    parser.add_argument('--base_model', type=str, default='Qwen2-7B')
    parser.add_argument('--tag', type=str, default='default')

    args = parser.parse_args()
    return args
