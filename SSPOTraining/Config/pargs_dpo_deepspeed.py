import argparse


def pargs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local rank for distributed training on GPUs(deepspeed required)")

    parser.add_argument('--data_file_name', type=str, default='dpo_glm4chat9b_th.json')
    parser.add_argument('--val_size', type=float, default=0.01)

    parser.add_argument('--ds_config', type=str, default='deepspeed_zero3_config.json')
    parser.add_argument('--initial_scale_power', type=int, default=12)

    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--target_modules', type=str, default="q_proj||k_proj||v_proj")
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--save_merged_model', action='store_true')

    # gradient_accumulation_steps is dynamically adjusted based on train_batch_size, train_micro_batch_size_per_gpu, and the number of Gpus used for training
    parser.add_argument('--template', type=str, default="qwen")
    parser.add_argument('--max_samples', type=int, default=350)
    parser.add_argument('--train_batch_size', type=int, default=64, help='total batch size')
    parser.add_argument('--train_micro_batch_size_per_gpu', type=int, default=1, help='batch size per gpu')
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--shuffle', action='store_true')

    parser.add_argument('--average_log_prob', action='store_true')
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--reference_free', action='store_true')
    parser.add_argument('--kl_penalty_weight', type=float, default=1e-4)
    parser.add_argument('--add_kl_penalty', action='store_true')

    parser.add_argument('--log_interval', type=int, default=4)
    parser.add_argument('--validate_interval', type=int, default=100)

    parser.add_argument('--model_path', type=str, default='/data/jovyan/work/yezang/TranslationPipeline/Models/llamafactory')
    parser.add_argument('--base_model', type=str, default='Qwen2.5-14B-Instruct/th/sft_default')
    parser.add_argument('--tag', type=str, default='default')

    args = parser.parse_args()
    return args
