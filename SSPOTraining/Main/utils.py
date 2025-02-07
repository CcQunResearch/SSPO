import random
import os
import os.path as osp
import json
import numpy as np
import deepspeed
import matplotlib.pyplot as plt
import torch
from torch.nn.utils.rnn import pad_sequence


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def pad_list(list, padding_value=0):
    tensor_list = [torch.tensor(d) for d in list]
    padded_tensor = pad_sequence(tensor_list, batch_first=True, padding_value=padding_value)
    return padded_tensor


def penalty(x_duration, y_duration, w1=0.8, w2=1.0):
    term1 = w1 * max(0, x_duration - y_duration)
    term2 = w2 * ((max(0, y_duration - x_duration) + 1) ** 2 - 1)
    return term1 + term2


def initialize_paths(task_name, dirname, args):
    save_name = task_name
    if task_name in ['dpo', 'ppo'] and args.lora:
        save_name += "-lora"
    else:
        save_name += "-full"
    save_path = osp.join(dirname, '..', "Save", save_name, f"{args.base_model.split('/')[0]}", args.tag)
    temp_save_path = osp.join(dirname, '..', "Save", save_name, f"TEMP-{args.base_model}")
    log_path = osp.join(save_path, 'training.log')
    total_data_path = osp.join(dirname, "..", "Data", task_name, args.data_file_name)
    train_data_path = osp.join(dirname, "..", "Data", task_name, f"train_{args.data_file_name}")
    val_data_path = osp.join(dirname, "..", "Data", task_name, f"val_{args.data_file_name}")
    base_model_path = osp.join(args.model_path, args.base_model)
    ds_config_path = osp.join(dirname, "..", "Config", args.ds_config)
    if not osp.exists(save_path):
        os.makedirs(save_path)
    return save_path, temp_save_path, log_path, total_data_path, train_data_path, val_data_path, base_model_path, ds_config_path


def plot_loss(train_loss, test_loss, save_path):
    plt.figure(figsize=(10, 6))
    if len(train_loss) > 0:
        train_steps, train_values = zip(*train_loss)
        plt.plot(train_steps, train_values, label='Train Loss', marker='o')
    if len(test_loss) > 0:
        test_steps, test_values = zip(*test_loss)
        plt.plot(test_steps, test_values, label='Test Loss', marker='o')

    plt.title('Training and Testing Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')

    plt.legend()
    plt.savefig(osp.join(save_path, "loss.png"))


def split_dataset(total_data_path, train_data_path, val_data_path, val_size, max_samples):
    total_data = json.load(open(total_data_path, "r", encoding="utf-8"))
    random.shuffle(total_data)
    total_data = total_data[:int(max_samples)]
    spilt_pos = int(len(total_data) * (1 - val_size))
    train_data, val_data = total_data[:spilt_pos], total_data[spilt_pos:]
    json.dump(train_data, open(train_data_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    json.dump(val_data, open(val_data_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)


def get_ds_config(ds_config_path, gradient_accumulation_steps, total_training_steps, args):
    ds_config = json.load(open(ds_config_path, "r", encoding="utf-8"))
    ds_config["train_batch_size"] = args.train_batch_size
    ds_config["train_micro_batch_size_per_gpu"] = args.train_micro_batch_size_per_gpu
    ds_config["gradient_accumulation_steps"] = gradient_accumulation_steps
    ds_config["fp16"]["initial_scale_power"] = args.initial_scale_power
    ds_config["optimizer"]["lr"] = args.lr
    ds_config["scheduler"]["params"]["warmup_max_lr"] = args.lr
    ds_config["scheduler"]["params"]["warmup_num_steps"] = args.warmup_ratio * total_training_steps
    return ds_config


def get_deepspeed_gpu_count():
    world_size = int(os.environ.get('WORLD_SIZE', 0))
    return world_size


def all_gather(tensor):
    group = deepspeed.comm.get_world_group()
    world_size = deepspeed.comm.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    deepspeed.comm.all_gather(gathered_tensors, tensor, group=group)
    return torch.cat(gathered_tensors, dim=0)


def all_gather_accelerate(accelerator, tensor):
    return accelerator.gather(tensor)

def formatted_dict(d):
    """Format a dictionary for printing."""
    return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}
