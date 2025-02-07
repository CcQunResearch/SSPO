import sys
import os
import os.path as osp
import warnings

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['MASTER_PORT'] = '61000'
dirname = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(dirname, ".."))

import logging
import shutil
import deepspeed
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from Config.pargs_sft_deepspeed import pargs
from Main.dataset import SFTDataset
from Main.collator import SFTCollator
from Main.utils import *


def save_checkpoint(model_engine, model, temp_save_path, hf_save_path):
    model_engine.save_checkpoint(temp_save_path)
    model_state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(temp_save_path)
    for key, value in model_state_dict.items():
        model_state_dict[key] = value.half()
    model.save_pretrained(hf_save_path, state_dict=model_state_dict, safe_serialization=True)
    tokenizer.save_pretrained(hf_save_path)


def validate(model_engine, val_dataloader):
    model_engine.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            outputs = model_engine(
                input_ids=batch["input_ids"].to(model_engine.device),
                attention_mask=batch["attention_mask"].to(model_engine.device),
                labels=batch["labels"].to(model_engine.device)
            )
            loss = outputs.loss
            valid_loss += loss.item()

    valid_loss /= len(val_dataloader)
    model_engine.train()
    return valid_loss


if __name__ == "__main__":
    args = pargs()
    seed_everything(args.seed)

    # define paths
    save_path, temp_save_path, log_path, total_data_path, train_data_path, val_data_path, base_model_path, ds_config_path = \
        initialize_paths('sft', dirname, args)

    # initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(name)s] [%(filename)s:%(lineno)s] [%(levelname)s] %(message)s',
        filename=log_path,
        filemode='w'
    )
    logger = logging.getLogger(__name__)

    # split training and validation sets
    logger.info("split training and validation sets")
    split_dataset(total_data_path, train_data_path, val_data_path, args.val_size, args.max_samples)

    # load tokenizer and model
    logger.info("load tokenizer and model")
    model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # create dataset and collator
    logger.info("create dataset and collator")
    train_dataset = SFTDataset(train_data_path, args.template)
    val_dataset = SFTDataset(val_data_path, args.template)
    logger.info(f"Training Dataset Size: {len(train_dataset)}, Validation Dataset Size: {len(val_dataset)}")
    collator = SFTCollator(tokenizer=tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=args.train_micro_batch_size_per_gpu, collate_fn=collator,
                                shuffle=False)

    # initialize deepspeed
    logger.info("initialize deepspeed")
    gpu_count = get_deepspeed_gpu_count()
    gradient_accumulation_steps = args.train_batch_size // (args.train_micro_batch_size_per_gpu * gpu_count)
    assert args.train_batch_size == args.train_micro_batch_size_per_gpu * gpu_count * gradient_accumulation_steps, \
        "train_batch_size must be divisible by train_micro_batch_size_per_gpu * gpu_count"
    total_training_steps = len(train_dataset) * args.num_epochs // args.train_batch_size
    ds_config = get_ds_config(ds_config_path, gradient_accumulation_steps, total_training_steps, args)
    model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=collator,
        config_params=ds_config
    )

    # training loop
    logger.info("start training")
    plot_train_loss = []
    plot_val_loss = []
    for epoch in range(1, args.num_epochs + 1):
        if model_engine.local_rank == 0:
            logger.info(f"epoch {epoch} start")
        model_engine.train()
        for batch in train_dataloader:
            outputs = model_engine(
                input_ids=batch["input_ids"].to(model_engine.device),
                attention_mask=batch["attention_mask"].to(model_engine.device),
                labels=batch["labels"].to(model_engine.device)
            )
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()

            if model_engine.micro_steps % (
                    args.log_interval * gradient_accumulation_steps) == 0 and model_engine.local_rank == 0:
                plot_train_loss.append((model_engine.global_steps, loss.item()))
                logger.info(
                    f"[Training] Epoch {epoch}, Step {model_engine.global_steps}, Train Loss: {loss.item()}")

            if model_engine.micro_steps % (args.validate_interval * gradient_accumulation_steps) == 0:
                valid_loss = validate(model_engine, val_dataloader)
                plot_val_loss.append((model_engine.global_steps, valid_loss))
                if model_engine.local_rank == 0:
                    logger.info(
                        f"[Validation] Epoch {epoch}, Step {model_engine.global_steps}, Val Loss: {valid_loss}")

        valid_loss = validate(model_engine, val_dataloader)
        if model_engine.local_rank == 0:
            logger.info(
                f"[EoE Validation] Epoch {epoch}, Step {model_engine.global_steps}, Val Loss: {valid_loss}")
        deepspeed.get_accelerator().empty_cache()

        if model_engine.local_rank == 0:
            logger.info('save model checkpoint')
        save_checkpoint(model_engine, model, temp_save_path, save_path)

    if model_engine.local_rank == 0:
        logger.info('save final model checkpoint')
    save_checkpoint(model_engine, model, temp_save_path, save_path)
    if osp.exists(temp_save_path) and model_engine.local_rank == 0:
        shutil.rmtree(temp_save_path)

    if model_engine.local_rank == 0:
        logger.info('plot loss')
        plot_loss(plot_train_loss, plot_val_loss, save_path)
