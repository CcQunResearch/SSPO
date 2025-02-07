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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model
from Config.pargs_dpo_deepspeed import pargs
from Main.dataset import DPODataset
from Main.collator import DPOCollator
from Main.utils import *


def save_checkpoint(model_engine, model, temp_save_path, hf_save_path, base_model_path, args):
    model_engine.save_checkpoint(temp_save_path)
    model_state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(temp_save_path)
    if args.lora and args.save_merged_model:
        model.save_pretrained(hf_save_path, state_dict=model_state_dict, safe_serialization=True)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True)
        peft_model = PeftModel.from_pretrained(base_model, hf_save_path)
        merged_model = peft_model.merge_and_unload()
        merged_model = merged_model.half()
        merged_model.save_pretrained(hf_save_path, safe_serialization=True)
    else:
        for key, value in model_state_dict.items():
            model_state_dict[key] = value.half()
        model.save_pretrained(hf_save_path, state_dict=model_state_dict, safe_serialization=True)
    tokenizer.save_pretrained(hf_save_path)


def validate(policy_engine, reference_engine, val_dataloader, args):
    policy_engine.eval()
    metrics = {"rewards of chosen": [], "rewards of rejected": [], "rewards accuracies": [], "rewards margins": [],
               "log probabilities of chosen": [], "log probabilities of rejected": [], "losses": []}
    if args.add_kl_penalty:
        metrics["kl losses"] = []
    with torch.no_grad():
        for batch in val_dataloader:
            if args.add_kl_penalty:
                losses, kl_losses, batch_metrics = get_batch_metrics(batch, policy_engine, reference_engine, args)
            else:
                losses, batch_metrics = get_batch_metrics(batch, policy_engine, reference_engine, args)
            for k, v in batch_metrics.items():
                metrics[k].extend(v)

    val_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
    policy_engine.train()
    return val_metrics


def get_batch_logps(logits, labels, average_log_prob=False):
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0
    labels = labels.to(torch.int64)
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def concatenated_forward(batch, model_engine, average_log_prob):
    outputs = model_engine(
        input_ids=batch["input_ids"].to(model_engine.device),
        attention_mask=batch["attention_mask"].to(model_engine.device)
    )
    batch_logits = outputs.logits.to(torch.float32)
    batch_logps = get_batch_logps(batch_logits, batch["labels"].to(model_engine.device),
                                  average_log_prob=average_log_prob)
    chosen_logps = batch_logps[:batch_logps.shape[0] // 2]
    rejected_logps = batch_logps[batch_logps.shape[0] // 2:]
    return chosen_logps, rejected_logps


def calculate_kl_penalty(batch, policy_engine, reference_engine, kl_penalty_weight):
    policy_complete_outputs = policy_engine(
        input_ids=batch["complete_input_ids"].to(policy_engine.device),
        attention_mask=batch["complete_attention_mask"].to(policy_engine.device)
    )
    policy_complete_logits = policy_complete_outputs.logits.to(torch.float32)  # [batch_size, seq_len, vocab_size]
    reference_complete_outputs = reference_engine(
        input_ids=batch["complete_input_ids"].to(reference_engine.device),
        attention_mask=batch["complete_attention_mask"].to(reference_engine.device)
    )
    reference_complete_logits = reference_complete_outputs.logits.to(torch.float32).to(policy_engine.device).detach()

    kl_divergence = F.kl_div(F.log_softmax(policy_complete_logits, dim=-1),
                             F.softmax(reference_complete_logits, dim=-1), reduction='none')  # [batch_size, seq_len]
    kl_divergence = kl_divergence[:, :-1, :]
    complete_labels = batch["complete_labels"].to(policy_engine.device)[:, 1:]  # [batch_size, seq_len]
    kl_mask = (complete_labels != -100).unsqueeze(-1).expand(*kl_divergence.shape).to(kl_divergence.device)
    kl_divergence = kl_divergence * kl_mask
    kl_losses = kl_penalty_weight * kl_divergence.sum(dim=(-2, -1))
    return kl_losses


def preference_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta,
                    label_smoothing=0.0, reference_free=False):
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps if not reference_free else 0

    # print(f"pi_logratios: {pi_logratios.tolist()}, ref_logratios: {ref_logratios.tolist()}")

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    # print('logits: ', logits.tolist())
    # print()

    # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
    losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def get_batch_metrics(batch, policy_engine, reference_engine, args):
    batch_metrics = {}

    policy_chosen_logps, policy_rejected_logps = concatenated_forward(batch, policy_engine, args.average_log_prob)
    with torch.no_grad():
        reference_chosen_logps, reference_rejected_logps = concatenated_forward(batch, reference_engine,
                                                                                args.average_log_prob)

    reference_chosen_logps = reference_chosen_logps.to(policy_engine.device)
    reference_rejected_logps = reference_rejected_logps.to(policy_engine.device)
    losses, chosen_rewards, rejected_rewards = preference_loss(policy_chosen_logps, policy_rejected_logps,
                                                               reference_chosen_logps, reference_rejected_logps,
                                                               args.beta, label_smoothing=args.label_smoothing,
                                                               reference_free=args.reference_free)

    all_chosen_rewards = all_gather(chosen_rewards)
    all_rejected_rewards = all_gather(rejected_rewards)
    all_policy_chosen_logps = all_gather(policy_chosen_logps)
    all_policy_rejected_logps = all_gather(policy_rejected_logps)
    all_losses = all_gather(losses)

    all_reward_accuracies = (all_chosen_rewards > all_rejected_rewards).float()
    all_rewards_margins = all_chosen_rewards - all_rejected_rewards

    batch_metrics["rewards of chosen"] = all_chosen_rewards.cpu().detach().numpy().tolist()
    batch_metrics["rewards of rejected"] = all_rejected_rewards.cpu().detach().numpy().tolist()
    batch_metrics["rewards accuracies"] = all_reward_accuracies.cpu().detach().numpy().tolist()
    batch_metrics["rewards margins"] = all_rewards_margins.cpu().detach().numpy().tolist()
    batch_metrics["log probabilities of chosen"] = all_policy_chosen_logps.cpu().detach().numpy().tolist()
    batch_metrics["log probabilities of rejected"] = all_policy_rejected_logps.cpu().detach().numpy().tolist()
    batch_metrics["losses"] = all_losses.cpu().detach().numpy().tolist()

    if args.add_kl_penalty:
        kl_losses = calculate_kl_penalty(batch, policy_engine, reference_engine, args.kl_penalty_weight)
        all_kl_losses = all_gather(kl_losses)
        batch_metrics["kl losses"] = all_kl_losses.cpu().detach().numpy().tolist()
        return losses, kl_losses, batch_metrics
    else:
        return losses, batch_metrics


if __name__ == "__main__":
    args = pargs()
    seed_everything(args.seed)

    # define paths
    save_path, temp_save_path, log_path, total_data_path, train_data_path, val_data_path, base_model_path, ds_config_path = \
        initialize_paths('dpo', dirname, args)

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
    policy_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True)
    reference_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # configure LoRA
    target_modules = args.target_modules.split("||") if "||" in args.target_modules else args.target_modules
    if args.lora:
        logger.info("configure LoRA")
        lora_config = LoraConfig(
            r=args.r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        policy_model = get_peft_model(policy_model, lora_config)

    # create dataset and collator
    logger.info("create dataset and collator")
    train_dataset = DPODataset(train_data_path, args.add_kl_penalty, args.template)
    val_dataset = DPODataset(val_data_path, args.add_kl_penalty, args.template)
    logger.info(f"Training Dataset Size: {len(train_dataset)}, Validation Dataset Size: {len(val_dataset)}")
    collator = DPOCollator(tokenizer=tokenizer, add_kl_penalty=args.add_kl_penalty)
    val_dataloader = DataLoader(val_dataset, batch_size=args.train_micro_batch_size_per_gpu, collate_fn=collator,
                                shuffle=False)

    # initialize deepspeed
    logger.info("initialize deepspeed")
    gpu_count = get_deepspeed_gpu_count()
    gradient_accumulation_steps = args.train_batch_size // (args.train_micro_batch_size_per_gpu * gpu_count)
    assert args.train_batch_size == args.train_micro_batch_size_per_gpu * gpu_count * gradient_accumulation_steps, \
        "train_batch_size must be divisible by train_micro_batch_size_per_gpu * gpu_count"
    total_training_steps = len(train_dataset) * args.num_epochs // args.train_batch_size
    print("total_training_steps:", total_training_steps)
    ds_config = get_ds_config(ds_config_path, gradient_accumulation_steps, total_training_steps, args)
    ds_config["dataloader_shuffle"] = args.shuffle
    policy_engine, optimizer, train_dataloader, _ = deepspeed.initialize(
        model=policy_model,
        model_parameters=policy_model.parameters(),
        training_data=train_dataset,
        collate_fn=collator,
        config_params=ds_config
    )
    reference_engine, _, _, _ = deepspeed.initialize(
        model=reference_model,
        model_parameters=reference_model.parameters(),
        collate_fn=collator,
        config_params=ds_config
    )

    # training loop
    logger.info("start training")
    reference_engine.eval()
    metrics = {"rewards of chosen": [], "rewards of rejected": [], "rewards accuracies": [], "rewards margins": [],
               "log probabilities of chosen": [], "log probabilities of rejected": [], "losses": []}
    if args.add_kl_penalty:
        metrics["kl losses"] = []
    plot_train_loss = []
    plot_val_loss = []
    for epoch in range(1, args.num_epochs + 1):
        if policy_engine.local_rank == 0:
            logger.info(f"epoch {epoch} start")
        policy_engine.train()
        for batch in train_dataloader:
            if args.add_kl_penalty:
                losses, kl_losses, batch_metrics = get_batch_metrics(batch, policy_engine, reference_engine, args)
                loss = losses.mean() + kl_losses.mean()
            else:
                losses, batch_metrics = get_batch_metrics(batch, policy_engine, reference_engine, args)
                loss = losses.mean()
            for k, v in batch_metrics.items():
                metrics[k].extend(v)
            policy_engine.backward(loss)
            policy_engine.step()
            del batch

            if policy_engine.micro_steps % (
                    args.log_interval * gradient_accumulation_steps) == 0 and policy_engine.local_rank == 0:
                mean_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
                plot_train_loss.append((policy_engine.global_steps, mean_metrics["losses"]))
                logger.info(
                    f"[Training] Epoch {epoch}, Step {policy_engine.global_steps}, Train Info: {formatted_dict(mean_metrics)}")

                for k, v in metrics.items():
                    metrics[k] = []

            if policy_engine.micro_steps % (args.validate_interval * gradient_accumulation_steps) == 0:
                val_metrics = validate(policy_engine, reference_engine, val_dataloader, args)
                plot_val_loss.append((policy_engine.global_steps, val_metrics["losses"]))
                if policy_engine.local_rank == 0:
                    logger.info(
                        f"[Validation] Epoch {epoch}, Step {policy_engine.global_steps}, Val Info: {formatted_dict(val_metrics)}")

        val_metrics = validate(policy_engine, reference_engine, val_dataloader, args)
        if policy_engine.local_rank == 0:
            logger.info(
                f"[EoE Validation] Epoch {epoch}, Step {policy_engine.global_steps}, Val Info: {formatted_dict(val_metrics)}")
        deepspeed.get_accelerator().empty_cache()

        if policy_engine.local_rank == 0:
            logger.info('save model checkpoint')
        save_checkpoint(policy_engine, policy_model, temp_save_path, save_path, base_model_path, args)

    if policy_engine.local_rank == 0:
        logger.info('save final model checkpoint')
    save_checkpoint(policy_engine, policy_model, temp_save_path, save_path, base_model_path, args)
    if osp.exists(temp_save_path) and policy_engine.local_rank == 0:
        shutil.rmtree(temp_save_path)

    if policy_engine.local_rank == 0:
        logger.info('plot loss')
        plot_loss(plot_train_loss, plot_val_loss, save_path)
