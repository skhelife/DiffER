import sys
import argparse
import os
import math
import traceback
import logging
import json
from datetime import datetime
from itertools import cycle

from datasets import load_dataset
from torch.optim import AdamW
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_scheduler, set_seed
import deepspeed

# ---------- 1. Logging Setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------- LLaDA Core Algorithm (Final Entity-Aware Masking Implementation) ----------
MASK_TOKEN_ID = 126336


def load_and_tokenize_entities(file_path, tokenizer, is_main_process):
    """
    Before training starts, load all entities once and convert them into token ID lists.
    Return a preprocessed entity list where each element is (entity_name, token_ids).
    """
    if not os.path.exists(file_path):
        if is_main_process:
            logger.warning(f"Entity file '{file_path}' was not found. Falling back to standard random masking mode.")
        return []

    entities = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if name:
                # Key point: add a space before the name before tokenization to simulate the
                # tokenization behavior when it appears in the middle of a sentence.
                # This is the key trick to ensure correct token-sequence matching.
                token_ids = tokenizer(" " + name, add_special_tokens=False).input_ids
                if token_ids:  # Ensure the tokenization result is not empty
                    entities.append((name, token_ids))

    # For better efficiency, sort by entity token length in descending order and match longer entities first
    entities.sort(key=lambda x: len(x[1]), reverse=True)

    if is_main_process:
        logger.info(f"Successfully loaded and preprocessed {len(entities)} entities for entity-aware masking.")
    return entities


def find_all_subsequences(main_list, sub_list):
    """Find all occurrences of a subsequence in the main list."""
    sub_len = len(sub_list)
    if sub_len == 0:
        return []
    return [i for i in range(len(main_list) - sub_len + 1) if main_list[i:i+sub_len] == sub_list]


def entity_aware_forward_process(input_ids, entity_token_list, eps=1e-3):
    """
    [Final optimized version] Entity-aware LLaDA forward noising process.
    Completely based on token IDs, efficient and accurate.
    """
    b, l = input_ids.shape
    device = input_ids.device

    # 1. Generate the initial random mask
    t = torch.rand(b, device=device)
    p_mask = (1 - eps) * t + eps
    p_mask_expanded = p_mask[:, None].repeat(1, l)
    masked_indices = torch.rand((b, l), device=device) < p_mask_expanded

    # If no entity list is provided, directly return the result of standard random masking
    if not entity_token_list:
        noisy_batch = torch.where(masked_indices, MASK_TOKEN_ID, input_ids)
        return noisy_batch, masked_indices, p_mask_expanded

    # 2. Entity mask expansion
    # Iterate over each sample in the batch
    for i in range(b):
        sample_ids_list = input_ids[i].tolist()

        # Use a boolean array with the same length as the sample to mark entity positions
        # that have already been processed, preventing overlapping entities from being reprocessed
        processed_mask = [False] * l

        # Iterate through the sorted entity list (longer entities first)
        for _, token_ids in entity_token_list:

            # Find all positions where the current entity appears in the sample
            start_indices = find_all_subsequences(sample_ids_list, token_ids)

            for start in start_indices:
                end = start + len(token_ids)

                # Check whether this entity span has already been processed. If it has been
                # covered by a longer entity, skip it
                if any(processed_mask[start:end]):
                    continue

                # Check whether the initial mask hits any token of this entity
                if masked_indices[i, start:end].any():
                    # If it does, set all tokens of this entity to be masked
                    masked_indices[i, start:end] = True

                    # Mark this region as processed
                    processed_mask[start:end] = [True] * len(token_ids)

    # 3. Apply the final mask after entity expansion
    noisy_batch = torch.where(masked_indices, MASK_TOKEN_ID, input_ids)

    return noisy_batch, masked_indices, p_mask_expanded


# ---------- ZeRO-3 Save Function ----------
def save_hf_checkpoint_zero3(model_engine, tokenizer, output_dir, is_main):
    """Robust ZeRO-3 model saving function."""
    import deepspeed
    if dist.is_initialized():
        dist.barrier()

    try:
        model_engine.eval()
        core = getattr(model_engine.module, "model", model_engine.module)
        if hasattr(core, "set_activation_checkpointing"):
            core.set_activation_checkpointing(None)
        if getattr(model_engine.module, "gradient_checkpointing_disable", None):
            model_engine.module.gradient_checkpointing_disable()
    except Exception as e:
        logger.warning(f"[Pre-save notice] A non-critical warning occurred while disabling checkpointing: {e}")

    try:
        if hasattr(torch, "npu"):
            torch.npu.synchronize()
    except Exception:
        pass

    params = list(model_engine.module.parameters())
    gathered_sd = None
    try:
        with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
            if is_main:
                gathered_sd = model_engine.module.state_dict()
    except Exception as e:
        if is_main:
            logger.error(f"[Error] Failed to obtain state_dict during ZeRO-3 aggregation: {e}")
            traceback.print_exc()

    if dist.is_initialized():
        dist.barrier()

    if is_main:
        try:
            os.makedirs(output_dir, exist_ok=True)
            if gathered_sd is not None:
                model_engine.module.save_pretrained(
                    output_dir,
                    state_dict=gathered_sd,
                    safe_serialization=True
                )
            else:
                logger.warning("[Warning] Aggregated state_dict was not obtained; falling back to direct save_pretrained (may be unstable).")
                model_engine.module.save_pretrained(
                    output_dir,
                    safe_serialization=True
                )
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Model has been fully saved to: {output_dir}")
        except Exception as e:
            logger.error(f"[Error] save_pretrained failed while writing to disk: {e}")
            traceback.print_exc()

    if dist.is_initialized():
        dist.barrier()


# ---------- Main Flow ----------
def main():
    start_time = datetime.now()

    parser = argparse.ArgumentParser(description="LLaDA pretraining script (entity-aware masking version)")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--ckpt_strategy", type=str, default="whole_layer", choices=["off", "whole_layer", "one_in_two", "one_in_three", "one_in_four", "fine_grained"], help="LLaDA activation checkpointing strategy. 'off' means disabled.")
    parser.add_argument("--entity_list_path", type=str, default="entity_names.txt", help="Path to the entity list file.")
    # <<< Change: add a new parameter to control the number of steps per epoch
    parser.add_argument("--steps_per_epoch", type=int, default=1000, help="Number of training steps to force in each epoch.")

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))

    # --- Environment Detection and Distributed Setup ---
    is_npu = hasattr(torch, "npu") and torch.npu.is_available()
    is_cuda = torch.cuda.is_available()
    device_type = "npu" if is_npu else "cuda" if is_cuda else "cpu"
    if not device_type:
        raise RuntimeError("No available NPU or GPU device was detected.")

    if args.local_rank != -1:
        if is_cuda:
            torch.cuda.set_device(args.local_rank)
        if is_npu:
            torch.npu.set_device(args.local_rank)

    deepspeed.init_distributed(dist_backend="hccl" if is_npu else "nccl")

    is_main_process = dist.get_rank() == 0

    def log_on_main(*a, **kw):
        if is_main_process:
            logger.info(*a, **kw)

    set_seed(args.seed)
    log_on_main(f"[Training Start] Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_on_main(f"Device type: {device_type.upper()} | World Size: {dist.get_world_size()}")

    # ----- Model and Tokenizer -----
    log_on_main(f"[Model Loading] Starting to load model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # ----- Load and broadcast entity list -----
    entity_tokens = load_and_tokenize_entities(args.entity_list_path, tokenizer, is_main_process)

    if dist.is_initialized():
        broadcast_list = [entity_tokens]
        dist.broadcast_object_list(broadcast_list, src=0)
        entity_tokens = broadcast_list[0]
        dist.barrier()

    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # ---------- Activation Checkpointing Setup ----------
    try:
        # <<< Completion: you need to ensure that the 'configuration_llada.py' file exists in the project
        # This file should contain the ActivationCheckpointingStrategy enum class
        from configuration_llada import ActivationCheckpointingStrategy as ACS
        strat_map = {
            "off": None,
            "whole_layer": ACS.whole_layer,
            "one_in_two": ACS.one_in_two,
            "one_in_three": ACS.one_in_three,
            "one_in_four": ACS.one_in_four,
            "fine_grained": ACS.fine_grained,
        }
        chosen = strat_map.get(args.ckpt_strategy)
        if hasattr(model, "model") and hasattr(model.model, "set_activation_checkpointing"):
            model.model.set_activation_checkpointing(chosen)
            if chosen is None:
                log_on_main("Activation checkpointing: disabled (off).")
            else:
                log_on_main(f"Activation checkpointing: enabled -> {args.ckpt_strategy}.")
        else:
            log_on_main("Activation checkpointing: LLaDA interface not found, trying to fall back to the HF generic switch...")
            if args.ckpt_strategy != "off" and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                log_on_main("HF gradient checkpointing: enabled (compatibility fallback).")
            else:
                log_on_main("Failed to enable any checkpointing interface.")
    except ImportError:
        log_on_main("[Warning] Unable to import 'configuration_llada'. Please ensure this file exists. Activation checkpointing setup will be skipped.")
    except Exception as e:
        log_on_main(f"Error while enabling checkpointing: {e}")

    # ----- Dataset -----
    log_on_main(f"[Dataset Loading] Starting to load data: {args.dataset_name}")
    raw_datasets = load_dataset('text', data_files={'train': args.dataset_name})

    if not is_main_process:
        dist.barrier()
    log_on_main("[Dataset Preprocessing] Starting tokenization and grouping...")

    def tokenize_function(examples):
        return tokenizer(examples["text"], add_special_tokens=False)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, num_proc=os.cpu_count(), remove_columns=["text"])

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // args.block_size) * args.block_size
        result = {k: [t[i: i + args.block_size] for i in range(0, total_length, args.block_size)] for k, t in concatenated_examples.items()}
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)
    lm_datasets.set_format("torch")

    if is_main_process:
        dist.barrier()

    train_dataset = lm_datasets["train"]
    log_on_main(f"Grouping completed, number of training samples: {len(train_dataset)}")

    sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=args.per_device_train_batch_size, num_workers=args.num_workers, pin_memory=(device_type == "cuda"))

    # ----- DeepSpeed Initialization -----
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    grad_accum_steps = 1
    if args.deepspeed_config:
        log_on_main(f"Loading DeepSpeed configuration from {args.deepspeed_config}...")
        with open(args.deepspeed_config, 'r') as f:
            ds_config = json.load(f)
        grad_accum_steps = ds_config.get("gradient_accumulation_steps", 1)

    log_on_main(f"Gradient accumulation steps: {grad_accum_steps}")

    # <<< Change: calculate the total number of training steps according to steps_per_epoch
    if args.max_train_steps is None:
        num_training_steps = args.steps_per_epoch * args.num_train_epochs
    else:
        num_training_steps = args.max_train_steps

    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    model_engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, args=args, lr_scheduler=lr_scheduler, dist_init_required=False)

    # ----- Training Loop -----
    log_on_main(f"[Training Begin] Total global steps: {num_training_steps} | Steps per epoch: {args.steps_per_epoch}")
    progress_bar = tqdm(range(num_training_steps), disable=not is_main_process)
    completed_steps = 0
    model_engine.train()

    # <<< Change: create an infinite iterator from the dataloader
    data_iterator = cycle(train_dataloader)

    for epoch in range(args.num_train_epochs):
        log_on_main(f"--- Epoch {epoch + 1}/{args.num_train_epochs} ---")
        sampler.set_epoch(epoch)  # Key point: reshuffle the data order at the start of each epoch

        # <<< Change: the inner loop now strictly executes `steps_per_epoch` times
        for step in range(args.steps_per_epoch):
            if completed_steps >= num_training_steps:
                break

            # <<< Change: fetch the next batch of data from the infinite iterator
            try:
                batch = next(data_iterator)
            except StopIteration:
                # Using cycle usually will not trigger this exception; this is a safety safeguard
                data_iterator = cycle(train_dataloader)
                batch = next(data_iterator)

            input_ids = batch['input_ids'].to(model_engine.device)

            if torch.rand(1, device=input_ids.device) < 0.01:
                random_length = torch.randint(1, input_ids.shape[1] + 1, (1,), device=input_ids.device)
                input_ids = input_ids[:, :random_length]

            # [Core change] Call the entity-aware forward process
            noisy_batch, masked_indices, p_mask = entity_aware_forward_process(
                input_ids,
                entity_tokens
            )

            outputs = model_engine(input_ids=noisy_batch)
            logits = outputs.logits

            # The loss calculation and subsequent steps remain unchanged
            loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none')
            weighted_loss = loss / p_mask[masked_indices]
            final_loss = weighted_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])

            if torch.isnan(final_loss) or torch.isinf(final_loss):
                model_engine.zero_grad()
                log_on_main(f"Abnormal loss value (NaN/Inf) detected at step {completed_steps}; skipped.")
                continue

            model_engine.backward(final_loss)

            if model_engine.is_gradient_accumulation_boundary():
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{final_loss.item():.4f}")
                completed_steps += 1

            model_engine.step()

        if completed_steps >= num_training_steps:
            break

    # ----- Saving -----
    log_on_main("\n[Model Saving] Training completed. Starting to save safetensors weights...")
    save_hf_checkpoint_zero3(
        model_engine=model_engine,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        is_main=is_main_process
    )

    log_on_main("\n[Training Finished]")


if __name__ == "__main__":
    main()
