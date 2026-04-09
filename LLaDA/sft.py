import argparse
import os
import math
import shutil
import traceback
import logging
from typing import List

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_scheduler, set_seed
import deepspeed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class SFTDataset(Dataset):
    def __init__(self, txt_file_path: str):
        with open(txt_file_path, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]


def find_subsequence(main_list: List[int], sub_list: List[int]) -> int:
    len_sub = len(sub_list)
    for i in range(len(main_list) - len_sub + 1):
        if main_list[i : i + len_sub] == sub_list:
            return i
    return -1


def sft_data_collator(batch_of_strings: List[str], tokenizer: AutoTokenizer, max_seq_length: int):
    pad_token_id = tokenizer.pad_token_id

    try:
        start_id = tokenizer.convert_tokens_to_ids("<start_id>")
        end_id = tokenizer.convert_tokens_to_ids("<end_id>")
        assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)
        assistant_header_ids = [start_id] + assistant_ids + [end_id]
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        len_newline = len(newline_ids)
    except Exception:
        raise RuntimeError("Error: Tokenizer is missing the special tokens required for SFT (<start_id>, <end_id>).")

    processed = []
    for line in batch_of_strings:
        input_ids = tokenizer.encode(line, add_special_tokens=False)
        if len(input_ids) > max_seq_length:
            continue
        start_index = find_subsequence(input_ids, assistant_header_ids)
        if start_index != -1:
            prompt_length = start_index + len(assistant_header_ids)
            end_of_header_pos = prompt_length
            if (
                end_of_header_pos + len_newline <= len(input_ids)
                and input_ids[end_of_header_pos : end_of_header_pos + len_newline] == newline_ids
            ):
                prompt_length += len_newline
            processed.append(
                {"input_ids": torch.tensor(input_ids, dtype=torch.long), "prompt_length": prompt_length}
            )

    if not processed:
        return {
            "input_ids": torch.empty(0, max_seq_length, dtype=torch.long),
            "prompt_lengths": torch.empty(0, dtype=torch.long),
        }

    bs = len(processed)
    padded = torch.full((bs, max_seq_length), pad_token_id, dtype=torch.long)
    prompt_lengths = torch.empty(bs, dtype=torch.long)
    for i, item in enumerate(processed):
        ids, L = item["input_ids"], item["input_ids"].shape[0]
        padded[i, :L] = ids
        prompt_lengths[i] = item["prompt_length"]
    return {"input_ids": padded, "prompt_lengths": prompt_lengths}


def manage_checkpoints(output_dir, keep_last_n=-1, save_total_limit=-1):
    """
    Manage the number of checkpoints and delete old checkpoints to save space.
    
    Args:
        output_dir: Main output directory.
        keep_last_n: Keep the most recent N checkpoints; -1 means keep all.
        save_total_limit: Maximum number of checkpoints to keep; -1 means unlimited.
    """
    import glob
    
    ckpt_pattern = os.path.join(output_dir, "checkpoint-epoch-*")
    ckpt_dirs = glob.glob(ckpt_pattern)
    
    if not ckpt_dirs:
        return
  
    def extract_epoch_num(path):
        try:
            return int(os.path.basename(path).split('-')[-1])
        except:
            return 0
    
    ckpt_dirs.sort(key=extract_epoch_num)

    to_delete = []
    
    if keep_last_n > 0 and len(ckpt_dirs) > keep_last_n:
        to_delete.extend(ckpt_dirs[:-keep_last_n])
    
    if save_total_limit > 0 and len(ckpt_dirs) > save_total_limit:
        to_delete.extend(ckpt_dirs[:-save_total_limit])

    to_delete = list(set(to_delete))
    
    for ckpt_dir in to_delete:
        try:
            shutil.rmtree(ckpt_dir)
            print(f"Deleted old checkpoint: {ckpt_dir}")
        except Exception as e:
            print(f"Failed to delete checkpoint {ckpt_dir}: {e}")

def forward_process(input_ids: torch.Tensor, eps: float = 1e-3, mask_id: int = 126336):
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy_batch = torch.where(
        masked_indices,
        torch.tensor(mask_id, device=input_ids.device, dtype=torch.long),
        input_ids,
    )
    return noisy_batch, masked_indices, p_mask

def save_hf_checkpoint_zero3(model_engine, tokenizer, output_dir, extra_files=None, is_main=lambda: True):
    if dist.is_initialized():
        dist.barrier()

    try:
        model_engine.eval()  
        if hasattr(model_engine.module, "model") and hasattr(model_engine.module.model, "set_activation_checkpointing"):
            model_engine.module.model.set_activation_checkpointing(None)
        if getattr(model_engine.module, "gradient_checkpointing_disable", None):
            model_engine.module.gradient_checkpointing_disable()
    except Exception as e:
        if is_main():
            logger.warning(f"[Pre-save notice] A warning occurred while disabling checkpointing: {e}")


    state_dict_cpu = None
    with deepspeed.zero.GatheredParameters(model_engine.module.parameters(), modifier_rank=0):
        if is_main():
            state_dict = model_engine.module.state_dict()
            state_dict_cpu = {k: v.detach().cpu() for k, v in state_dict.items()}

    if is_main():
        os.makedirs(output_dir, exist_ok=True)
        model_engine.module.save_pretrained(output_dir, state_dict=state_dict_cpu, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)

        for file_name in (extra_files or []):
            src = os.path.join(model_engine.module.config._name_or_path, file_name)
            dst = os.path.join(output_dir, file_name)
            if os.path.exists(src):
                shutil.copyfile(src, dst)
                print(f"Copied custom code: {file_name}")

        print(f"Model has been saved to: {output_dir}")

    if dist.is_initialized():
        dist.barrier()

def main():
    parser = argparse.ArgumentParser(description="LLaDA full-parameter SFT")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_txt_path", type=str, required=True, help="Path to the preprocessed TXT data file")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--mask_eps", type=float, default=1e-3)
    parser.add_argument("--local_rank", type=int, default=-1)

    parser.add_argument(
        "--ckpt_strategy",
        type=str,
        default="off",
        choices=["off", "whole_layer", "one_in_two", "one_in_three", "one_in_four", "fine_grained"],
        help="LLaDA activation checkpointing strategy. 'off' means disabled."
    )
    
    parser.add_argument("--save_every_epoch", action="store_true", help="Whether to save a checkpoint after every epoch")
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=-1, help="Keep the most recent N checkpoints; -1 means keep all")
    parser.add_argument("--save_total_limit", type=int, default=-1, help="Maximum number of checkpoints to keep; -1 means unlimited")

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))

    if not torch.cuda.is_available():
        raise RuntimeError("No available GPU device was detected.")
    device_type = "cuda"
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
    deepspeed.init_distributed(dist_backend="nccl")

    def is_main_process():
        return dist.get_rank() == 0 if dist.is_initialized() else True

    def print_on_main(*a, **kw):
        if is_main_process():
            print(*a, **kw)
    
    set_seed(args.seed)
    MASK_TOKEN_ID = 126336

    print_on_main("[Script Start] SFT (DeepSpeed ZeRO-3)")
    if dist.is_initialized():
        print_on_main(f"Device: {device_type.upper()}:{args.local_rank} | world_size={dist.get_world_size()}")

    if dist.is_initialized():
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    try:
        model = AutoModel.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        print_on_main("`attn_implementation` set to `eager` for compatibility.")
    except TypeError:
        print_on_main("`attn_implementation` not supported, loading model without it.")
        model = AutoModel.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "eager"

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    
    if args.ckpt_strategy != "off":
        try:
            from configuration_llada import ActivationCheckpointingStrategy as ACS
            strat_map = {
                "whole_layer": ACS.whole_layer,
                "one_in_two": ACS.one_in_two,
                "one_in_three": ACS.one_in_three,
                "one_in_four": ACS.one_in_four,
                "fine_grained": ACS.fine_grained,
            }
            chosen_strategy = strat_map.get(args.ckpt_strategy)

            if hasattr(model, "model") and hasattr(model.model, "set_activation_checkpointing"):
                model.model.set_activation_checkpointing(chosen_strategy)
                print_on_main(f"Activation checkpointing: enabled with the LLaDA-specific strategy -> {args.ckpt_strategy}.")

            elif getattr(model, "gradient_checkpointing_enable", None):
                model.gradient_checkpointing_enable()
                print_on_main("Activation checkpointing: LLaDA interface not found; fell back to and enabled the HF generic strategy.")
            else:
                print_on_main("Failed to enable any checkpointing interface.")
        except ImportError:
            if getattr(model, "gradient_checkpointing_enable", None):
                model.gradient_checkpointing_enable()
                print_on_main("Activation checkpointing: LLaDA configuration not found; fell back to and enabled the HF generic strategy.")
            else:
                print_on_main("Failed to enable any checkpointing interface.")
        except Exception as e:
            print_on_main(f"Error while enabling checkpointing: {e}")
    else:
        print_on_main("Activation checkpointing: disabled (off).")


    if dist.is_initialized():
        dist.barrier()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    if dist.is_initialized():
        dist.barrier()
    dataset = SFTDataset(args.dataset_txt_path)
    if dist.is_initialized():
        dist.barrier()
    print_on_main(f"[Dataset] Loaded {len(dataset)} samples from {args.dataset_txt_path}.")

    data_collator = lambda data: sft_data_collator(data, tokenizer, args.max_seq_length)
    sampler = DistributedSampler(dataset) if dist.is_initialized() else None

    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=data_collator,
        drop_last=False,
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    num_training_steps = updates_per_epoch * args.num_train_epochs
    num_warmup_steps = max(1, int(num_training_steps * args.warmup_ratio))

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        dist_init_required=False,
    )

    print_on_main(
        f"[Training Config] epoch={args.num_train_epochs} | batch_per_device={args.per_device_train_batch_size} | "
        f"grad_accum={args.gradient_accumulation_steps} | updates/epoch={updates_per_epoch} | "
        f"total_updates={num_training_steps} | warmup_steps={num_warmup_steps}"
    )

    progress_bar = tqdm(range(num_training_steps), disable=not is_main_process())
    model_engine.train()

    for epoch in range(args.num_train_epochs):
        print_on_main(f"--- Epoch {epoch + 1} / {args.num_train_epochs} ---")
        if sampler:
            sampler.set_epoch(epoch)

        for step, batch in enumerate(dataloader):
            if batch["input_ids"].shape[0] == 0:
                continue

            input_ids = batch["input_ids"].to(model_engine.device)
            prompt_lengths = batch["prompt_lengths"].to(model_engine.device)

            noisy_batch, _, p_mask = forward_process(
                input_ids=input_ids, eps=args.mask_eps, mask_id=MASK_TOKEN_ID
            )

            token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(
                noisy_batch.size(0), noisy_batch.size(1)
            )
            prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
            noisy_batch[prompt_mask] = input_ids[prompt_mask]

            masked_indices = (noisy_batch == MASK_TOKEN_ID)

            outputs = model_engine(input_ids=noisy_batch)
            logits = outputs.logits

            pm_i64 = prompt_mask.to(torch.int64)
            answer_lengths = torch.sum((1 - pm_i64), dim=-1, keepdim=True)
            answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])

            if not masked_indices.any():
                ce_loss = logits.sum() * 0.0
            else:
                token_loss = F.cross_entropy(
                    logits[masked_indices],
                    input_ids[masked_indices],
                    reduction="none",
                ) / p_mask[masked_indices]
                ce_loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]

            if torch.isnan(ce_loss) or torch.isinf(ce_loss):
                model_engine.zero_grad(set_to_none=True)
                continue

            model_engine.backward(ce_loss)
            
            if model_engine.is_gradient_accumulation_boundary():
                progress_bar.update(1)
                try:
                    progress_bar.set_postfix(loss=f"{float(ce_loss):.4f}")
                except Exception:
                    pass
            model_engine.step()
        

        if args.save_every_epoch and (epoch + 1) >= 30:

            epoch_output_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch + 1}")
            print_on_main(f"\n[Epoch {epoch + 1} Save] Starting to save checkpoint to {epoch_output_dir}...")
            
            try:
                save_hf_checkpoint_zero3(
                    model_engine=model_engine,
                    tokenizer=tokenizer,
                    output_dir=epoch_output_dir,
                    extra_files=["configuration_llada.py", "modeling_llada.py", "generation_utils.py"],
                    is_main=is_main_process,
                )
                
                if is_main_process():
                    manage_checkpoints(
                        output_dir=args.output_dir,
                        keep_last_n=args.keep_last_n_checkpoints,
                        save_total_limit=args.save_total_limit
                    )
                    
            except Exception as e:
                if is_main_process():
                    print(f"!!!!!! Failed to save checkpoint for epoch {epoch + 1}: {e}")
                    traceback.print_exc()
            
            model_engine.train()

    final_output_dir = args.output_dir if not args.save_every_epoch else os.path.join(args.output_dir, "final-model")
    print_on_main(f"\n[Final Save] Training completed. Starting to save the final model to {final_output_dir}...")
    
    try:
        save_hf_checkpoint_zero3(
            model_engine=model_engine,
            tokenizer=tokenizer,
            output_dir=final_output_dir,
            extra_files=["configuration_llada.py", "modeling_llada.py", "generation_utils.py"],
            is_main=is_main_process,
        )
    except Exception as e:
        if is_main_process():
            print(f"!!!!!! An error occurred during the [Final Save] step: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
