import argparse
import os
import math
import shutil
import traceback
import logging
import json
from datetime import datetime
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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)



MASK_TOKEN_ID = 126336

def forward_process(input_ids, eps=1e-3):
    """
    对输入进行随机掩盖（加噪声），LLaDA 预训练核心。
    此函数现在只负责加噪，不再处理随机截断。
    """
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy_batch = torch.where(masked_indices, MASK_TOKEN_ID, input_ids)
    return noisy_batch, masked_indices, p_mask


def save_hf_checkpoint_zero3(model_engine, tokenizer, output_dir, is_main):

    import os, traceback
    import torch
    import deepspeed
    import torch.distributed as dist

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
        logger.warning(f"[保存前提示] 关闭 checkpointing 出现可忽略警告: {e}")

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
            logger.error(f"[错误] 在 ZeRO-3 聚合阶段获取 state_dict 失败：{e}")
            traceback.print_exc()

    if dist.is_initialized():
        dist.barrier()

    if is_main:
        try:
            os.makedirs(output_dir, exist_ok=True)

            try:
                cfg = model_engine.module.config
                if not hasattr(cfg, "model_type"):
                    setattr(cfg, "model_type", "llada")
                if not getattr(cfg, "architectures", None):
                    setattr(cfg, "architectures", [type(model_engine.module).__name__])
                auto_map = dict(getattr(cfg, "auto_map", {}) or {})
                if "AutoModel" not in auto_map:
                    auto_map["AutoModel"] = f"modeling_llada.{type(model_engine.module).__name__}"
                    setattr(cfg, "auto_map", auto_map)
            except Exception:
                pass

            if gathered_sd is not None:
                model_engine.module.save_pretrained(
                    output_dir,
                    state_dict=gathered_sd,
                    safe_serialization=True
                )
            else:
                logger.warning("[警告] 未拿到聚合后的 state_dict，退回直接 save_pretrained（可能不稳定）。")
                model_engine.module.save_pretrained(
                    output_dir,
                    safe_serialization=True
                )

            tokenizer.save_pretrained(output_dir)
            logger.info(f"✅ 模型已完整保存到: {output_dir}")
        except Exception as e:
            logger.error(f"[错误] save_pretrained 写盘失败：{e}")
            traceback.print_exc()

    if dist.is_initialized():
        dist.barrier()


def main():
    start_time = datetime.now()

    parser = argparse.ArgumentParser(description="LLaDA 预训练脚本 (DeepSpeed 框架)")
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
    parser.add_argument("--ckpt_strategy",type=str,default="whole_layer",choices=["off", "whole_layer", "one_in_two", "one_in_three", "one_in_four", "fine_grained"],help="LLaDA 激活检查点策略。off 表示关闭。")

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))

    is_npu = hasattr(torch, "npu") and torch.npu.is_available()
    is_cuda = torch.cuda.is_available()
    if is_npu:
        device_type = "npu"
    elif is_cuda:
        device_type = "cuda"
    else:
        raise RuntimeError("未检测到任何可用的 NPU 或 GPU 设备。")

    if args.local_rank != -1:
        if device_type == "npu":
            torch.npu.set_device(args.local_rank)
        else:
            torch.cuda.set_device(args.local_rank)
    
    deepspeed.init_distributed(dist_backend="hccl" if is_npu else "nccl")
    
    is_main_process = dist.get_rank() == 0
    def log_on_main(*a, **kw):
        if is_main_process:
            logger.info(*a, **kw)

    set_seed(args.seed)
    log_on_main(f"【训练启动】时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_on_main(f"设备类型: {device_type.upper()} | World Size: {dist.get_world_size()}")

    log_on_main(f"【模型加载】开始加载模型: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    
    try:

        from configuration_llada import ActivationCheckpointingStrategy as ACS
        strat_map = {
            "off": None,
            "whole_layer": ACS.whole_layer,
            "one_in_two": ACS.one_in_two,
            "one_in_three": ACS.one_in_three,
            "one_in_four": ACS.one_in_four,
            "fine_grained": ACS.fine_grained,
    }
        chosen = strat_map[args.ckpt_strategy]
        if hasattr(model, "model") and hasattr(model.model, "set_activation_checkpointing"):
            model.model.set_activation_checkpointing(chosen)
            if chosen is None:
                log_on_main("Activation checkpointing: 已关闭 (off).")
            else:
                log_on_main(f"Activation checkpointing: 已启用 -> {args.ckpt_strategy}.")
        else:
            log_on_main("Activation checkpointing: 未找到 LLaDA 接口，尝试退回 HF 通用开关…")
            if getattr(model, "gradient_checkpointing_enable", None):
                model.gradient_checkpointing_enable()
                log_on_main("HF gradient checkpointing: 已启用（兼容回退）。")
            else:
                log_on_main("未能启用任何 checkpointing 接口。")
    except Exception as e:
        log_on_main(f"启用 checkpointing 时出错：{e}")

    log_on_main(f"【数据集加载】开始加载数据: {args.dataset_name}")
    raw_datasets = load_dataset('text', data_files={'train': args.dataset_name})

    if args.local_rank != 0:
        dist.barrier()
    log_on_main("【数据集预处理】开始分词和分组...")

    def tokenize_function(examples):
        return tokenizer(examples["text"], add_special_tokens=False)

    tokenized_datasets = raw_datasets.map(
        tokenize_function, batched=True, num_proc=os.cpu_count(), remove_columns=["text"]
    )

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // args.block_size) * args.block_size
        result = {
            k: [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)
    lm_datasets.set_format("torch")
    
    if args.local_rank == 0:
        dist.barrier()
    
    train_dataset = lm_datasets["train"]
    log_on_main(f"分组完成，训练集样本数: {len(train_dataset)}")
    
    sampler = DistributedSampler(train_dataset)
    pin_mem = device_type == "cuda"
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    grad_accum_steps = 1
    if args.deepspeed_config:
        log_on_main(f"从 {args.deepspeed_config} 加载 DeepSpeed 配置...")
        with open(args.deepspeed_config, 'r') as f:
            ds_config = json.load(f)
        grad_accum_steps = ds_config.get("gradient_accumulation_steps", 1)
    
    log_on_main(f"梯度累积步数 (Gradient Accumulation Steps): {grad_accum_steps}")

    updates_per_epoch = math.ceil(len(train_dataloader) / grad_accum_steps)
    if args.max_train_steps is None:
        num_training_steps = updates_per_epoch * args.num_train_epochs
    else:
        num_training_steps = args.max_train_steps

    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        dist_init_required=False,
    )
    
    log_on_main(f"【训练开始】全局总步数: {num_training_steps}")
    progress_bar = tqdm(range(num_training_steps), disable=not is_main_process)
    completed_steps = 0
    model_engine.train()

    for epoch in range(args.num_train_epochs):
        log_on_main(f"--- 第 {epoch + 1}/{args.num_train_epochs} 轮 ---")
        sampler.set_epoch(epoch)
        
        for step, batch in enumerate(train_dataloader):
            if completed_steps >= num_training_steps:
                break
            
            input_ids = batch['input_ids'].to(model_engine.device)
            
            if torch.rand(1, device=input_ids.device) < 0.01:
                random_length = torch.randint(1, input_ids.shape[1] + 1, (1,), device=input_ids.device)
                input_ids = input_ids[:, :random_length]

   
            noisy_batch, masked_indices, p_mask = forward_process(input_ids)

            outputs = model_engine(input_ids=noisy_batch)
            logits = outputs.logits

            loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none')
            weighted_loss = loss / p_mask[masked_indices]
            final_loss = weighted_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
            
            if torch.isnan(final_loss) or torch.isinf(final_loss):
                model_engine.zero_grad()
                log_on_main(f"检测到异常损失值 (NaN/Inf) 在 step {completed_steps}，已跳过。")
                continue

            model_engine.backward(final_loss)
            
            if model_engine.is_gradient_accumulation_boundary():
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{final_loss.item():.4f}")
                completed_steps += 1
                
            model_engine.step()

        if completed_steps >= num_training_steps:
            break

    log_on_main("\n【模型保存】训练完成，开始保存 safetensors 权重...")
    try:
        save_hf_checkpoint_zero3(
            model_engine=model_engine,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            is_main=is_main_process
        )
    except Exception:
        if is_main_process:
            logger.error(f"!!!!!! 在【模型保存】步骤发生错误 !!!!!!")
            traceback.print_exc()

    log_on_main(f"\n【训练结束】")

if __name__ == "__main__":
    main()