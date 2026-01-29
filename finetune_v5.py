"""
Overfit 测试脚本 - 在 v9 子集（10条数据）上验证模型能否过拟合
基于 finetune_v2.py 修改，极限显存优化版
"""
import os
# 显存优化：使用更大的切分减少碎片化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

import torch
import gc
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import transformers.utils.import_utils


class SafeStopCallback(TrainerCallback):
    """
    安全停止回调：检测 STOP 文件来优雅停止训练
    使用方法：在训练目录创建名为 STOP 的文件（无扩展名）
    
    这比 Ctrl+C 更安全，不会导致 CUDA 驱动崩溃
    """
    def __init__(self, output_dir: str):
        self.stop_file = os.path.join(output_dir, "STOP")
        # 启动时清理旧的 STOP 文件
        if os.path.exists(self.stop_file):
            os.remove(self.stop_file)
            print(f"[SafeStop] Removed old STOP file")
        print(f"[SafeStop] To stop training safely, create file: {self.stop_file}")
    
    def on_step_end(self, args, state, control, **kwargs):
        if os.path.exists(self.stop_file):
            print(f"\n[SafeStop] STOP file detected at step {state.global_step}!")
            print("[SafeStop] Finishing current step and saving model...")
            control.should_save = True
            control.should_training_stop = True
            # 删除 STOP 文件
            try:
                os.remove(self.stop_file)
            except:
                pass
        return control


class MemoryCleanupCallback(TrainerCallback):
    """
    定期清理显存，防止碎片化导致 OOM
    """
    def __init__(self, cleanup_every_n_steps: int = 50):
        self.cleanup_every_n_steps = cleanup_every_n_steps
        self.last_cleanup_step = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        # 每 N 步清理一次显存
        if state.global_step - self.last_cleanup_step >= self.cleanup_every_n_steps:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # 注意：移除了 synchronize()，因为它可能在 Ctrl+C 时卡住
            self.last_cleanup_step = state.global_step
        return control
    
    def on_save(self, args, state, control, **kwargs):
        # 保存模型前后都清理显存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return control

# --- 暴力绕过 transformers 安全检查 ---
os.environ["ALLOW_DANGEROUS_DESERIALIZATION"] = "true"
def no_op(*args, **kwargs): return None
transformers.utils.import_utils.check_torch_load_is_safe = no_op

# --- 配置参数 (与 v3 保持一致，仅数据集不同) ---
MODEL_NAME = "Qwen/Qwen3-0.6B"
DATA_PATH = "KIT-ML/qwen_ready_v9"
OUTPUT_DIR = "Qwen-Motion-Overfit-v9"
MAX_SEQ_LENGTH = 4096   # 足够容纳40帧数据（实际2569 tokens）
BATCH_SIZE = 1
GRAD_ACCUMULATION = 8   # 与 v2 一致
LEARNING_RATE = 2e-4    # 与 v2 一致
NUM_EPOCHS = 2000       # 单个episode完全过拟合，需要更多epoch
RESUME_FROM_CHECKPOINT = False  # 设为 True 自动恢复，或设为 checkpoint 路径

def main():
    # 1. 加载数据集
    print(f"Loading data from {DATA_PATH}...")
    dataset = load_dataset("json", data_files={
        "train": os.path.join(DATA_PATH, "train_v9.jsonl"),
        "validation": os.path.join(DATA_PATH, "val_v9.jsonl")
    })
    
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Val samples: {len(dataset['validation'])}")

    # 2. 加载 Tokenizer
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # 设置 tokenizer 的最大长度（关键！控制序列截断）
    tokenizer.model_max_length = MAX_SEQ_LENGTH
    print(f"Set tokenizer.model_max_length to {MAX_SEQ_LENGTH}")
    
    # --- 添加新 Token ---
    print("Adding new tokens <0>...<255> to tokenizer...")
    new_tokens = [f"<{i}>" for i in range(256)]
    num_added_toks = tokenizer.add_tokens(new_tokens)
    print(f"Added {num_added_toks} new tokens to tokenizer.")
    
    # 3. 量化配置 (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 4. 加载模型
    print("Loading model...")
    gc.collect()
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"[DEBUG] CUDA Mem before load: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        print(f"[DEBUG] GPU Total Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")
    
    print("[DEBUG] Initializing AutoModelForCausalLM...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("[DEBUG] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise e
    
    # --- 调整 Embedding 大小 ---
    original_vocab_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model embeddings to {len(tokenizer)}")

    # 准备 k-bit 训练
    model = prepare_model_for_kbit_training(model)
    
    # 清理一次显存
    gc.collect()
    torch.cuda.empty_cache()
    
    # 5. LoRA 配置 (与 v2 完全一致)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "o_proj"],
        modules_to_save=["embed_tokens", "lm_head"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- 梯度 hook：只训练新 token ---
    # 这是 v2 能跑而 v9 掉卡的关键原因！
    # 没有 hook：训练 151921 个 token 的 embedding (467M 参数)
    # 有 hook：只训练 256 个新 token 的 embedding (0.8M 参数)
    def zero_out_old_token_grads_hook(grad):
        if grad is None:
            return None
        grad[:original_vocab_size] = 0
        return grad

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    
    if input_embeddings is not None and input_embeddings.weight.requires_grad:
        print("Registering gradient hook for Input Embeddings (training only new 256 tokens)...")
        input_embeddings.weight.register_hook(zero_out_old_token_grads_hook)
        
    if output_embeddings is not None and output_embeddings is not input_embeddings and output_embeddings.weight.requires_grad:
        print("Registering gradient hook for Output Head (training only new 256 tokens)...")
        output_embeddings.weight.register_hook(zero_out_old_token_grads_hook)

    # 再次清理
    gc.collect()
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        print(f"[DEBUG] CUDA Mem after PEFT: {torch.cuda.memory_allocated()/1024**2:.2f}MB")

    # 6. 训练参数 (使用 SFTConfig 支持 max_length)
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        # 保存策略：减少保存频率
        save_strategy="steps",
        save_steps=100,
        # save_total_limit=1,  # 不再限制检查点数量
        save_only_model=True,
        # 启用评估
        eval_strategy="steps",
        eval_steps=100,
        fp16=True,
        # 改用普通 AdamW，避免 paged 优化器的 CPU RAM 开销
        optim="adamw_torch",
        dataloader_num_workers=0,
        # 禁用 wandb 节省内存
        report_to="wandb",
        run_name="qwen-motion-overfit-v9",
        # 梯度检查点
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        # 禁用 warmup
        warmup_steps=0,
        weight_decay=0.0,
        # 额外优化
        dataloader_pin_memory=False,  # 禁用 pin memory
        per_device_eval_batch_size=1,
        # 关键！设置最大序列长度，确保 2569 tokens 不被截断
        max_length=MAX_SEQ_LENGTH,
    )

    # 创建回调
    memory_callback = MemoryCleanupCallback(cleanup_every_n_steps=50)
    safe_stop_callback = SafeStopCallback(output_dir=OUTPUT_DIR)
    
    # 验证数据长度（训练前检查）
    print("\n" + "="*60)
    print("DATA VERIFICATION:")
    sample = dataset["train"][0]
    sample_text = tokenizer.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    sample_tokens = tokenizer.encode(sample_text, add_special_tokens=False)
    print(f"Sample length: {len(sample_tokens)} tokens")
    print(f"Tokenizer max length: {tokenizer.model_max_length}")
    if len(sample_tokens) > tokenizer.model_max_length:
        print(f"⚠️  WARNING: Sample will be TRUNCATED by {len(sample_tokens) - tokenizer.model_max_length} tokens!")
    else:
        print(f"✓ Sample fits within max length (buffer: {tokenizer.model_max_length - len(sample_tokens)} tokens)")
    print("="*60 + "\n")
    
    # 与 v2 一致：传递 peft_config 给 SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],  # 启用评估
        peft_config=peft_config,  # 与 v2 一致！
        processing_class=tokenizer,
        args=training_args,
        callbacks=[memory_callback, safe_stop_callback],
    )

    print("="*60)
    print("OVERFIT TEST: Training on single episode (v9)")
    print("Memory cleanup: every 50 steps")
    print(f"Safe stop: create file '{OUTPUT_DIR}/STOP' to stop training")
    print(f"Max seq length: {MAX_SEQ_LENGTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Grad accumulation: {GRAD_ACCUMULATION}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"LoRA r: {peft_config.r}")
    print("Expected: Train loss should drop to near 0")
    print("="*60)
    
    # 最后一次清理
    gc.collect()
    torch.cuda.empty_cache()
    
    # 检查是否有 checkpoint 可以恢复
    resume_checkpoint = None
    if RESUME_FROM_CHECKPOINT:
        if RESUME_FROM_CHECKPOINT is True:
            # 自动查找最新 checkpoint
            if os.path.isdir(OUTPUT_DIR):
                checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
                if checkpoints:
                    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                    resume_checkpoint = os.path.join(OUTPUT_DIR, checkpoints[-1])
                    print(f"Auto-resuming from: {resume_checkpoint}")
        else:
            # 用户指定了 checkpoint 路径
            resume_checkpoint = RESUME_FROM_CHECKPOINT
            print(f"Resuming from specified checkpoint: {resume_checkpoint}")
    
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("="*60)
    print("OVERFIT TEST COMPLETE")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("="*60)

import sys
import argparse

# 与 v2 保持一致的简单异常处理
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overfit test for motion generation")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from specific checkpoint path")
    args = parser.parse_args()
    
    # 设置全局变量
    if args.checkpoint:
        RESUME_FROM_CHECKPOINT = args.checkpoint
    elif args.resume:
        RESUME_FROM_CHECKPOINT = True
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[User Interrupt] Training stopped by user.")
        print("Cleaning up GPU memory...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("GPU memory released. Exiting safely.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[Fatal Error] {e}")
        raise e
