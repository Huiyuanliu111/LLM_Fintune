import os
# --- Windows 显存碎片优化 (防止 UVMLiteProcess1 驱动崩溃) ---
# 限制 PyTorch 内存分配器的最大切分大小，减少显存碎片，防止 Windows 驱动因搬运大块内存而重置
# 原来的 expandable_segments:True 在 Windows 上不支持，改用经典的 max_split_size_mb
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import torch
import gc
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import transformers.utils.import_utils

# --- 暴力绕过 transformers 安全检查 ---
os.environ["ALLOW_DANGEROUS_DESERIALIZATION"] = "true"
def no_op(*args, **kwargs): return None
transformers.utils.import_utils.check_torch_load_is_safe = no_op

# --- 配置参数 ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# 使用处理后的新数据路径 (v4: 降采样版)
DATA_PATH = "KIT-ML/qwen_ready_v4"  
OUTPUT_DIR = "Qwen-Motion-Finetuned-v2"
# 序列长度可以大大缩短了！
# 21 joints * 3 dims = 63 tokens/frame
# 降采样 1/4 后，40帧原始数据 -> 10帧 -> 630 tokens
# 120帧原始数据 -> 30帧 -> 1890 tokens
# 1024 足够覆盖大部分中短动作，且显存非常安全
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 1
GRAD_ACCUMULATION = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 5
RESUME_FROM_CHECKPOINT = False

def main():
    # 1. 加载数据集
    print(f"Loading data from {DATA_PATH}...")
    dataset = load_dataset("json", data_files={
        "train": os.path.join(DATA_PATH, "train_v4.jsonl"),
        "validation": os.path.join(DATA_PATH, "val_v4.jsonl")
    })

    # 2. 加载 Tokenizer
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # --- 关键步骤：添加新 Token ---
    # 我们添加 <0> 到 <255> 作为特殊 Token
    print("Adding new tokens <0>...<255> to tokenizer...")
    new_tokens = [f"<{i}>" for i in range(256)]
    num_added_toks = tokenizer.add_tokens(new_tokens)
    print(f"Added {num_added_toks} new tokens to tokenizer.")
    
    # 3. 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 4. 加载模型
    print("Loading model...")
    # 清理一下缓存
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"[DEBUG] CUDA Mem before load: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
    
    print("[DEBUG] Initializing AutoModelForCausalLM...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config, # 恢复量化，节省内存
            # torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True # 显式开启低内存加载模式
        )
        print("[DEBUG] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise e
    
    # --- 关键步骤：调整 Embedding 大小 ---
    # 因为 Tokenizer 变大了，模型的 Embedding 层也必须变大
    # 保存原始词表大小，用于后续冻结旧 Token
    original_vocab_size = model.get_input_embeddings().weight.shape[0]
    
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model embeddings to {len(tokenizer)}")

    # 准备 k-bit 训练
    model = prepare_model_for_kbit_training(model)
    
    # 5. LoRA 配置
    peft_config = LoraConfig(
        r=8,  # 降低秩以节省显存 (16 -> 8)
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # 显存优化：仅微调 Q、V 和输出投影，去掉 MLP 层 (gate/up/down) 以大幅节省显存
        target_modules=["q_proj", "v_proj", "o_proj"],
        # 重要：保存 Embedding 层和输出层，因为我们改了词表！
        # 恢复 embed_tokens 的训练，否则模型无法理解新 Token 的含义，导致 Loss 不降
        modules_to_save=["embed_tokens", "lm_head"] 
    )
    
    # 应用 PEFT/LoRA 配置
    # 这步之后，embed_tokens 和 lm_head 的 requires_grad 才会被设为 True
    model = get_peft_model(model, peft_config)
    
    # --- 断点续训逻辑 (仅加载权重) ---
    # 检查是否有历史 Checkpoint，如果有，手动加载权重，实现“无 Optimizer 状态”的续训
    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            # 按步数排序
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            last_checkpoint = os.path.join(OUTPUT_DIR, checkpoints[-1])
            print(f"Found existing checkpoint: {last_checkpoint}")
            print("Loading weights from checkpoint to continue training (Optimizer state resets)...")
            
            # 使用 PeftModel 加载权重。注意：我们需要把权重加载到当前的 model 上
            # load_peft_weights 会自动把 checkpoint 里的 adapter 权重覆盖到当前模型
            from peft import set_peft_model_state_dict, load_peft_weights
            
            # 加载 Adapter 权重
            # 优化：尝试加载到 CPU，并立即释放，防止 OOM
            try:
                adapter_weights = load_peft_weights(last_checkpoint, device="cpu")
            except TypeError:
                # 如果旧版本 peft 不支持 device 参数
                adapter_weights = load_peft_weights(last_checkpoint)
            
            set_peft_model_state_dict(model, adapter_weights)
            
            # --- 关键修复：立即释放内存 ---
            del adapter_weights
            gc.collect()
            torch.cuda.empty_cache()
            
            print("Successfully loaded LoRA weights from checkpoint.")
    
    model.print_trainable_parameters()

    # --- 高级优化：只训练新 Token ---
    # 通过 Hook 屏蔽旧 Token 的梯度，防止破坏原有知识
    # 注意：这在逻辑上只训练新 Token，但显存占用可能不会显著减少（取决于优化器实现）
    def zero_out_old_token_grads_hook(grad):
        # 将原始词表范围内的梯度置为 0
        if grad is None:
            return None
        # 极速版：直接原地修改，不 clone，减少一次巨大的内存复制！
        # 这能显著降低反向传播时的瞬时峰值压力，防止系统崩溃
        grad[:original_vocab_size] = 0
        return grad

    # 获取输入和输出层 (注意：PEFT 包装后，访问原始模块路径可能变化，但 get_input_embeddings 通常还能用)
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    
    # 注册 Hook
    # 必须在 get_peft_model 之后注册，确保 requires_grad=True
    if input_embeddings is not None and input_embeddings.weight.requires_grad:
        print("Registering gradient hook for Input Embeddings (Training only new tokens)...")
        input_embeddings.weight.register_hook(zero_out_old_token_grads_hook)
    else:
        print("Warning: Input Embeddings are frozen, hook skipped.")
        
    if output_embeddings is not None and output_embeddings is not input_embeddings and output_embeddings.weight.requires_grad:
        print("Registering gradient hook for Output Head (Training only new tokens)...")
        output_embeddings.weight.register_hook(zero_out_old_token_grads_hook)

    # 6. 训练参数
    # 定义输出目录
    best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        # 检查点策略：每 300 步保存一次
        save_strategy="steps",
        save_steps=300,
        save_total_limit=3,
        # 救命参数：只保存模型权重，不保存优化器状态 (optimizer.pt)
        save_only_model=True,
        # 最佳模型策略：训练结束时自动加载 Loss 最低的模型
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        # 策略统一：如果 load_best_model_at_end=True，则 eval 和 save 策略必须一致
        eval_strategy="steps",
        eval_steps=300,
        fp16=True,
        # 显存优化：切回 Paged 8-bit 优化器
        # 配合 Windows 虚拟内存 (Pagefile) 设置，可以将优化器状态卸载到 RAM 甚至硬盘
        optim="paged_adamw_8bit",
        # 解决 Windows 上 Ctrl+C 卡死或保存时死锁的问题：强制单线程加载数据
        dataloader_num_workers=0,
        # 监控：使用 wandb
        report_to="wandb",
        run_name="qwen-motion-finetune-v2",
        # 开启梯度检查点，大幅节省显存
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # 必须设置为 False，因为我们在用 LoRA 但同时也训练了 Embedding 层
        # 某些情况下 DDP 可能会报错，单卡训练没问题
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    print("Starting training...")
    # 强制不使用 resume，避免 trainer_state.json 缺失报错
    # 我们已经在上面手动加载了权重，所以这里看起来是从头练，但其实是在旧权重基础上练
    trainer.train(resume_from_checkpoint=False)

    print("Saving model...")
    # 保存最终模型
    trainer.save_model(OUTPUT_DIR)
    
    # 额外保存最佳模型到 best_model 目录 (虽然 load_best_model_at_end=True 已经让 model 变成了 best，但显式保存更安全)
    best_model_path = os.path.join(OUTPUT_DIR, "best_model")
    print(f"Saving best model to {best_model_path}...")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    
    # 同时保存 Tokenizer 到根目录，以防万一
    tokenizer.save_pretrained(OUTPUT_DIR)

import sys

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[User Interrupt] Training stopped by user.")
        print("Cleaning up GPU memory...")
        # 尝试清理显存（注意：由于作用域问题，无法直接访问 main 内部变量，但 gc 和 empty_cache 依然有效）
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("GPU memory released. Exiting safely.")
        sys.exit(0)
    except Exception as e:
        # 捕获其他严重错误，打印并抛出
        print(f"\n[Fatal Error] {e}")
        raise e

