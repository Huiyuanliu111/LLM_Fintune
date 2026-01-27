import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import argparse
import sys
import os
import numpy as np
from visualize_motion import parse_motion_tokens, plot_motion

# Constants - 全参数微调版本
MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_PATH = "Qwen3-Motion-Finetuned-v3/best_model"
MAX_NEW_TOKENS = 512  # 缩短以节省显存
# 反量化参数
BINS = 256
V_MIN = -3000.0
V_MAX = 3000.0

def dequantize(motion_data, bins=BINS, v_min=V_MIN, v_max=V_MAX):
    """
    反量化：将 [0, bins-1] 的整数映射回 [v_min, v_max]
    """
    norm = motion_data.astype(float) / (bins - 1)
    v_range = v_max - v_min
    denorm_data = (norm * v_range) + v_min
    return denorm_data

def load_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # --- 添加新 Token ---
    print("Adding new tokens <0>...<255> to tokenizer...")
    new_tokens = [f"<{i}>" for i in range(256)]
    tokenizer.add_tokens(new_tokens)
    print(f"Added {len(new_tokens)} tokens.")

    print("Loading fine-tuned model...")
    # 全参数微调：直接加载完整的微调模型
    print("Loading model in bfloat16 for inference...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    model.eval()
    return model, tokenizer

def generate_motion(model, tokenizer, text_prompt):
    messages = [
        {"role": "system", "content": "You are a motion generation assistant. Generate motion sequences as integer tokens based on text descriptions."},
        {"role": "user", "content": text_prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 初始化 Streamer
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print(f"Generating motion for: '{text_prompt}'...")
    print("--- Stream Output Start ---")
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # Greedy decoding for determinism
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer
        )
    print("\n--- Stream Output End ---")

    # Extract only the newly generated tokens
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def main():
    parser = argparse.ArgumentParser(description="Generate motion sequences from text descriptions using Qwen3 full-finetuned model.")
    parser.add_argument("prompt", type=str, help="Text description of the motion (e.g., 'A person walks forward')")

    # 确保输出目录存在
    output_dir = "Generation_v3"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    default_output = os.path.join(output_dir, "generated_motion.gif")
    parser.add_argument("--output", type=str, default=default_output,
                       help=f"Output filename for the visualization (default: {default_output})")
    parser.add_argument("--no_viz", action="store_true", help="Skip visualization, only print tokens")

    args = parser.parse_args()

    model, tokenizer = load_model()

    motion_tokens_str = generate_motion(model, tokenizer, args.prompt)

    print("\nGenerated Tokens:")
    print(motion_tokens_str[:200] + "..." if len(motion_tokens_str) > 200 else motion_tokens_str)

    if not args.no_viz:
        try:
            # Clean up the string to ensure only numbers are passed
            import re
            tokens = re.findall(r'\d+', motion_tokens_str)
            clean_input = " ".join(tokens)

            if not clean_input:
                print("No valid motion tokens generated.")
                return

            motion_data = parse_motion_tokens(clean_input)
            print(f"Parsed motion data shape: {motion_data.shape}")

            # Denormalization (Dequantization)
            print("Applying dequantization...")
            motion_data = dequantize(motion_data)

            if motion_data.shape[0] > 0:
                # Save the motion data for analysis
                npy_output = args.output.replace('.gif', '.npy')
                if npy_output == args.output:
                    npy_output = args.output + ".npy"
                np.save(npy_output, motion_data)
                print(f"Motion data saved to {npy_output}")

                plot_motion(motion_data, output_file=args.output)
                print(f"Motion visualization saved to {args.output}")
            else:
                print("Generated sequence resulted in empty motion data.")
        except Exception as e:
            print(f"Error visualizing motion: {e}")
    else:
        # Even if no_viz is True, we might want to save the numpy array if possible
        try:
            import re
            tokens = re.findall(r'\d+', motion_tokens_str)
            clean_input = " ".join(tokens)
            if clean_input:
                motion_data = parse_motion_tokens(clean_input)
                # Denormalization (Dequantization)
                motion_data = dequantize(motion_data)
                if motion_data.shape[0] > 0:
                    npy_output = args.output.replace('.gif', '.npy')
                    if npy_output == args.output:
                        npy_output = args.output + ".npy"
                    np.save(npy_output, motion_data)
                    print(f"Motion data saved to {npy_output} (Visualization skipped)")
        except:
            pass

if __name__ == "__main__":
    main()
