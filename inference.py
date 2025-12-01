import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel
import argparse
import sys
import os
import numpy as np
from visualize_motion import parse_motion_tokens, plot_motion

# Constants
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "Qwen-Motion-Finetuned"
MAX_NEW_TOKENS = 4096 # Adjust based on your needs
# 去掉 MEAN/STD，改用反量化参数
BINS = 256
# 假设的物理范围，KIT-ML 通常单位是毫米，范围大致在 -3000 到 3000 左右
# 如果发现动作幅度太大或太小，可以调整这两个值
V_MIN = -3000.0
V_MAX = 3000.0

def dequantize(motion_data, bins=BINS, v_min=V_MIN, v_max=V_MAX):
    """
    反量化：将 [0, bins-1] 的整数映射回 [v_min, v_max]
    """
    # motion_data 是整数 token
    norm = motion_data.astype(float) / (bins - 1)
    # 映射回物理范围
    v_range = v_max - v_min
    denorm_data = (norm * v_range) + v_min
    return denorm_data

def load_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    
    print("Loading base model...")
    # Use 4-bit quantization if possible to match training environment, or load in full precision for inference
    # Here we try to load it in a way that fits in memory. 
    try:
        # from transformers import BitsAndBytesConfig
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.float16,
        # )
        # model = AutoModelForCausalLM.from_pretrained(
        #     BASE_MODEL_NAME,
        #     quantization_config=quantization_config,
        #     device_map="auto",
        #     trust_remote_code=True
        # )
        # 如果显存足够，直接使用 fp16 加载，推理速度会快很多
        print("Loading model in float16 for faster inference...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    except ImportError:
        print("bitsandbytes not found, loading in float16...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer

def generate_motion(model, tokenizer, text_prompt):
    messages = [
        {"role": "system", "content": "You are a motion generation assistant. Generate the motion sequence (flattened integer tokens) based on the text description."},
        {"role": "user", "content": text_prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 初始化 Streamer，这样可以在生成时实时打印 Token
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    print(f"Generating motion for: '{text_prompt}'...")
    print("--- Stream Output Start ---")
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, # Greedy decoding for determinism, or use True for variety
            # top_p=0.9,
            # temperature=0.7,
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
    parser = argparse.ArgumentParser(description="Generate motion sequences from text descriptions.")
    parser.add_argument("prompt", type=str, help="Text description of the motion (e.g., 'A person walks forward')")
    # 确保输出目录存在
    output_dir = "Generation"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    default_output = os.path.join(output_dir, "generated_motion.gif")
    parser.add_argument("--output", type=str, default=default_output, help=f"Output filename for the visualization (default: {default_output})")
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
        # But since we need to parse tokens first, we duplicate a bit of logic or refactor.
        # Let's just do a quick save if tokens are available.
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

