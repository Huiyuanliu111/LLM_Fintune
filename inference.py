import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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

def load_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    
    print("Loading base model...")
    # Use 4-bit quantization if possible to match training environment, or load in full precision for inference
    # Here we try to load it in a way that fits in memory. 
    try:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
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
    
    print(f"Generating motion for: '{text_prompt}'...")
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, # Greedy decoding for determinism, or use True for variety
            # top_p=0.9,
            # temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
        
    # Extract only the newly generated tokens
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def main():
    parser = argparse.ArgumentParser(description="Generate motion sequences from text descriptions.")
    parser.add_argument("prompt", type=str, help="Text description of the motion (e.g., 'A person walks forward')")
    parser.add_argument("--output", type=str, default="generated_motion.gif", help="Output filename for the visualization (default: generated_motion.gif)")
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
            
            if motion_data.shape[0] > 0:
                plot_motion(motion_data, output_file=args.output)
                print(f"Motion visualization saved to {args.output}")
            else:
                print("Generated sequence resulted in empty motion data.")
        except Exception as e:
            print(f"Error visualizing motion: {e}")
    
if __name__ == "__main__":
    main()

