import json
import os
from vllm import LLM, SamplingParams

# 1. Environment Variable Configuration
# Set to "1" to test Static Batching, or "0" for Continuous Batching.
os.environ["VLLM_STATIC_BATCHING"] = "0" 

# 2. Load Model and Initialize vLLM Engine
model_id = "/group-volume/Meta-Llama-3-8B-Instruct/" 
llm = LLM(model=model_id, max_num_seqs=5, trust_remote_code=True) # max_num_seqs = batch size
tokenizer = llm.get_tokenizer()

# 3. Load JSON Data
data_path = "/group-volume/sykim/vllm_test/ShareGPT_V3_unfiltered_cleaned_split.json"
with open(data_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

prompts = []
target_samples = 20
min_answer_length = 200
print(f"Starting dataset filtering: Collecting {target_samples} valid samples...")

for sample in raw_data:
    # Stop once we have reached the target number of samples
    if len(prompts) >= target_samples:
        break
        
    conversations = sample.get("conversations", [])
    
    # Select samples with at least 10 turns where the first message is from 'human' and second is from 'gpt'
    if len(conversations) >= 10:
        if conversations[0]["from"] == "human" and conversations[1]["from"] == "gpt":
            expected_answer = conversations[1]["value"]
            
            if len(expected_answer) >= min_answer_length:
                user_msg = conversations[0]["value"]
                
                messages = [{"role": "user", "content": user_msg}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                prompts.append(formatted_prompt)

if len(prompts) < target_samples:
    print(f"Warning: Only found {len(prompts)} valid samples. (Insufficient total data)")
else:
    print(f"Success: {len(prompts)} prompts are ready.")

# 4. Inference Configuration
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=256, # Set high enough to allow variation in output length
    stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
)

# 5. Execute Inference and Display Results
batch_mode = 'STATIC' if os.environ.get('VLLM_STATIC_BATCHING') == '1' else 'CONTINUOUS'
print(f"Starting inference. (Batch Mode: {batch_mode})")
outputs = llm.generate(prompts, sampling_params)

# Quick verification of results (Top 3)
print("\n" + "="*50)
print("Inference Complete: Sample Results (Top 3)")
print("="*50)
for i, output in enumerate(outputs[:3]):
    print(f"\n[Request {i}]")
    print(f"Prompt: {output.prompt[:80]}...")
    print(f"Generated: {output.outputs[0].text[:80]}...")