import argparse
import ast
import sys

import torch
import os

import time
from tqdm import tqdm
import pandas as pd
import numpy as np

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    LlamaConfig
)

from utils_real_drop.modify_llama import H2OLlamaForCausalLM, H2OLlamaAttention

# ENVS
PATH_TO_YOUR_SAVE_DIR = os.getenv('HF_HUB_CACHE', './')
CACHE_DIR = PATH_TO_YOUR_SAVE_DIR

# ---- CONFIGS ---- #

parser = argparse.ArgumentParser()
parser.add_argument('--batch_prompt_output', type=str, required=True, help='list of (batch_size, prompt_length, output_length)')
parser.add_argument('--ratio', type=float, required=True, help='heavy hitter and recent ratio')
parser.add_argument('--prefix', type=str, required=True, help='output file prefix')
args = parser.parse_args()

# H2O CONFIGS
RATIO = args.ratio

# MODEL CONFIGS
model_name = 'llama'
model_name_or_path = 'meta-llama/Llama-2-7b-chat-hf'
num_repeats = 1
num_warmups = 0

ENABLE_Heavy_Hitter_FUNCTIONS = {
    "llama": H2OLlamaForCausalLM,
}

TARGET_MODULES = {
    "llama": H2OLlamaAttention,
}

def get_quantiles(data):
    return {
        "mean": np.mean(data),
        "min": np.min(data),
        "Q1": np.percentile(data, 25),
        "median": np.median(data),
        "Q3": np.percentile(data, 75),
        "max": np.max(data),
        "std": np.std(data)
    }

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

config = LlamaConfig.from_pretrained(model_name_or_path)
config.use_flash = True
config.heavy_ratio = RATIO
config.recent_ratio = RATIO

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, 
    use_fast=False, 
    trust_remote_code=True, 
    tokenizer_type='llama'
)

model = ENABLE_Heavy_Hitter_FUNCTIONS["llama"].from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            config=config,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.float16,
        )

model.half().eval().cuda()
output_csv = f"outputs/h2o_{args.prefix}_ratio_{args.ratio:.2f}.csv"

# warmup LLM
print(f"[INFO] Warming up {num_warmups} times for {model_name_or_path}[h_ratio={RATIO},r_ratio={RATIO}].")
context = []
for _ in range(1):
    string = 't,' * (128)
    context.append(string[:-1])

inputs = tokenizer(context, return_tensors="pt").to('cuda')
with torch.no_grad():
    for _ in range(num_warmups):
        model.generate(**inputs,     
                        max_new_tokens=1024, 
                        num_beams=1, 
                        do_sample=False,
                        temperature=1.0)

for batch_size, prompt_length, output_length in ast.literal_eval(args.batch_prompt_output):
    context = []
    for _ in range(batch_size):
        string = 't,' * (prompt_length // 2)
        context.append(string[:-1])

    inputs = tokenizer(context, return_tensors="pt").to('cuda')

    print(f"bs: {batch_size}, seqlen: {prompt_length}+{output_length}\nmodel:{model_name_or_path}")
    
    latency = []
    used_mem = []
    reserved_mem = []
    oom_error = False
    try:
        with torch.no_grad():
            for i in tqdm(range(num_repeats), desc="repeats"):
                st = time.perf_counter()
                outputs = model.generate(**inputs,     
                                        max_new_tokens=output_length, 
                                        num_beams=1, 
                                        do_sample=False,
                                        temperature=1.0)
                latency.append(time.perf_counter() - st)
                used_mem.append(torch.cuda.max_memory_allocated())
                reserved_mem.append(torch.cuda.max_memory_reserved())

                try:
                    for name, m in model.named_modules():
                        if "llama" in model_name_or_path.lower() and isinstance(m, TARGET_MODULES['llama']):
                            m._clean_cache()
                except AttributeError:
                    print(f"[WARNING] Cannot clear cache, expected if this is a full model.")

                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

    except torch.cuda.OutOfMemoryError:
        print(f"[ERROR] CUDA OOM encountered at bs: {batch_size}, seqlen: {prompt_length}+{output_length}\nmodel:{model_name_or_path}")
        oom_error = True
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    latency_stats = get_quantiles(latency) if not oom_error else {}
    used_mem_stats = get_quantiles([mem / (1024 ** 3) for mem in used_mem]) if not oom_error else {}
    reserved_mem_stats = sum(reserved_mem) / ((1024 ** 3) * num_repeats) if not oom_error else None

    metric = {
        "strategy": "H2O",
        "model": model_name,
        "batch_size": batch_size,
        "prompt_length": prompt_length,
        "output_length": output_length,
        "quantization": f"K=16, V=16",
        "ratio": RATIO,
        "OOM": "Yes" if oom_error else "No",
        "latency_mean": latency_stats.get("mean", None),
        "latency_std": latency_stats.get("std", None),
        "latency_min": latency_stats.get("min", None),
        "latency_Q1": latency_stats.get("Q1", None),
        "latency_median": latency_stats.get("median", None),
        "latency_Q3": latency_stats.get("Q3", None),
        "latency_max": latency_stats.get("max", None),
        "peak_mem_usage_mean": used_mem_stats.get("mean", None),
        "peak_mem_usage_std": used_mem_stats.get("std", None),
        "peak_mem_usage_min": used_mem_stats.get("min", None),
        "peak_mem_usage_Q1": used_mem_stats.get("Q1", None),
        "peak_mem_usage_median": used_mem_stats.get("median", None),
        "peak_mem_usage_Q3": used_mem_stats.get("Q3", None),
        "peak_mem_usage_max": used_mem_stats.get("max", None),
        "peak_mem_reserved_mean": reserved_mem_stats if not oom_error else None,
        "tokens/second": (prompt_length + output_length) * batch_size / latency_stats["mean"] if not oom_error else None
    }

    if not os.path.isfile(output_csv):
        pd.DataFrame([metric]).to_csv(output_csv, header=True, index=False)
    else:            
        pd.DataFrame([metric]).to_csv(output_csv, mode='a', header=False, index=False)

    if oom_error:
        # exit on OOM, don't even try higher sequences
        sys.exit(0)