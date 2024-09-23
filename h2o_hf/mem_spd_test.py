import argparse
import logging
import numpy as np
import torch
import pandas as pd
import time
import os
import math
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    LlamaConfig
)
from utils_real_drop.modify_llama import H2OLlamaForCausalLM, H2OLlamaAttention

# --- CONFIGS ---
PATH_TO_YOUR_SAVE_DIR = os.getenv('HF_HUB_CACHE', './')
CACHE_DIR = PATH_TO_YOUR_SAVE_DIR

# H2O Configs
RATIOS = [0.05, 0.25]

# Model configs
MODELS = {'llama': 'meta-llama/Llama-2-7b-chat-hf'}
BATCH_PROMPT = [(1, 256), (4, 256), (8, 256), (16, 256), (32, 256), (64, 256), (128, 256), (256, 256)]
output_length = 128
num_repeats = 20
num_warmups = 15

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

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

ENABLE_Heavy_Hitter_FUNCTIONS = {
    "llama": H2OLlamaForCausalLM,
}

TARGET_MODULES = {
    "llama": H2OLlamaAttention,
}

metrics_list = []

for model_name in MODELS:
    set_seed(42)
    model_name_or_path = MODELS[model_name]
    config = LlamaConfig.from_pretrained(model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            use_fast=False, 
            trust_remote_code=True, 
        )
    
    for ratio in RATIOS:
        config.heavy_ratio = ratio
        config.recent_ratio = ratio

        model = ENABLE_Heavy_Hitter_FUNCTIONS["llama"].from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            config=config,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.float16,
        )

        model.half().eval().cuda()
        
        # warmup LLM
        print(f"[INFO] Warming up {num_warmups} times for {model_name_or_path}[h_ratio={ratio},r_ratio={ratio}].")
        string = 'this is a random prompt that we want our LLM to use during warmup. This will be repeated multiple times. this is a random prompt that we want our LLM to use during warmup. This will be repeated multiple times.'
        inputs = tokenizer(string, return_tensors="pt").to('cuda')
        input_ids = inputs['input_ids']
        with torch.no_grad():
            for i in range(num_warmups):
                outputs = model.generate(**inputs, max_new_tokens=output_length)

        for batch_size, prompt_length in BATCH_PROMPT:
            context = []
            for _ in range(batch_size):
                string = 't,' * (prompt_length // 2)
                context.append(string[:-1])

            inputs = tokenizer(context, return_tensors="pt").to('cuda')            
            input_ids = inputs['input_ids']
            print(f"bs: {batch_size}, seqlen: {input_ids.shape[1]}+{output_length}\nmodel:{model_name_or_path}")
            torch.cuda.reset_peak_memory_stats()

            latency = []
            used_mem = []
            reserved_mem = []

            with torch.no_grad():
                torch.cuda.synchronize()
                for i in tqdm(range(num_repeats), desc = "repeats"):
                    st = time.perf_counter()
                    outputs = model.generate(input_ids, max_new_tokens=output_length, num_beams = 1, do_sample = False, temperature = 1.0)
                    latency.append(time.perf_counter() - st)

                    used_mem.append(torch.cuda.max_memory_allocated())
                    reserved_mem.append(torch.cuda.max_memory_reserved())

                    for name, m in model.named_modules():
                        if "llama" in model_name_or_path.lower() and isinstance(m, TARGET_MODULES['llama']):
                            m._clean_cache()
                
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    torch.cuda.reset_peak_memory_stats()

                torch.cuda.synchronize()

                latency_stats = get_quantiles(latency)
                used_mem_stats = get_quantiles([mem / (1024 ** 3) for mem in used_mem])  # Convert to GB
                reserved_mem_stats = sum(reserved_mem) / ((1024 ** 3) * num_repeats)

                metrics_list.append({
                    "strategy": "H2O",
                    "model": model_name,
                    "batch_size": batch_size,
                    "quantization": f"K=16, V=16",
                    "h_ratio&r_ratio": ratio,
                    "prompt_length": prompt_length,
                    "output_length": output_length,
                    "latency_mean": latency_stats["mean"],
                    "latency_std": latency_stats["std"],
                    "latency_min": latency_stats["min"],
                    "latency_Q1": latency_stats["Q1"],
                    "latency_median": latency_stats["median"],
                    "latency_Q3": latency_stats["Q3"],
                    "latency_max": latency_stats["max"],
                    "peak_mem_usage_mean": used_mem_stats["mean"],
                    "peak_mem_usage_std": used_mem_stats["std"],
                    "peak_mem_usage_min": used_mem_stats["min"],
                    "peak_mem_usage_Q1": used_mem_stats["Q1"],
                    "peak_mem_usage_median": used_mem_stats["median"],
                    "peak_mem_usage_Q3": used_mem_stats["Q3"],
                    "peak_mem_usage_max": used_mem_stats["max"],
                    "peak_mem_reserved_mean": reserved_mem_stats,
                    "tokens/second": (prompt_length + output_length) * batch_size / latency_stats["mean"]
                })

metrics = pd.DataFrame(metrics_list)
metrics.to_csv("h2o-large-batch-metrics.csv", index=False)
