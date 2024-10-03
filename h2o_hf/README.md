# H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models

## Installation

**Requirements**

- PyTorch >= 1.12

## H2O + MiniKV

### Setup

```
git clone https://github.com/neeldani/H2O.git
cd H2O
git checkout feature/batching
cd h2o_hf
conda env create -f environment.yml

# build transformers from source
cd transformers-4.35.2
pip install -e . -U

# build flash attention from source
git clone https://github.com/jpli02/flash-attention/tree/main
cd flash-attention
git checkout accum
pip install . --no-build-isolation -v
```

### How to run
We will be using the code inside `h2o_hf`. The main parts of H2O are defined in the `h2o_hf/utils_real_drop/modify_llama.py` file. Use the branch `feature/batching`. 

`h2o_hf/mem_spd_test.py` is the file that we use for benchmarking H2O. Modify `h2o_batch_0.05.slurm` file and run it as an example.

## Usage and Examples

### Streaming with H2O

This section provides a simple demo to generate with H2O.

```
# Full KV Cache
bash scripts/streaming/eval.sh full

# H2O
bash scripts/streaming/eval.sh h2o
```

### Benchmarking on summarization tasks with real KV dropping implementation

This section provides the code with real KV dropping implementation, rather than masking. 

```
# Full baseline on XSUM
bash scripts/summarization/eval.sh xsum ${shots} full ${GPU-ID}

# H2O KV Cache on XSUM
bash scripts/summarization/eval.sh xsum ${shots} h2o ${GPU-ID} ${HH-size} ${Local-size} 
```



### Text-Generation with custom prompts

In this code, you can generate text with your own prompts, The models will automatically downloaded from Hugging Face.

```
python -u run_text_generation.py \
    --model_arch llama \
    --model_name huggyllama/llama-13b \
    --recent_ratio 0.1 \ # kv cache size for the most recent ones (num = 0.1 * length_of_prompt)
    --heavy_ratio 0.1 \ # kv cache size heavy hitters (num = 0.1 * length_of_prompt)
```

You can change the prompt by modifying **prompt_text** in **run_test_generation.py**, more examples are available in **scripts/generation**.

### Evaluation on tasks from [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework

Here we provide an example to evaluate the 5-shot performance of LLaMA-7b on OpenbookQA, more examples can be found at **scripts/lm_eval/experiments.sh**

```
# Step 1: Prepare inference text
task=openbookqa
shots=5
python -u generate_task_data.py \
  --output-file ${task}-${shots}.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots}

# Step 2 (Full Cache Baseline): Generate the output from LLaMA-7b with Full Cache
model=huggyllama/llama-7b
model_arch=llama
python -u run_lm_eval_harness.py \
  --input-path ${task}-${shots}.jsonl \
  --output-path ${task}-${shots}-${model_arch}.jsonl \
  --model-name ${model} \
  --model-type ${model_arch}
  
# Step 2 ("Local" Baseline): Generate the output from LLaMA-7b with 20% kv of the most recent tokens
model=huggyllama/llama-7b
model_arch=llama
python -u run_lm_eval_harness.py \
  --input-path ${task}-${shots}.jsonl \
  --output-path ${task}-${shots}-${model_arch}.jsonl \
  --model-name ${model} \
  --model-type ${model_arch} \
  --enable_small_cache \
  --heavy_ratio 0 \
  --recent_ratio 0.2

# Step 2 (H2O): Generate the output from LLaMA-7b with H2O
model=huggyllama/llama-7b
model_arch=llama
python -u run_lm_eval_harness.py \
  --input-path ${task}-${shots}.jsonl \
  --output-path ${task}-${shots}-${model_arch}.jsonl \
  --model-name ${model} \
  --model-type ${model_arch} \
  --enable_small_cache \
  --heavy_ratio 0.1 \
  --recent_ratio 0.1

# Step 3: Evaluate the performance of generated text
python -u evaluate_task_result.py \
  --result-file ${task}-${shots}-${model_arch}.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots} \
  --model-type ${model_arch}
```

### Evaluation on tasks from [HELM](https://crfm.stanford.edu/helm/latest/) framework

To evaluate the performance of tasks from HELM framework, the pipeline is similar with lm-eval-harness. An example is provided in the following, and more experiments can be found at **scripts/helm/experiments.sh**

```
# Step 1: prepare inference text
# Examples of converting inference data to jsonl format is provided in helm/command/get_data.sh
# And the data is provided in data/

# Step 2 (Full Cache Baseline): Generate the output from LLaMA-7b with Full Cache
model=huggyllama/llama-7b
model_arch=llama
python -u run_helm.py \
  --input_path data/xsum.jsonl \
  --output_path generate_xsum_llama7b.jsonl \
  --model_name ${model} \
  --model_arch ${model_arch} 

# Step 2 ("Local" Baseline): Generate the output from LLaMA-7b with 20% kv of the most recent tokens
model=huggyllama/llama-7b
model_arch=llama
python -u run_helm.py \
  --input_path data/xsum.jsonl \
  --output_path generate_xsum_llama7b_local.jsonl \
  --model_name ${model} \
  --model_arch ${model_arch} \
  --enable_small_cache \
  --heavy_ratio 0 \
  --recent_ratio 0.2
  
# Step 2 (H2O): Generate the output from LLaMA-7b with H2O
model=huggyllama/llama-7b
model_arch=llama
python -u run_helm.py \
  --input_path data/xsum.jsonl \
  --output_path generate_xsum_llama7b_h20.jsonl \
  --model_name ${model} \
  --model_arch ${model_arch} \
  --enable_small_cache \
  --heavy_ratio 0.1 \
  --recent_ratio 0.1
  
# Step 3: Evaluate the performance of generated text (refer helm/command/eval.sh)
cd helm
TASK=xsum
JSONL=generate_xsum_llama7b.jsonl
OUTPUT=xsum_llama7b_result
ARCH=llama
python scripts/offline_eval/import_results.py together ${JSONL} --cache-dir prod_env/cache
helm-run --conf src/helm/benchmark/presentation/${TASK}/run_specs_${ARCH}.conf --local --max-eval-instances 100 --num-train-trials=1 --suite ${OUTPUT} -n 1
helm-summarize --suite ${OUTPUT} 
# The results are writted into a tex file that can be found in benchmark_output/runs/xsum_llama7b_result/groups/latex/ 
```

