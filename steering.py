from constants import experiment_path, results_path, llm_cache, models
from pipeline import ImplicaturePipeline

import os, sys, pickle
import pandas as pd
import torch as t
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import trange

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


model_name, candidate, hook_type = sys.argv[1:4]

candidates = t.load(f"{results_path}/{model_name}-base/steering_vectors.pt", weights_only=True)
v = candidates[int(candidate)]
data = pd.read_json(f"{experiment_path}/test_data.jsonl", orient="records", lines=True)
outpath = f"{results_path}/{model_name}-instruct/steered_c{candidate}_{hook_type}"
if os.path.exists(f"{outpath}.pt"):
    print("results already exist")
    sys.exit(0)
# load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    models[f"{model_name}-instruct"],
    torch_dtype=t.bfloat16,
    device_map="auto",
    cache_dir=llm_cache,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    models[f"{model_name}-instruct"],
    cache_dir=llm_cache
)


# hooks
def v_rejection(module, input, output):
    global v
    state = output[0]
    v = v.to(state.device).to(state.dtype)
    proj = ((state @ v).unsqueeze(-1) / (v @ v)) * v
    rej = state - proj
    return (rej, output[1])
n_layers = len(model.model.layers)
if hook_type == "all":
    for l in range(1, n_layers):
        module = model.model.layers[l]
        module.register_forward_hook(v_rejection)
elif hook_type == "last":
    module = model.model.layers[n_layers-1]
    module.register_forward_hook(v_rejection)


# pipeline
pipeline = ImplicaturePipeline(model, tokenizer, "zero_shot", "instruct")
results, answers, max_new_tokens = [], [], 128
for i in trange(len(data)):
    messages = [
        {"role": "user", "content": data.at[i, "prompt"]},
        {"role": "assistant", "content": "answer: The response implies \""}
    ]
    answer, x = pipeline(messages, max_new_tokens=max_new_tokens)
    x = F.pad(x, (0, 0, 0, max_new_tokens-len(x)), mode="constant", value=-1)
    results.append(x.cpu())
    answers.append(answer)


# save results
results = t.stack(results, dim=0)
t.save(results, f"{outpath}.pt")
# save answers if possible
if len(answers) > 0:
    with open(f"{outpath}_answers.pkl", "wb") as outfile: pickle.dump(answers, outfile)