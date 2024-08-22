import os, sys, pickle
from pathlib import Path
import pandas as pd
from constants import experiment_path, results_path, llm_cache, models
from tqdm import trange

HF_TOKEN = os.environ.get("HF_TOKEN")
from pipeline import ImplicaturePipeline

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


dataset, model_name, method, choice = sys.argv[1:5]
data = pd.read_json(f"{experiment_path}/{dataset}_data.jsonl", orient="records", lines=True)
model_type = model_name.split("-")[-1]
outpath = f"{results_path}/{model_name}"
Path(outpath).mkdir(exist_ok=True, parents=True)
outpath += f"/{method}"
if method == "harvest": outpath += f"_{choice}"
if os.path.exists(f"{outpath}.pt"):
    print("results already exist")
    sys.exit(0)

# load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    models[model_name],
    torch_dtype=t.bfloat16,
    device_map="auto",
    cache_dir=llm_cache,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    models[model_name],
    cache_dir=llm_cache
)

pipeline = ImplicaturePipeline(model, tokenizer, method, model_type)
results, answers, max_new_tokens = [], [], 128
for i in trange(len(data)):
    messages = [
        {"role": "user", "content": data.at[i, "prompt"]}
    ]
    if model_type == "instruct":
        messages += [
            {"role": "assistant", "content": "answer: The response implies "}
        ]
    elif model_type == "base":
        messages[0]["content"] += f"\nanswer: The response implies "
    if method == "zero_shot":
        answer, x = pipeline(messages, max_new_tokens=max_new_tokens)
        x = F.pad(x, (0, 0, 0, max_new_tokens-len(x)), mode="constant", value=-1)
        results.append(x.cpu())
        answers.append(answer)
    elif method == "harvest":
        messages[-1]["content"] += str(choice)
        x = pipeline(messages)
        results.append(x.cpu())

# save results
results = t.stack(results, dim=0)
t.save(results, f"{outpath}.pt")
# save answers if possible
if len(answers) > 0:
    with open(f"{outpath}_answers.pkl", "wb") as outfile: pickle.dump(answers, outfile)