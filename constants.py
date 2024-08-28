experiment_path = "/gws/nopw/j04/ai4er/users/maiush/implicature"
results_path = "/gws/nopw/j04/ai4er/users/maiush/implicature_results"
llm_cache = "/gws/nopw/j04/ai4er/users/maiush/LLMs"

models = {
    "qwen-2-72b-instruct": "Qwen/Qwen2-72B-Instruct", # 72b
    "qwen-2-72b-base": "Qwen/Qwen2-72B", # 72b
    
    "mistral-nemo-12b-instruct": "mistralai/Mistral-Nemo-Instruct-2407", # 12b
    "mistral-nemo-12b-base": "mistralai/Mistral-Nemo-Base-2407", # 12b

    "llama-3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct", # 8b
    "llama-3.1-8b-base": "meta-llama/Meta-Llama-3.1-8B", # 8b

    "gemma-2-2b-instruct": "google/gemma-2-2b-it", # 2b
    "gemma-2-2b-base": "google/gemma-2-2b", # 2b
}