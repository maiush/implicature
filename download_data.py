import pandas as pd
from datasets import load_dataset
from constants import experiment_path


prompt_template = """\
Does the following response to the question imply yes or no?

question: {QUESTION}
response: {RESPONSE}"""


def build_prompt(row: pd.Series) -> str:
    q, r = row["utterance"], row["response"]
    prompt = prompt_template.format(QUESTION=q, RESPONSE=r)
    return prompt


data = load_dataset("UCL-DARK/ludwig", "0-shot")["test"].to_pandas()
data["prompt"] = data.apply(build_prompt, axis=1)
data[["prompt", "implicature"]].to_json(f"{experiment_path}/test_data.jsonl", orient="records", lines=True)

data = load_dataset("UCL-DARK/ludwig", "0-shot")["validation"].to_pandas()
data["prompt"] = data.apply(build_prompt, axis=1)
data[["prompt", "implicature"]].to_json(f"{experiment_path}/train_data.jsonl", orient="records", lines=True)