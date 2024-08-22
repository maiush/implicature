import pickle, string
from constants import results_path, models
import pandas as pd
import torch as t


df = pd.DataFrame()
for model in models.keys():
    path = f"{results_path}/{model}/zero_shot_answers.pkl"
    with open(path, "rb") as f: answers = pickle.load(f)
    # lowercase
    answers = [a.lower() for a in answers]
    # remove punctuation
    full_answers = [a.translate(str.maketrans('', '', string.punctuation)) for a in answers]
    yes_counts = [a.count(" yes ") for a in full_answers]
    no_counts = [a.count(" no ") for a in full_answers]
    answers, unk_ixs = [], []
    counter = 0
    for answer, y, n in zip(full_answers, yes_counts, no_counts):
        answer = answer.replace("\n", " ")
        if answer.split(" ")[0] == "yes" or answer.split(" ")[-1] == "yes":
            answers.append("yes")
        elif answer.split(" ")[0] == "no" or answer.split(" ")[-1] == "no":
            answers.append("no")
        elif y > 0 and n == 0:
            answers.append("yes")
        elif y == 0 and n > 0:
            answers.append("no")
        else:
            answers.append("unknown")
            unk_ixs.append(counter)
        counter += 1
    df[model] = answers
df.to_json(f"{results_path}/zero_shot_answers.jsonl", orient="records", lines=True)

df = pd.DataFrame()
for model in models.keys():
    path = f"{results_path}/{model}/zero_shot.pt"
    logits = t.load(path)
    choices = logits[:, 0, :].argmax(dim=-1).tolist()
    answers = [["yes", "no"][ix] for ix in choices]
    df[model] = answers
df.to_json(f"{results_path}/logits.jsonl", orient="records", lines=True)