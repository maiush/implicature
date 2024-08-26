from constants import results_path, models
models = [m for m in list(models.keys()) if "base" in m]

import numpy as np
import pandas as pd
import torch as t


data = pd.read_json("train_data.jsonl", orient="records", lines=True)
for model in models:

    yes_acts = t.load(f"{results_path}/{model}/harvest_yes_train.pt", weights_only=True)
    no_acts = t.load(f"{results_path}/{model}/harvest_no_train.pt", weights_only=True)
    true_acts = t.where(t.tensor(data["implicature"] == "yes")[:, None], yes_acts, no_acts)
    false_acts = t.where(t.tensor(data["implicature"] == "yes")[:, None], no_acts, yes_acts)
    c1 = true_acts.mean(dim=0) - false_acts.mean(dim=0)
    c2 = (true_acts - false_acts).mean(dim=0)

    yes_ixs = np.argwhere(data["implicature"] == "yes").flatten()
    no_ixs = np.argwhere(data["implicature"] == "no").flatten()
    acts = t.load(f"{results_path}/{model}/harvest_train.pt", weights_only=True)
    c3 = acts[yes_ixs].mean(dim=0) - acts[no_ixs].mean(dim=0)

    candidates = t.stack([c1, c2, c3])
    t.save(candidates, f"{results_path}/{model}/steering_vectors.pt")