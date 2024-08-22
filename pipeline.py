import gc
import torch as t
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from jaxtyping import Float
from typing import Union, List, Tuple, Dict

def free_mem(vars):
    for v in vars: del v
    gc.collect()
    t.cuda.empty_cache()


class ImplicaturePipeline:

    def __init__(
            self,
            model: AutoModelForCausalLM,
            tokenizer: AutoTokenizer,
            method: str,
            model_type: str
    ):
        self.model = model; self.model.eval()
        self.tokenizer = tokenizer
        self.method = method
        if method == "zero_shot":
            print(f"zero-shot. returning generations and processed logits.")
            self.logit_ids = [tokenizer.encode(ans, return_tensors="pt", add_special_tokens=False).flatten()[-1].item() for ans in ["yes", "no"]]
        elif method == "harvest":
            print(f"harvest. returning activations.")
        self.model_type = model_type

    def __call__(
            self,
            messages: List[Dict[str, str]],
            **kwargs
    ) -> Union[Tuple[str, Float[Tensor, "n_seq n_vocab"]], Float[Tensor, "d_model"]]:
        if self.model_type == "instruct":
            # apply chat template
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # if necessary, allow for continuation instead of QA
            prompt = self.check_continue(messages, prompt)
            # set add_special_tokens
            ast = False
        elif self.model_type == "base":
            # no prompt template for base models
            prompt = messages[0]["content"]
            # set add_special_tokens
            ast = True
        # tokenize
        tks = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=ast).to(self.model.device)
        # return logits or residual stream, depending on mode
        with t.inference_mode(): 
            if self.method == "zero_shot":
                # default value for max_new_tokens
                max_new_tokens = kwargs.pop("max_new_tokens", 32)
                out = self.model.generate(
                    tks.input_ids,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **kwargs
                )
                logits = t.stack(out["scores"]).squeeze(1).to(self.model.dtype)
                answer = self.tokenizer.decode(
                    logits.argmax(dim=-1),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                ).strip()
                scores = logits[:, self.logit_ids]
                free_mem([prompt, tks, out, logits])
                return answer, scores
            elif self.method == "harvest":
                out = self.model(tks.input_ids, output_hidden_states=True)
                activations = out["hidden_states"][-1].squeeze(0)[-1, :]
                free_mem([prompt, tks, out])
                return activations

    def check_continue(
            self,
            messages: List[Dict[str, str]],
            prompt: str
    ) -> str:
        '''
        if we want continuation of the prompt instead of QA, we need to modify the prompt a bit.
        '''
        if messages[-1]["role"] != "assistant": return prompt
        message = messages[-1]["content"]
        # check if we want to end with a space
        space = False
        if message[-1] == " ":
            space = True
            message = message[:-1]
        ix = prompt.rindex(message) + len(message)
        prompt = f"{prompt[:ix]} " if space else prompt[:ix]
        return prompt