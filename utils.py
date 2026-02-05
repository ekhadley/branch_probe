import json
import os
import re
import IPython
import tqdm

import transformer_lens
from transformer_lens import HookedTransformer
import datasets

import torch as t
from torch import Tensor

IPYTHON = IPython.get_ipython()
if IPYTHON is not None:
    IPYTHON.run_line_magic('load_ext', 'autoreload')
    IPYTHON.run_line_magic('autoreload', '2')

purple = '\x1b[38;2;255;0;255m'
blue = '\x1b[38;2;0;0;255m'
brown = '\x1b[38;2;128;128;0m'
cyan = '\x1b[38;2;0;255;255m'
lime = '\x1b[38;2;0;255;0m'
yellow = '\x1b[38;2;255;255;0m'
red = '\x1b[38;2;255;0;0m'
pink = '\x1b[38;2;255;51;204m'
orange = '\x1b[38;2;255;51;0m'
green = '\x1b[38;2;5;170;20m'
gray = '\x1b[38;2;127;127;127m'
magenta = '\x1b[38;2;128;0;128m'
white = '\x1b[38;2;255;255;255m'
bold = '\033[1m'
underline = '\033[4m'
endc = '\033[0m'

def get_test_response(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens=256,
    do_sample=True,
    give_toks:bool = True,
    completion_only:bool = False,
    skip_special_tokens:bool = False,
    verbose:bool=False,
) -> Tensor:
    conv_toks = model.tokenizer.apply_chat_template(
        conversation = [{"role": "user", "content":prompt}],
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.cfg.device)

    resp_toks = model.generate(
        conv_toks,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=model.tokenizer.eot_token_id,
        verbose=verbose,
    )[0]
    
    toks_out = resp_toks[conv_toks.shape[-1]:] if completion_only else resp_toks

    if give_toks:
        out = toks_out
    else:
        out = model.tokenizer.decode(toks_out, skip_special_tokens=skip_special_tokens)
    t.cuda.empty_cache()
    return out

def load_jsonl(file_path: str) -> list:
    """Load a JSONL file and return a list of dictionaries."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data

def save_jsonl(data: list, file_path: str) -> None:
    """Save a list of dictionaries to a JSONL file."""
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def extract_answer(text: str) -> float: # grabs an int from the gsmm8k answer field
    pattern = r'\\boxed\{([^}]+)\}'
    match = re.search(pattern, text)
    
    if not match:
        raise ValueError("No non-empty \\boxed{} expression found in text")
    
    content = match.group(1).strip()
    
    if not content:
        return None
    
    try:
        return int(content)
    except ValueError:
        return None

def parse_answer_value(answer_str: str) -> int: # grabs an int value from a non-empty \boxed{}
    """Parse the answer value from the answer string.
    The answer is always at the very end as "#### answer_value"."""
    assert "####" in answer_str, f"Answer string '{answer_str}' does not contain '####'"
    val_str = answer_str.split("####")[-1].strip()
    val_str = val_str.replace(",", "")
    try:
        return int(val_str)
    except ValueError:
        return None
