# coding=utf-8
# Copyright 2023-present the International Business Machines.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import numpy as np
import requests
import tqdm
import os
import re
import random
import torch
import transformers

from typing import Any, Dict, List, Union

import yaml

DEFAULT_PROMPT_BEGIN = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
DEFAULT_PROMPT_END = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# GPU related utils    
def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A6 GPU|grep Free >tmp_smi')
    memory_available = [int(x.split()[2])+5*i for i, x in enumerate(open('tmp_smi', 'r').readlines())]
    os.remove("tmp_smi")
    return np.argmax(memory_available)

def select_freer_gpu():
    freer_gpu = str(get_freer_gpu())
    print("Will use GPU: %s" % (freer_gpu))
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""+freer_gpu
    return freer_gpu

# Set the random seed globally
def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)

# String manipulation utils
def join_segments(*args: Union[str, List[str]], separator: str = '\n\n\n') -> str:
    """Joins an unspecified number of strings using the separator."""
    all_segments = []

    for arg in args:
        if isinstance(arg, list):
            all_segments.extend(arg)
        else:
            all_segments.append(strip_string(str(arg)))

    return strip_string(separator.join(all_segments))


def strip_string(s: str) -> str:
    """Strips a string of newlines and spaces."""
    return s.strip(' \n')

def punctuation_only_inside_quotes(text):
    # find all quoted sections (single or double quotes)
    quoted_spans = [match.span() for match in re.finditer(r'"[^"]*"|\'[^\']*\'', text)]

    def is_inside_quotes(index):
        return any(start < index < end for start, end in quoted_spans)

    # check each comma and semicolon
    for i, char in enumerate(text):
        if char in [",", ";"]:
            if not is_inside_quotes(i):
                return False  # found punctuation outside quotes
    return True


def extract_first_square_brackets(input_string: str) -> str:
    """Extracts the contents of the FIRST string between square brackets."""
    raw_result = re.findall(r'\[.*?\]', input_string, flags=re.DOTALL)

    if raw_result:
        return raw_result[0][1:-1]
    else:
        return ''

def extract_last_square_brackets(input_string: str) -> str:
    """Extracts the contents of the FIRST string between square brackets."""
    raw_result = re.findall(r'\[.*?\]', input_string, flags=re.DOTALL)

    if raw_result:
        return raw_result[-1][1:-1]
    else:
        return ''

def extract_last_wrapped_response(input_string: str) -> str:
    """Extracts the contents of the LAST string between pairs of ###."""
    raw_result = re.findall(r'###.*?###', input_string, flags=re.DOTALL)

    if raw_result:
        return raw_result[-1][3:-3]
    else:
        return ''


def extract_first_code_block(
    input_string: str, ignore_language: bool = False
) -> str:
    """Extracts the contents of a string between the first code block (```)."""
    if ignore_language:
        pattern = re.compile(r'```(?:\w+\n)?(.*?)```', re.DOTALL)
    else:
        pattern = re.compile(r'```(.*?)```', re.DOTALL)

    match = pattern.search(input_string)
    return strip_string(match.group(1)) if match else ''


def batcher(iterator, batch_size=4, progress=False):
    if progress:
        iterator = tqdm.tqdm(iterator)

    batch = []
    for elem in iterator:
        batch.append(elem)
        if len(batch) == batch_size:
            final_batch = batch
            batch = []
            yield final_batch
    if len(batch) > 0: # Leftovers
        yield batch

# Google Drive related

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def get_models_config() -> Dict[str, Any]:
    """
    Return the models config from configs/models.yaml.
    """

    d = Path(__file__).resolve().parent
    filename = Path.joinpath(d, "configs", "models.yaml")
    with open(filename, "r") as f:
        config = yaml.safe_load(f)
    return config
