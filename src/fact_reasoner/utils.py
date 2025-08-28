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

import contextlib
import numpy as np
import requests
import tqdm
from tqdm.contrib import DummyTqdmFile
import os
import re
import random
import sys
import torch
import transformers

from typing import List, Union

DEFAULT_PROMPT_BEGIN = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
DEFAULT_PROMPT_END = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

# GPU-related business

model_map = {
    "boychaboy/SNLI_roberta-base": {"entailment_idx": 0, "contradiction_idx": 2},
    "boychaboy/SNLI_roberta-large": {"entailment_idx": 0, "contradiction_idx": 2},
    "microsoft/deberta-base-mnli": {"entailment_idx": 2, "contradiction_idx": 0},
    "roberta-large-mnli": {"entailment_idx": 2, "contradiction_idx": 0},
    "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli": {"entailment_idx": 0, "contradiction_idx": 2},
    "tals/albert-base-vitaminc-mnli": {"entailment_idx": 0, "contradiction_idx": 1},
    "tals/albert-xlarge-vitaminc-mnli": {"entailment_idx": 0, "contradiction_idx": 1},
    "tals/albert-xlarge-vitaminc": {"entailment_idx": 0, "contradiction_idx": 1},
}

# Maps the RITS supported models to their IDs and APIs
RITS_MODELS = {
    "gpt-oss-120b": {
        "model_id": "openai/openai/gpt-oss-120b",
        "api_base": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/gpt-oss-120b/v1",
        "max_new_tokens": 128000,
        "prompt_template": "{}",
        "prompt_begin": "",
        "prompt_end": "",
    },
    "gpt-oss-20b": {
        "model_id": "openai/openai/gpt-oss-20b",
        "api_base": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/gpt-oss-20b/v1",
        "max_new_tokens": 128000,
        "prompt_template": "{}",
        "prompt_begin": "",
        "prompt_end": "",
    },
    "deepseek-v3": {
        "model_id": "openai/deepseek-ai/DeepSeek-V3",
        "api_base": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/deepseek-v3-h200/v1",
        "max_new_tokens": 4096,
        "prompt_template": "<|begin_of_text|><｜begin▁of▁sentence｜>user<｜end▁of▁sentence｜>\n\n{}<|end|><｜begin▁of▁sentence｜>assistant<｜end▁of▁sentence｜>",
        "prompt_begin": "<|begin_of_text|><｜begin▁of▁sentence｜>user<｜end▁of▁sentence｜>",
        "prompt_end": "<|end|><｜begin▁of▁sentence｜>assistant<｜end▁of▁sentence｜>"
    },
    "granite-3.0-8b-instruct": {
        "model_id": "openai/ibm-granite/granite-3.0-8b-instruct",
        "api_base": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-0-8b-instruct/v1",
        "max_new_tokens": 4096,
        "prompt_template": "<|start_of_role|>user<|end_of_role|>{}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
        "prompt_begin": "<|start_of_role|>user<|end_of_role|>",
        "prompt_end": "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
        # "prompt_template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    },
    "granite-3.1-8b-instruct": {
        "model_id": "openai/ibm-granite/granite-3.1-8b-instruct",
        "api_base": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-1-8b-instruct/v1",
        "max_new_tokens": 128000,
        "prompt_template": "<|start_of_role|>user<|end_of_role|>{}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
        "prompt_begin": "<|start_of_role|>user<|end_of_role|>",
        "prompt_end": "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
        # "prompt_template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    },
    "granite-3.2-8b-instruct": {
        "model_id": "openai/ibm-granite/granite-3.2-8b-instruct",
        "api_base": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-2-8b-instruct/v1",
        "max_new_tokens": 128000,
        "prompt_template": "<|start_of_role|>user<|end_of_role|>{}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
        "prompt_begin": "<|start_of_role|>user<|end_of_role|>",
        "prompt_end": "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
        # "prompt_template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    },
    "granite-3.3-8b-instruct": {
        "model_id": "openai/ibm-granite/granite-3.3-8b-instruct",
        "api_base": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-3-8b-instruct/v1",
        "max_new_tokens": 128000,
        "prompt_template": "<|start_of_role|>user<|end_of_role|>{}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
        "prompt_begin": "<|start_of_role|>user<|end_of_role|>",
        "prompt_end": "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
        # "prompt_template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    },
    "granite-guardian-3.2-5b": {
        "model_id": "openai/ibm-granite/granite-guardian-3.2-5b",
        "api_base": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-guardian-3-2-5b-ris/v1",
        "max_new_tokens": 128000,
        "prompt_template": "<|start_of_role|>user<|end_of_role|>{}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
        "prompt_begin": "<|start_of_role|>user<|end_of_role|>",
        "prompt_end": "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
        # "prompt_template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    },
    "granite-guardian-3.2-3b-a800m": {
        "model_id": "openai/ibm-granite/granite-guardian-3.2-3b-a800m",
        "api_base": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-guardian-3-2-3b-a800m/v1",
        "max_new_tokens": 128000,
        "prompt_template": "<|start_of_role|>user<|end_of_role|>{}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
        "prompt_begin": "<|start_of_role|>user<|end_of_role|>",
        "prompt_end": "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
        # "prompt_template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    },
    "llama-3.1-8b-instruct": {
        "model_id": "openai/ibm-granite/Meta-Llama-3.1-8B-Instruct",
        "api_base": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/meta-llama-3-1-8b-instruct/v1",
        "max_new_tokens": 4096,
        "prompt_template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    },
    "llama-3.1-405b-instruct": {
        "model_id": "openai/meta-llama/llama-3-1-405b-instruct-fp8",
        "api_base": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-1-405b-instruct-fp8/v1",
        "max_new_tokens": 8196,
        "prompt_template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        "prompt_begin": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>",
        "prompt_end": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    },
    "llama-3.2-11b-instruct": {
        "model_id": "openai/meta-llama/Llama-3.2-11B-Vision-Instruct",
        "api_base": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-2-11b-instruct/v1",
        "max_new_tokens": 8196,
        "prompt_template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        "prompt_begin": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>",
        "prompt_end": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    },
    "llama-3.2-90b-instruct": {
        "model_id": "openai/meta-llama/Llama-3.2-90B-Vision-Instruct",
        "api_base": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-2-90b-instruct/v1",
        "max_new_tokens": 8196,
        "prompt_template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        "prompt_begin": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>",
        "prompt_end": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    },
    "llama-3.3-70b-instruct": {
        "model_id": "openai/meta-llama/llama-3-3-70b-instruct",
        "api_base": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct/v1",
        "max_new_tokens": 128000,
        "prompt_template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        "prompt_begin": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>",
        "prompt_end": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    },
    "llama-4-maverick-17b-128e-instruct-fp8": {
        "model_id": "openai/meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        "api_base": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-4-mvk-17b-128e-fp8/v1",
        "max_new_tokens": 128000,
        "prompt_template": "<|begin_of_text|><|header_start|>user<|header_end|>\n\n{}<|eot|><|header_start|>assistant<|header_end|>",
        "prompt_begin": "<|begin_of_text|><|header_start|>user<|header_end|>",
        "prompt_end": "<|eot|><|header_start|>assistant<|header_end|>"
    },
    "llama-4-scout-17b-16e": {
        "model_id": "openai/meta-llama/llama-4-scout-17b-16e",
        "api_base": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-4-scout-17b-16e/v1",
        "max_new_tokens": 128000,
        "prompt_template": "<|begin_of_text|><|header_start|>user<|header_end|>\n\n{}<|eot|><|header_start|>assistant<|header_end|>",
        "prompt_begin": "<|begin_of_text|><|header_start|>user<|header_end|>",
        "prompt_end": "<|eot|><|header_start|>assistant<|header_end|>"
    },
    "mixtral-8x7b-instruct": {
        "model_id": "openai/mistralai/mixtral-8x7B-instruct-v0.1",
        "api_base": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mixtral-8x7b-instruct-v01/v1",
        "max_new_tokens": 16384,
        "prompt_template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        "prompt_begin": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>",
        "prompt_end": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    },
    "mixtral-8x22b-instruct": {
        "model_id": "openai/mistralai/mixtral-8x22B-instruct-v0.1",
        "api_base": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mixtral-8x22b-instruct-v01/v1",
        "max_new_tokens": 16384,
        "prompt_template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        "prompt_begin": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>",
        "prompt_end": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    },
    "mixtral-8x22b-instruct-priority": {
        "model_id": "openai/mistralai/mixtral-8x22B-instruct-v0.1",
        "api_base": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mixtral-8x22b-test/v1",
        "max_new_tokens": 16384,
        "prompt_template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        "prompt_begin": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>",
        "prompt_end": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    },
    "phi-4": {
        "model_id": "openai/microsoft/phi-4",
        "api_base": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/microsoft-phi-4/v1",
        "max_new_tokens": 8196,
        "prompt_template": "<|begin_of_text|><|im_start|>user<|im_end|>\n\n{}<|end|><|im_start|>assistant<|im_end|>",
        "prompt_begin": "<|begin_of_text|><|im_start|>user<|im_end|>",
        "prompt_end": "<|end|><|im_start|>assistant<|im_end|>"
    },
    "Qwen2.5-72B-Instruct": {
        "model_id": "openai/Qwen/Qwen2.5-72B-Instruct",
        "api_base": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/qwen2-5-72b-instruct/v1",
        "max_new_tokens": 65536,
        "prompt_template": "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        "prompt_begin": "<|im_start|>user\n",
        "prompt_end": "<|im_end|>\n<|im_start|>assistant\n"
    }
}

HF_MODELS = {
    "mixtral-8x22b-instruct": {
        "model_id": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "max_new_tokens": 16384,
        "prompt_template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        "prompt_begin": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>",
        "prompt_end": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    },
    "mixtral-8x7b-instruct": {
        "model_id": "mistralai/mixtral-8x7B-instruct-v0.1",
        "max_new_tokens": 16384,
        "prompt_template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        "prompt_begin": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>",
        "prompt_end": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    },
    "facebook/opt-350m": {
        "model_id": "facebook/opt-350m",
        "max_new_tokens": 16384,
        "prompt_template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        "prompt_begin": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>",
        "prompt_end": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    }
}

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
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

################################################################################
#                             STRING MANIPULATION                              #
################################################################################
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

@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    """
    Redirects stdout and stderr to tqdm.write() for correct tqdm progress display
    """
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    except Exception as e:
        raise e
    finally:
        sys.stdout, sys.stderr = orig_out_err