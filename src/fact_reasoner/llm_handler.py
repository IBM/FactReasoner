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

import os

import litellm
import torch
from dotenv import load_dotenv
from vllm import LLM, SamplingParams

# Local imports
from src.fact_reasoner.utils import (
    DEFAULT_PROMPT_BEGIN,
    DEFAULT_PROMPT_END,
    get_models_config,
)

GPU = torch.cuda.is_available()
DEVICE = GPU*"cuda" + (not GPU)*"cpu"

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class LLMHandler:
    """
    A handler for LLMs that can switch between different backends like rits,
    huggingface, or watsonx. It is possible to extend the handler to support
    additional backends like openai or anthropic.
    """

    def __init__(self, model_id: str, backend: str = "rits", dtype="auto", **default_kwargs):
        """
        Initializes the LLM handler.

        Args:
            model_id: str
                The model name or path to load.
            backend: str
                The model's backend such as [rits, hf, wx].
            dtype: str
                The data type for the model (e.g., "auto", "half", "bfloat16").
            default_kwargs: dict
                Default parameters to pass to completion calls (e.g., temperature, max_tokens).
        """

        self.backend = backend  # The model's backend: one of [rits, hf, wx]
        self.default_kwargs = default_kwargs  # Store common parameters for completions
        assert backend in ["rits", "hf", "wx"], \
            f"Model backend {backend} is not supported yet. Use `rits`, `hf` or `wx` only."
        
        self.models_config = get_models_config()
        if self.backend == "hf":
            self.HF_model_info = self.models_config["HF_MODELS"][model_id]
            self.model_id = self.HF_model_info.get("model_id", None)
            assert self.model_id is not None
            print(f"Loading local model with vLLM: {self.model_id}...")
            self.llm = LLM(model=self.model_id, device=DEVICE, dtype=dtype)  # Load model using vLLM
        # It's an API provider
        else:
            # Load API params from env
            if not os.environ.get("_DOTENV_LOADED"):
                load_dotenv(override=True)
                os.environ["_DOTENV_LOADED"] = "1"
            
            self.extra_headers = {}
            self.model_id = model_id

            if self.backend == "rits":
                self.api_key = os.getenv("RITS_API_KEY")
                self.model_info = self.models_config["RITS_MODELS"][model_id]
                self.extra_headers["RITS_API_KEY"] = self.api_key
            elif self.backend == "wx":
                self.api_key = os.getenv("WX_API_KEY")
                self.model_info = self.models_config["WX_MODELS"][model_id]
            else:
                raise ValueError(f"Uknown backend value: {self.backend}")
            
            # Generic attributes
            self.prompt_template = self.model_info.get("prompt_template", None)
            self.max_new_tokens = self.model_info.get("max_new_tokens", None)
            self.api_base = self.model_info.get("api_base", None)
            self.model_id = self.model_info.get("model_id", None)
            self.prompt_begin = self.model_info.get("prompt_begin", DEFAULT_PROMPT_BEGIN)
            self.prompt_end = self.model_info.get("prompt_end", DEFAULT_PROMPT_END)

            assert self.prompt_template is not None \
                and self.max_new_tokens is not None \
                and self.model_id is not None
            
            if self.backend == "rits":
                assert self.api_base is not None
            
            print(f"[LLMHandler] Using API key: {self.api_key}")
            print(f"[LLMHandler] Using model id: {self.model_id}")
            print(f"[LLMHandler] Using model info: {self.model_info}")
            print(f"[LLMHandler] Initialization completed.")

    def get_prompt_begin(self):
        """
        Returns the prompt begin template for the model.
        """

        if self.backend in ["rits", "wx"]:
            return self.prompt_begin
        else:
            return ""  # vLLM does not use a prompt begin template
    
    def get_prompt_end(self):
        """
        Returns the prompt end template for the model.
        """

        if self.backend in ["rits", "wx"]:
            return self.prompt_end
        else:
            return "" # vLLM does not use a prompt end template

    def completion(self, prompt, **kwargs):
        """
        Generate a response using the RITS API (if RITS=True) or the local model.

        Args:
            prompt: str
                The prompt or a list of prompts to generate responses for.
            kwargs: dict
                Additional parameters for completion (e.g., temperature, max_tokens).
        """
        return self._call_model(prompt, **kwargs)

    def batch_completion(self, prompts, **kwargs):
        """
        Generate responses in batch using the RITS API (if RITS=True) or the local model.

        Args:
            prompts: list of str
                A list of prompts to generate responses for.
            kwargs: dict
                Additional parameters for batch completion (e.g., temperature, max_tokens).
        """
        return self._call_model(prompts, **kwargs)

    def _call_model(self, prompts, num_retries=5, **kwargs):
        """
        Handles both single and batch generation.

        Args:
            prompts: str or list of str
                The prompt or a list of prompts to generate responses for.
            num_retries: int
                Number of retries for the API call in case of failure.
            kwargs: dict
                Additional parameters for completion (e.g., temperature, max_tokens).               
        """

        params = {
            "temperature": 0,
            "seed": 42,
            # the two above are overwritten if passed
            # as kwargs
            **self.default_kwargs,
            **kwargs
        }  # Merge defaults with provided params


        if self.backend in ["rits", "wx"]:
            # Ensure we always send a list to batch_completion
            if isinstance(prompts, str):
                return litellm.completion(
                    model=self.model_id,
                    api_base=self.api_base,
                    messages=[{"role": "user", "content": prompts}],  # Wrap prompt for compatibility
                    api_key=self.api_key,
                    num_retries=num_retries,
                    extra_headers=self.extra_headers,
                    **params
                )
            return litellm.batch_completion(
                model=self.model_id,
                api_base=self.api_base,
                messages=[[{"role": "user", "content": p}] for p in prompts],  # Wrap each prompt
                api_key=self.api_key,
                num_retries=num_retries,
                extra_headers=self.extra_headers,
                **params
            )

        elif self.backend == "hf":
            # Ensure prompts is always a list for vLLM
            if isinstance(prompts, str):
                prompts = [prompts]

            sampling_params = SamplingParams(**params)
            outputs = self.llm.generate(prompts, sampling_params)

            #print("\n=== FULL OUTPUT STRUCTURE ===\n")
            #self.recursive_print(outputs)

            #import pickle
            #with open("saved_vllm_response.pkl",'wb') as f:
            #    pickle.dump(outputs,f)

            # Convert vLLM outputs to match litellm format
            responses = [self.transform_vllm_response(output) for output in outputs]
            
            return responses if len(prompts) > 1 else responses[0]
            #return [output.outputs[0].text for output in outputs] #TODO: make output consistent with that of RITS

    def transform_vllm_response(self, response_obj):
        """
        Transform the vLLM response to match the expected litellm format.
        """
        output_obj = response_obj.outputs[0]  

        # Extract the generated text
        text = output_obj.text

        # Convert logprobs into the expected structure
        logprobs = []
        for token_dict in output_obj.logprobs:
            best_token_id = max(token_dict, key=lambda k: token_dict[k].rank)  # Select top-ranked token
            logprobs.append({
                "logprob": token_dict[best_token_id].logprob,
                "decoded_token": token_dict[best_token_id].decoded_token
            })

        # Create the transformed response
        transformed_response = dotdict({
            "choices": [
                dotdict({
                    "message": dotdict({"content": text}),
                    "logprobs": {"content": logprobs}
                })
            ]
        })

        return transformed_response

    def recursive_print(self, obj, indent=0):
        """
        Recursively print objects, lists, and dicts for deep inspection.
        """
        
        prefix = "  " * indent  # Indentation for readability

        if isinstance(obj, list):
            print(f"{prefix}[")
            for item in obj:
                self.recursive_print(item, indent + 1)
            print(f"{prefix}]")
        elif isinstance(obj, dict):
            print(f"{prefix}{{")
            for key, value in obj.items():
                print(f"{prefix}  {key}: ", end="")
                self.recursive_print(value, indent + 1)
            print(f"{prefix}}}")
        elif hasattr(obj, "__dict__"):  # Print class attributes
            print(f"{prefix}{obj.__class__.__name__}(")
            for key, value in vars(obj).items():
                print(f"{prefix}  {key}: ", end="")
                self.recursive_print(value, indent + 1)
            print(f"{prefix})")
        else:
            print(f"{prefix}{repr(obj)}")  # Print basic values

if __name__ == "__main__":

    """
    Test to compare remote (rits) and local (hf) outputs.
    """
    test_prompt = "What is the capital of France?"

    # RITS (litellm) API
    print(f"Test LLMHandler on remote backend (rits)...")
    remote_handler = LLMHandler(
        model_id="llama-3.3-70b-instruct",
        backend="rits",
    )
    
    remote_response = remote_handler.completion(
        test_prompt,
        logprobs=True,
        seed=12345
        )
    print("\nREMOTE RESPONSE:")
    print(remote_response)

    # Local (vLLM) - Using a small model for testing
    """
    local_handler = LLMHandler(
        model="mixtral-8x7b-instruct",
        backend="hf",
        dtype="half",
        logprobs=1
    )
    """

    print(f"Test LLMHandler on local backend (hf)...")
    local_handler = LLMHandler(
        model_id="facebook/opt-350m",
        backend="hf",
        logprobs=1
    )

    local_response = local_handler.completion(test_prompt)
    print("\nLOCAL RESPONSE:")
    print(local_response)

    # Ensure the response has 'choices' attribute
    assert hasattr(remote_response, "choices"), "Remote response missing 'choices'"
    assert hasattr(local_response, "choices"), "Local response missing 'choices'"

    # Ensure 'choices' is a list
    assert isinstance(remote_response.choices, list), "'choices' should be a list in remote response"
    assert isinstance(local_response.choices, list), "'choices' should be a list in local response"
    assert len(remote_response.choices) > 0, "Remote response 'choices' is empty"
    assert len(local_response.choices) > 0, "Local response 'choices' is empty"

    # Ensure the first choice has 'message' and 'logprobs'
    assert hasattr(remote_response.choices[0], "message"), "Remote response missing 'message' in choices[0]"
    assert hasattr(local_response.choices[0], "message"), "Local response missing 'message' in choices[0]"
    assert hasattr(remote_response.choices[0].message, "content"), "Remote response missing 'content' in message"
    assert hasattr(local_response.choices[0].message, "content"), "Local response missing 'content' in message"

    assert hasattr(remote_response.choices[0], "logprobs"), "Remote response missing 'logprobs' in choices[0]"
    assert hasattr(local_response.choices[0], "logprobs"), "Local response missing 'logprobs' in choices[0]"
    assert isinstance(remote_response.choices[0].logprobs, dict), "'logprobs' should be a dictionary in remote response"
    assert isinstance(local_response.choices[0].logprobs, dict), "'logprobs' should be a dictionary in local response"
    assert "content" in remote_response.choices[0].logprobs, "Remote response missing 'content' in logprobs"
    assert "content" in local_response.choices[0].logprobs, "Local response missing 'content' in logprobs"

    print("\n✅ Test passed: Both remote and local responses follow the same structure.")
