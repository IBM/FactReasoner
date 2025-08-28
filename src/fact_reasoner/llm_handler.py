from copy import deepcopy
import os
import litellm
from litellm.exceptions import (
    Timeout,
    RateLimitError,
    APIConnectionError,
    APIError,
    ServiceUnavailableError,
    InternalServerError,
)
from litellm.types.utils import Choices, ModelResponse
import torch
from dotenv import load_dotenv
from src.fact_reasoner.utils import RITS_MODELS, DEFAULT_PROMPT_BEGIN, DEFAULT_PROMPT_END, HF_MODELS
from tenacity import Retrying, stop_after_attempt, wait_random_exponential
from typing import cast

GPU = torch.cuda.is_available()
DEVICE = GPU*"cuda" + (not GPU)*"cpu"

GPT_RETRY_PROMPT = """IMPORTANT: This is a retry because the previous attempt got stuck in a reasoning loop and exceeded the maximum token limit. To avoid this, please keep your reasoning very short and aim to provide an answer as early as possible, even if you are uncertain. It is paramount that you return a response in the correct format quickly, regardless of whether it might be imperfect.\n\n"""

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class LLMHandler:
    def __init__(self, 
                 model: str,  
                 RITS: bool = True, 
                 dtype="auto",
                 **default_kwargs
    ):
        """
        Initializes the LLM handler.

        :param model_id: Model name or path.
        :param RITS: If True, use RITS model; if False, load using vLLM.
        :param default_kwargs: Default parameters (e.g., temperature, max_tokens) to pass to completion calls.
        """
        self.RITS = RITS
        self.default_kwargs = default_kwargs  # Store common parameters for completions

        if not self.RITS:
            from vllm import LLM
            self.HF_model_info = HF_MODELS[model]
            self.model_id = self.HF_model_info["model_id"]
            assert self.model_id is not None
            print(f"Loading local model with vLLM: {self.model_id}...")
            self.llm = LLM(model=self.model_id, device=DEVICE, dtype=dtype)  # Load model using vLLM
        else:

            if not os.environ.get("_DOTENV_LOADED"):
                load_dotenv(override=True) 
                os.environ["_DOTENV_LOADED"] = "1"
            
            self.RITS_API_KEY = os.getenv("RITS_API_KEY")

            self.rits_model_info = RITS_MODELS[model]

            self.prompt_template = self.rits_model_info.get("prompt_template", None)
            self.max_new_tokens = self.rits_model_info.get("max_new_tokens", None)
            self.api_base = self.rits_model_info.get("api_base", None)
            self.model_id = self.rits_model_info["model_id"]
            self.prompt_begin = self.rits_model_info.get("prompt_begin", DEFAULT_PROMPT_BEGIN)
            self.prompt_end = self.rits_model_info.get("prompt_end", DEFAULT_PROMPT_END)
            assert self.prompt_template is not None \
                and self.max_new_tokens is not None \
                and self.api_base is not None \
                and self.model_id is not None


    def completion(self, prompt: str | None = None, messages: list[dict[str, str]] = [], **kwargs):
        """
        Generate a response using the RITS API (if RITS=True) or the local model.

        :param prompt: The prompt.
        :param messages: A list of messages in a chat format.
        :param kwargs: Additional parameters for completion (overrides defaults).
        """
        prompts = prompt if prompt is not None else []
        return self._call_model(prompts, messages=messages, **kwargs)

    def batch_completion(self, prompts, **kwargs):
        """
        Generate responses in batch using the RITS API (if RITS=True) or the local model.

        :param prompts: List of prompts.
        :param kwargs: Additional parameters for batch completion.
        """
        responses = self._call_model(prompts, **kwargs)
        assert responses is not None, "Unexpected None responses from model."
        return responses


    def _call_model(
        self,
        prompts: str | list[str] = [],
        messages: list[dict[str, str]] = [],
        num_internal_retries=5,
        retry_after=3,
        num_outer_retries=4,
        expect_content=False,
        **kwargs,
    ):
        """
        Handles both single and batch generation.

        :param prompts: A single string or a list of strings.
        :param messages: A list of messages in a chat format.
        :param kwargs: Additional parameters.
        :param num_internal_retries: The number of litellm completion retries.
        :param retry_after: Retry after value for litellm.
        :param num_outer_retries: The number of outer retries using Tenacity.
        :param expect_content: Whether to expect non-null content on the response.
        """
        params = {
            "temperature": 0,
            "seed": 42,
            # the two above are overwritten if passed
            # as kwargs
            **self.default_kwargs,
            **kwargs
        }  # Merge defaults with provided params

        if self.RITS:
            # Ensure we always send a list to batch_completion
            if messages or isinstance(prompts, str):
                if not messages:
                    prompts = cast(str, prompts)
                    messages = [{"role": "user", "content": prompts}]  # Wrap prompt for compatibility
                got_empty_content = False
                for attempt in Retrying(
                    stop=stop_after_attempt(num_outer_retries),
                    wait=wait_random_exponential(multiplier=30, max=180),
                ):
                    with attempt:
                        current_messages = deepcopy(messages)
                        attempt_number = attempt.retry_state.attempt_number
                        if (
                            "gpt-oss" in self.model_id
                            and got_empty_content
                            and attempt_number > 1
                            and attempt_number > (num_outer_retries - 2)
                        ):
                            # gpt-oss occasionaly starts cycling in its reasoning process
                            # without ever generating an answer. We try to nudge it to
                            # shorten its reasoning here.
                            last_message = current_messages[-1]
                            if last_message["role"] != "user":
                                print(
                                    "WARNING: Unable to adjust prompt for gpt-oss retry."
                                )
                            else:
                                last_message["content"] = (
                                    GPT_RETRY_PROMPT + last_message["content"]
                                )
                                print("INFO: Modified model prompt:")
                                print(last_message["content"])
                        got_empty_content = False
                        response = litellm.completion(
                            model=self.model_id,
                            api_base=self.api_base,
                            messages=current_messages,
                            api_key=self.RITS_API_KEY,
                            num_retries=num_internal_retries,
                            retry_after=retry_after,
                            extra_headers={"RITS_API_KEY": self.RITS_API_KEY},
                            **params,
                        )
                        if any(
                            isinstance(response, c)
                            for c in [
                                Timeout,
                                RateLimitError,
                                APIConnectionError,
                                APIError,
                                ServiceUnavailableError,
                                InternalServerError,
                            ]
                        ):
                            error_name = type(response).__name__
                            print(
                                f"WARNING: Retrying completion due to {error_name}!"
                            )
                            raise ValueError(
                                f"Retrying due to {error_name}"
                            )
                        if expect_content and (
                            not isinstance(response, ModelResponse)
                            or not isinstance(response.choices[0], Choices)
                            or not isinstance(response.choices[0].message.content, str)
                        ):
                            got_empty_content = True
                            if isinstance(response, ModelResponse) and isinstance(
                                response.choices[0], Choices
                            ):
                                # Remove logprobs from the response, as they result
                                # in overly verbose output
                                response.choices[0].logprobs = None
                            print(
                                f"WARNING: Retrying due to unexpected response without content: {response}"
                            )
                            raise ValueError(
                                f"Retrying due to unexpected response without content: {response}"
                            )
                        return response

            got_empty_content = False
            for attempt in Retrying(
                stop=stop_after_attempt(num_internal_retries),
                wait=wait_random_exponential(multiplier=30, max=180),
            ):
                with attempt:
                    current_prompts = deepcopy(prompts)
                    attempt_number = attempt.retry_state.attempt_number
                    if (
                        "gpt-oss" in self.model_id
                        and got_empty_content
                        and attempt_number > 1
                        and attempt_number > (num_outer_retries - 2)
                    ):
                        # gpt-oss occasionaly starts cycling in its reasoning process
                        # without ever generating an answer. We try to nudge it to
                        # shorten its reasoning here.
                        current_prompts = [GPT_RETRY_PROMPT + p for p in current_prompts]
                        print("INFO: Modified model prompts, e.g.:")
                        print(current_prompts[0])
                    got_empty_content = False
                    responses = litellm.batch_completion(
                        model=self.model_id,
                        api_base=self.api_base,
                        messages=[
                            [{"role": "user", "content": p}] for p in current_prompts
                        ],  # Wrap each prompt
                        api_key=self.RITS_API_KEY,
                        num_retries=num_internal_retries,
                        retry_after=retry_after,
                        extra_headers={"RITS_API_KEY": self.RITS_API_KEY},
                        **params,
                    )
                    error_response = next(
                        (
                            r
                            for r in responses
                            if any(
                                isinstance(r, c)
                                for c in [
                                    Timeout,
                                    RateLimitError,
                                    APIConnectionError,
                                    APIError,
                                    ServiceUnavailableError,
                                    InternalServerError,
                                ]
                            )
                        ),
                        None,
                    )
                    if error_response is not None:
                        error_name = type(error_response).__name__
                        print(
                            f"WARNING: Retrying batch completion due to {error_name}!"
                        )
                        raise ValueError(f"Retrying due to {error_name}.")
                    empty_response = next(
                        (
                            r
                            for r in responses
                            if expect_content
                            and (
                                not isinstance(r, ModelResponse)
                                or not isinstance(r.choices[0], Choices)
                                or not isinstance(r.choices[0].message.content, str)
                            )
                        ),
                        None,
                    )
                    if empty_response is not None:
                        got_empty_content = True
                        if isinstance(empty_response, ModelResponse) and isinstance(
                            empty_response.choices[0], Choices
                        ):
                            # Remove logprobs from the response, as they result
                            # in overly verbose output
                            empty_response.choices[0].logprobs = None
                        print(
                            f"WARNING: Retrying batch completion due to unexpected response without content {empty_response}"
                        )
                        raise ValueError(
                            f"Retrying due to unexpected response without content: {empty_response}"
                        )
                    return responses
        else:
            from vllm import SamplingParams

            if messages:
                raise NotImplementedError("Chat-style inputs are not yet supported for vLLM.")

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
        """Recursively print objects, lists, and dicts for deep inspection."""
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
    Test to compare RITS (litellm) and local (vLLM) outputs.
    """
    test_prompt = "What is the capital of France?"

    # RITS (litellm) API
    remote_handler = LLMHandler(
        model="llama-3.1-70b-instruct",
        RITS=True,
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
        RITS=False,
        dtype="half",
        logprobs=1
    )
    """

    local_handler = LLMHandler(
        model="facebook/opt-350m",
        RITS=False,
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
