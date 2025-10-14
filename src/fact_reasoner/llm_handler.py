import os

import torch
from dotenv import load_dotenv
from fm_factual.utils import DEFAULT_PROMPT_BEGIN, DEFAULT_PROMPT_END, MODELS

import litellm

GPU = torch.cuda.is_available()
DEVICE = GPU*"cuda" + (not GPU)*"cpu"
# litellm._turn_on_debug()

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class LLMHandler:
    def __init__(self, 
                 model: str,  
                 llm_inference_provider: str,
                 dtype="auto",
                 **default_kwargs
    ):
        """
        Initializes the LLM handler.

        :param model_id: Model name or path.
        :param RITS: If True, use RITS model; if False, load using vLLM.
        :param default_kwargs: Default parameters (e.g., temperature, max_tokens) to pass to completion calls.
        """
        self.llm_inference_provider = llm_inference_provider
        self.default_kwargs = default_kwargs  # Store common parameters for completions

        if not os.environ.get("_DOTENV_LOADED"):
            load_dotenv(override=True) 
            os.environ["_DOTENV_LOADED"] = "1"
            
        if self.llm_inference_provider == "vllm":
            from vllm import LLM
            self.model_info = MODELS["huggingface"][model]
            self.model_id = self.model_info.get("model_id", None)
            assert self.model_id is not None
            print(f"Loading local model with vLLM: {self.model_id}...")
            self.llm = LLM(model=self.model_id, device=DEVICE, dtype=dtype)  # Load model using vLLM

        elif self.llm_inference_provider == "rits":
            self.API_KEY = os.getenv("RITS_API_KEY")
        elif llm_inference_provider == "watsonx":
            self.API_KEY = os.getenv("WATSONX_API_KEY")
        elif llm_inference_provider == "openai":
            self.API_KEY = os.getenv("OPENAI_API_KEY")

        self.model_info = MODELS[self.llm_inference_provider][model]
        self.prompt_template = self.model_info.get("prompt_template", None)
        self.max_new_tokens = self.model_info.get("max_new_tokens", None)
        self.api_base = self.model_info.get("api_base", None)
        self.model_id = self.model_info.get("model_id", None)
        self.prompt_begin = self.model_info.get("prompt_begin", DEFAULT_PROMPT_BEGIN)
        self.prompt_end = self.model_info.get("prompt_end", DEFAULT_PROMPT_END)

        if llm_inference_provider == "rits" or llm_inference_provider == "watsonx" or llm_inference_provider == "openai":
            assert self.prompt_template is not None \
                and self.max_new_tokens is not None \
                and self.api_base is not None \
                and self.model_id is not None


    def completion(self, prompt, credentials, **kwargs):
        """
        Generate a response using the RITS API (if RITS=True) or the local model.

        :param message: The prompt.
        :param kwargs: Additional parameters for completion (overrides defaults).
        """
        return self._call_model(prompt, credentials, **kwargs)

    def batch_completion(self, prompts, credentials, **kwargs):
        """
        Generate responses in batch using the RITS API (if RITS=True) or the local model.

        :param prompts: List of prompts.
        :param kwargs: Additional parameters for batch completion.
        """
        return self._call_model(prompts, credentials, **kwargs)


    def _call_model(self, prompts, credentials, num_retries=5, **kwargs):
        """
        Handles both single and batch generation using passed-in credentials.

        :param prompts: A single string or a list of strings.
        :param kwargs: Additional parameters.
        """
        # If no credentials are provided, use an empty dict to avoid errors
        if credentials is None:
            credentials = {}

        params = {
            "temperature": 0,
            "seed": 42,
            "top_p": 0.95,
            **self.default_kwargs,
            **kwargs
        }

        auth_params = {} 
        extra_headers = {}
        api_inference_provider = True

        if self.llm_inference_provider == "openai":
            auth_params = {
                "api_key": credentials.get("OPENAI_API_KEY"),
            }
        elif self.llm_inference_provider == "watsonx":
            auth_params = {
                "api_key": credentials.get("WATSONX_API_KEY"),
            }
        elif self.llm_inference_provider == "rits":
            if "RITS_API_KEY" in credentials:
                extra_headers["RITS_API_KEY"] = credentials["RITS_API_KEY"]
        else:
            # if we are not using these providers specifically it will trigger the vllm call.
            api_inference_provider = False

        # Filter out any keys that were not provided
        auth_params = {k: v for k, v in auth_params.items() if v is not None}
        
        merged_params = {**params, **auth_params}
        if api_inference_provider:
            # Ensure we always send a list to batch_completion
            if isinstance(prompts, str):
                return litellm.completion(
                    model=self.model_id,
                    api_base=self.api_base,
                    messages=[{"role": "user", "content": prompts}],
                    num_retries=num_retries,
                    extra_headers=extra_headers,
                    **merged_params,
                )

            return litellm.batch_completion(
                model=self.model_id,
                api_base=self.api_base,
                messages=[[{"role": "user", "content": p}] for p in prompts],
                num_retries=num_retries,
                extra_headers=extra_headers,
                **merged_params,
            )

        else:
            from vllm import SamplingParams

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
        model="llama-3.3-70b-instruct",
        llm_inference_provider="watsonx" # rits
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

    # local_handler = LLMHandler(
    #     model="facebook/opt-350m",
    #     RITS=False,
    #     logprobs=1
    # )

    # local_response = local_handler.completion(test_prompt)
    # print("\nLOCAL RESPONSE:")
    # print(local_response)

    # # Ensure the response has 'choices' attribute
    # assert hasattr(remote_response, "choices"), "Remote response missing 'choices'"
    # assert hasattr(local_response, "choices"), "Local response missing 'choices'"

    # # Ensure 'choices' is a list
    # assert isinstance(remote_response.choices, list), "'choices' should be a list in remote response"
    # assert isinstance(local_response.choices, list), "'choices' should be a list in local response"
    # assert len(remote_response.choices) > 0, "Remote response 'choices' is empty"
    # assert len(local_response.choices) > 0, "Local response 'choices' is empty"

    # # Ensure the first choice has 'message' and 'logprobs'
    # assert hasattr(remote_response.choices[0], "message"), "Remote response missing 'message' in choices[0]"
    # assert hasattr(local_response.choices[0], "message"), "Local response missing 'message' in choices[0]"
    # assert hasattr(remote_response.choices[0].message, "content"), "Remote response missing 'content' in message"
    # assert hasattr(local_response.choices[0].message, "content"), "Local response missing 'content' in message"

    # assert hasattr(remote_response.choices[0], "logprobs"), "Remote response missing 'logprobs' in choices[0]"
    # assert hasattr(local_response.choices[0], "logprobs"), "Local response missing 'logprobs' in choices[0]"
    # assert isinstance(remote_response.choices[0].logprobs, dict), "'logprobs' should be a dictionary in remote response"
    # assert isinstance(local_response.choices[0].logprobs, dict), "'logprobs' should be a dictionary in local response"
    # assert "content" in remote_response.choices[0].logprobs, "Remote response missing 'content' in logprobs"
    # assert "content" in local_response.choices[0].logprobs, "Local response missing 'content' in logprobs"

    # print("\n✅ Test passed: Both remote and local responses follow the same structure.")
