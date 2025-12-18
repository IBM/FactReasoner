from __future__ import annotations

import json
import time
import uuid
from typing import List, Union
from concurrent import futures

import boto3
from botocore.config import Config

from litellm.types.utils import (
    ModelResponse,
    Choices,
    Message,
    Usage,
    ChoiceLogprobs,
    ChatCompletionTokenLogprob,
    TopLogprob,
)


class BedrockLlamaWithLogprobsClient:
    """
    Thin wrapper around a Bedrock imported Llama model that:
    - Calls Bedrock directly via boto3
    - Requests logprobs
    - Returns responses in LiteLLM's ModelResponse format

    completion(prompts):
      - if prompts is str  -> ModelResponse
      - if prompts is list -> List[ModelResponse]
    """

    def __init__(
        self,
        model_arn: str,
        prompt_begin: str,
        prompt_end: str,
        region: str = "us-east-1",
    ):
        self.model_arn = model_arn
        self.prompt_begin = prompt_begin
        self.prompt_end = prompt_end

        config = Config(
            region_name=region,
            retries={"max_attempts": 20, "mode": "adaptive"},
        )
        self.client = boto3.client(
            "bedrock-runtime",
            config=config,
        )

    def completion(
        self,
        prompts: Union[str, List[str]],
        max_gen_len: int = 4096,
        temperature: float = 0.0,
        **kwargs,
    ) -> Union[ModelResponse, List[ModelResponse]]:
        """
        Accept either a single prompt (str) or list of prompts.
        Returns a single ModelResponse or a list[ModelResponse] accordingly.

        NOTE: the BedRock chat API does not return logprobs if the `seed` parameter
        is set so it will be passed to the `_single_completion` helper.
        """
        if isinstance(prompts, str):
            return self._single_completion(
                prompt=prompts,
                temperature=temperature,
                max_gen_len=max_gen_len,
            )

        with futures.ThreadPoolExecutor(max_workers=25) as executor:
            responses = list(
                executor.map(
                    lambda prompt: self._single_completion(
                        prompt=prompt,
                        temperature=temperature,
                        max_gen_len=max_gen_len,
                    ),
                    prompts,
                ),
            )

        return responses

    def render_prompt(self, user_input: str) -> str:
        """
        Renders the full prompt given user input, using the prompt template and prompt end.
        """
        return f"{self.prompt_begin}\n\n{user_input}{self.prompt_end}"

    def _single_completion(
        self,
        prompt: str,
        temperature: float,
        max_gen_len: int = 2048,
    ) -> ModelResponse:
        """
        Single-prompt helper:
        - Calls Bedrock with return_logprobs=True
        - Converts the response into a LiteLLM ModelResponse
        """

        STOP = "<|eot_id|>"

        body = {
            "prompt": f"{self.render_prompt(prompt)}{STOP}",
            "max_gen_len": max_gen_len,
            "temperature": temperature,
            "return_logprobs": True,
        }

        start = time.time()
        try:
            resp = self.client.invoke_model(
                modelId=self.model_arn,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
        except self.client.exceptions.ModelNotReadyException:
            raise Exception(
                "Error: Bedrock model is not ready. Please try again in a few minutes"
            )
        except Exception as e:
            print("Unhandled Bedrock exception:", str(e))

        request_id = resp["ResponseMetadata"]["RequestId"]
        raw_bytes = resp["body"].read()
        payload = json.loads(raw_bytes)

        # Example payload shape:
        # {
        #   "generation": "- The Moon is a spherical rocky body.\n- The Moon has a mean equatorial radius of approximately 1,738
        #                  kilometers. # \n- The Moon has a mean equatorial radius of approximately 1, 080 miles.\n- The Moon has a
        #                  diameter of roughly 3, 476 kilometers. # \n- The Moon has a diameter of roughly 2, 160 miles.",
        #   "prompt_token_count": 820,
        #   "generation_token_count": 72,
        #   "stop_reason": "stop",
        #   "logprobs": [
        #         {
        #             "12": -0.002968668704852462
        #         },
        #         {
        #             "578": -0.006308285985141993
        #         },
        #         ...
        #         {
        #             "13": -0.012077654711902142
        #         },
        #         {
        #             "128009": -0.6430296301841736
        #         }
        #     ]
        # }

        # -------------------------

        # Detect response envelope
        # -------------------------
        is_native = "generation" in payload or "prompt_token_count" in payload
        is_openai_like = "choices" in payload and isinstance(payload["choices"], list)

        # -------------------------
        # Extract core fields
        # -------------------------
        if is_native:
            # Native CMI llama schema
            generation_text = payload.get("generation", "").split(STOP, 1)[0]
            finish_reason = payload.get("stop_reason", "stop")

            prompt_tokens = int(payload.get("prompt_token_count", 0) or 0)
            completion_tokens = int(payload.get("generation_token_count", 0) or 0)
            total_tokens = prompt_tokens + completion_tokens

            # Logprobs: list[dict[token_id -> logprob]]
            raw_logprobs = payload.get("logprobs")

            response_id = f"chatcmpl-{request_id}"
            created = int(time.time())
            model_name = str(payload.get("model") or self.model_arn)

        elif is_openai_like:
            # OpenAI-style text_completion schema
            choices = payload.get("choices") or []
            first = choices[0] if choices else {}

            generation_text = first.get("text", "") or ""
            finish_reason = first.get("finish_reason", "stop") or "stop"

            usage = payload.get("usage", {}) or {}
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            total_tokens = int(
                usage.get("total_tokens", prompt_tokens + completion_tokens) or 0
            )

            # In this envelope, logprobs (if any) live on the choice
            raw_logprobs = first.get("logprobs")

            response_id = payload.get("id", f"chatcmpl-{request_id}")
            created = int(payload.get("created", int(time.time())))
            model_name = str(payload.get("model") or self.model_arn)

        else:
            # Unknown shape: fall back safely
            generation_text = ""
            finish_reason = "stop"
            prompt_tokens = completion_tokens = total_tokens = 0
            raw_logprobs = None
            response_id = f"chatcmpl-{request_id}"
            created = int(time.time())
            model_name = str(self.model_arn)

        # -------------------------
        # Map token logprobs → LiteLLM
        # -------------------------
        token_logprobs_list: List[ChatCompletionTokenLogprob] = []
        if raw_logprobs:
            # Expecting list of dicts: one per generated token position.
            for token_dist in raw_logprobs:
                if not token_dist:
                    continue
                # Most likely token id + its logprob
                token_id, logp = max(token_dist.items(), key=lambda kv: kv[1])
                # Full top-k distribution we received (often it's all tokens considered)
                top_logprobs_objs: List[TopLogprob] = [
                    TopLogprob(token=str(tid), logprob=float(lp), bytes=None)
                    for tid, lp in token_dist.items()
                ]
                token_logprobs_list.append(
                    ChatCompletionTokenLogprob(
                        token=str(
                            token_id
                        ),  # numeric id as string; decode only if you wire a tokenizer
                        logprob=float(logp),
                        top_logprobs=top_logprobs_objs,
                        bytes=None,
                    )
                )

        choice_logprobs = (
            ChoiceLogprobs(content=token_logprobs_list) if token_logprobs_list else None
        )

        # -------------------------
        # Build LiteLLM ModelResponse
        # -------------------------
        model_response = ModelResponse(
            id=response_id,
            created=created,
            model=model_name,
            object="chat.completion",
            choices=[
                Choices(
                    index=0,
                    finish_reason=finish_reason,
                    message=Message(role="assistant", content=generation_text),
                    logprobs=choice_logprobs,
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
        )

        # Keep the raw for debugging
        model_response._hidden_params = {"original_response": payload}
        return model_response
