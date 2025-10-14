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

# Context summarization using LLMs

import os
import string
import time
from typing import Any, Dict, List

import numpy as np
from dotenv import load_dotenv
from fm_factual.llm_handler import LLMHandler
from fm_factual.utils import (DEFAULT_PROMPT_BEGIN, DEFAULT_PROMPT_END, MODELS,
                              dotdict, extract_first_code_block, strip_string)
from tqdm import tqdm

from litellm.exceptions import RateLimitError

# v1
CONTEXT_SUMMARIZATION_PROMPT1 = """{_PROMPT_BEGIN_PLACEHOLDER}

Your task is to summarize the CONTEXT with respect to the ATOM.

Instructions: 
Follow the steps below for CONTEXT summarization:
1. The ATOM can be true, false or not verifiable according to the SUMMARY.
2. It is very possible that no relevant information about the ATOM or related to the ATOM can be found in the CONTEXT. In this case, the SUMMARY must be: "None".
3. If the CONTEXT does not provide information about the ATOM, or if the CONTEXT does not mention anything related to the ATOM, the SUMMARY must be: "None".
4. If the CONTEXT provides information about the ATOM, the SUMMARY must contain the most relevant information of the CONTEXT and be such that we can fact-check the ATOM using this SUMMARY. 
5. The SUMMARY must not use reported speech to refer to the CONTEXT, for instance the SUMMARY must NOT state: "according to the context", "this context mentions", or "this article outlines", but instead the SUMMARY must only summarize the CONTEXT.
6. If the CONTEXT provides information about the ATOM, provide the SUMMARY and wrap the SUMMARY in a markdown code block.
7. If the CONTEXT does not provide information about the ATOM, the SUMMARY must only provide "None". Provide "None" and wrap it in a markdown code block. Do not mention that the context does not provide any information about the atom. Do not provide anything else.


Example 1:
CONTEXT:
+ Sense and Sensibility + Sense and Sensibility is a novel by Jane \
Austen , published in 1811 . + Jane Austen + Jane Austen ( 16 December 1775 - 18 July \
1817 ) was an English novelist known primarily for her six major novels , which interpret , \
critique and comment upon the British landed gentry at the end of the 18th century .

ATOM:
Sense and Sensibility was published in the summer of 1811.

SUMMARY:
```
Sense and Sensibility was published in 1811, however it is not known whether it \
has been published in summer.
```

Example 2:
CONTEXT:
+ Filmfare + Filmfare is an English-language , tabloid-sized magazine \
about Hindi-language cinema , popularly known as Bollywood . + Bollywood + Bollywood \
is the sobriquet for India 's Hindi language film industry , based in the city of Mumbai , \
Maharashtra .

ATOM: 
Filmfare is about cheese.

SUMMARY: 
```
Filmfare is about Hindi-language cinema, not about cheese.
```

Example 3:
CONTEXT:
+ 19th G7 summit + The Group of Seven ( G7 ) was an unofficial forum \
which brought together the heads of the richest industrialized countries : France , Germany \
, Italy , Japan , the United Kingdom , the United States , Canada ( since 1976 ) and the \
President of the European Commission ( starting officially in 1981 ) .

ATOM:
The 19th G7 summit only included Russia.

SUMMARY: 
```
The 19th G7 summit did not only include Russia, but also the heads of the six \
other richest industrialized countries and the President of the European Commission.
```

Example 4:
CONTEXT:
The Amazon rainforest, often referred to as the "lungs of the Earth," spans over 5.5 million square kilometers across nine countries. \
It is home to millions of species, many of which are yet to be discovered. The rainforest plays a crucial role in global oxygen production \
and carbon dioxide absorption. However, it faces severe threats from deforestation, illegal mining, and climate change. Conservation efforts \
are ongoing, with governments, environmental organizations, and indigenous communities working together to protect this vital ecosystem. 

ATOM:
Quantum mechanics describes the behavior of particles at the smallest scales, where classical physics no longer applies.

SUMMARY:
```
None
```

Example 5:
CONTEXT:
+ Artemis + She was the Hellenic goddess of the hunt , wild animals , \
wilderness , childbirth , virginity and protector of young girls , bringing and relieving \
disease in women ; she often was depicted as a huntress carrying a bow and arrows .

ATOM:
Zeus was the creator of Nazgul.

SUMMARY:
```
None
```

YOUR TASK:
CONTEXT:
{_CONTEXT_PLACEHOLDER}

ATOM:
{_ATOM_PLACEHOLDER}

SUMMARIZED CONTEXT:{_PROMPT_END_PLACEHOLDER}
"""


class ContextSummarizer:
    """
    Context summarization given the atom.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-instruct",
        method: str = "logprobs",
        prompt_version: str = "v1",
        llm_inference_provider: str = "rits",
    ):
        """
        Args:
            mode: str
                The name/id of the model.
            prompt_version: str
                The prompt version used. Allowed values are v1.
        """

        self.model = model
        self.prompt_version = prompt_version

        self.model_info = MODELS[llm_inference_provider][
            model
        ]  # only used to ger prompt begin/end
        self.prompt_begin = self.model_info.get("prompt_begin", DEFAULT_PROMPT_BEGIN)
        self.prompt_end = self.model_info.get("prompt_end", DEFAULT_PROMPT_END)
        self.model_id = self.model_info.get("model_id", None)

        self.llm_handler = LLMHandler(
            self.model, llm_inference_provider=llm_inference_provider
        )

        print(
            f"[ContextSummarizer] Using LLM on {llm_inference_provider}: {self.model_id}"
        )
        print(f"[ContextSummarizer] Using prompt version: {self.prompt_version}")

    def make_prompt(self, atom: str, context: str):
        if self.prompt_version == "v1":
            prompt = CONTEXT_SUMMARIZATION_PROMPT1.format(
                _ATOM_PLACEHOLDER=atom,
                _CONTEXT_PLACEHOLDER=context,
                _PROMPT_BEGIN_PLACEHOLDER=self.prompt_begin,
                _PROMPT_END_PLACEHOLDER=self.prompt_end,
            )
        else:
            raise ValueError(
                f"ContextSummarizer: Unknow prompt version {self.prompt_version}"
            )
        prompt = strip_string(prompt)
        return prompt

    def run(self, contexts: List[str], atom: str, credentials: Dict[str, Any]):
        generated_texts = []
        generated_logprobs = []
        prompts = [
            self.make_prompt(atom, context) for context in contexts if context != ""
        ]
        print(f"[ContextSummarizer] Prompts created: {len(prompts)}")

        batch_size = 8
        with tqdm(total=len(prompts), desc="Summarization", unit="prompts") as pbar:
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i : i + batch_size]
                try:
                    batch_responses = self.llm_handler.batch_completion(
                        batch_prompts,
                        credentials=credentials,
                        logprobs=True,
                        temperature=0,
                        seed=42,
                    )
                    for response in batch_responses:
                        generated_texts.append(response.choices[0].message.content)
                        generated_logprobs.append(response.choices[0].logprobs['content'])
                    time.sleep(1) # Add a 1-second delay between batches
                except (RateLimitError, Exception) as e:
                    print(f"An error occurred on batch starting at index {i}: {e}")
                    # Re-raise the exception to halt the entire process
                    raise
                finally:
                    pbar.update(len(batch_prompts))

        summaries = []
        for text, logprobs in zip(generated_texts, generated_logprobs):

            if text is not None and logprobs is not None:
                summary = extract_first_code_block(text, ignore_language=True)
                logprob_sum = 0.0
                generated_tokens = logprobs[:-1]
                for token in generated_tokens:  # last token is just <|eot_id|>
                    token = dotdict(token)
                    logprob_sum += token.logprob
                probability = np.exp(logprob_sum / len(generated_tokens))
            else:
                summary = ""
                probability = 0.0
            summaries.append({"summary": summary, "probability": probability})

        final_summaries = [
            {"summary": context, "probability": 1.0} for context in contexts
        ]
        j = 0
        # swap out the summaries
        for i in range(len(final_summaries)):
            if final_summaries[i]["summary"] != "":
                final_summaries[i]["summary"] = summaries[j]["summary"]
                final_summaries[i]["probability"] = summaries[j]["probability"]
                j += 1

        for summary in final_summaries:
            if (
                (len(summary["summary"]) > 0)
                and (summary["summary"] != "None")
                and (not summary["summary"][-1] in string.punctuation)
            ):
                summary["summary"] += "."

        outputs = []
        for i in range(len(contexts)):
            if (
                len(final_summaries[i]["summary"]) > 0
                and final_summaries[i]["summary"] != "None"
            ):
                outputs.append(
                    dict(
                        summary=final_summaries[i]["summary"],
                        context=contexts[i],
                        probability=final_summaries[i]["probability"],
                    )
                )
            else:
                outputs.append(
                    dict(
                        summary="",
                        context=contexts[i],
                        probability=final_summaries[i]["probability"],
                    )
                )

        return outputs


    def runall(self, contexts: List[List[str]], atoms: List[str], credentials: Dict[str, Any]):

        n = len(contexts)
        generated_texts = []
        generated_logprobs = []
        prompts = [
            self.make_prompt(atom, context)
            for i, atom in enumerate(atoms)
            for context in contexts[i]
            if context != ""
        ]
        messages = [[dict(role="user", content=prompt)] for prompt in prompts]
        print(f"[ContextSummarizer] Prompts created: {len(prompts)}")

        batch_size = 8
        with tqdm(total=len(prompts), desc="Summarization", unit="prompts") as pbar:
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i : i + batch_size]
                try:
                    batch_responses = self.llm_handler.batch_completion(
                        batch_prompts,
                        credentials=credentials,
                        logprobs=True,
                        temperature=0,
                        seed=42
                    )
                    for response in batch_responses:
                        generated_texts.append(response.choices[0].message.content)
                        generated_logprobs.append(response.choices[0].logprobs["content"])
                    time.sleep(1)
                except (RateLimitError, Exception) as e:
                    print(f"An error occurred on batch starting at index {i}: {e}")
                    # Re-raise the exception to halt the entire process
                    raise
                finally:
                    pbar.update(len(batch_prompts))

        summaries = []
        for text, logprobs in zip(generated_texts, generated_logprobs):

            if text is not None and logprobs is not None:
                summary = extract_first_code_block(text, ignore_language=True)
                logprob_sum = 0.0
                generated_tokens = logprobs[:-1]
                for token in generated_tokens:  # last token is just <|eot_id|>
                    token = dotdict(token)
                    logprob_sum += token.logprob
                probability = np.exp(logprob_sum / len(generated_tokens))
            else:
                summary = ""
                probability = 0.0
            summaries.append({"summary": summary, "probability": probability})

        final_summaries = [
            {"summary": context, "probability": 1.0}
            for contex in contexts
            for context in contex
        ]

        j = 0
        for i in range(len(final_summaries)):
            if final_summaries[i]["summary"] != "":
                final_summaries[i]["summary"] = summaries[j]["summary"]
                final_summaries[i]["probability"] = summaries[j]["probability"]
                j += 1

        for summary in final_summaries:
            if (
                (len(summary["summary"]) > 0)
                and (summary["summary"] != "None")
                and (not summary["summary"][-1] in string.punctuation)
            ):
                summary["summary"] += "."

        k = 0
        outputs = []
        for j in range(n):
            output = [
                (
                    {
                        "summary": final_summaries[k + i]["summary"],
                        "context": contexts[j][i],
                        "probability": final_summaries[k + i]["probability"],
                    }
                    if (
                        len(final_summaries[k + i]["summary"]) > 0
                        and final_summaries[k + i]["summary"] != "None"
                    )
                    else {
                        "summary": "",
                        "context": contexts[j][i],
                        "probability": final_summaries[k + i]["probability"],
                    }
                )
                for i in range(len(contexts[j]))
            ]
            outputs.append(output)
            k += len(contexts[j])
        return outputs


if __name__ == "__main__":

    model = "llama-3.3-70b-instruct"
    prompt_version = "v1"
    llm_inference_provider: str = "watsonx"
    credentials = {"WATSONX_API_KEY": ".."}
    summarizer = ContextSummarizer(
        model=model,
        prompt_version=prompt_version,
        llm_inference_provider=llm_inference_provider,
    )

    atom = "The city council has approved new regulations for electric scooters."
    contexts = [
        "In the past year, the city had seen a rapid increase in the use of electric scooters. They seemed like a perfect solution to reduce traffic and provide an eco-friendly transportation option. However, problems arose quickly. Riders often ignored traffic laws, riding on sidewalks, and causing accidents. Additionally, the scooters were frequently left haphazardly around public spaces, obstructing pedestrians. City officials were under increasing pressure to act, and after numerous public consultations and debates, the council finally passed new regulations. The new rules included mandatory helmet use, restricted riding areas, and designated parking zones for scooters. The implementation of these regulations was expected to improve safety and the overall experience for both scooter users and pedestrians.",
        "With the rise of shared electric scooters and bikes in cities across the country, municipal governments have been scrambling to develop effective policies to handle this new form of transportation. Many cities, including the local area, were caught off guard by the sudden popularity of scooters, and their original infrastructure was ill-prepared for this new trend. Early attempts to regulate the scooters were chaotic and ineffective, often leading to public frustration. Some cities took drastic steps, such as banning scooters altogether, while others focused on infrastructure improvements, like adding dedicated lanes for scooters and bicycles. The city council's recent approval of new regulations was part of a larger effort to stay ahead of the curve and provide a balanced approach to regulating modern transportation options while encouraging their growth. These regulations were designed not only to ensure the safety of riders but also to integrate the scooters more seamlessly into the city's broader transportation network.",
        "",
        "The sun hung low in the sky, casting a warm golden glow over the city as Emily wandered through the bustling streets, her mind drifting between thoughts of the past and the uncertain future. She passed the familiar old bookstore that always smelled like aged paper and adventure, a place she used to frequent with her grandmother, whose absence still left a hollow ache in her chest. The air was thick with the scent of coffee wafting from nearby cafés, mingling with the earthy smell of rain that had yet to fall. Despite the noise of the traffic, the chatter of pedestrians, and the hum of city life, there was a strange sense of stillness around her. It was as if time had slowed down, giving her a moment to breathe, to collect her scattered thoughts. She glanced up at the towering buildings that seemed to stretch endlessly into the sky, their glass facades reflecting the fading light. Everything around her was in constant motion, yet she felt an unexpected calm. Her phone buzzed in her pocket, pulling her back to reality, and she sighed, reluctantly slipping it out. It was a message from her best friend, asking if they still wanted to meet up later.",
    ]

    results = summarizer.run(contexts, atom, credentials)
    for i, elem in enumerate(results):
        context = elem["context"]
        summary = elem["summary"]
        probability = elem["probability"]
        print(
            f"\n\nContext #{i + 1}: {context}\n--> Summary #{i + 1}: {summary}\n--> Probability #{i + 1}: {probability}"
        )

    print()

    # from summac.model_summac import SummaCZS, SummaCConv
    # import nltk
    # nltk.download('punkt_tab')

    # model_zs = SummaCZS(granularity="sentence", model_name="vitc", device="cpu") # If you have a GPU: switch to: device="cuda"
    # model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cpu", start_file="default", agg="mean")

    # score_zs1 = model_zs.score([results[0]["context"]], [results[0]["summary"]])
    # score_conv1 = model_conv.score([results[0]["context"]], [results[0]["summary"]])
    # print("[Summary 1] SummaCZS Score: %.3f; SummacConv score: %.3f" % (score_zs1["scores"][0], score_conv1["scores"][0])) # [Summary 1] SummaCZS Score: 0.582; SummacConv score: 0.536

    # score_zs2 = model_zs.score([results[0]["context"]], [results[0]["summary"]])
    # score_conv2 = model_conv.score([results[0]["context"]], [results[0]["summary"]])
    # print("[Summary 2] SummaCZS Score: %.3f; SummacConv score: %.3f" % (score_zs2["scores"][0], score_conv2["scores"][0])) # [Summary 2] SummaCZS Score: 0.877; SummacConv score: 0.709

    # atoms = [
    #     "The city council has approved new regulations for electric scooters.",
    #     "The team announced a new partnership with a major tech company.",
    #     "They've developed a new app that helps manage personal finances more effectively."
    # ]

    # contexts = [
    #     [
    #         "In the past year, the city had seen a rapid increase in the use of electric scooters. They seemed like a perfect solution to reduce traffic and provide an eco-friendly transportation option. However, problems arose quickly. Riders often ignored traffic laws, riding on sidewalks, and causing accidents. Additionally, the scooters were frequently left haphazardly around public spaces, obstructing pedestrians. City officials were under increasing pressure to act, and after numerous public consultations and debates, the council finally passed new regulations. The new rules included mandatory helmet use, restricted riding areas, and designated parking zones for scooters. The implementation of these regulations was expected to improve safety and the overall experience for both scooter users and pedestrians.",
    #         "",
    #         "As the sun began to set, Sarah made her way to the park to meet up with friends after work. As she walked past the entrance, she noticed several electric scooters parked in random spots. A few of them were right in the middle of the sidewalk, forcing pedestrians to step around them. She rolled her eyes, knowing that the city had been discussing new regulations for scooters for months. Sarah, who had lived in the city for several years, had witnessed how technology could both improve and complicate life. She remembered the early days of rideshare programs like Uber, which were initially unregulated and caused a similar public uproar. Just like with scooters, city officials had scrambled to come up with solutions that balanced convenience and safety. The new scooter regulations were an important step, but Sarah couldn't help but wonder if it would be enough to prevent further accidents. She had heard stories of people crashing into trees or getting hurt due to careless riders. With a sigh, she grabbed her phone to send a quick text to her friends, secretly hoping they wouldn't decide to ride scooters tonight.",
    #     ],
    #     [
    #         "When the announcement was made, it sent ripples of excitement through both the sports and tech communities. The team had been in talks with several major companies, but this deal with the tech giant was unexpected. The new partnership was part of a larger strategy to modernize the team's infrastructure and fan experience. With the help of the tech company, the team would implement advanced analytics to improve training techniques, player performance tracking, and even fan engagement through cutting-edge virtual reality experiences. As part of the deal, the team also planned to unveil a revamped app that would offer fans personalized content, live stats, and direct interactions with players. For the tech company, this partnership was a prime opportunity to showcase its innovative solutions on a global stage, potentially leading to millions of new customers in the sports sector.",
    #         "Behind the scenes, the negotiations had been intense. The team’s management, along with advisors from the tech company, spent months hammering out the terms of the deal. Initially, there had been resistance on both sides, with each party trying to secure the most advantageous terms. The team was looking for more than just financial support; they wanted access to cutting-edge technologies that could set them apart from their competitors. The tech company, on the other hand, was eager to tap into the rapidly growing sports market, which had proven to be highly lucrative. After several rounds of talks, including visits to the tech company’s headquarters and multiple brainstorming sessions, a partnership was finally agreed upon. Both parties celebrated the deal, knowing that this collaboration could change the way the team trained and interacted with its fans. The team would be the first in their league to introduce such a robust tech-driven approach to player development, and the partnership was expected to serve as a model for other organizations to follow.",
    #         "Jonathan had always been skeptical of corporate sponsorships in sports. To him, they felt like a distraction from the real essence of the game. He had grown up watching his favorite teams battle it out on the field without the constant bombardment of tech ads or virtual reality experiences. As he sat in the stadium, surrounded by fans excited about the team's new partnership, he couldn't help but feel uneasy. While he understood the business side of things, he worried that the essence of the sport would be lost in the shuffle of corporate interests. Jonathan's concerns were not unique; many fans shared his belief that sports should remain a pure form of entertainment, untainted by outside influences. But as the announcement about the new partnership came over the loudspeaker, he tried to push aside his doubts. Maybe, just maybe, the team would find a way to balance innovation with tradition.",
    #     ],
    #     [
    #         "",
    #         "Ellen had always been diligent about saving for the future, but when her financial advisor retired, she found herself struggling to keep track of her savings and investments. The spreadsheets she once relied on seemed outdated, and she couldn't find a budgeting tool that worked for her lifestyle. It was then that a friend recommended the new app, which promised to simplify everything. Intrigued, Ellen downloaded it and began the process of linking her bank accounts. To her surprise, the app immediately pulled in her transaction history and categorized her expenses. She was impressed by the level of detail the app provided. It offered insights into her spending habits, helping her identify areas where she could cut back. The best part? The app also included tips on how to invest her savings, using algorithms to recommend strategies that aligned with her risk tolerance. Ellen felt empowered by the app’s comprehensive approach to personal finance and began using it religiously. She also recommended it to her friends and family, confident that it could help them take control of their financial futures as well.",
    #         "Brian had never been particularly interested in finances. As a freelance graphic designer, his income varied from month to month, making it difficult to plan his spending. He lived in a small apartment, often struggled to pay bills on time, and would occasionally splurge on a new piece of equipment for his studio without thinking about the long-term impact on his budget. While he had heard about the new finance app, he was skeptical. After all, he didn’t want a program telling him what to do with his money. However, after an unexpected tax bill arrived, Brian realized that he needed to take a more serious approach to managing his finances. He decided to give the app a try. At first, he found it annoying that the app tracked every single transaction, but soon he began to appreciate its guidance. The app helped him set realistic financial goals, and with its alerts and reminders, he managed to avoid late fees. While Brian still wasn’t thrilled by the idea of budgeting, he had to admit that the app had made his financial life significantly easier."
    #     ]

    # ]

    # results = summarizer.runall(contexts, atoms)
    # print(f"Number of results: {len(results)}")
    # for i, result in enumerate(results):
    #     for j, elem in enumerate(result):
    #         context = elem["context"]
    #         summary = elem["summary"]
    #         probability = elem["probability"]
    #         print(f"\n\nContext #{i + 1}.{j + 1}: {context}\n--> Summary #{i + 1}.{j + 1}: {summary}\n--> Probability #{i + 1}.{j + 1}: {probability}")

    # print()

    print("Done.")
