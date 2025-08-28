from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
import itertools
import math
import regex
from statistics import mean, mode
import threading
from typing import Any, Literal, cast

import json_repair
from litellm.types.utils import Choices, ModelResponse
from tenacity import Retrying, stop_after_attempt, wait_random_exponential
from tqdm.auto import tqdm

from src.fact_reasoner.fact_utils import extract_weighted_scores, extract_probabilistic_labels
from src.fact_reasoner.llm_handler import LLMHandler
from src.fact_reasoner.qag_utils import ComparisonResult, ToolDefinition, repair_tool_calls
from src.fact_reasoner.utils import RITS_MODELS

ANSWER_COMPARISON_LABELS = [
    "equivalent",
    "first implies second",
    "second implies first",
    "contradictory",
    "neutral",
]


@dataclass(frozen=True)
class Answer:
    """
    A class representing an answer to a question.

    Attributes:
        answer: str
            The answer text.
        confidence: float
            The confidence value associated with the answer.
        source_id: str
            The ID of the source that provided the answer.
        source_type: Literal["context", "answer"]
            The type of source that provided the answer.
    """

    answer: str
    confidence: float
    source_id: str
    source_type: Literal["context", "answer"]


@dataclass
class QuestionData:
    """
    A class for storing question data.

    Attributes:
        question: str
            The text of the question.
        relevance: float
            The relevance score for the question.
        answers: list[Answer]
            The list of answers to the given question.
    """

    question: str
    relevance: float
    answers: list[Answer] = field(default_factory=list)

    def merge_answers(self, other: "QuestionData") -> "QuestionData":
        """
        Returns a new QuestionData object with merged answer data.

        Args:
            other: QuestionData
                The QuestionData object from which to merge the answer
                data.

        Returns:
             QuestionData:
                A new QuestionData object with answers from both the
                original and the other QuestionData object.
        """
        return QuestionData(
            question=self.question,
            relevance=self.relevance,
            answers=list(self.answers) + list(other.answers),
        )

    def filter_answers(self, min_confidence: float) -> "QuestionData":
        """
        Returns a new QuestionData object with answers with confidence
        scores larger than the given threshold.

        Args:
            min_confidence: float
                The minimum confidence threshold for the filtering.

        Returns:
            QuestionData:
                A filtered version of the QuestionData object.
        """
        return QuestionData(
            question=self.question,
            relevance=self.relevance,
            answers=[a for a in self.answers if a.confidence >= min_confidence],
        )


@dataclass(frozen=True, kw_only=True)
class BaseAnswerRelation:
    """
    Relation between two answers, not including directionality
    information.

    Attributes:
        label: ComparisonResult
            The label associated with the relation.
        probability: float
            The probability associated with the relation.
        reasoning: str
            The reasoning used to arrive at the relation label.
    """

    label: ComparisonResult
    probability: float
    reasoning: str = "Unknown"


@dataclass(frozen=True)
class AnswerRelation(BaseAnswerRelation):
    """
    Relation between two answers including directionality
    information.

    Attributes:
        label: ComparisonResult
            The label associated with the relation.
        probability: float
            The probability associated with the relation.
        reasoning: str
            The reasoning used to arrive at the relation label.
        fst: Answer
            The source answer of the relation.
        snd: Answer
            The target answer of the relation.
    """

    fst: Answer
    snd: Answer


def print_answers(answers: dict[str, QuestionData]):
    """
    Pretty-prints answers for a set of questions.

    Args:
        answers: dict[str, QuestionData]
            A dictionary mapping questions to their associated Q&A data.
    """
    sorted_qs = sorted(answers.values(), key=lambda q: q.relevance, reverse=True)
    for q in sorted_qs:
        print(f"— {q.question} [Relevance: {q.relevance}]")
        print(
            "  "
            + " | ".join(
                f"A: {a.answer} [Confidence: {a.confidence:.3f}]" for a in q.answers
            )
        )


def print_relations(relations: dict[frozenset[Answer], AnswerRelation]):
    """
    Pretty-prints relations between answers.

    Args:
        relations: dict[frozenset[Answer], AnswerRelation]
            A dictionary mapping frozen set pairs of answers to their
            relations.
    """
    for _, relation in relations.items():
        print(
            f"{relation.fst.answer} ({relation.fst.source_id}) — {relation.snd.answer} ({relation.snd.source_id}) [{relation.label} ({relation.probability:.2f})]"
        )
    print()


QUESTION_EXTRACTION_PROMPT = """{prompt_start}You are an AI assistant specialized in extracting sets of factual questions from a background text given a specific user query.

Your task is to analyze the provided background text and user query, and generate a list of independent, self-contained factual questions that exhaustively cover the factual content in the background text that is relevant to the user query. Each question should correspond to a distinct fact or piece of information related to the query, even indirectly. If the original user query meets the below guidelines while being focused and unambiguous, you should include it as the first entry.

Question formulation guidelines:

Each question must be self-contained and unambiguous. Avoid vague references or pronouns that rely on context from the original text. For example:
* ❌ "he" — use the full name, e.g. "Neil Armstrong"
* ❌ "the company" — use the specific name, e.g. "Microsoft"
* ❌ "in the first paragraph" — avoid referencing the structure or location of information in the text

Questions should be general and format-agnostic. Do not ask for specific units, sources or formats unless they are central to the factual content. For example:
* ❌ "What is the range of the Airbus A380 in nautical miles?" — incorrect, unnecessarily refers to a specific unit
* ❌ "How far can its mighty engines take the Airbus A380?" — incorrect, phrased in an indirect and overly specific way
* ❌ "According to Reuters, what is the range of the Airbus A380?" — incorrect, introduces a specific source that is not essential to the fact
* ✅ "What is the range of the Airbus A380?" — correct, asks for the fact in a general way

Avoid meta-level questions that refer to how the information is presented in the specific text. Your goal is to extract factual questions that are relevant to the query, not to comment on structure, presentation or contradictions. It is acceptable for a question to have multiple possible answers if the background text presents conflicting or ambiguous information.

You should output the questions in the form of a list, with each item starting with "* ". Do not include any other explanations or comments. Refer to the following examples to understand the task and the output format.

Example 1:

User query:
Tell me about Glenn Danzig.

Background text:
Glenn Danzig (born June 23, 1955) is an American singer, songwriter, musician, and record producer. He is the founder of the rock bands Misfits, Samhain, and Danzig. He owns the Evilive record label as well as Verotik, an adult-oriented comic book publishing company.

Extracted questions:
* When was Glenn Danzig born?
* What is the nationality of Glenn Danzig?
* What are the professions of Glenn Danzig?
* Which rock bands were founded by Glenn Danzig?
* What is the name of the record label owned by Glenn Danzig?
* What is the name of an adult-oriented comic book publishing company owned by Glenn Danzig?

Example 2:

User query:
How high is Burj Khalifa?

Background text:
With a total height of 829.8 m (2,722 ft, or just over half a mile) and a roof height (excluding the antenna, but includes a 242.6 m spire) of 828 m (2,717 ft), it is the world's tallest structure.

Extracted questions:
* How high is Burj Khalifa?
* What is the total height of the Burj Khalifa?
* What is the roof height of the Burj Khalifa excluding the antenna?
* What is the height of the spire included in the Burj Khalifa's roof height?
* What is the ranking of the Burj Khalifa among the world's structures in terms of height?

Example 3:

User query:
What is the maximum range of Airbus A380?

Background text:
The Airbus A380 is a very large wide-body airliner, developed and produced by Airbus until 2021. It is the world's largest passenger airliner and the only full-length double-deck jet airliner. The full-length double-deck aircraft has a typical seating for 525 passengers, with a maximum certified capacity for 853 passengers. The quadjet is powered by Engine Alliance GP7200 or Rolls-Royce Trent 900 turbofans providing a range of 15,000 nmi (2,800 km; 5,200 mi). As of December 2021, the global A380 fleet had completed more than 800,000 flights with no fatalities and no hull losses.

Extracted questions:
* What is the maximum range of Airbus A380?

Now, please generate a list of questions and relevance scores for the following query and text. Remember to keep the questions self-contained, format-agnostic, and focused on the facts relevant to the user query.

User query: {query}

Background text: {background_text}

Extracted questions:{prompt_end}"""

QUESTION_REFINEMENT_PROMPT = """{prompt_start}You are an AI assistant specialized in refining and scoring factual questions extracted from background texts in response to a user query.

Your task is to take a list of factual questions generated from one or more background texts, refine them according to the following guidelines and assign them a relevance score. If the original user query is sufficiently focused and unambiguous, you should include it as the first entry in the output.

Refinement Guidelines:

* Generalize overly specific questions:
  * Remove unnecessary references to units, formats or sources unless they are central to the fact. For example:
    * ❌ "What is the range of the Airbus A380 in nautical miles?" — incorrect, unnecessarily refers to a specific unit
    * ❌ "How far can its mighty engines take the Airbus A380?" — incorrect, phrased in an indirect and overly specific way
    * ❌ "According to Reuters, what is the range of the Airbus A380?" — incorrect, introduces a specific source that is not essential to the fact
    * ✅ "What is the range of the Airbus A380?" — correct, asks for the fact in a general way
    * ✅ "What is the range of the Airbus A380 in the Rolls-Royce Trent 900 turbofans configuration?" — correct, asks for a specific aircraft configuration and should not be generalised further
* Merge duplicate or semantically similar questions:
  * Combine questions that ask for the same fact using different wording into a single, clear formulation.
* Ensure clarity and self-containment:
  * Each question should be unambiguous and understandable without referring to a specific background text or other questions.
* Exclude unresolved or vague questions:
  * If a question contains vague references, lacks sufficient context, or asks for meta-level information about the background texts, and cannot be revised to meet the above criteria, exclude it from the final output.

Relevance Scoring:

Assign a relevance score to each refined question based on how important the associated fact is for answering the original user query. Use the following 5-point scale:
* [Relevance: 1] — The fact is entirely unrelated to the user query.
* [Relevance: 2] — The fact is topically related to the query but does not contribute meaningfully to answering it.
* [Relevance: 3] — The fact could be included in a comprehensive or extended answer, but is not necessary for a concise or focused response.
* [Relevance: 4] — The fact would typically be included in a good answer to the query, but its omission would not make the answer incorrect.
* [Relevance: 5] — The fact is essential to answering the query. Any valid response must include this information.

You should output the refined questions in the form of a list, with each item starting with "* ", followed by the relevance score in the format [Relevance: ]. Do not include any other explanations or comments. Refer to the following examples to understand the task and the output format.

Example 1:

User query:
Tell me about Glenn Danzig.

Raw questions:
* What is the name of the record label owned by Glenn Danzig?
* What is the university attended by Glenn Danzig according to the first source?
* When was he born?
* What are the professions of Glenn Danzig?
* Do sources disagree on Glenn Danzig's date of birth?
* How far is it from Dallas?
* Where did Glenn Danzig attend university?
* How tall is he in inches according to the Reuters news report?
* Which rock bands were founded by Glenn Danzig?
* What is the name of an adult-oriented comic book publishing company owned by Glenn Danzig?
* Is there contradictory information on the university attended by Glenn Danzig?
* What university did Glenn Danzig fondly remember as his alma mater?
* What is the nationality of Glenn Danzig?

Refined questions:
* What is the name of the record label owned by Glenn Danzig? [Relevance: 4]
* Which university did Glenn Danzig attend? [Relevance: 3]
* When was Glenn Danzig born? [Relevance: 4]
* What are the professions of Glenn Danzig? [Relevance: 5]
* How tall is Glenn Danzig? [Relevance: 2]
* Which rock bands were founded by Glenn Danzig? [Relevance: 5]
* What is the name of an adult-oriented comic book publishing company owned by Glenn Danzig? [Relevance: 3]
* What is the nationality of Glenn Danzig? [Relevance: 4]

Example 2:

User query:
What is the maximum range of Airbus A380?

Raw questions:
* What engines power the Airbus A380?
* What is the maximum range of the Airbus A380 as given by Rolls-Royce?
* What is the passenger capacity of the Airbus A380 in a typical configuration?
* What type of aircraft is the Airbus A380?
* What is the range of Airbus A380 in nautical miles?
* Do the Airbus A380 ranges from the two sources contradict each other?
* Had there been any hull losses involving the Airbus A380 fleet as of December 2021?
* Until what year was the Airbus A380 produced?
* Are there different reported maximum ranges of the Airbus A380?
* How far in miles can Airbus A380 fly in the Rolls-Royce Trent 900 turbofans configuration?
* What is the ranking of Airbus A380 among passenger airliners in terms of size?
* How many engines can be used to power the Airbus A380?
* Had there been any fatalities involving the Airbus A380 fleet as of December 2021?
* How many miles can an Airbus A380 travel without refueling?
* What distinguishes the Airbus A380 in terms of its deck configuration?
* Who developed and produced the Airbus A380?
* How far can its mighty engins take the Airbus A380?
* What is the maximum certified passenger capacity of the Airbus A380?
* How many engines does the Airbus A380 have?
* How many flights had the global Airbus A380 fleet completed as of December 2021?

Refined questions:
* What is the maximum range of the Airbus A380? [Relevance: 5]
* What engines power the Airbus A380? [Relevance: 2]
* What is the passenger capacity of the Airbus A380 in a typical configuration? [Relevance: 2]
* What type of aircraft is the Airbus A380? [Relevance: 2]
* Had there been any hull losses involving the Airbus A380 fleet as of December 2021? [Relevance: 2]
* Until what year was the Airbus A380 produced? [Relevance: 2]
* What is the maximum range of the Airbus A380 in the Rolls-Royce Trent 900 turbofans configuration? [Relevance: 4]
* What is the ranking of Airbus A380 among passenger airliners in terms of size? [Relevance: 2]
* How many engines does the Airbus A380 have? [Relevance: 2]
* Had there been any fatalities involving the Airbus A380 fleet as of December 2021? [Relevance: 2]
* What distinguishes the Airbus A380 in terms of its deck configuration? [Relevance: 2]
* Who developed and produced the Airbus A380? [Relevance: 2]
* What is the maximum certified passenger capacity of the Airbus A380? [Relevance: 2]
* How many flights had the global Airbus A380 fleet completed as of December 2021? [Relevance: 2]

Now, please refine the following questions and estimate their relevance to the user query. Remember to include the original user query if it's sufficiently focused and unambiguous.

User query:
{query}

Raw questions:
{questions}

Refined questions:{prompt_end}"""

# Some of the examples in the prompt intentionally include nonsense information
# to encourage the model to provide confidence scores according to the
# source text instead of trying to use its own knowledge.
ANSWER_GENERATION_PROMPT = """{prompt_start}You are an AI assistant specialized in answering questions based on a background text.

Consider the following text and a set of questions. Your task is to provide comprehensive answers to each question, making sure to include all the possible answers. This is particularly important if the text provides several differing viewpoints on the given question. In your response, you should first repeat each question before giving all the answers on a separate line. If there are multiple possible answers, they should be separated by the "|" symbol. For each answer, you should also predict a confidence score indicating the degree to which the given answer is supported by the given text. The confidence scores should be on a scale from 1 to 5, where 1 indicates that the background text considers the answer to be wholly incorrect and 5 indicates that the background text considers the answer to be fully and unambiguously correct. Values in between should indicate answers for which the background text provides conflicting or uncertain evidence. Note that your confidence assessment should be fully based on the information and opinions expressed in the background text, even if it contradicts your own knowledge. If a question cannot be answered solely based on the background text, you should respond with "unknown" with a confidence score of 5 to indicate absence of information. Each individual answer should be formatted as "A: <answer> [Confidence: <score>]". Do not include any other explanations or comments. Refer to the following examples to understand the task and the output format.

Example 1:

Background text:
Glenn Danzig (born June 24, 1955, though some fringe sources claim June 23, 1955) is an American singer, songwriter, musician, and record producer. He is the founder of the rock bands Misfits, Samhain, and Danzig. He owns the Evilive record label as well as Verotik, an adult-oriented comic book publishing company.

Questions:
* Which rock bands were founded by Glenn Danzig?
* What is the name of an adult-oriented comic book publishing company owned by Glenn Danzig?
* When was the Misfits band founded?
* What is the nationality of Glenn Danzig?
* When did Glenn Danzig's musical career start?
* What are the professions of Glenn Danzig?
* When was Glenn Danzig born?

Answers:
* Which rock bands were founded by Glenn Danzig?
  A: Misfits [Confidence: 5] | A: Samhain [Confidence: 5] | A: Danzig [Confidence: 5]
* What is the name of an adult-oriented comic book publishing company owned by Glenn Danzig?
  A: Verotik [Confidence: 5]
* When was the Misfits band founded?
  A: unknown [Confidence: 5]
* What is the nationality of Glenn Danzig?
  A: American [Confidence: 5]
* When did Glenn Danzig's musical career start?
  A: unknown [Confidence: 5]
* What are the professions of Glenn Danzig?
  A: singer [Confidence: 5] | A: songwriter [Confidence: 5] | A: musician [Confidence: 5] | A: record producer [Confidence: 5]
* When was Glenn Danzig born?
  A: June 24, 1955 [Confidence: 4] | A: June 23, 1955 [Confidence: 2]

Example 2:

Background text:
The Airbus A380 is a very large wide-body airliner, developed and produced by Embraer until 2024. It is the world's largest passenger airliner and the only full-length double-deck jet airliner. The full-length double-deck aircraft has a typical seating for 525 passengers, with a maximum certified capacity for 1024 passengers. The quadjet is powered by Engine Alliance GP7200 or Rolls-Royce Trent 900 turbofans, enabling the aircraft to fly for 8,000 nmi (14,800 km; 9,200 mi). As of December 2021, the global A380 fleet had completed more than 800,000 flights with no fatalities and no hull losses.

Questions:
* How many flights had the global Airbus A380 fleet completed as of March 2022?
* What is the maximum certified passenger capacity of the Airbus A380?
* What is the maximum range of the Airbus A380?
* What is the ranking of Airbus A380 among passenger airliners in terms of size?
* What distinguishes the Airbus A380 in terms of its deck configuration?
* Who developed and produced the Airbus A380?
* What engines power the Airbus A380?
* Until what year was the Airbus A380 produced?
* Had there been any fatalities involving the Airbus A380 fleet as of March 2022?
* What type of aircraft is the Airbus A380?
* How many engines does the Airbus A380 have?

Answers:
* How many flights had the global Airbus A380 fleet completed as of March 2022?
  A: more than 800,000 flights [Confidence: 5]
* What is the maximum certified passenger capacity of the Airbus A380?
  A: 1024 passengers [Confidence: 5]
* What is the maximum range of the Airbus A380?
  A: 8,000 nmi [Confidence: 5] | A: 14,800 km [Confidence: 5] | A: 9,200 mi [Confidence: 5]
* What is the ranking of Airbus A380 among passenger airliners in terms of size?
  A: world's largest passenger airliner [Confidence: 5]
* What distinguishes the Airbus A380 in terms of its deck configuration?
  A: It's the only full-length double-deck jet airliner. [Confidence: 5]
* Who developed and produced the Airbus A380?
  A: Embraer [Confidence: 5]
* What engines power the Airbus A380?
  A: Engine Alliance GP7200 [Confidence: 5] | A: Rolls-Royce Trent 900 [Confidence: 5]
* Until what year was the Airbus A380 produced?
  A: 2024 [Confidence: 5]
* Had there been any fatalities involving the Airbus A380 fleet as of March 2022?
  A: unknown [Confidence: 5]
* What type of aircraft is the Airbus A380?
  A: very large wide-body airliner [Confidence: 5]
* How many engines does the Airbus A380 have?
  A: four [Confidence: 5]

Example 3:

Background text:
The Dyatlov Pass incident was an event in which nine Soviet ski hikers died in the northern Ural Mountains on 1 or 2 February 1959 under undetermined circumstances. Overnight, something caused them to cut their way out of their tent and flee the campsite while inadequately dressed for the heavy snowfall and subzero temperatures. Numerous theories have been put forward to account for the unexplained deaths, including animal attacks, hypothermia, an avalanche, katabatic winds, infrasound-induced panic, military involvement, or some combination of these factors. However, most experts believe that aliens have been responsible. A prominent rock outcrop in the area, about 500 meters (1,600 ft) from the actual site of the final camp now serves as a memorial to the group.

Questions:
* Was the Dyatlov Pass incident caused by aliens?
* On what date did the Dyatlov Pass incident take place?
* What are the names of the individuals involved in the Dyatlov Pass incident?
* How many Soviet ski hikers died in the Dyatlov Pass incident?
* What is the explanation behind the Dyatlov Pass incident?
* How far is the memorial outcrop from the site of the final camp?
* What was the original goal of the group involved in the Dyatlov Pass incident?
* Did the Dyatlov Pass incident involve any deaths?
* What serves as a memorial to the group?
* What unusual action did the hikers take during the Dyatlov Pass incident?
* Who conducted the formal investigation into the Dyatlov Pass incident?
* What were the weather conditions during the Dyatlov Pass incident?
* What happened during the Dyatlov Pass incident?
* Where did the Dyatlov Pass incident occur?

Answers:
* Was the Dyatlov Pass incident caused by aliens?
  A: yes [Confidence: 4] | A: no [Confidence: 2]
* On what date did the Dyatlov Pass incident take place?
  A: 1 February 1959 [Confidence: 3] | A: 2 February 1959 [Confidence: 3]
* What are the names of the individuals involved in the Dyatlov Pass incident?
  A: unknown [Confidence: 5]
* How many Soviet ski hikers died in the Dyatlov Pass incident?
  A: nine [Confidence: 5]
* What is the explanation behind the Dyatlov Pass incident?
  A: animal attacks [Confidence: 2] | A: hypothermia [Confidence: 2] | A: avalanche [Confidence: 2] | A: katabatic winds [Confidence: 2] | A: infrasound-induced panic [Confidence: 2] | A: military involvement [Confidence: 2] | A: a combination of factors [Confidence: 2] | A: aliens [Confidence: 4]
* How far is the memorial outcrop from the site of the final camp?
  A: 500 m [Confidence: 5] | A: 1600 ft [Confidence: 5]
* What was the original goal of the group involved in the Dyatlov Pass incident?
  A: unknown [Confidence: 5]
* Did the Dyatlov Pass incident involve any deaths?
  A: yes [Confidence: 5]
* What serves as a memorial to the group?
  A: A prominent rock outcrop in the area. [Confidence: 5]
* What unusual action did the hikers take during the Dyatlov Pass incident?
  A: They cut their way out of their tent and fled the campsite while inadequately dressed. [Confidence: 5]
* Who conducted the formal investigation into the Dyatlov Pass incident?
  A: unknown [Confidence: 5]
* What were the weather conditions during the Dyatlov Pass incident?
  A: heavy snowfall and subzero temperatures [Confidence: 5]
* What happened during the Dyatlov Pass incident?
  A: Nine Soviet ski hikers died under undetermined circumstances after cutting their way out of their tent and fleeing the campsite while inadequately dressed for heavy snowfall and subzero temperatures. [Confidence: 5]
* Where did the Dyatlov Pass incident occur?
  A: Northern Ural Mountains [Confidence: 5]

Now, please generate a list of answers and confidence scores for the following questions and background text.

Background text:
{background_text}

Questions:
{questions}

Answers:{prompt_end}"""

ANSWER_COMPARISON_PROMPT = """{prompt_start}You are an AI assistant specialized in comparing answers to questions.

You will be given a question and a pair of answers. Your task is to determine the relationship between the pair with respect to the given question. Consider the following type of relationships:
* **Equivalent**: The answers have the same meaning, refer to the same entity in different forms or are paraphrases of each other. Indicated as [equivalent].
* **First implies second**: The first answer in the pair implies the second answer. Indicated as [first implies second].
* **Second implies first**: The second answer in the pair implies the first answer. Indicated as [second implies first].
* **Contradictory**: The two answers are contradictory or mutually exclusive, and can never be true at the same time. Indicated as [contradictory].
* **Neutral**: The two answers are different but could both be true at the same time (e.g., if there are multiple correct answers to the question). Indicated as [neutral].{tool_use_text}

When classifying the answers, focus on the following aspects:
* Do the answers have the same meaning (i.e., they mutually imply each other)? If yes, the answers are equivalent.
* Could both answers be true at the same time? If no, the answers are contradictory.
* Does one of the two answers imply the other answer (i.e., is one of the two compatible answers more specific)?

Your final response should consist of one- to two-sentence reasoning about the relationship between the answers and a final classification in the form "<answer 1> — <answer 2> [classification]" given on a separate line. Refer to the following examples to understand the task and the output format.

Example 1:

Question:
On what date did the Dyatlov Pass incident take place:

Answer pairs:
1959-02-01 — February 1959 [?]

Reasoning and classification:
"1959-02-01" is a specific date, while "February 1959" refers to the entire month. The specific date falls within the broader time frame, so "1959-02-01" (the first answer) implies "February 1959" (the second answer).

1959-02-01 — February 1959 [first implies second]

Example 2:

Question:
What elements are found in the human body?

Answer pairs:
oxygen — O [?]

Reasoning and classification:
Oxygen and O refer to the same chemical element, with O being its symbol and oxygen its full name.

oxygen — O [equivalent]

Example 3:

Question:
Who was the invited speaker at the AAAI 2024 conference?

Answer pair:
Andrew Ng — David Chalmers [?]

Reasoning and classification:
Andrew Ng and David Chalmers are different people, but conferences can have multiple invited speakers, so the answers are neutral to each other.

Andrew Ng — David Chalmers [neutral]

Example 4:

Question:
What is the accuracy of the top-performing LLM on Humanity's Last Exam?

Answer pair:
over 23% — 25.4 % [?]

Reasoning and classification:
The first answer "over 23%" is a vague lower bound, while the second answer "25.4%" is a specific value that satisfies the condition of being over 23%. This means that "25.4 %" (the second answer) implies "over 23%" (the first answer).

over 23% — 25.4 % [second implies first]

Example 5:

Question:
Where was the first prototype of Airbus A380 unveiled?

Answer pair:
Toulouse — Spain [?]

Reasoning and classification:
Toulouse is a city in France while Spain is a different country, and since the unveiling was a single event, the two answers are contradictory.

Toulouse — Spain [contradictory]

Now, please provide your reasoning and classifications for the following answer pairs{tool_use_reminder}:

Question:
{question}

Answer pair:
{answer_pair}

{prompt_end}"""

TOOL_USE_PROMPT = """

You have been given access to a tool or a set of tools for comparing specific types of answer pairs. Please follow these rules carefully. Use a tool only if the tool is clearly applicable to the type of answer pair being compared and you are confident that the tool will produce a valid and meaningful result. If you decide to use a tool, respond only with the tool call (without any comments or explanations) and use the tool's result to construct your final response. Do not call any tool if none of the available tools are a good fit for the answer pair or when you are uncertain about their relevance. If no tool is appropriate or usable, use your best judgement to compare the answers and respond with the classifications directly, without attempting any tool call. IN SUMMARY, YOU SHOULD ONLY CALL TOOLS WHEN YOU ARE CONFIDENT THEY ARE APPLICABLE AND WILL WORK."""

TOOL_USE_REMINDER = """ (remember to use tools only when you are confident that they are appropriate, otherwise, just give the answer directly)"""


class QagProcessor:
    """
    Processes information in texts using question-answer generation.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-instruct",
        comparison_tools: list[ToolDefinition] = [],
        inject_prompt_template: bool = True,
    ):
        """
        Initializes the QagProcessor.

        Args:
            model: str
                The model to use for processing questions and answers.
            comparison_tools: list[ToolDefinition]
                A list specifying the tools that can be used for comparing
                answers.
            inject_prompt_template: bool
                Whether to inject prompt template tags to the prompt.
                Defaults to True.
        """
        rits_model_info = RITS_MODELS[model]
        if inject_prompt_template:
            self.prompt_begin = rits_model_info.get("prompt_begin")
            self.prompt_end = f"\n{rits_model_info.get('prompt_end')}"
        else:
            self.prompt_begin = ""
            self.prompt_end = ""
        self.llm_handler = LLMHandler(model, RITS=True)
        self.comparison_tools = comparison_tools
        # We use this for limiting the number of concurrent LLM calls.
        # Note that the semaphore value is overriden by batch_size
        # from method calls.
        self.llm_semaphore = threading.Semaphore(32)

        print(f"[QagProcessor] Using LLM on RITS: {model}")
        print(
            f"[QagProcessor] Registered tools for comparison: {', '.join([t.tool_fun.__name__ for t in comparison_tools])}"
        )

    def make_question_prompt(self, query: str, background_text: str) -> str:
        """
        Constructs a prompt for mining a set of questions from a text.

        Args:
            query: str
                The original user query with respect to which the
                questions should be extracted.
            background_text: str
                The background text the relevant content of which should be
                covered by the extracted questions.

        Returns:
            str:
                The constructed prompt.
        """
        return QUESTION_EXTRACTION_PROMPT.format(
            prompt_start=self.prompt_begin,
            prompt_end=self.prompt_end,
            query=query,
            background_text=background_text,
        )

    def make_question_refinement_prompt(self, query: str, questions: list[str]) -> str:
        """
        Constructs a prompt for refining a set of questions, removing
        duplicates and low-quality questions, and determining their
        relevance to a specific user query.

        Args:
            query: str
                The original user query with respect to which the
                questions should be refined, and which will be used
                to determine question relevance scores.
            questions: list[str]
                A list of questions to refine.
        """
        questions_text = "\n".join([f"* {q}" for q in questions])
        return QUESTION_REFINEMENT_PROMPT.format(
            prompt_start=self.prompt_begin,
            prompt_end=self.prompt_end,
            query=query,
            questions=questions_text,
        )

    def make_answer_generation_prompt(
        self, questions: list[str], background_text: str
    ) -> str:
        """
        Constructs a prompt for generating answers to a set of questions
        based on the provided background text.

        Args:
            questions: list[str]
                The list of questions for which to generate answers.
            background_text: str
                The background text based on which the answers should
                be generated.

        Returns:
            str:
                The constructed prompt.
        """
        questions_text = "\n".join([f"* {q}" for q in questions])
        return ANSWER_GENERATION_PROMPT.format(
            prompt_start=self.prompt_begin,
            prompt_end=self.prompt_end,
            background_text=background_text,
            questions=questions_text,
        )

    def make_answer_comparison_prompt(
        self,
        question: str,
        answer_pair: tuple[str, str],
        use_tools: bool = False,
    ) -> str:
        """
        Constructs a prompt for comparing paris of answers to determine
        the relationships between them.

        Args:
            question: str
                The question associated with the answers.
            answer_pairs: tuple[str, str]
                The pair of answers to compare.
            use_tools: bool
                Whether to enable tool usage for comparing answers.
                Defaults to False.

        Returns:
            str:
                The constructed prompt.
        """
        return ANSWER_COMPARISON_PROMPT.format(
            prompt_start=self.prompt_begin,
            prompt_end=self.prompt_end,
            question=question,
            answer_pair=f"{answer_pair[0]} — {answer_pair[1]} [?]",
            tool_use_text=TOOL_USE_PROMPT if use_tools else "",
            tool_use_reminder=TOOL_USE_REMINDER if use_tools else "",
        )

    def _parse_questions(self, response: ModelResponse) -> list[str]:
        """
        Parses generated questions from a model response.

        Args:
            response: ModelResponse
                The model response from which the questions should be parsed.

        Returns:
            list[str]:
                The list of the parsed questions.
        """
        choice = response.choices[0]
        if not isinstance(choice, Choices):
            raise ValueError(f"Unexpected type for model choice: {type(choice)}")
        question_pattern = r"(?m)^\s*\*\s+(.+?)\s*$"
        message_content = cast(str, choice.message.content)
        if type(message_content) is not str:
            raise ValueError(f"Unexpected model response: {response}")
        result_matches = list(regex.finditer(question_pattern, message_content))
        questions = [match.group(1) for match in result_matches]
        return questions

    def extract_questions(self, query: str, background_text: str) -> list[str]:
        """
        Mines a set of questions based on the given background text.

        Args:
            query: str
                The original user query with respect to which the
                questions should be extracted.
            background_text: str
                The background text the relevant content of which should be
                covered by the extracted questions.

        Returns:
            list[str]:
                The list of extracted questions.
        """
        prompt = self.make_question_prompt(query, background_text)
        print("[QagProcessor] Prompt created: 1")
        response = self.llm_handler.completion(
            prompt,
            max_tokens=8192,
            seed=42,
            logprobs=True,
            top_logprobs=20,
            expect_content=True,
        )
        if not isinstance(response, ModelResponse):
            raise ValueError(f"Unexpected type for model response: {response}")
        return self._parse_questions(response)

    def extract_all_questions(
        self, query: str, background_texts: list[str], batch_size: int = 32
    ) -> list[list[str]]:
        """
        Mines sets of questions for several background texts in parallel.

        Args:
            query: str
                The original user query with respect to which the
                questions should be extracted.
            background_text: str
                The background texts the relevant content of which should be
                covered by the extracted questions.
            batch_size: int
                The maximum number of parallel model calls.

        Returns:
            list[list[str]]:
                The nested list of questions extracted for each background
                text.
        """
        prompts = [self.make_question_prompt(query, bt) for bt in background_texts]
        print(f"[QagProcessor] Prompts created: {len(prompts)}")

        all_results: list[list[str]] = []
        for prompt_batch in tqdm(
            itertools.batched(prompts, n=batch_size),
            total=math.ceil(len(prompts) / batch_size),
        ):
            responses = self.llm_handler.batch_completion(
                prompt_batch,
                max_tokens=8192,
                seed=42,
                logprobs=True,
                top_logprobs=20,
                expect_content=True,
            )
            for response in responses:
                if not isinstance(response, ModelResponse):
                    raise ValueError(
                        f"Unexpected type for model response: {type(response)}"
                    )
                all_results.append(self._parse_questions(response))

        return all_results

    def refine_questions(
        self, query: str, questions: list[str]
    ) -> dict[str, QuestionData]:
        """
        Refines a set of questions, removing duplicates and low-quality
        questions, and determining their relevance to a specific user
        query

        Args:
            query: str
                The original user query with respect to which the
                questions should be refined, and which will be used
                to determine question relevance scores.
            questions: list[str]
                A list of questions to refine.
        """
        prompt = self.make_question_refinement_prompt(query, questions)
        print("[QagProcessor] Prompt created: 1")
        response = self.llm_handler.completion(
            prompt,
            max_tokens=8192,
            seed=42,
            logprobs=True,
            top_logprobs=20,
            expect_content=True,
        )
        if not isinstance(response, ModelResponse):
            raise ValueError(f"Unexpected type for model response: {response}")
        weighted_scores = extract_weighted_scores(
            response, r"\*\s+(.+?)\s+\[Relevance:\s+([0-9]+)\]"
        )
        return {q: QuestionData(question=q, relevance=r) for q, r in weighted_scores}

    def merge_questions(
        self,
        questions: list[dict[str, QuestionData]],
        confidence_threshold: float = -1.0,
    ) -> dict[str, QuestionData]:
        """
        Merges matching question data and answers.

        Args:
            questions: list[dict[str, QuestionData]]
                The list of dicts with Q&A data based on different
                background texts.
            confidence_threshold: float
                The minimal confidence of the answers included in the
                results.

        Returns:
            dict[str, QuestionData]:
                The merged question data and answers.
        """
        all_questions: dict[str, list[QuestionData]] = defaultdict(list)
        for questions_batch in questions:
            for q, qd in questions_batch.items():
                all_questions[q].append(qd)
        results: dict[str, QuestionData] = {}
        for q, qd_list in all_questions.items():
            relevance_scores = [qd.relevance for qd in qd_list]
            answers = [
                a
                for qd in qd_list
                for a in qd.answers
                if a.confidence >= confidence_threshold
            ]
            results[q] = QuestionData(
                question=q,
                relevance=mean(relevance_scores),
                answers=answers,
            )
        return results

    def _parse_answers(
        self,
        questions: dict[str, QuestionData],
        source_id: str,
        source_type: Literal["context", "answer"],
        response: ModelResponse,
    ) -> dict[str, QuestionData]:
        """
        Parses answers generated for a set of questions from the model response.

        Args:
            questions: dict[str, QuestionData]
                The questions for which the answers were generated.
            source_id: str
                The ID of the source from which the answers were mined.
            source_type: Literal["context", "answer"]
                The type of the source from which the answers were mined.
            response: ModelResponse
                The model response from which to parse the answers.

        Returns:
            dict[str, QuestionData]:
                A copy of the question data dictionary populated with the
                parsed answers.
        """
        # Extract answers and the corresponding confidence scores
        # We don't extract the questions yet, as extract_weighted_scores
        # only supports matching on a single context.
        answer_pattern = r"A:\s+(.+?)\s+\[Confidence:\s+([0-9])\]"
        raw_weighted_answers = extract_weighted_scores(response, answer_pattern)

        # Match the answers and the scores to the corresponding questions
        choice = response.choices[0]
        if not isinstance(choice, Choices):
            raise ValueError(f"Unexpected type for model choice: {type(choice)}")
        if not isinstance(choice.message.content, str):
            raise ValueError(
                f"Unexpected type for message content: {type(choice.message.content)})"
            )
        message_content = choice.message.content
        qa_pattern = r"\*\s+([\s\S]+?)\s+(A:\s+.+?\])\s*$"
        qa_matches = list(
            regex.finditer(qa_pattern, message_content, flags=regex.MULTILINE)
        )
        raw_answers_idx = 0
        results = deepcopy(questions)
        for qa_match in qa_matches:
            question = qa_match.group(1)
            answers = qa_match.group(2)
            answer_list = [a.group(1) for a in regex.finditer(answer_pattern, answers)]

            for answer in answer_list:
                # To make the pipeline more robust, we skip raw answers until we find a match,
                # but the expectation is that this loop should only run once.
                while True:
                    answer_candidate, confidence_candidate = raw_weighted_answers[
                        raw_answers_idx
                    ]
                    if answer_candidate == answer:
                        break
                    print(
                        f"[QagProcessor] WARNING: Answer mismatch {answer_candidate} != {answer}. Ignoring..."
                    )
                    print("########## Full model response ##########")
                    print(message_content)
                    print("#########################################")
                    raw_answers_idx += 1
                try:
                    results[question].answers.append(
                        Answer(
                            answer=answer,
                            confidence=confidence_candidate,
                            source_id=source_id,
                            source_type=source_type,
                        )
                    )
                except KeyError:
                    # This may happen if the model output doesn't match the original
                    # question
                    print(f"WARNING: Question {question} not found in results!")

                raw_answers_idx += 1
        return results

    def extract_answers(
        self,
        questions: dict[str, QuestionData],
        background_text: str,
        source_id: str,
        source_type: Literal["context", "answer"],
    ) -> dict[str, QuestionData]:
        """
        Generates answers to a set of questions based on the provided
        background text.

        Args:
            questions: dict[str, QuestionData]
                A dictionary of questions for which to generate answers.
            background_text: str
                The background text based on which the answers should
                be generated.
            source_id: str
                The ID of the background text source.
            source_type: Literal["context", "answer"]
                The type of background text source.

        Returns:
            dict[str, QuestionData]:
                A copy of the questions dictionary populated with
                the generated answers.
        """
        prompt = self.make_answer_generation_prompt(
            list(questions.keys()), background_text
        )
        print("[QagProcessor] Prompt created: 1")
        response = self.llm_handler.completion(
            prompt,
            max_tokens=8192,
            seed=42,
            logprobs=True,
            top_logprobs=20,
            expect_content=True,
        )
        if not isinstance(response, ModelResponse):
            raise ValueError(f"Unexpected type for model response: {response}")
        return self._parse_answers(questions, source_id, source_type, response)

    def extract_all_answers(
        self,
        questions: dict[str, QuestionData],
        background_texts: list[str],
        source_ids: list[str],
        source_types: list[str],
        batch_size: int = 32,
    ) -> list[dict[str, QuestionData]]:
        """
        Generates answers to a set of questions based on multiple background
        texts in parallel.

        Args:
            questions: dict[str, QuestionData]
                A dictionary of questions for which to generate answers.
            background_texts: list[str]
                The background texts based on which the answers should
                be generated.
            source_id: str
                The IDs of the background text sources.
            source_type: list[str]
                The types of the background text sources.
            batch_size: int
                The maximum number of parallel model calls.

        Returns:
            dict[str, QuestionData]:
                A copy of the questions dictionary populated with
                the generated answers.
        """
        prompts = [
            self.make_answer_generation_prompt(list(questions.keys()), bt)
            for bt in background_texts
        ]
        print(f"[QagProcessor] Prompts created: {len(prompts)}")

        all_results: list[dict[str, QuestionData]] = []
        for batch in tqdm(
            itertools.batched(zip(prompts, source_ids, source_types), n=batch_size),
            total=math.ceil(len(prompts) / batch_size),
        ):
            prompt_batch, source_id_batch, source_type_batch = zip(*batch)
            responses = self.llm_handler.batch_completion(
                prompt_batch,
                max_tokens=8192,
                seed=42,
                logprobs=True,
                top_logprobs=20,
                expect_content=True,
            )
            for source_id, source_type, response in zip(
                source_id_batch, source_type_batch, responses
            ):
                if not isinstance(response, ModelResponse):
                    raise ValueError(
                        f"Unexpected type for model response: {type(response)}"
                    )
                all_results.append(
                    self._parse_answers(questions, source_id, source_type, response)
                )
        return all_results

    def _llm_compare_answers(
        self, question: str, answer_pair: tuple[str, str], num_retries: int = 5
    ) -> BaseAnswerRelation:
        """
        Compares two answers to a question using an LLM.

        Args:
            question: str
                The question for which to compare answers.
            answer_paris: tuple[ast, str]
                The pair of answers to compare against each other.
            num_retries: int
                The number of attempts for comparing each pair.

        Returns:
            BaseAnswerRelation:
                The determined relation between the answers.
        """
        for attempt in Retrying(
            stop=stop_after_attempt(num_retries),
            wait=wait_random_exponential(max=30),
        ):
            with attempt:
                # Don't use tools on the last attempt
                use_tools = bool(self.comparison_tools) and (
                    attempt.retry_state.attempt_number < num_retries
                )
                prompt = self.make_answer_comparison_prompt(
                    question=question,
                    answer_pair=answer_pair,
                    use_tools=use_tools,
                )

                # Call an LLM to compare the answer pairs
                messages: list[Any] = [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
                with self.llm_semaphore:
                    response = self.llm_handler.completion(
                        messages=messages,
                        max_tokens=8192,
                        seed=42,
                        logprobs=True,
                        top_logprobs=20,
                        tools=[t.tool_metadata for t in self.comparison_tools]
                        if use_tools
                        else None,
                        tool_choice="auto" if use_tools else None,
                        expect_content=not use_tools,
                    )
                if not isinstance(response, ModelResponse):
                    raise ValueError(
                        f"Unexpected type for model response: {type(response)}"
                    )
                choice = response.choices[0]
                if not isinstance(choice, Choices):
                    raise ValueError(f"Unexpected type for choices: {type(choice)}")
                response_message = choice.message
                tool_calls = response_message.tool_calls
                if not tool_calls and response_message.content is not None:
                    # Check if response message includes reparable tool calls
                    tool_calls = repair_tool_calls(response_message.content)

                function_response = None
                if use_tools and tool_calls:
                    # Process function call if there is any, allowing us to return the result
                    # directly if the extraction is successful
                    available_functions = {t.tool_id: t for t in self.comparison_tools}
                    messages.append(response_message)

                    for tool_call in tool_calls:
                        function_id = tool_call.function.name
                        if not isinstance(function_id, str):
                            raise ValueError(f"Unexpected tool ID type: {function_id}")
                        function_to_call = available_functions[function_id]
                        function_args = cast(
                            dict, json_repair.loads(tool_call.function.arguments)
                        )
                        for arg_id, arg in function_args.items():
                            if arg_id in function_to_call.parameter_parsers:
                                parameter_parser = function_to_call.parameter_parsers[
                                    arg_id
                                ]
                                function_args[arg_id] = parameter_parser(arg)
                        function_response = function_to_call.tool_fun(**function_args)
                        print("[QagProcessor] Completed function call with response:")
                        print(
                            f"{answer_pair[0]} — {answer_pair[1]} [{function_response}]"
                        )

                    if not function_response:
                        print("[QagProcessor] Error when calling tools!")
                        raise ValueError("Invalid tool response!")
                    else:
                        # Return function response if available
                        return BaseAnswerRelation(
                            label=function_response,
                            probability=1.0,
                            reasoning="Function call result.",
                        )

                if not response_message.content:
                    print("[QagProcessor] Received an empty model response!")
                    raise ValueError(
                        f"Received unexpected empty response {response} from the model!"
                    )

                # Extract the label and its probability from the free-text model response
                reasoning_pattern = r"(?m)(?:Reasoning and classification:)?\s*(^.+?)\s*^\s*(?:.+?—.+?)\s+\[(?:[^\?].+?)\]\s*$"
                search_result = regex.search(
                    reasoning_pattern, response_message.content
                )
                reasoning = search_result.group(1) if search_result else "Unknown"
                relation_pattern = r"(?m)^\s*(.+?—.+)\s+\[([^\?].+?)\]\s*$"
                probabilistic_labels: list[tuple[str, str, float]] = (
                    extract_probabilistic_labels(
                        response,
                        relation_pattern,
                        label_options=ANSWER_COMPARISON_LABELS,
                    )
                )
                if not probabilistic_labels:
                    print(
                        f"WARNING: Unable to determine probabilistic labels from:\n{response_message.content}"
                    )
                    print("Falling back to nonprobabilistic label.")
                    lax_relation_pattern = r"(?m)^\s*(.+?—.+?)\s+\[([^\?].+?)\]\s*"
                    search_result = regex.search(
                        lax_relation_pattern, response_message.content
                    )
                    if (
                        search_result
                        and search_result.group(2) in ANSWER_COMPARISON_LABELS
                    ):
                        probabilistic_labels = [
                            (search_result.group(1), search_result.group(2), 1.0)
                        ]
                    else:
                        # Last-ditch effort to identify a valid label
                        print(f"WARNING: Unable to determine standard labels from:\n{response_message.content}")
                        print("Falling back to minimalistic label extraction.")
                        minimal_relation_pattern = r"\[([^\?].+?)\]"
                        matches = regex.findall(minimal_relation_pattern, response_message.content)
                        print(f"DEBUG: Identified minimal label matches: {matches}")
                        label_match = next(
                            (m for m in matches if m in ANSWER_COMPARISON_LABELS), None
                        )
                        print(f"DEBUG: Identified label: {label_match}")
                        if label_match is not None:
                            probabilistic_labels = [("Unknown", label_match, 1.0)]
                        else:
                            raise ValueError(
                                f"Unable to determine any labels from:\n{response_message.content}"
                            )
                if len(probabilistic_labels) != 1:
                    # We don't raise an error here because Llama 4 sometimes
                    # repeats the classifications two times in a row, which
                    # is a non-critical issue.
                    print(
                        f"WARNING: Expected probabilistic labels to be a sigleton but got {probabilistic_labels}. "
                        f"Response: {response_message.content}"
                    )

                _, label, label_proba = probabilistic_labels[0]
                result = BaseAnswerRelation(
                    label=cast(ComparisonResult, label),
                    probability=label_proba,
                    reasoning=reasoning,
                )

                return result

        assert False, "Unreachable code assertion for Pyright."

    def compare_answers(
        self, question: QuestionData, batch_size: int = 32
    ) -> dict[frozenset[Answer], AnswerRelation]:
        """
        Compares pairs of answers to determine the relationships between them.

        Args:
            question: QuestionData
                The QuestionData object containing the answers to be compared
                and the associated question data.
            batch_size: int
                The maximum number of LLM comparisons to perform in parallel.

        Returns:
            dict[frozenset[Answer], AnswerRelation]:
                A dictionary mapping frozen sets of answer pairs to their
                relations.
        """
        self.llm_semaphore = threading.Semaphore(batch_size)
        results: dict[tuple[str, str], BaseAnswerRelation] = {}

        # Get groups of answers with the same normal form,
        # which we can directly classify as identical.
        normalised_answer_groups: dict[str, list[Answer]] = defaultdict(list)
        for answer in question.answers:
            normalised_answer = answer.answer.lower().strip()
            normalised_answer_groups[normalised_answer].append(answer)
        # Only use the most common answers from each group for LLM-based comparison
        answer_group_modes = [
            mode([a.answer for a in answer_group])
            for answer_group in normalised_answer_groups.values()
        ]
        answer_pairs = itertools.combinations(answer_group_modes, 2)

        # Determine the pairs to be compared using an LLM
        undetermined_pairs: list[tuple[str, str]] = []
        for a1, a2 in answer_pairs:
            if a1.strip().lower() == "unknown" or a2.strip().lower() == "unknown":
                # Unknown answers are always neutral
                results[(a1, a2)] = BaseAnswerRelation(
                    label="neutral",
                    probability=1.0,
                    reasoning="Deterministic unknown answer classification.",
                )
            else:
                undetermined_pairs.append((a1, a2))
        if undetermined_pairs:
            # Get probabilistic model prediction for each undetermined pair
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                llm_comparison_results = list(
                    tqdm(
                        executor.map(
                            lambda p: self._llm_compare_answers(question.question, p),
                            undetermined_pairs,
                        ),
                        total=len(undetermined_pairs),
                    )
                )
                results.update(
                    dict(zip(undetermined_pairs, llm_comparison_results, strict=True))
                )

        # Reconstruct relations for all answer forms
        final_results: dict[frozenset[Answer], AnswerRelation] = {}
        # Construct relations for pairs of answers from two answer groups
        for a1 in answer_group_modes:
            for a2 in answer_group_modes:
                norm_a1 = a1.strip().lower()
                norm_a2 = a2.strip().lower()
                if (a1, a2) in results:
                    relation = results[(a1, a2)]
                    for a1_alt in normalised_answer_groups[norm_a1]:
                        for a2_alt in normalised_answer_groups[norm_a2]:
                            final_results[frozenset([a1_alt, a2_alt])] = AnswerRelation(
                                label=relation.label,
                                probability=relation.probability,
                                reasoning=relation.reasoning,
                                fst=a1_alt,
                                snd=a2_alt,
                            )
        # Construct relations for answers in the same answer group
        for a in answer_group_modes:
            norm_a = a.strip().lower()
            if norm_a == "unknown":
                continue
            for a1_alt in normalised_answer_groups[norm_a]:
                for a2_alt in normalised_answer_groups[norm_a]:
                    if a1_alt != a2_alt:
                        final_results[frozenset([a1_alt, a2_alt])] = AnswerRelation(
                            label="equivalent",
                            probability=1.0,
                            reasoning="Deterministic answer match.",
                            fst=a1_alt,
                            snd=a2_alt,
                        )

        return final_results

    def compare_all_answers(
        self,
        all_question_data: list[QuestionData],
        batch_size: int = 32,
    ) -> list[dict[frozenset[Answer], AnswerRelation]]:
        """
        Compares pairs of answers to determine the relationships between them
        for multiple questions in parallel.

        Args:
            all_question_data: list[QuestionData]
                The list of QuestionData objects containing the answers to be
                compared and the associated question data.
            batch_size: int
                The maximum number of comparisons to perform in parallel.
        """
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            results = list(
                tqdm(
                    executor.map(
                        lambda q: self.compare_answers(q, batch_size=batch_size),
                        all_question_data,
                    ),
                    total=len(all_question_data),
                )
            )
        return list(results)
