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


# v1
ATOM_EXTRACTION_PROMPT_V1 = """{_PROMPT_BEGIN_PLACEHOLDER}
Instructions:
1. You are given a paragraph. Your task is to break the sentence down into a list of atomic statements without adding any new information.
2. An atomic statement is a sentence containing a singular piece of information directly extracted from the provided paragraph.
3. Atomic statements may contradict one another.
4. The paragraph may contain information that is factually incorrect. Even in such cases, you are not to alter any information contained in the paragraph and must produce atomic statements that are completely faithful to the information in the paragraph.
5. Each atomic statement in the outputted list should check a different piece of information found explicitly in the paragraph.
6. Each atomic statement is standalone in that any actual nouns or proper nouns should be used in place of pronouns or anaphors.
7. Each atomic statement must not include any information beyond what is explicitly stated in the provided paragraph.
8. Where possible, avoid paraphrasing and instead try to only use language used in the paragraph without introducing new words. 
9. Use the previous examples to learn how to do this.
10. You should only output the atomic statement as a list, with each item starting with "- ". Do not include other formatting.
11. Your task is to do this for the last paragraph that is given. 

Example 1:
Please breakdown the following paragraph into independent statements: Glenn Allen Anzalone (born June 23, 1955), better known by his stage name Glenn Danzig, is an American singer, songwriter, musician, and record producer. He is the founder of the rock bands Misfits, Samhain, and Danzig. He owns the Evilive record label as well as Verotik, an adult-oriented comic book publishing company.
- Glenn Allen Anzalone was born on June 23, 1955.
- Glenn Allen Anzalone is better known by his stage name Glenn Danzig.
- Glenn Danzig is an American singer, songwriter, musician, and record producer.
- Glenn Danzig is the founder of several rock bands, including Misfits, Samhain, and Danzig.
- Glenn Danzig owns the Evilive record label.
- Glenn Danzig owns Verotik, which is an adult-oriented comic book publishing company.

Example 2:
Please breakdown the following paragraph into independent statements: Luiz Inácio Lula da Silva (born 27 October 1945), also known as Lula da Silva or simply Lula, is a Brazilian politician who is the 39th and current president of Brazil since 2023. A member of the Workers' Party, Lula was also the 35th president from 2003 to 2010. He also holds the presidency of the G20 since 2023. Lula quit school after second grade to work, and did not learn to read until he was ten years old. As a teenager, he worked as a metalworker and became a trade unionist.
- Luiz Inácio Lula da Silva was born on October 27, 1945.
- Luiz Inácio Lula da Silva is also known as Lula da Silva or simply Lula.
- Lula is a Brazilian politician.
- Lula is the 39th and current president of Brazil since 2023.
- Lula is a member of the Workers' Party.
- Lula served as the 35th president of Brazil from 2003 to 2010.
- Lula holds the presidency of the G20 since 2023.
- Lula quit school after the second grade to work.
- Lula did not learn to read until he was ten years old.
- As a teenager, Lula worked as a metalworker.
- Lula became a trade unionist.

Your task:
Please breakdown the following paragraph into independent statements: {_RESPONSE_PLACEHOLDER}{_PROMPT_END_PLACEHOLDER}
"""

# v2
ATOM_EXTRACTION_PROMPT_V2 = """{_PROMPT_BEGIN_PLACEHOLDER}

Instructions: 
- Exhaustively break down the following text into independent content units. Each content unit can take one of the following forms:
  a. Fact: An objective piece of information that can be proven or verified.
  b. Claim: A statement or assertion that expresses a position or viewpoint on a particular topic.
  c. Instruction: A directive or guidance on how to perform a specific task.
  d. Data Format: Any content presented in a specific format, including code, mathematical notations, equations, variables, technical symbols, tables, or structured data formats.
  e. Meta Statement: Disclaimers, acknowledgments, or any other statements about the nature of the response or the responder.
  f. Question: A query or inquiry about a particular topic.
  g. Other: Any other relevant content that doesn't fit into the above categories.
- Label each content unit with its corresponding unit type using the format: [content unit]: [content unit type]
- You should only output the independent content units as a list, with each item starting with "- ". Do not include other formatting or preamble text.
- Refer to the following examples to understand the task and output formats. 

Example 1:
TEXT: Zhejiang Huafang Pharmaceutical Co., Ltd. is a leading chemical company based in China that specializes in the research, manufacturing, and sales of various pharmaceutical products, including excipients and intermediates. The company was founded in 2018 and is located in Hangzhou, a city with a rich history in eastern China. Zhejiang Huafang Pharmaceutical Co., Ltd. is committed to providing high-quality products to its customers in the healthcare industry. The company's manufacturing facilities are equipped with state-of-the-art technology and infrastructure that ensure the production of high-quality products. Overall, Zhejiang Huafang Pharmaceutical Co., Ltd. is a reputable pharmaceutical company with a long history of success in the healthcare industry. The company's commitment to quality, innovation, and customer service has made it a leader in the field of pharmaceutical research and development.

UNITS:
- Zhejiang Huafang Pharmaceutical Co., Ltd. is a leading chemical company: Fact
- Zhejiang Huafang Pharmaceutical Co., Ltd. is based in China: Fact
- Zhejiang Huafang Pharmaceutical Co., Ltd. specializes in the research of various pharmaceutical products: Fact
- Zhejiang Huafang Pharmaceutical Co., Ltd. specializes in the manufacturing of various pharmaceutical products: Fact
- Zhejiang Huafang Pharmaceutical Co., Ltd. specializes in the sales of various pharmaceutical products: Fact
- excipients are the pharmaceutical products of the Zhejiang Huafang Pharmaceutical Co., Ltd.: Fact
- intermediates are the pharmaceutical products of the Zhejiang Huafang Pharmaceutical Co., Ltd.: Fact
- The company was founded in 2018: Fact
- The company is located in Hangzhou: Fact
- Hangzhou is a city: Fact
- Hangzhou has a rich history in eastern China: Fact
- Zhejiang Huafang Pharmaceutical Co., Ltd. is committed to providing high-quality products to its customers in the healthcare industry: Claim
- The company's manufacturing facilities are equipped with state-of-the-art technology: Fact
- The company's manufacturing facilities are equipped with state-of-the-art infrastructure: Fact
- The company's manufacturing facilities are equipped with state-of-the-art technology and infrastructure that ensure the production of high-quality products: Claim
- Zhejiang Huafang Pharmaceutical Co., Ltd. is a reputable pharmaceutical company: Claim
- Zhejiang Huafang Pharmaceutical Co., Ltd. has a long history of success in the healthcare industry: Claim
- The company is committed to quality: Claim
- The company is committed to innovation: Claim
- The company is committed to customer service: Claim
- The company's commitment to quality, innovation, and customer service has made it a leader in the field of pharmaceutical research: Claim
- The company's commitment to quality, innovation, and customer service has made it a leader in the field of pharmaceutical development: Claim

Example 2:
TEXT: I'm here to help you make an informed decision. Both the RTX 3060 Ti and RTX 3060 are powerful GPUs, and the difference between them lies in their performance. The RTX 3060 Ti has more CUDA cores (4864 vs 3584) but a lower boost clock speed (1665 MHz vs 1777 MHz) compared to the RTX 3060. In terms of memory bandwidth, the RTX 3060 Ti has a slight edge over the RTX 3060 with a bandwidth of 448 GB/s compared to 360 GB/s. However, the difference is relatively small. It's important to consider other factors such as the power consumption, cooling system, and compatibility with your system when making a decision."

UNITS: 
- I'm here to help you make an informed decision: Meta Statement
- The RTX 3060 Ti is a powerful GPU: Claim
- The RTX 3060 is a powerful GPU: Claim
- The difference between them lies in their performance: Claim
- The RTX 3060 Ti has more CUDA cores compared to the RTX 3060: Fact
- The RTX 3060 Ti has 4864 CUDA cores: Fact
- The RTX 3060 has 3584 CUDA cores: Fact
- The RTX 3060 Ti has a lower boost clock speed compared to the RTX 3060: Fact
- The RTX 3060 Ti has a boost clock speed of 1665 MHz: Fact
- The RTX 3060 has a boost clock speed of 1777 MHz: Fact
- The RTX 3060 Ti has a slight edge over the RTX 3060 in terms of memory bandwidth: Fact
- The RTX 3060 Ti has a memory bandwidth of 448 GB/s: Fact
- The RTX 3060 has a memory bandwidth of 360 GB/s: Fact
- The difference is relatively small: Claim
- It's important to consider other factors such as power consumption when making a decision: Instruction
- It's important to consider other factors such as cooling system when making a decision: Instruction
- It's important to consider other factors such as compatibility with your system when making a decision: Instruction

Your Task:
TEXT: {_RESPONSE_PLACEHOLDER}

UNITS:{_PROMPT_END_PLACEHOLDER}
# """

# ATOM_EXTRACTION_PROMPT_V2 = """{_PROMPT_BEGIN_PLACEHOLDER}

# Instructions: 
# - Exhaustively break down the following text into independent content units. Each content unit can take one of the following forms:
#   a. Fact: An objective piece of information that can be proven or verified.
#   b. Claim: A statement or assertion that expresses a position or viewpoint on a particular topic.
#   c. Instruction: A directive or guidance on how to perform a specific task.
#   d. Data Format: Any content presented in a specific format, including code, mathematical notations, equations, variables, technical symbols, tables, or structured data formats.
#   e. Meta Statement: Disclaimers, acknowledgments, or any other statements about the nature of the response or the responder.
#   f. Question: A query or inquiry about a particular topic.
#   g. Other: Any other relevant content that doesn't fit into the above categories.
# - Label each content unit with its corresponding unit type using the format: [content unit]: [content unit type]
# - You should only output the independent content units as a list, with each item starting with "- ". Do not include other formatting or preamble text.
# - Refer to the following correct and incorrect examples to understand the task and output formats. 

# Example 1 (correct):
# TEXT: Zhejiang Huafang Pharmaceutical Co., Ltd. is a leading chemical company based in China that specializes in the research, manufacturing, and sales of various pharmaceutical products, including excipients and intermediates. The company was founded in 2018 and is located in Hangzhou, a city with a rich history in eastern China. Zhejiang Huafang Pharmaceutical Co., Ltd. is committed to providing high-quality products to its customers in the healthcare industry. The company's manufacturing facilities are equipped with state-of-the-art technology and infrastructure that ensure the production of high-quality products. Overall, Zhejiang Huafang Pharmaceutical Co., Ltd. is a reputable pharmaceutical company with a long history of success in the healthcare industry. The company's commitment to quality, innovation, and customer service has made it a leader in the field of pharmaceutical research and development.

# UNITS:
# - Zhejiang Huafang Pharmaceutical Co., Ltd. is a leading chemical company: Fact
# - Zhejiang Huafang Pharmaceutical Co., Ltd. is based in China: Fact
# - Zhejiang Huafang Pharmaceutical Co., Ltd. specializes in the research of various pharmaceutical products: Fact
# - Zhejiang Huafang Pharmaceutical Co., Ltd. specializes in the manufacturing of various pharmaceutical products: Fact
# - Zhejiang Huafang Pharmaceutical Co., Ltd. specializes in the sales of various pharmaceutical products: Fact
# - excipients are the pharmaceutical products of the Zhejiang Huafang Pharmaceutical Co., Ltd.: Fact
# - intermediates are the pharmaceutical products of the Zhejiang Huafang Pharmaceutical Co., Ltd.: Fact
# - The company was founded in 2018: Fact
# - The company is located in Hangzhou: Fact
# - Hangzhou is a city: Fact
# - Hangzhou has a rich history in eastern China: Fact
# - Zhejiang Huafang Pharmaceutical Co., Ltd. is committed to providing high-quality products to its customers in the healthcare industry: Claim
# - The company's manufacturing facilities are equipped with state-of-the-art technology: Fact
# - The company's manufacturing facilities are equipped with state-of-the-art infrastructure: Fact
# - The company's manufacturing facilities are equipped with state-of-the-art technology and infrastructure that ensure the production of high-quality products: Claim
# - Zhejiang Huafang Pharmaceutical Co., Ltd. is a reputable pharmaceutical company: Claim
# - Zhejiang Huafang Pharmaceutical Co., Ltd. has a long history of success in the healthcare industry: Claim
# - The company is committed to quality: Claim
# - The company is committed to innovation: Claim
# - The company is committed to customer service: Claim
# - The company's commitment to quality, innovation, and customer service has made it a leader in the field of pharmaceutical research: Claim
# - The company's commitment to quality, innovation, and customer service has made it a leader in the field of pharmaceutical development: Claim

# Example 2 (correct):
# TEXT: I'm here to help you make an informed decision. Both the RTX 3060 Ti and RTX 3060 are powerful GPUs, and the difference between them lies in their performance. The RTX 3060 Ti has more CUDA cores (4864 vs 3584) but a lower boost clock speed (1665 MHz vs 1777 MHz) compared to the RTX 3060. In terms of memory bandwidth, the RTX 3060 Ti has a slight edge over the RTX 3060 with a bandwidth of 448 GB/s compared to 360 GB/s. However, the difference is relatively small. It's important to consider other factors such as the power consumption, cooling system, and compatibility with your system when making a decision."

# UNITS: 
# - I'm here to help you make an informed decision: Meta Statement
# - The RTX 3060 Ti is a powerful GPU: Claim
# - The RTX 3060 is a powerful GPU: Claim
# - The difference between them lies in their performance: Claim
# - The RTX 3060 Ti has more CUDA cores compared to the RTX 3060: Fact
# - The RTX 3060 Ti has 4864 CUDA cores: Fact
# - The RTX 3060 has 3584 CUDA cores: Fact
# - The RTX 3060 Ti has a lower boost clock speed compared to the RTX 3060: Fact
# - The RTX 3060 Ti has a boost clock speed of 1665 MHz: Fact
# - The RTX 3060 has a boost clock speed of 1777 MHz: Fact
# - The RTX 3060 Ti has a slight edge over the RTX 3060 in terms of memory bandwidth: Fact
# - The RTX 3060 Ti has a memory bandwidth of 448 GB/s: Fact
# - The RTX 3060 has a memory bandwidth of 360 GB/s: Fact
# - The difference is relatively small: Claim
# - It's important to consider other factors such as power consumption when making a decision: Instruction
# - It's important to consider other factors such as cooling system when making a decision: Instruction
# - It's important to consider other factors such as compatibility with your system when making a decision: Instruction


# Example 3 (incorrect):
# TEXT: The experiment measured leaf photosynthesis under three temperature treatments: 20 °C, 25 °C, and 30 °C. Results showed that photosynthetic rate peaked at 25 °C (12 μmol CO₂ m⁻² s⁻¹) and declined at higher temperatures. Leaf respiration increased steadily with temperature.

# UNITS: 
# - The experiment measured leaf photosynthesis under different temperature treatments: Fact  
# - Photosynthetic rate peaked at 25 °C: Fact  
# - Leaf respiration increased with temperature: Claim

# Your Task:
# TEXT: {_RESPONSE_PLACEHOLDER}

# UNITS:{_PROMPT_END_PLACEHOLDER}
# """



# v1
ATOM_REVISER_PROMPT_V1 = """{_PROMPT_BEGIN_PLACEHOLDER}

You task is to decontextualize a UNIT to make it standalone. \
Each UNIT is an independent content unit extracted from the broader context of a RESPONSE.   

Vague References:
- Pronouns (e.g., "he", "she", "they", "it")
- Demonstrative pronouns (e.g., "this", "that", "these", "those")
- Unknown entities (e.g., "the event", "the research", "the invention")
- Incomplete names (e.g., "Jeff..." or "Bezos..." when referring to Jeff Bezos)

Instructions: 
Follow the steps below for unit decontextualization:
1. If the UNIT contains vague references, minimally revise them with respect to the specific subjects they refer to in the RESPONSE.
2. The decontextualized UNIT should be minimally revised by ONLY resolving vague references. No additional information must be added.
3. UNIT extraction might decompose a conjunctive statement into multiple units (e.g. Democracy treats citizens as equals regardless of their race or religion -> (1) Democracy treats citizens as equals regardless of their race, (2) Democracy treats citizens as equals regardless of their religion). Avoid adding what is potentially part of another UNIT.
4. Provide a reasoning of the revisions you made to the UNIT, justifying each decision.
5. After showing your reasoning, provide the revised unit and wrap it in a markdown code block.

Example 1: 
UNIT:
Acorns is a financial technology company

RESPONSE:
Acorns is a financial technology company founded in 2012 by Walter Cruttenden, \
Jeff Cruttenden, and Mark Dru that provides micro-investing services. The \
company is headquartered in Irvine, California.

REVISED UNIT:
This UNIT does not contain any vague references. Thus, the unit does not require any further decontextualization.
```
Acorns is a financial technology company
```

Example 2: 
UNIT:
The victim had previously suffered a broken wrist.

RESPONSE:
The clip shows the victim, with his arm in a cast, being dragged to the floor \
by his neck as his attacker says "I'll drown you" on a school playing field, while forcing water from a bottle into the victim's mouth, \
simulating waterboarding. The video was filmed in a lunch break. The clip shows the victim walking away, without reacting, as the attacker \
and others can be heard continuing to verbally abuse him. The victim, a Syrian refugee, had previously suffered a broken wrist; this had also been \
investigated by the police, who had interviewed three youths but took no further action.

REVISED UNIT:
The UNIT contains a vague reference, "the victim." This is a reference to an unknown entity, \
since it is unclear who the victim is. From the RESPONSE, we can see that the victim is a Syrian refugee. \
Thus, the vague reference "the victim" should be replaced with "the Syrian refugee victim."
```
The Syrian refugee victim had previously suffered a broken wrist.
```

Example 3:
UNIT:
The difference is relatively small.

RESPONSE:
Both the RTX 3060 Ti and RTX 3060 are powerful GPUs, and the difference between them lies in their performance. \
The RTX 3060 Ti has more CUDA cores (4864 vs 3584) but a lower boost clock speed (1665 MHz vs 1777 MHz) compared to the RTX 3060. \
In terms of memory bandwidth, the RTX 3060 Ti has a slight edge over the RTX 3060 with a bandwidth of 448 GB/s compared to 360 GB/s. \
However, the difference is relatively small and may not be noticeable in real-world applications.

REVISED UNIT:
The UNIT contains a vague reference, "The difference." From the RESPONSE, we can see that the difference is in memory bandwidth between the RTX 3060 Ti and RTX 3060. \
Thus, the vague reference "The difference" should be replaced with "The difference in memory bandwidth between the RTX 3060 Ti and RTX 3060." \
The sentence from which the UNIT is extracted includes coordinating conjunctions that potentially decompose the statement into multiple units. Thus, adding more context to the UNIT is not necessary.
```
The difference in memory bandwidth between the RTX 3060 Ti and RTX 3060 is relatively small.
```

YOUR TASK:
UNIT:
{_UNIT_PLACEHOLDER}

RESPONSE:
{_RESPONSE_PLACEHOLDER}

REVISED UNIT:{_PROMPT_END_PLACEHOLDER}
"""

ATOM_REVISER_PROMPT_V2 = """{_PROMPT_BEGIN_PLACEHOLDER}\
Instructions:
1. You are given a statement and a context that the statement belongs to. Your task is to modify the \
statement so that any pronouns or anaphora (words like "it," "they," "this") are replaced with the noun \
or proper noun that they refer to, such that the sentence remains clear without referring to the \
original context.
2. Return only the revised, standalone version of the statement without adding any information that is not \
already contained within the original statement.
3. If the statement requires no changes, return the original statement as-is without any explanation.  
4. The statement that you return must start with ### and finish with ### as follows: ###<statement>###.
5. Do not include any explanation or any additional formatting including any lead-in or sign-off text.
6. Learn from the provided examples below and use that knowledge to amend the last example yourself.

Example 1:
Context: John went to the store.
Statement: He bought some apples.
Standalone: ###John bought some apples.###

Example 2:
Context: The presentation covered various aspects of climate change, including sea level rise.
Statement: This was a key part of the discussion.
Standalone: ###Sea level rise was a key part of the discussion.###

Example 3:
Context: Maria Sanchez is a renowned marine biologist known for her groundbreaking research on coral reef ecosystems. \
Her work has contributed to the preservation of many endangered coral species, and she is often invited to speak at \
international conferences on environmental conservation.
Statement: She presented her findings at the conference last year.
Standalone: ###Maria Sanchez presented her findings at the conference last year.###

Example 4:
Context: Nathan Carter is a best-selling science fiction author famous for his dystopian novels that explore the \
intersection of technology and society. His latest book, The Edge of Something, received widespread critical acclaim \
for its imaginative world-building and its poignant commentary on artificial cacti.
Statement: It was praised for its thought-provoking themes.
Standalone: ###The Edge of Tomorrow was praised for its thought-provoking themes.###

Now perform the task for the following example:
Context: {_RESPONSE_PLACEHOLDER}
Statement: {_UNIT_PLACEHOLDER}
Standalone:{_PROMPT_END_PLACEHOLDER}        
"""

# v1: Single turn version
QUERY_BUILDER_PROMPT_V1 = """{_PROMPT_BEGIN_PLACEHOLDER}
Instructions:
Your task is to generate a Google Search query about a given STATEMENT. \
Optionally, you are also given a list of previous queries and results called KNOWLEDGE. \
Your goal is to generate a high quality query that is most likely to retrieve the relevant information about the STATEMENT.

QUERY CONSTRUCTION CRITERIA: a well-crafted query should:
  - Retrieve information to verify the STATEMENT's factual accuracy.
  - Seek new information not present in the current KNOWLEDGE.
  - Balance specificity for targeted results with breadth to avoid missing critical information.

Process:
1. Construct a Useful Google Search Query: 
  - Craft a query based on the QUERY CONSTRUCTION CRITERIA.
  - Prioritize natural language queries that a typical user might enter.
  - Use special operators (quotation marks, "site:", Boolean operators, intitle:, etc.) selectively and only when they significantly enhance the query's effectiveness.

2. Provide Query Rationale (2-3 sentences): 
  Explain how this query builds upon previous efforts and/or why it's likely to uncover new, relevant information about the STATEMENT's accuracy.

3. Format Final Query: 
  Finally, present your query enclosed in square brackets, like [QUERY].

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""

# v2: Multi-turn version (experimental)
QUERY_BUILDER_PROMPT_V2 = """{_PROMPT_BEGIN_PLACEHOLDER}
Instructions:
You are engaged in a multi-round process to refine Google Search queries about a given STATEMENT. \
Each round builds upon KNOWLEDGE (a list of previous queries and results, starting empty in round 1). \
Your goal is to improve query quality and relevance over successive rounds.

QUERY CONSTRUCTION CRITERIA: a well-crafted query should:
  - Retrieve information to verify the STATEMENT's factual accuracy.
  - Seek new information not present in the current KNOWLEDGE.
  - Balance specificity for targeted results with breadth to avoid missing critical information.
  - In rounds 2+, leverage insights from earlier queries and outcomes.

Process:
1. Construct a Useful Google Search Query: 
  - Craft a query based on the QUERY CONSTRUCTION CRITERIA.
  - Prioritize natural language queries that a typical user might enter.
  - Use special operators (quotation marks, "site:", Boolean operators, intitle:, etc.) selectively and only when they significantly enhance the query's effectiveness.

2. Provide Query Rationale (2-3 sentences): 
  Explain how this query builds upon previous efforts and/or why it's likely to uncover new, relevant information about the STATEMENT's accuracy.

3. Format Final Query: 
  Present your query enclosed in square brackets, like [QUERY].

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""

# v1
NLI_EXTRACTION_PROMPT_V1 = """{_PROMPT_BEGIN_PLACEHOLDER}

Instructions:
1. You are given a premise and a hypothesis. Your task is to identify the relationship \
between them: does the premise entail, contradict, or remain neutral toward the hypothesis?
2. Your only output must be one of: (entailment | contradiction | neutral) without any \
lead-in, sign-off, new lines or any other formatting.
3. Do not provide any explanation or rationale to your output.
4. Use the following examples to learn how to do this, and provide your output for the last \
example given.

Premise: The weather forecast said it will rain tomorrow.
Hypothesis: It will be sunny tomorrow.
Output: contradiction

Premise: The company hired three new software engineers this month.
Hypothesis: The company did not hire any new employees.
Output: contradiction

Premise: Sarah bought a new book and has been reading it every night.
Hypothesis: Sarah enjoys reading her new book in the evenings.
Output: entailment

Premise: The museum is open from 9 AM to 5 PM on weekdays.
Hypothesis: The museum is open until 6 PM on Saturdays.
Output: neutral

Premise: The company announced a new product line featuring eco-friendly materials in their \
latest press release.
Hypothesis: The company is expanding its product offerings with a focus on sustainability.
Output: Entailment

Premise: The event was canceled due to the severe storm that hit the city.
Hypothesis: The event went on as planned, with no major disruptions.
Output: Contradiction

Premise: The CEO of the tech company gave a keynote speech at the conference yesterday.
Hypothesis: The keynote speech was well-received by the audience.
Output: Neutral

Premise: {_PREMISE_PLACEHOLDER}
Hypothesis: {_HYPOTHESIS_PLACEHOLDER}
Output:{_PROMPT_END_PLACEHOLDER}
"""

# v2
NLI_EXTRACTION_PROMPT_V2 = """{_PROMPT_BEGIN_PLACEHOLDER}

Instructions:
You are provided with a PREMISE and a HYPOTHESIS. \
Your task is to evaluate the relationship between the PREMISE and the HYPOTHESIS, following the steps outlined below:

1. Evaluate Relationship:
- If the PREMISE strongly implies or directly supports the HYPOTHESIS, explain the supporting evidence.
- If the PREMISE contradicts the HYPOTHESIS, identify and explain the conflicting evidence.
- If the PREMISE is insufficient to confirm or deny the HYPOTHESIS, explain why the evidence is inconclusive.
2. Provide the reasoning behind your evaluation of the relationship between PREMISE and HYPOTHESIS, justifying each decision.
3. Final Answer: Based on your reasoning, the HYPOTHESIS and the PREMISE, determine your final answer. \
Your final answer must be one of the following, wrapped in square brackets:
- [entailment] if the PREMISE strongly implies, directly supports or entails the HYPOTHESIS.
- [contradiction] if the PREMISE contradicts the HYPOTHESIS.
- [neutral] if the PREMISE and the HYPOTHESIS neither entail nor contradict each other.

Use the following examples to better understand your task.

Example 1:
PREMISE: Robert Haldane Smith, Baron Smith of Kelvin, KT, CH, FRSGS is a British businessman and former Governor of the British Broadcasting Corporation. Smith was knighted in 1999, appointed to the House of Lords as an independent crossbench peer in 2008, and appointed Knight of the Thistle in the 2014 New Year Honours.
HYPOTHESIS: Robert Smith holds the title of Baron Smith of Kelvin.
1. Evaluate Relationship:
The PREMISE states that Robert Haldane Smith, Baron Smith of Kelvin, KT, CH, FRSGS is a British businessman and former Governor of the British Broadcasting Corporation. It also mentions that Smith was appointed to the House of Lords as an independent crossbench peer in 2008. This information directly supports the HYPOTHESIS that Robert Smith holds the title of Baron Smith of Kelvin.
2: Reasoning:
The PREMISE explicitly mentions that Robert Smith is Baron Smith of Kelvin, which directly supports the HYPOTHESIS. The additional information about his knighthood, appointment to the House of Lords, and other titles further confirms his status as a peer, but it is not necessary to support the specific HYPOTHESIS about him holding the title of Baron Smith of Kelvin.
3. Final Answer: 
[entailment]

Example 2:
PREMISE: In 2022, Passover begins in Israel at sunset on Friday, 15 April, and ends at sunset on Friday, 22 April 2022.
HYPOTHESIS: Passover in 2022 begins at sundown on March 27.
1. Evaluate Relationship:
The PREMISE states that Passover in 2022 begins at sunset on Friday, 15 April, and ends at sunset on Friday, 22 April 2022. The HYPOTHESIS claims that Passover in 2022 begins at sundown on March 27. 
Upon analyzing the information, I found that the dates mentioned in the PREMISE and the HYPOTHESIS do not match. Since the dates provided in the PREMISE and the HYPOTHESIS are different, the HYPOTHESIS is contradicted by the PREMISE.
2. Reasoning:
The PREMISE provides specific information about the start date of Passover in 2022, which is April 15. The HYPOTHESIS, on the other hand, claims a different start date, March 27. This discrepancy indicates that the PREMISE and the HYPOTHESIS cannot both be true.
3. Final Answer:
[contradiction]

Example 3:
PREMISE: Little India in the East Village: Two restaurants ablaze with tiny colored lights stand at the top of a steep staircase.
HYPOTHESIS: The village had colorful decorations on every street corner.
1. Evaluate Relationship:
The PREMISE describes a specific scene in Little India in the East Village, where two restaurants are decorated with tiny colored lights at the top of a steep staircase. The HYPOTHESIS makes a broader claim that the village had colorful decorations on every street corner.
The PREMISE provides evidence of colorful decorations in one specific location, but it does not provide information about the decorations on every street corner in the village. The PREMISE is insufficient to confirm or deny the HYPOTHESIS, as it only describes a small part of the village.
2. Reasoning:
The PREMISE and HYPOTHESIS are related in that they both mention colorful decorations, but the scope of the HYPOTHESIS is much broader than the PREMISE. The PREMISE only provides a glimpse into one specific location, whereas the HYPOTHESIS makes a general claim about the entire village. Without more information, it is impossible to determine whether the village had colorful decorations on every street corner.
3. Final Answer:
[neutral]

YOUR TASK:
PREMISE: {_PREMISE_PLACEHOLDER}
HYPOTHESIS: {_HYPOTHESIS_PLACEHOLDER}{_PROMPT_END_PLACEHOLDER}
"""

# v3
# TODO: we need a prompt specific to Google search results
NLI_EXTRACTION_PROMPT_V3_FEW_SHOTS = [
    {"claim": "Characters Lenny and Carl on The Simpsons are hearing but are depicted as close friends of the Simpsons family.", "search_result": "Title: Character Spotlight: Lenny Leonard and Carl Carlson (& Barflies)\nContent: Their friendship is a pretty singular aspect on the show -- save Bart and Milhouse (or to some degree, Mr. Burns and Smithers) -- they always ...\nLink: https://nohomers.net/forums/index.php?threads/character-spotlight-lenny-leonard-and-carl-carlson-barflies.23798/", "human_label": "Inconclusive"},
    {"claim": "The championship match of the FIFA World Cup 2026 will be hosted by the United States.", "search_result": "Title: World Cup 2026 | New York New Jersey to host final - FIFA\nContent: New York New Jersey Stadium has been confirmed as the location for the FIFA World Cup 26™ final on Sunday, 19 July 2026. The full match schedule for the ...\nLink: https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/canadamexicousa2026/articles/new-york-new-jersey-stadium-host-world-cup-2026-final", "human_label": "Supported"},
    {"claim": "It is essential to understand the limitations of heating a dead battery to temporarily revive its function.", "search_result": "Title: Why do batteries come back to life if you let them rest?\nContent: By letting the battery rest, you give the reaction products a chance to dissipate. The higher the drain on the battery, the faster the products ...\nLink: https://electronics.howstuffworks.com/everyday-tech/question390.htm", "human_label": "Inconclusive"},
    {"claim": "Sarah and James were shot execution-style in the living room.", "search_result": "Title: By Taufik | There were string of armed robberies and free murders ...\nContent: handgun been in the execution-style murder but we ... quickly cleared the living room, went ...\nLink: https://www.facebook.com/Haru6789.cv/videos/watch-killer-siblings-season-3-episode-5-allridges/411826944038089/", "human_label": "Inconclusive"},
    {"claim": "Vikings used their longships to transport livestock.", "search_result": "Title: How did the Vikings transport animals on their ships? - Quora\nContent: The Vikings transported horses overseas in boats very similar to Viking longships, but with flat flooring built within the hulls, which allowed ...\nLink: https://www.quora.com/How-did-the-Vikings-transport-animals-on-their-ships", "human_label": "Contradicted"},
    {"claim": "Romário has scored a total of 92 international goals.", "search_result": "Title: Romário - Wikipedia\nContent: A prolific striker renowned for his clinical finishing, he scored over 700 goals and is one of the few players to score at least 100 goals for three clubs. He ...\nLink: https://en.wikipedia.org/wiki/Rom%C3%A1rio", "human_label": "Contradicted"},
    {"claim": "Utopia portrays a society that values education and learning.", "search_result": "Title: Utopia Education, Science, Philosophy Summary & Analysis\nContent: The Utopians believe that it is through education that the values and dispositions of citizens are molded. The success of the Utopian educational system is ...\nLink: https://www.sparknotes.com/philosophy/utopia/section10/", "human_label": "Supported"},
    {"claim": "The higher density of water can cause sound waves to be reflected or refracted differently.", "search_result": "Title: How does sound in air differ from sound in water?\nContent: Sounds in water and sounds in air that have the same pressures have very different intensities because the density of water is much greater than ...\nLink: https://dosits.org/science/sounds-in-the-sea/how-does-sound-in-air-differ-from-sound-in-water/", "human_label": "Supported"},
    {"claim": "Mount Katahdin is 6,288.2 feet (1,917.6 meters) tall.", "search_result": "Title: Mount Katahdin - Wikipedia\nContent: Mount Katahdin is the highest mountain in the U.S. state of Maine at 5,269 feet (1,606 m). Named Katahdin, which means \"Great Mountain\", by the Penobscot ...\nLink: https://en.wikipedia.org/wiki/Mount_Katahdin", "human_label": "Contradicted"}
]

NLI_EXTRACTION_PROMPT_V3 = """{_PROMPT_BEGIN_PLACEHOLDER}

You need to judge whether a claim is supported or contradicted by a Google search result, or whether there is no enough information to make the judgement. When doing the task, take into consideration whether the link of the search result is of a trustworthy source. Place your answer in square brackets.

Below are the definitions of the three categories:

Supported: A claim is supported by the search results if everything in the claim is supported and nothing is contradicted by the search results. There can be some search results that are not fully related to the claim.
Contradicted: A claim is contradicted by the search results if something in the claim is contradicted by some search results. There should be no search result that supports the same part.
Inconclusive: A claim is inconclusive based on the search results if:
- a part of a claim cannot be verified by the search results,
- a part of a claim is supported and contradicted by different pieces of evidence,
- the entity/person mentioned in the claim has no clear referent (e.g., "the approach", "Emily", "a book").

Here are some examples:

Claim: {}

{}

Your decision: [{}]

Claim: {}

{}

Your decision: [{}]

Claim: {}

{}

Your decision: [{}]

Claim: {}

{}

Your decision: [{}]

Claim: {}

{}

Your decision: [{}]

Claim: {}

{}

Your decision: [{}]

Claim: {}

{}

Your decision: [{}]

Claim: {}

{}

Your decision: [{}]

Claim: {}

{}

Your decision: [{}]

Your task:
Claim: {_CLAIM_PLACEHOLDER}

{_SEARCH_RESULTS_PLACEHOLDER}

Your decision:{_PROMPT_END_PLACEHOLDER}
"""

# v1
CONTEXT_SUMMARIZATION_PROMPT_V1 = """{_PROMPT_BEGIN_PLACEHOLDER}

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
