
import re
import json

from mellea import MelleaSession
from mellea.backends.types import ModelOption
from mellea.stdlib.requirement import req, check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

from mellea_ibm.rits import RITSBackend, RITS


def validate_json_code_block(input_string: str) -> bool:
    """
    Checks if the input string is a valid JSON dictionary
    and contains the 'atomic_units' key.

    Parameters:
        input_string (str): The string to check.

    Returns:
        bool: True if valid JSON dictionary with 'atomic_units' key, False otherwise.
    """
    try:

        def _strip_code_fences(s: str) -> str:
            s = s.strip()

            # Try a strict fenced block: ```json\n ... \n```
            m = re.match(r"^```(?:json|JSON)?\s*\n(.*?)\n```$", s, re.DOTALL)
            if m:
                return m.group(1).strip()

            # Fallback: starts with ``` but may have irregular spacing/line breaks
            if s.startswith("```"):
                lines = s.splitlines()
                # Remove the opening fence line
                content_lines = lines[1:]
                # If the last line is a closing fence, drop it
                if content_lines and content_lines[-1].strip().startswith("```"):
                    content_lines = content_lines[:-1]
                return "\n".join(content_lines).strip()

            # No fences detected; return as-is
            return s


        # Attempt to parse the string as JSON
        # Remove markdown fences if present
        cleaned = _strip_code_fences(input_string)
        data = json.loads(cleaned)

        # Check if it's a dictionary and has the 'atomic_units' key
        return isinstance(data, dict) and 'atomic_units' in data
    except json.JSONDecodeError:
        # If parsing fails, it's not valid JSON
        print(f"Parsing failed!")
        return False


PROMPT = """
Instructions:
Your task is to break down a given paragraph into a set of atomic units without adding any new information.

Rules:
- An atomic unit is the smallest sentence containing a singular piece of information directly extracted from the provided paragraph.
- Atomic units may contradict one another.
- The paragraph may contain information that is factually incorrect. Even in such cases, you are not to alter any information contained in the paragraph and must produce atomic units that are completely faithful to the information in the paragraph.
- Each atomic unit in the output must check a different piece of information found explicitly in the paragraph.
- Each atomic unit is standalone in that any actual nouns or proper nouns should be used in place of pronouns or anaphors.
- Each atomic unit must not include any information beyond what is explicitly stated in the provided paragraph.
- Where possible, avoid paraphrasing and instead try to only use language used in the paragraph without introducing new words. 
- The output must be a JSON dictionary with the following format and markdown code fences:

```json
{
  "atomic_units": [
    {"id": 1, "text": "<first atomic unit>."},
    {"id": 2, "text": "<second atomic unit>."},
    ...
  ]
}
```

Use the following examples to learn your task.

Example 1:
INPUT: Glenn Allen Anzalone (born June 23, 1955), better known by his stage name Glenn Danzig, is an American singer, songwriter, musician, and record producer. He is the founder of the rock bands Misfits, Samhain, and Danzig. He owns the Evilive record label as well as Verotik, an adult-oriented comic book publishing company.
OUTPUT:
```json
{
  "atomic_units": [
    {"id": 1, "text": "Glenn Allen Anzalone was born on June 23, 1955."},
    {"id": 2, "text": "Glenn Allen Anzalone is better known by his stage name Glenn Danzig."},
    {"id": 3, "text": "Glenn Danzig is an American singer, songwriter, musician, and record producer."},
    {"id": 4, "text": "Glenn Danzig is the founder of several rock bands, including Misfits, Samhain, and Danzig."},
    {"id": 5, "text": "Glenn Danzig owns the Evilive record label."},
    {"id": 6, "text": "Glenn Danzig owns Verotik, which is an adult-oriented comic book publishing company."}
  ]
}
```

Example 2:
INPUT: Luiz Inácio Lula da Silva (born 27 October 1945), also known as Lula da Silva or simply Lula, is a Brazilian politician who is the 39th and current president of Brazil since 2023. A member of the Workers' Party, Lula was also the 35th president from 2003 to 2010. He also holds the presidency of the G20 since 2023. Lula quit school after second grade to work, and did not learn to read until he was ten years old. As a teenager, he worked as a metalworker and became a trade unionist.
OUTPUT:
```json
{
  "atomic_units": [
    {"id": 1, "text": "Luiz Inácio Lula da Silva was born on October 27, 1945."},
    {"id": 2, "text": "Luiz Inácio Lula da Silva is also known as Lula da Silva or simply Lula."},
    {"id": 3, "text": "Lula is a Brazilian politician."},
    {"id": 4, "text": "Lula is the 39th and current president of Brazil since 2023."},
    {"id": 5, "text": "Lula is a member of the Workers' Party."},
    {"id": 6, "text": "Lula served as the 35th president of Brazil from 2003 to 2010."},
    {"id": 7, "text": "Lula holds the presidency of the G20 since 2023."},
    {"id": 8, "text": "Lula quit school after the second grade to work."},
    {"id": 9, "text": "Lula did not learn to read until he was ten years old."},
    {"id": 10, "text": "As a teenager, Lula worked as a metalworker."},
    {"id": 11, "text": "Lula became a trade unionist."}
  ]
}
```

Your task:
INPUT: {{response}}
OUTPUT:
"""

response = "The Apollo 14 mission to the Moon took place on January 31, 1971. \
        This mission was significant as it marked the third time humans set \
        foot on the lunar surface, with astronauts Alan Shepard and Edgar \
        Mitchell joining Captain Stuart Roosa, who had previously flown on \
        Apollo 13. The mission lasted for approximately 8 days, during which \
        the crew conducted various experiments and collected samples from the \
        lunar surface. Apollo 14 brought back approximately 70 kilograms of \
        lunar material, including rocks, soil, and core samples, which have \
        been invaluable for scientific research ever since."


m = MelleaSession(
    backend=RITSBackend(
        RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS: 300}
    )
)

model_output = m.instruct(
    PROMPT,
    requirements=[
        check(
            "The output must be a valid JSON object containing an 'atomic_units' key.",
            validation_fn=simple_validate(lambda x: validate_json_code_block(x))
        ),
    ],
    user_variables={"response": response},
    strategy=RejectionSamplingStrategy(loop_budget=3),
    return_sampling_results=True
)

print("***" * 20)
print(model_output)
print("***" * 20)
print(str(model_output))