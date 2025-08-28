import ast
from dataclasses import dataclass, field
import math
import regex
import traceback
from typing import Any, Callable, Literal

import json_repair
from litellm.types.utils import ChatCompletionMessageToolCall, Function
from pint import Quantity

from src.fact_reasoner import ureg

type ComparisonResult = Literal[
    "equivalent",
    "contradictory",
    "first implies second",
    "second implies first",
    "neutral",
]


def repair_tool_calls(message: str) -> list[ChatCompletionMessageToolCall]:
    """
    Attempts to repair and parse imperfect tool calls from an LLM message.

    Args:
        message: str
            The message from which to extract tool calls.

    Returns:
        The list of chat completion tool calls extracted from the LLM
        message (empty if no valid tool calls can be extracted).
    """
    call_pattern = r"['\"]type['\"]:\s*['\"]function['\"]"
    tool_calls = []
    if regex.search(call_pattern, message):
        try:
            tool_call_json = json_repair.loads(message)
            if not tool_call_json:
                return []
            if type(tool_call_json) is not list:
                tool_call_json = [tool_call_json]
            for tool_call in tool_call_json:
                try:
                    function_id = tool_call["name"]  # type: ignore
                    function_parameters = tool_call["parameters"]  # type: ignore
                    function = Function(
                        arguments=function_parameters,
                        name=function_id,
                    )
                    tool_calls.append(
                        ChatCompletionMessageToolCall(
                            function=function,
                        )
                    )
                except Exception as e:
                    print(f"[QagUtils] Error repairing tool calls: {e}")
                    traceback.print_exception(e)
        except Exception as e:
            print(f"[QagUtils] Error repairing tool calls: {e}")
            traceback.print_exception(e)
    if tool_calls:
        print("[QagUtils] Attempted tool call repair.")
    return tool_calls


@dataclass
class ToolDefinition:
    """
    A definition of a tool for comparing answers to a question.

    Attributes:
        tool_id: str
            The string ID of the tool.
        tool_fun: Callable[..., ComparisonResult | None]
            The function performing the comparison and returning the result.
        tool_metadata: dict[str, Any]
            Tool definition and metadata to be passed to LiteLLM chat
            completion API.
        parameter_parsers: dict[str, Callable]
            A dictionary mapping argument names to functions that parse
            the corresponding arguments for a tool call.
        tool_config: dict[str, Any] | None
            Additional configuration associated with the tool.
    """

    tool_id: str
    tool_fun: Callable[..., ComparisonResult | None]
    tool_metadata: dict[str, Any]
    parameter_parsers: dict[str, Callable] = field(default_factory=dict)
    tool_config: dict[str, Any] | None = None


def _parse_list[T](
    element_fun: Callable[[Any], T] = lambda e: e,
) -> Callable[[Any], list[T]]:
    """
    Constructs a function for parsing a list and its elements.

    Args:
        element_fun: Callable[[Any], T]
            The function to parse the list elements. Defaults to identity.

    Returns:
        Callable[[Any], list[T]]:
            The constructed parsing function.
    """

    def parsing_fun(parsed_value: Any) -> list[T]:
        if not isinstance(parsed_value, list):
            parsed_value = ast.literal_eval(parsed_value)
        return list(map(element_fun, parsed_value))

    return parsing_fun


def _parse_quantity(magnitudes: list[float], units: list[str]) -> Quantity:
    """
    Parses a physical quantity using Pint.

    Args:
        magnitudes: list[float]
            The list of magnitudes associated with the quantity.
        units: list[str]
            The list of units associated with the quantity.

    Returns:
        Quantity:
            The parsed Pint quantity.
    """
    accumulator = None
    for magnitude, unit in zip(magnitudes, units):
        parsed_component = magnitude * ureg(unit)
        if accumulator is None:
            accumulator = parsed_component
        else:
            accumulator += parsed_component
    return accumulator


def compare_quantities_with_units(
    pair: str,
    first_magnitudes: list[float],
    first_units: list[str],
    second_magnitudes: list[float],
    second_units: list[str],
    rel_tol: float = 0.01,
) -> ComparisonResult | None:
    """
    Compares a pair of physical quantities using Pint.

    Args:
        pair: str
            A string representation of the pair of quantities to compare.
        first_magnitudes: list[float]
            The list of magnitudes associated with the first quantity in the pair.
        first_units: list[str]
            The list of units associated with the first quantity in the pair, one for
            each value in `first_magnitudes`.
        second_magnitudes: list[float]
            The list of magnitudes associated with the second quantity in the pair.
        second_units: list[str]
            The list of units associated with the second quantity in the pair, one for
            each value in `second_magnitudes`.
        rel_tol: float
            The relative tolerance used when comparing quantities.
        verbalization_map: dict[bool | None, str]
            The dictionary mapping comparison results to verbalized
            relation types.

    Returns:
        str:
            The formatted string of the comparison result to be passed
            back to the LLM.
    """
    try:
        parsed_q1 = _parse_quantity(first_magnitudes, first_units)
        parsed_q2 = _parse_quantity(second_magnitudes, second_units).to(parsed_q1.units)
        if str(parsed_q1.units) in ["year", "month", "day"] or str(parsed_q2.units) in [
            "year",
            "month",
            "day",
        ]:
            # Override rel_tol for dates
            rel_tol = 1e-5
        comparison_result = (
            "equivalent"
            if math.isclose(parsed_q1.magnitude, parsed_q2.magnitude, rel_tol=rel_tol)
            else "contradictory"
        )
    except Exception as e:
        print("Exception occured while comparing quantities:")
        print(e)
        comparison_result = None
    return comparison_result


COMPARE_QUANTITIES_WITH_UNITS_DESCRIPTION = """Compares a pair of physical quantities with units. UNSUITABLE FOR DATES, YEARS, TEXTUAL INPUTS, INEQUALITIES (e.g., 'over 25 feet') OR NON-STANDARD UNITS (e.g., 'people', 'apples') — DO NOT USE THIS FUNCTION FOR SUCH CASES.

Example:
6075 ft 7 in (1851 m 84 cm) — 1 nmi

Example call parameters:
{
    "pair": "6075 ft 7 in (1851 m 84 cm) — 1 nmi",
    "first_magnitudes": [6075.0, 7.0],
    "first_units": ["foot", "inch"],
    "second_magnitudes": [1.0],
    "second_units": ["nautical_mile"]
}

Note that units should use standardized names, in singular, with underscores for multi-word units ("nautical_mile"). Also notice that duplicate versions of the same quantity (such as 1851 m 84 cm in the example above) shouldn't be included in the function calls. Use the *, / and ^ operators to represent complex units, such as "meter^2", "kilometer^2", or "meter/second^2"."""

COMPARE_QUANTITIES_WITH_UNITS_METADATA = {
    "type": "function",
    "function": {
        "name": "compare_quantities_with_units",
        "description": "Compares a pair of physical quantities with units. UNSUITABLE FOR DATES, YEARS, TEXTUAL INPUTS, INEQUALITIES (e.g., 'over 25 feet') OR NON-STANDARD UNITS (e.g., 'people', 'apples') — DO NOT USE THIS FUNCTION FOR SUCH CASES.",
        "parameters": {
            "type": "object",
            "properties": {
                "pair": {
                    "type": "string",
                    "description": "A string representation of the pair of quantities to compare.",
                },
                "first_magnitudes": {
                    "type": "list[float]",
                    "description": "The list of magnitudes associated with the first quantity in the pair.",
                },
                "first_units": {
                    "type": "list[str]",
                    "description": "The list of units associated with the first quantity in the pair, one for each value in `first_magnitudes`.",
                },
                "second_magnitudes": {
                    "type": "list[float]",
                    "description": "The list of magnitudes associated with the second quantity in the pair.",
                },
                "second_units": {
                    "type": "list[str]",
                    "description": "The list of units associated with the second quantity in the pair, one for each value in `second_magnitudes`.",
                },
            },
            "required": [
                "pair",
                "first_magnitudes",
                "first_units",
                "second_magnitudes",
                "second_units",
            ],
        },
    },
}

compare_quantities_with_units_definition = ToolDefinition(
    tool_id="compare_quantities_with_units",
    tool_fun=compare_quantities_with_units,
    tool_metadata=COMPARE_QUANTITIES_WITH_UNITS_METADATA,
    parameter_parsers={
        "first_magnitudes": _parse_list(float),
        "first_units": _parse_list(str),
        "second_magnitudes": _parse_list(float),
        "second_units": _parse_list(str),
    },
)
