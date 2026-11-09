"""Prompts. Every reply layout the harness asks for is parsed by grids.parse_outputs / grids.parse_program.

Layouts
  direct            reason, then <output_k> grids
  direct+program    reason, then <output_k> grids AND one ```python block with transform(grid)
  rescue            thinking off; the task plus the tail of the cut reasoning; same answer layout
  repair            new user turn: the program's results on the demonstrations (Poetiq feedback); revise
  poetiq            Poetiq's solver prompt (condensed, MIT, Copyright 2025 Poetiq, Inc.)
"""
from __future__ import annotations

from typing import List, Optional

from .grids import grid_text


def format_task(task: dict, seed: Optional[int] = None, style: str = "diagram") -> str:
    """The task as text. seed shuffles the demonstration order (Poetiq does this every round);
    the numbering in the text is the shuffled numbering, and feedback uses the same numbering."""
    train, test = task["train"], task["test"]
    order = list(range(len(train)))
    if seed is not None:
        import numpy as np            # only when shuffling: the serving Python may run without numpy
        order = list(np.random.default_rng(seed).permutation(len(train)))
    s = []
    for n, i in enumerate(order, 1):
        s.append(f"Example #{n}\nInput:\n{_g(train[i]['input'], style)}\nOutput:\n{_g(train[i]['output'], style)}\n")
    for n, t in enumerate(test, 1):
        s.append(f"Test input #{n}\n{_g(t['input'], style)}\n")
    return "\n".join(s)


def _g(grid, style):
    if style == "diagram":
        return f"<Diagram>\n{grid_text(grid)}\n</Diagram>"
    return grid_text(grid)


ANSWER_GRIDS = """Write each test output grid as rows of digits separated by spaces, inside <output_1> ... </output_1>{more}.
Every row must have the same length. Do not write anything else inside the tags."""


def _more(n_test: int) -> str:
    return "" if n_test == 1 else " and " + ", ".join(f"<output_{k}> ... </output_{k}>" for k in range(2, n_test + 1))


def direct_prompt(task: dict, seed: Optional[int] = None, style: str = "diagram") -> str:
    n = len(task["test"])
    return (
        "You are solving an ARC-AGI puzzle. Each example shows an input grid and the output grid produced by the "
        "same hidden rule. Grids are rows of digits 0-9 (0 is usually the background). Work out the rule from the "
        "examples, check it against every example, then apply it to the test input"
        + ("s" if n > 1 else "") + ".\n\n" + format_task(task, seed, style) + "\n\n"
        + ANSWER_GRIDS.format(more=_more(n))
    )


PROGRAM_SPEC = """Then write the rule as a Python function `transform(grid: np.ndarray) -> np.ndarray` in ONE
```python block (numpy and scipy are available; return a 2-D array of ints 0-9). The function must reproduce every
example output from its input and must give the same test output as the grid you wrote. If you cannot write a
faithful program, still write the grid; a partial program is better than none."""


def think_hint(tokens: int) -> str:
    words = int(tokens * 0.6)
    return (f"\n\nYou have room for about {words:,} words of reasoning before you must answer. Spend them on "
            "checking a candidate rule against every example, not on re-reading the grids; when the rule fits all "
            "examples, apply it and answer.")


def direct_program_prompt(task: dict, seed: Optional[int] = None, style: str = "diagram") -> str:
    n = len(task["test"])
    return (
        "You are solving an ARC-AGI puzzle. Each example shows an input grid and the output grid produced by the "
        "same hidden rule. Grids are rows of digits 0-9 (0 is usually the background). Work out the rule from the "
        "examples, check it against every example, then apply it to the test input"
        + ("s" if n > 1 else "") + ".\n\n" + format_task(task, seed, style) + "\n\n"
        + "First, w" + ANSWER_GRIDS.format(more=_more(n))[1:] + "\n\n" + PROGRAM_SPEC
    )


def rescue_prompt(task: dict, reasoning_tail: str, with_program: bool, style: str = "diagram") -> str:
    """Thinking off. The model's own cut reasoning is shown so it can finish from where it stopped."""
    n = len(task["test"])
    body = (
        "You were solving this ARC-AGI puzzle and your reasoning was cut off by a length limit. The end of your "
        "reasoning is shown below. Do not start over: use what you had found and write the answer now.\n\n"
        + format_task(task, None, style)
        + "\n\n<your_reasoning_so_far>\n...\n" + reasoning_tail + "\n</your_reasoning_so_far>\n\n"
        + ANSWER_GRIDS.format(more=_more(n))
    )
    if with_program:
        body += "\n\n" + PROGRAM_SPEC.replace("Then write", "After the grids, write")
    return body


def repair_prompt(task: dict, program: Optional[str], feedback: str, score: float,
                  prev_grids: List[Optional[list]], style: str = "diagram") -> str:
    """One Poetiq feedback round on the model's own program, as a new user turn."""
    n = len(task["test"])
    parts = [
        "You proposed a rule for this ARC-AGI puzzle, written as a Python program. It was run on the examples and "
        "it does NOT reproduce all of them. Study the feedback, work out what the real rule is, then write a "
        "corrected answer.\n",
        format_task(task, None, style),
    ]
    if program:
        parts.append(f"\nYour previous program:\n```python\n{program}\n```")
    else:
        parts.append("\n(Your previous reply had no runnable program.)")
    parts.append(f"\nResults on the examples (score {score:.2f}, 1.0 = all examples correct):\n{feedback}")
    shown = [(k, g) for k, g in enumerate(prev_grids, 1) if g is not None]
    if shown:
        parts.append("\nThe test output grid(s) you wrote last time (numbered as the test inputs):\n" + "\n\n".join(
            f"<previous_output_{k}>\n{grid_text(g)}\n</previous_output_{k}>" for k, g in shown))
    parts.append(
        "\nThe examples are the ground truth. If your previous grid is consistent with the corrected rule, keep it; "
        "otherwise change it. Write a program that encodes the rule; do not hard-code the example outputs.\n\n"
        "First, w" + ANSWER_GRIDS.format(more=_more(n))[1:] + "\n\n" + PROGRAM_SPEC
    )
    return "\n".join(parts)


# --- Poetiq (condensed from poetiq-ai/poetiq-arc-agi-solver, MIT, Copyright 2025 Poetiq, Inc.) ---

POETIQ_SOLVER = """You solve ARC tasks by writing Python. Study the examples, find the rule, and write
`transform(grid: np.ndarray) -> np.ndarray` that turns every input into its output.
Prefer simple rules. Return the complete function in one ```python block.

{problem}"""

POETIQ_FEEDBACK = """

EXISTING PARTIAL/INCORRECT SOLUTIONS, each with its code, feedback on the examples and a
score from 0 (worst) to 1 (best). Study them and write a new solution that fixes all the issues.

{feedback}"""


def poetiq_prompt(task: dict, seed: int, tries: List[dict], max_shown: int = 5) -> str:
    problem = format_task(task, seed, "diagram").replace("Test input #", "Challenge #")
    prompt = POETIQ_SOLVER.format(problem=problem)
    if tries:
        shown = sorted(tries, key=lambda t: t["score"])[-max_shown:]  # best ones, worst to best
        prompt += POETIQ_FEEDBACK.format(feedback="\n".join(
            f"<solution_{k}>\n```python\n{t['code']}\n```\n{t['feedback']}\nScore: {t['score']:.2f}\n</solution_{k}>"
            for k, t in enumerate(shown, 1)))
    return prompt
