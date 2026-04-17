from __future__ import annotations

import re


MATH_SYSTEM_PROMPT = (
    "You are a careful mathematical reasoning assistant. "
    "Solve the problem step by step and end with 'Final answer: <answer>'."
)

CODE_SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Write correct, concise Python solutions and return code only."
)


def gsm8k_final_answer(solution: str) -> str:
    match = re.search(r"####\s*(.+)$", solution, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return solution.strip().splitlines()[-1].strip()


def build_math_training_messages(question: str, solution: str) -> list[dict[str, str]]:
    final_answer = gsm8k_final_answer(solution)
    assistant = solution.strip()
    if "Final answer:" not in assistant:
        assistant = f"{assistant}\nFinal answer: {final_answer}"
    return [
        {"role": "system", "content": MATH_SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": assistant},
    ]


def build_math_eval_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": MATH_SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]


def build_code_training_messages(prompt: str, code: str) -> list[dict[str, str]]:
    assistant = f"```python\n{code.strip()}\n```"
    return [
        {"role": "system", "content": CODE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt.strip()},
        {"role": "assistant", "content": assistant},
    ]


def build_code_eval_messages(prompt: str) -> list[dict[str, str]]:
    user_prompt = (
        "Complete the following Python function. Return only Python code.\n\n"
        f"{prompt.rstrip()}"
    )
    return [
        {"role": "system", "content": CODE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

