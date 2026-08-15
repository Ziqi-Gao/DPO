"""Conservative exact-answer normalization."""

from __future__ import annotations

import re
from fractions import Fraction


def normalize_answer(text: str) -> str:
    value = text.strip().replace(",", "")
    value = re.sub(r"^\\boxed\{(.*)\}$", r"\1", value)
    try:
        return str(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return " ".join(value.lower().split())
