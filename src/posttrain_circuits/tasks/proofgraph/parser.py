"""Strict but whitespace-tolerant ProofGraph response parser."""

from __future__ import annotations

import re

from posttrain_circuits.tasks.proofgraph.schemas import Literal, ParsedResponse, ProofStep

_RESPONSE = re.compile(
    r"^\s*<proof>\s*(?P<proof>.*?)\s*</proof>\s*<answer>\s*(?P<answer>[01])\s*</answer>\s*$",
    re.DOTALL,
)
_STEP = re.compile(r"^(?P<step>S\d+):\s*(?P<rule>R\d+)\((?P<citations>[^)]*)\)\s*->\s*(?P<literal>.+?)\s*$")


def parse_response(text: str) -> ParsedResponse:
    match = _RESPONSE.match(text)
    if match is None:
        return ParsedResponse(False, [], None, text, "response_syntax")
    steps: list[ProofStep] = []
    proof_text = match.group("proof").strip()
    if proof_text:
        for line in proof_text.splitlines():
            step_match = _STEP.match(line.strip())
            if step_match is None:
                return ParsedResponse(False, [], int(match.group("answer")), text, "step_syntax")
            citations = tuple(
                citation.strip() for citation in step_match.group("citations").split(",") if citation.strip()
            )
            try:
                conclusion = Literal.parse(step_match.group("literal"))
            except ValueError:
                return ParsedResponse(False, [], int(match.group("answer")), text, "literal_syntax")
            steps.append(
                ProofStep(
                    step_id=step_match.group("step"),
                    rule_id=step_match.group("rule"),
                    citations=citations,
                    conclusion=conclusion,
                )
            )
    return ParsedResponse(True, steps, int(match.group("answer")), text)
