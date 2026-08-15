"""Deterministic seed and RNG-state management."""

from __future__ import annotations

import base64
import pickle
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass
class RNGState:
    python: str
    numpy: str
    torch_cpu: list[int]
    torch_cuda: list[list[int]]

    @classmethod
    def capture(cls) -> RNGState:
        def encode(value):
            return base64.b64encode(pickle.dumps(value)).decode("ascii")

        return cls(
            python=encode(random.getstate()),
            numpy=encode(np.random.get_state()),
            torch_cpu=torch.get_rng_state().tolist(),
            torch_cuda=[state.tolist() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else [],
        )

    def restore(self) -> None:
        def decode(value):
            return pickle.loads(base64.b64decode(value.encode("ascii")))

        random.setstate(decode(self.python))
        np.random.set_state(decode(self.numpy))
        torch.set_rng_state(torch.tensor(self.torch_cpu, dtype=torch.uint8))
        if self.torch_cuda and torch.cuda.is_available():
            states = [torch.tensor(state, dtype=torch.uint8) for state in self.torch_cuda]
            torch.cuda.set_rng_state_all(states)

    def as_dict(self) -> dict[str, Any]:
        return {
            "python": self.python,
            "numpy": self.numpy,
            "torch_cpu": self.torch_cpu,
            "torch_cuda": self.torch_cuda,
        }
