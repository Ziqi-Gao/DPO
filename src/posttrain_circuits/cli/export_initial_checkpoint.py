"""Export a pinned base HF model as the repository's checkpoint file format."""

from __future__ import annotations

import argparse
from pathlib import Path

from posttrain_circuits.core.config import compose_config
from posttrain_circuits.core.hashing import sha256_file
from posttrain_circuits.models.loading import load_model_and_tokenizer
from posttrain_circuits.training.checkpointing import atomic_torch_save


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export a pinned initial model checkpoint")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-production", action="store_true")
    args = parser.parse_args(argv)
    config = compose_config(args.overrides)
    if not args.confirm_production:
        raise SystemExit("initial production checkpoint export requires --confirm-production")
    if str(config["model"]["model_name_or_path"]).startswith("local/"):
        raise ValueError("production initial checkpoint cannot use a local/tiny model")
    loaded = load_model_and_tokenizer(config["model"], for_training=False)
    atomic_torch_save(
        args.output,
        {
            "format": "initial_hf_full_state_v1",
            "model": loaded.model.state_dict(),
            "model_id": loaded.model_id,
            "model_revision": loaded.requested_model_revision,
            "resolved_model_commit": loaded.resolved_model_commit,
            "tokenizer_id": loaded.tokenizer_id,
            "tokenizer_revision": loaded.requested_tokenizer_revision,
            "resolved_tokenizer_commit": loaded.resolved_tokenizer_commit,
            "tokenizer_hash": loaded.tokenizer_hash,
        },
    )
    print(f"{args.output} {sha256_file(args.output)}")


if __name__ == "__main__":
    main()
