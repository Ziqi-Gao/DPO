PYTHON ?= .venv/bin/python

.PHONY: test lint format typecheck validate-configs smoke-factorial smoke-sft \
	smoke-grpo smoke-local-fork smoke-resume smoke-circuits readiness \
	test-scientific-design smoke-repaired-g0

test:
	$(PYTHON) -m pytest -q -m "not gpu and not slow and not network"

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

typecheck:
	$(PYTHON) -m mypy src

test-scientific-design:
	$(PYTHON) -m pytest -q tests/unit/test_scientific_updates.py tests/unit/test_scientific_repair_v2.py

validate-configs:
	$(PYTHON) -m posttrain_circuits.cli.validate_configs --output outputs/validation/configs.json

smoke-factorial:
	bash scripts/smoke/run_factorial.sh

smoke-sft:
	$(PYTHON) -m posttrain_circuits.cli.build_teacher_demos experiment=canonical_sft task.num_examples=4 state_source.num_candidates=2 --output outputs/smoke-sft/teacher-demos
	$(PYTHON) -m posttrain_circuits.cli.train experiment=canonical_sft model=tiny_qwen state_source.store_path=outputs/smoke-sft/teacher-demos trainer.max_steps=2 --output outputs/smoke-sft/run

smoke-grpo:
	$(PYTHON) -m posttrain_circuits.cli.run_grpo experiment=grpo_random_reward model=tiny_qwen task.num_examples=4 trainer.max_steps=1 trainer.batch_size=4 supervision.num_generations=2 supervision.gradient_accumulation_steps=1 supervision.max_completion_length=8 --output outputs/smoke-grpo

smoke-local-fork:
	$(PYTHON) -m posttrain_circuits.cli.create_fork_bundle experiment=local_fork model=tiny_qwen --seed 42 --output outputs/smoke-local-fork/bundle.pt
	$(PYTHON) -m posttrain_circuits.cli.run_local_fork --bundle outputs/smoke-local-fork/bundle.pt --output outputs/smoke-local-fork/results.json --horizons 1

smoke-resume:
	$(PYTHON) -m pytest -q tests/unit/test_checkpointing.py tests/unit/test_state_source_resume.py

smoke-circuits:
	$(PYTHON) -m posttrain_circuits.cli.discover_circuit task=proofgraph_small circuit=eap_ig model=tiny_qwen circuit.smoke_steps=2 --output outputs/smoke-circuits/circuit.json
	$(PYTHON) -m posttrain_circuits.cli.evaluate_circuit task=proofgraph_small circuit.random_mask_repeats=2 circuit.prompt_bootstrap_samples=20 --circuit-artifact outputs/smoke-circuits/circuit.json --output outputs/smoke-circuits/exact-patching.json

smoke-repaired-g0:
	bash scripts/smoke/run_repaired_g0.sh

readiness:
	$(PYTHON) -m posttrain_circuits.cli.readiness --output outputs/readiness
