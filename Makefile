OPD_DATA_ROOT ?= /data/del6500/OPD
OPD_SCRATCH_ROOT ?= /scr/del6500/OPD
OUTPUT_ROOT ?= $(OPD_DATA_ROOT)/outputs
SMOKE_ROOT ?= $(OUTPUT_ROOT)/smoke
SMOKE_DATASET_ROOT ?= $(SMOKE_ROOT)/dataset
PYTHON ?= $(OPD_SCRATCH_ROOT)/envs/opd/bin/python

.PHONY: test lint format typecheck validate-configs smoke-factorial smoke-sft \
	smoke-grpo smoke-local-fork smoke-resume smoke-circuits readiness \
	test-scientific-design smoke-repaired-g0 smoke-dataset

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
	$(PYTHON) -m posttrain_circuits.cli.validate_configs --output $(OUTPUT_ROOT)/validation/configs.json

smoke-factorial:
	bash scripts/smoke/run_factorial.sh

smoke-dataset:
	$(PYTHON) -m posttrain_circuits.cli.build_splits task=proofgraph_small task.split_sizes.train=20 task.split_sizes.validation=20 task.split_sizes.iid_test=20 task.split_sizes.ood_depth_test=20 task.split_sizes.ood_structure_test=20 task.split_sizes.circuit_discovery=20 task.split_sizes.circuit_validation=20 --output $(SMOKE_DATASET_ROOT)

smoke-sft: smoke-dataset
	$(PYTHON) -m posttrain_circuits.cli.build_teacher_demos experiment=canonical_sft task=proofgraph_small task.dataset_family_path=$(SMOKE_DATASET_ROOT) task.num_examples=4 state_source.num_candidates=2 --output $(SMOKE_ROOT)/sft/teacher-demos
	$(PYTHON) -m posttrain_circuits.cli.train experiment=canonical_sft model=tiny_qwen task=proofgraph_small task.dataset_family_path=$(SMOKE_DATASET_ROOT) task.num_examples=4 state_source.store_path=$(SMOKE_ROOT)/sft/teacher-demos trainer.max_steps=2 --output $(SMOKE_ROOT)/sft/run

smoke-grpo: smoke-dataset
	$(PYTHON) -m posttrain_circuits.cli.run_grpo experiment=grpo_random_reward model=tiny_qwen task=proofgraph_small task.dataset_family_path=$(SMOKE_DATASET_ROOT) task.num_examples=4 trainer.max_steps=1 trainer.batch_size=4 supervision.num_generations=2 supervision.gradient_accumulation_steps=1 supervision.max_completion_length=8 --output $(SMOKE_ROOT)/grpo

smoke-local-fork:
	$(PYTHON) -m posttrain_circuits.cli.create_fork_bundle experiment=local_fork model=tiny_qwen --seed 42 --output $(SMOKE_ROOT)/local-fork/bundle.pt
	$(PYTHON) -m posttrain_circuits.cli.run_local_fork --bundle $(SMOKE_ROOT)/local-fork/bundle.pt --output $(SMOKE_ROOT)/local-fork/results.json --horizons 1

smoke-resume:
	$(PYTHON) -m pytest -q tests/unit/test_checkpointing.py tests/unit/test_state_source_resume.py

smoke-circuits:
	$(PYTHON) -m posttrain_circuits.cli.discover_circuit task=proofgraph_small circuit=eap_ig model=tiny_qwen circuit.smoke_steps=2 --output $(SMOKE_ROOT)/circuits/circuit.json
	$(PYTHON) -m posttrain_circuits.cli.evaluate_circuit task=proofgraph_small circuit.random_mask_repeats=2 circuit.prompt_bootstrap_samples=20 --circuit-artifact $(SMOKE_ROOT)/circuits/circuit.json --output $(SMOKE_ROOT)/circuits/exact-patching.json

smoke-repaired-g0:
	bash scripts/smoke/run_repaired_g0.sh

readiness:
	$(PYTHON) -m posttrain_circuits.cli.readiness --output $(OUTPUT_ROOT)/readiness
