# Reproducibility and provenance

Datasets are reconstructed from generator version, task configuration, split namespace, and seed
range. Manifests bind canonicalized example content with SHA-256. Rollout banks bind the behavior
policy, prompt manifest, sampler, verifier, teacher, top-k, reward/length distributions, and every
Parquet/safetensors shard hash. Integrity checks run before training.

Run directories always contain resolved configuration, manifest, local JSONL metrics, environment,
git diff, checkpoints, and evaluations. W&B is optional and off. Official model/tokenizer revisions
must be immutable commit hashes; `main`, `master`, empty, and `latest` are rejected unless an
explicit `allow_unpinned_revision` override is recorded. Resolved model and tokenizer hashes belong
in the run manifest after Hub resolution.

CPU tests use a local WordLevel tokenizer and a random Qwen2Config model. They make no network call.
The login-node environment uses the official PyTorch CPU wheel. Cluster users resolve
`environment.yml` without changing system CUDA. Generated datasets, banks, weights, checkpoints,
W&B caches, secrets, and local cluster settings are ignored.

Training seeds, prompt examples, and circuit-discovery bootstrap subsets are separate variation
levels. Edge scores are not independent statistical samples. Aggregation supports hierarchical
bootstrap, paired mask-transfer permutation tests, CPR/CMD and locking intervals, and
Benjamini-Hochberg correction for secondary families.

