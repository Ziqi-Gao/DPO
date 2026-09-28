"""Real Accelerate 1.10.1 launcher/plugin regression; no GPU or process group.

The two-H100 preflight constructs FSDP directly, whereas calibration launches
Accelerate and lets its environment construct the plugin. Keep this different
call path covered: a missing YAML option otherwise becomes the CLI default true.
"""

import importlib.util
import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

accelerate = pytest.importorskip("accelerate")
torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

accelerate_launch = pytest.importorskip("accelerate.commands.launch")
accelerate_utils = pytest.importorskip("accelerate.utils")
launch_utils = pytest.importorskip("accelerate.utils.launch")
torch_fsdp = pytest.importorskip("torch.distributed.fsdp")
_validate_launch_command = accelerate_launch._validate_launch_command
launch_command_parser = accelerate_launch.launch_command_parser
FullyShardedDataParallelPlugin = accelerate_utils.FullyShardedDataParallelPlugin
prepare_multi_gpu_env = launch_utils.prepare_multi_gpu_env
ShardingStrategy = torch_fsdp.ShardingStrategy

ROOT = Path(__file__).resolve().parents[2]
OLD_CONFIG = ROOT / "configs/accelerate/fsdp_2gpu_server_scheduler.yaml"
NEW_CONFIG = ROOT / "configs/accelerate/fsdp_2gpu_adapted_student_v2.yaml"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


worker = module("adapted_accelerate_worker", ROOT / "tools/sdsc_adapted_calibration.py")
contract = module(
    "adapted_accelerate_fsdp_contract",
    ROOT / "src/posttrain_circuits/learning/training/fsdp_contract.py",
)


def worker_launch_arguments():
    args = SimpleNamespace(
        python=Path(sys.executable).resolve(),
        science_root=ROOT,
        output_dir=Path("/scratch/calibration-fixture/outputs/qwen3-v2"),
        hf_home=Path("/scratch/calibration-fixture/huggingface"),
        student_protocol_sha256="a" * 64,
    )
    command = worker.build_plan(args, [], initial_checkpoint_sha256="b" * 64)["train_argv"]
    assert command[:4] == [str(args.python), "-B", "-m", "accelerate.commands.launch"]
    return command[4:]


def launch_with_config(path, *extra):
    """Replay the actual worker argv, changing only the tested config selector."""
    arguments = worker_launch_arguments()
    arguments[arguments.index("--config_file") + 1] = str(path)
    # Launcher options must precede its positional training module/arguments.
    position = arguments.index("-m")
    arguments[position:position] = extra
    return arguments


def resolve_launcher_plugin(arguments, inherited=None):
    assert accelerate.__version__ == "1.10.1", "regression requires the deployed Accelerate pin"
    assert torch.__version__ == "2.8.0+cu128", "regression requires the deployed PyTorch pin"
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("FSDP_", "ACCELERATE_"))
        and key not in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
    }
    environment.update(CUDA_VISIBLE_DEVICES="", **(inherited or {}))
    with (
        patch.dict(os.environ, environment, clear=True),
        # Only ephemeral-port allocation/probing is replaced. The actual parser,
        # YAML merge, generated environment and plugin conversion execute intact.
        patch("accelerate.utils.launch.get_free_port", return_value=29599),
        patch("accelerate.utils.launch.is_port_in_use", return_value=False),
        patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("GPU initialization forbidden")),
        patch.object(
            torch.distributed,
            "init_process_group",
            side_effect=AssertionError("distributed allocation forbidden"),
        ),
    ):
        args = launch_command_parser().parse_args(arguments)
        args, defaults, _ = _validate_launch_command(args)
        generated = prepare_multi_gpu_env(args)
        assert args.use_fsdp is True and args.num_processes == 2
        assert args.num_cpu_threads_per_process == 12
        assert generated["ACCELERATE_USE_FSDP"] == "true"
        assert generated["FSDP_SHARDING_STRATEGY"] == "FULL_SHARD"
        assert generated["FSDP_AUTO_WRAP_POLICY"] == "TRANSFORMER_BASED_WRAP"
        assert generated["FSDP_STATE_DICT_TYPE"] == "FULL_STATE_DICT"
        assert generated["FSDP_VERSION"] == "1"
        assert generated["FSDP_SYNC_MODULE_STATES"] == "true"
        assert generated["FSDP_CPU_RAM_EFFICIENT_LOADING"] == "true"
        assert generated["CUDA_VISIBLE_DEVICES"] == ""
        os.environ.update(generated)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The `cpu_ram_efficient_loading` flag.*")
            # These two explicit CPU-only construction arguments avoid CUDA
            # synchronization; the original generated values are asserted above.
            # Crucially, use_orig_params is NOT passed to this real plugin.
            plugin = FullyShardedDataParallelPlugin(sync_module_states=False, cpu_ram_efficient_loading=False)
        assert not torch.cuda.is_initialized()
        return args, defaults, generated, plugin


class Qwen3DecoderLayer(torch.nn.Module):
    """CPU structural fixture; no model weights or forward computation."""


class CpuFsdpWrapper(torch.nn.Module):
    """Expose the plugin value to the real post-prepare contract validator."""

    def __init__(self, wrapped, plugin):
        super().__init__()
        self.module = wrapped
        self._use_orig_params = plugin.use_orig_params
        self.sharding_strategy = ShardingStrategy.FULL_SHARD


def validate_cpu_wrapper_tree(plugin):
    root = CpuFsdpWrapper(
        torch.nn.Sequential(
            CpuFsdpWrapper(Qwen3DecoderLayer(), plugin),
            CpuFsdpWrapper(Qwen3DecoderLayer(), plugin),
        ),
        plugin,
    )
    return contract.validate_model_fsdp_sharding(root, world_size=2, fsdp_type=CpuFsdpWrapper)


def test_failed_original_worker_configuration_reproduces_real_plugin_and_guard_failure():
    args, defaults, generated, plugin = resolve_launcher_plugin(launch_with_config(OLD_CONFIG))
    assert "fsdp_use_orig_params" not in defaults.fsdp_config
    assert args.fsdp_use_orig_params == "true"
    assert generated["FSDP_USE_ORIG_PARAMS"] == "true"
    assert plugin.use_orig_params is True
    with pytest.raises(RuntimeError, match="prepared FSDP wrappers must all expose _use_orig_params=False"):
        validate_cpu_wrapper_tree(plugin)


def test_parent_environment_false_cannot_fix_the_original_launcher_default():
    _, _, generated, plugin = resolve_launcher_plugin(
        launch_with_config(OLD_CONFIG), inherited={"FSDP_USE_ORIG_PARAMS": "false"}
    )
    assert generated["FSDP_USE_ORIG_PARAMS"] == "true"
    assert plugin.use_orig_params is True


def test_explicit_cli_false_reaches_the_real_plugin():
    _, _, generated, plugin = resolve_launcher_plugin(
        launch_with_config(OLD_CONFIG, "--fsdp_use_orig_params", "false")
    )
    assert generated["FSDP_USE_ORIG_PARAMS"] == "false"
    assert plugin.use_orig_params is False
    assert validate_cpu_wrapper_tree(plugin)["fsdp_wrapper_count"] == 3


def test_revised_yaml_changes_only_the_missing_fsdp_option():
    original = yaml.safe_load(OLD_CONFIG.read_text())
    revised = yaml.safe_load(NEW_CONFIG.read_text())
    assert revised["fsdp_config"].pop("fsdp_use_orig_params") is False
    assert revised == original
    _, _, old_environment, _ = resolve_launcher_plugin(launch_with_config(OLD_CONFIG))
    _, _, new_environment, plugin = resolve_launcher_plugin(launch_with_config(NEW_CONFIG))
    assert {
        key: (old_environment.get(key), new_environment.get(key))
        for key in old_environment.keys() | new_environment.keys()
        if old_environment.get(key) != new_environment.get(key)
    } == {"FSDP_USE_ORIG_PARAMS": ("true", "false")}
    assert plugin.use_orig_params is False


def test_actual_calibration_worker_argv_uses_revised_yaml_and_passes_same_guard():
    arguments = worker_launch_arguments()
    assert arguments[arguments.index("--config_file") + 1] == str(NEW_CONFIG)
    args, defaults, generated, plugin = resolve_launcher_plugin(arguments)
    assert defaults.fsdp_config["fsdp_use_orig_params"] is False
    assert args.fsdp_use_orig_params is False
    assert generated["FSDP_USE_ORIG_PARAMS"] == "false"
    assert plugin.use_orig_params is False
    assert validate_cpu_wrapper_tree(plugin) == {
        "requested_fsdp_sharding_strategy": "FULL_SHARD",
        "effective_fsdp_sharding_strategy": "FULL_SHARD",
        "fsdp_wrapper_count": 3,
    }
