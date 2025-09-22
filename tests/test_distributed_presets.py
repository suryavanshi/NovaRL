from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.distributed import (
    DeepSpeedZeROConfig,
    MegatronAdapterConfig,
    MegatronAdapterRegistry,
    NullMegatronAdapter,
    get_default_fsdp_preset,
)


def test_fsdp_default_and_overrides():
    preset = get_default_fsdp_preset()
    default_cfg = preset.to_hf_dict()
    assert default_cfg["mixed_precision"] == "bf16"
    assert default_cfg["activation_checkpointing"] is True
    override = preset.with_overrides(
        gradient_checkpointing=False, mixed_precision="fp32", cpu_offload=True
    )
    override_cfg = override.to_hf_dict()
    assert override_cfg["activation_checkpointing"] is False
    assert override_cfg["cpu_offload"] is True
    assert override_cfg["mixed_precision"] is None


def test_deepspeed_stage3_config(tmp_path: Path):
    cfg = DeepSpeedZeROConfig.stage3_default(
        offload_optimizer=True, offload_parameters=True, optimizer_sharding=False
    )
    payload = cfg.to_dict()
    zero = payload["zero_optimization"]
    assert zero["stage"] == 3
    assert zero["offload_optimizer"]["device"] == "cpu"
    assert zero["offload_param"]["device"] == "cpu"
    assert zero["reduce_scatter"] is False
    out = cfg.dump_json(tmp_path / "zero3.json")
    loaded = json.loads(out.read_text())
    assert loaded["zero_optimization"]["stage"] == 3


def test_deepspeed_stage2_config():
    cfg = DeepSpeedZeROConfig.stage2_default()
    payload = cfg.to_dict()
    zero = payload["zero_optimization"]
    assert zero["stage"] == 2
    assert "offload_param" not in zero


def test_megatron_registry():
    registry = MegatronAdapterRegistry()
    registry.register("null", NullMegatronAdapter)
    adapter = registry.create("null")
    assert isinstance(adapter, NullMegatronAdapter)
    assert "null" in registry.available()
    with pytest.raises(ValueError):
        registry.register("null", NullMegatronAdapter)

    with pytest.raises(KeyError):
        registry.create("missing")

    cfg = MegatronAdapterConfig(tensor_parallel_size=2, pipeline_parallel_size=2)
    null_adapter = NullMegatronAdapter(cfg)
    strategy = null_adapter.configure_distributed_strategy()
    assert strategy["tensor_parallel_size"] == 2
    assert strategy["pipeline_parallel_size"] == 2
