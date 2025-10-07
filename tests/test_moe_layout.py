import pytest

from core.distributed import MoERouterConfig, ParallelLayout


def test_parallel_layout_from_world_size():
    layout = ParallelLayout.from_world_size(
        total_world_size=8, tensor_parallel_size=2, pipeline_parallel_size=2, expert_parallel_size=2
    )
    assert layout.data_parallel_size == 1
    assert layout.describe()["world"] == 8


def test_parallel_layout_validation_failure():
    with pytest.raises(ValueError):
        ParallelLayout.from_world_size(total_world_size=7, tensor_parallel_size=2)


def test_moe_router_to_deepspeed():
    layout = ParallelLayout.from_world_size(total_world_size=8, tensor_parallel_size=2, expert_parallel_size=2)
    router = MoERouterConfig(num_experts=4, top_k=2, capacity_factor=1.3)
    cfg = router.to_deepspeed(layout)
    assert cfg["num_experts"] == 4
    assert cfg["expert_parallel_size"] == 2
