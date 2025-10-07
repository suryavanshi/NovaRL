import time

from examples.ppo_async import WeightSyncCadence, WeightSyncController


def test_weight_sync_interval_triggers():
    controller = WeightSyncController(WeightSyncCadence(interval=3))
    assert controller.should_sync(3, [])
    controller.mark_synced(3)
    assert not controller.should_sync(4, [0, 1])
    assert controller.should_sync(6, [])


def test_weight_sync_timeout():
    controller = WeightSyncController(WeightSyncCadence(interval=100, timeout_s=0.01))
    controller.mark_synced(0)
    time.sleep(0.02)
    assert controller.should_sync(1, [])


def test_weight_sync_staleness():
    controller = WeightSyncController(WeightSyncCadence(interval=10, max_staleness=5))
    controller.mark_synced(0)
    assert controller.should_sync(1, [6])
