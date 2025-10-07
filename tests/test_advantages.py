import torch

from core.utils.advantages import GAEAdvantageEstimator, normalize_advantages
from core.utils.stats import compute_gae


def test_gae_estimator_matches_reference():
    torch.manual_seed(0)
    rewards = torch.randn(5, 3)
    values = torch.randn(5, 3)
    bootstrap = torch.randn(3)
    estimator = GAEAdvantageEstimator(gamma=0.9, lam=0.8)
    result = estimator.estimate(
        rewards=rewards,
        values=values,
        dones=torch.zeros_like(rewards),
        bootstrap_value=bootstrap,
    )
    reference = compute_gae(
        rewards=rewards,
        values=torch.cat([values, bootstrap.unsqueeze(0)], dim=0),
        dones=torch.zeros_like(rewards),
        gamma=0.9,
        lam=0.8,
    )
    assert torch.allclose(result.advantages, reference)
    assert torch.allclose(result.returns, reference + values)


def test_normalize_advantages_zero_mean():
    data = torch.tensor([1.0, 2.0, 3.0])
    normalized = normalize_advantages(data)
    assert torch.isclose(normalized.mean(), torch.tensor(0.0), atol=1e-6)
