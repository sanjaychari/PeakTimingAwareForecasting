import torch

from ptaf.losses import EnhancedPeakLoss, LossConfig, MeanSquaredError, build_loss


def test_mse_runs():
    y = torch.tensor([[[1.0], [2.0]]])
    y_hat = torch.tensor([[[1.0], [0.0]]])
    assert torch.isclose(MeanSquaredError()(y, y_hat), torch.tensor(2.0))


def test_ep_matches_published_equation_16():
    # T=5, Fu=2, Fo=1.  Three points exercise: below threshold,
    # peak underprediction, and peak overprediction.
    y = torch.tensor([[[4.0], [10.0], [10.0]]])
    y_hat = torch.tensor([[[2.0], [8.0], [13.0]]])
    loss = EnhancedPeakLoss(threshold=5.0, under_weight=2.0, over_weight=1.0)

    # Base squared errors: 4, 4, 9.
    # EP extras: 0, 2*4, 1*9 => totals 4, 12, 18.
    expected = torch.tensor((4.0 + 12.0 + 18.0) / 3.0)
    assert torch.isclose(loss(y, y_hat), expected)


def test_ep_uses_strict_threshold_and_strict_direction_indicators():
    y = torch.tensor([[[5.0], [10.0]]])
    y_hat = torch.tensor([[[0.0], [10.0]]])
    loss = EnhancedPeakLoss(threshold=5.0, under_weight=100.0, over_weight=100.0)
    # y==T does not activate the EP term; y==y_hat activates neither direction.
    assert torch.isclose(loss(y, y_hat), torch.tensor(12.5))


def test_ep_supports_per_channel_thresholds():
    y = torch.tensor([[[10.0, 10.0]]])
    y_hat = torch.tensor([[[8.0, 8.0]]])
    loss = EnhancedPeakLoss(threshold=(5.0, 20.0), under_weight=2.0, over_weight=1.0)
    # Channel 0: 4 + 2*4 = 12. Channel 1: base 4 only. Mean = 8.
    assert torch.isclose(loss(y, y_hat), torch.tensor(8.0))


def test_only_paper_losses_are_enabled():
    assert build_loss(LossConfig("mse"))
    assert build_loss(LossConfig("ep", ep_threshold=1.0))
