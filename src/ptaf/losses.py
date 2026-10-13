from __future__ import annotations

from dataclasses import dataclass

import torch


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


class _PointLoss(torch.nn.Module):
    outputsize_multiplier = 1
    output_names = [""]
    is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor) -> torch.Tensor:
        return y_hat


class MeanSquaredError(_PointLoss):
    def forward(self, y, y_hat, mask=None, y_insample=None):
        del y_insample
        return _masked_mean((y - y_hat) ** 2, mask)


def _channel_parameter(value, reference: torch.Tensor, name: str) -> torch.Tensor:
    """Broadcast a scalar or per-channel parameter over NeuralForecast [B,H,N]."""
    parameter = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if parameter.ndim == 0:
        return parameter
    parameter = parameter.flatten()
    if reference.ndim != 3:
        if parameter.numel() != 1:
            raise ValueError(f"{name} has per-channel values but loss input is not [B, H, N]")
        return parameter[0]
    if parameter.numel() != reference.shape[-1]:
        raise ValueError(
            f"{name} has {parameter.numel()} values but loss input has "
            f"{reference.shape[-1]} channels"
        )
    return parameter.view(1, 1, -1)


class ModifiedEnhancedPeakLoss(_PointLoss):
    """Modified Enhanced Peak (MEP) loss for upper and lower excursions.

    The additional EP penalty is activated when the target is either above
    ``high_threshold`` or below ``low_threshold``. Underprediction and
    overprediction retain their ordinary numerical meanings for both peaks
    and troughs.
    """

    def __init__(
        self,
        low_threshold,
        high_threshold,
        under_weight: float = 2.0,
        over_weight: float = 1.0,
    ):
        super().__init__()
        low = torch.as_tensor(low_threshold, dtype=torch.float64).flatten()
        high = torch.as_tensor(high_threshold, dtype=torch.float64).flatten()
        if low.numel() == 0 or high.numel() == 0:
            raise ValueError("MEP thresholds must not be empty")
        if not torch.all(torch.isfinite(low)) or not torch.all(torch.isfinite(high)):
            raise ValueError("MEP thresholds must contain finite values")
        if low.numel() != high.numel() and low.numel() != 1 and high.numel() != 1:
            raise ValueError("MEP low/high thresholds must have compatible channel counts")
        if torch.any(low >= high):
            raise ValueError("MEP requires low_threshold < high_threshold")
        if under_weight < 0 or over_weight < 0:
            raise ValueError("MEP penalty factors must be non-negative")
        self.low_threshold = tuple(float(v) for v in low.tolist()) if low.numel() > 1 else float(low[0])
        self.high_threshold = tuple(float(v) for v in high.tolist()) if high.numel() > 1 else float(high[0])
        self.under_weight = float(under_weight)
        self.over_weight = float(over_weight)

    def forward(self, y, y_hat, mask=None, y_insample=None):
        del y_insample
        squared_error = (y - y_hat) ** 2
        low = _channel_parameter(self.low_threshold, y, "MEP low threshold")
        high = _channel_parameter(self.high_threshold, y, "MEP high threshold")
        excursion = ((y > high) | (y < low)).to(squared_error.dtype)
        under = (y_hat < y).to(squared_error.dtype)
        over = (y_hat > y).to(squared_error.dtype)
        loss = squared_error * (
            1.0
            + self.under_weight * excursion * under
            + self.over_weight * excursion * over
        )
        return _masked_mean(loss, mask)


class EnhancedPeakLoss(_PointLoss):
    """Enhanced Peak (EP) loss from Hajiabbasi Somehsaraie et al. (2025).

    Equation (16) in the paper is

        MSE
        + Fu * MSE * 1(y > T) * 1(y > y_hat)
        + Fo * MSE * 1(y > T) * 1(y_hat > y).

    ``threshold`` may be scalar (univariate) or one value per output channel
    (full multivariate forecasting). The threshold and penalty factors are
    hyperparameters in the paper and are therefore configurable here.
    """

    def __init__(self, threshold, under_weight: float = 2.0, over_weight: float = 1.0):
        super().__init__()
        values = torch.as_tensor(threshold, dtype=torch.float64).flatten()
        if values.numel() == 0 or not torch.all(torch.isfinite(values)):
            raise ValueError("threshold must contain finite values")
        if under_weight < 0 or over_weight < 0:
            raise ValueError("EP penalty factors must be non-negative")
        self.threshold = (
            tuple(float(v) for v in values.tolist())
            if values.numel() > 1
            else float(values[0])
        )
        self.under_weight = float(under_weight)
        self.over_weight = float(over_weight)

    def forward(self, y, y_hat, mask=None, y_insample=None):
        del y_insample
        squared_error = (y - y_hat) ** 2
        threshold = _channel_parameter(self.threshold, y, "EP threshold")
        peak = (y > threshold).to(squared_error.dtype)
        under = (y > y_hat).to(squared_error.dtype)
        over = (y_hat > y).to(squared_error.dtype)
        loss = squared_error * (
            1.0
            + self.under_weight * peak * under
            + self.over_weight * peak * over
        )
        return _masked_mean(loss, mask)


@dataclass(frozen=True)
class LossConfig:
    name: str
    ep_threshold: float | tuple[float, ...] | None = None
    ep_under_weight: float = 2.0
    ep_over_weight: float = 1.0
    mep_low_threshold: float | tuple[float, ...] | None = None
    mep_high_threshold: float | tuple[float, ...] | None = None


def build_loss(config: LossConfig) -> _PointLoss:
    name = config.name.lower()
    if name == "mse":
        return MeanSquaredError()
    if name == "ep":
        if config.ep_threshold is None:
            raise ValueError("EP requires a training-calibrated threshold")
        return EnhancedPeakLoss(
            threshold=config.ep_threshold,
            under_weight=config.ep_under_weight,
            over_weight=config.ep_over_weight,
        )
    if name == "mep":
        if config.mep_low_threshold is None or config.mep_high_threshold is None:
            raise ValueError("MEP requires training-calibrated lower and upper thresholds")
        return ModifiedEnhancedPeakLoss(
            low_threshold=config.mep_low_threshold,
            high_threshold=config.mep_high_threshold,
            under_weight=config.ep_under_weight,
            over_weight=config.ep_over_weight,
        )
    raise ValueError("loss must be one of: mse, ep, mep")
