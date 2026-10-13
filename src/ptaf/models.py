from __future__ import annotations

from dataclasses import dataclass

from .losses import MeanSquaredError

SUPPORTED_MODELS = ("dlinear", "patchtst", "tsmixer", "itransformer", "timemixer")
MULTIVARIATE_MODELS = ("tsmixer", "itransformer", "timemixer")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    horizon: int
    input_size: int
    max_steps: int = 1000
    learning_rate: float = 1e-3
    seed: int = 1
    n_series: int = 1


def build_model(config: ModelConfig, loss):
    from neuralforecast.models import DLinear, PatchTST, TSMixer, TimeMixer, iTransformer

    name = config.name.lower()
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model {config.name!r}; choose from {SUPPORTED_MODELS}")
    if config.n_series > 1 and name not in MULTIVARIATE_MODELS:
        raise ValueError(
            f"Model {config.name!r} is not enabled for full multivariate forecasting; "
            f"choose from {MULTIVARIATE_MODELS}"
        )

    common = dict(
        h=config.horizon,
        input_size=config.input_size,
        loss=loss,
        valid_loss=MeanSquaredError(),
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        scaler_type="identity",
        random_seed=config.seed,
        alias=f"{name}_{loss.__class__.__name__.lower()}",
    )
    if name == "dlinear":
        return DLinear(**common)
    if name == "patchtst":
        return PatchTST(**common)
    if name == "tsmixer":
        return TSMixer(n_series=config.n_series, **common)
    if name == "itransformer":
        return iTransformer(n_series=config.n_series, **common)
    return TimeMixer(n_series=config.n_series, channel_independence=0, **common)
