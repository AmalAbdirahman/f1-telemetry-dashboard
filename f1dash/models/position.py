"""Predict a driver's position a few laps ahead, and test it honestly.

The model trains on earlier laps of the same race and is scored on later laps.
Tree models cannot extrapolate, so LapNumber adds little in that later period,
and the naive "position stays the same" baseline is often hard to beat. The
result always carries that baseline so the UI can say which one won.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from f1dash import config

FEATURES = ["LapNumber", "TyreLife", "Position", "CompoundCode"]


@dataclass
class Evaluation:
    model: GradientBoostingRegressor
    mae: float
    baseline_mae: float
    test: pd.DataFrame
    n_train: int
    n_test: int
    cut_lap: int
    beats_baseline: bool


def build_dataset(laps: pd.DataFrame, horizon: int = config.PREDICTION_HORIZON) -> pd.DataFrame:
    """Pair each lap with the same driver's position ``horizon`` laps later.

    The target is looked up by driver and lap number, so it cannot come from
    another driver. Laps with no known future are dropped, not filled in.
    """
    future = laps[["Driver", "LapNumber", "Position"]].dropna().rename(columns={"Position": "Target"})
    future = future.assign(LapNumber=future["LapNumber"] - horizon)
    df = laps.merge(future, on=["Driver", "LapNumber"], how="inner")
    df = df.assign(CompoundCode=df["Compound"].map(config.COMPOUND_CODES))
    return df.dropna(subset=[*FEATURES, "Target"]).reset_index(drop=True)


def temporal_split(
    dataset: pd.DataFrame,
    test_fraction: float = config.TEST_FRACTION,
    horizon: int = config.PREDICTION_HORIZON,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Split by lap number. Returns (train, test, last lap before the test period).

    Training rows whose target lap falls in the test period are removed, so
    nothing from those laps can leak into the model.
    """
    cut = int(np.floor(dataset["LapNumber"].max() * (1 - test_fraction)))
    train = dataset[dataset["LapNumber"] + horizon <= cut]
    test = dataset[dataset["LapNumber"] > cut]
    return train, test, cut


def train_and_evaluate(dataset: pd.DataFrame) -> Evaluation | None:
    """Fit the model and score it against the baseline, or None if data is thin."""
    if len(dataset) < config.MIN_DATASET_ROWS:
        return None
    train, test, cut = temporal_split(dataset)
    if train.empty or test.empty:
        return None
    model = GradientBoostingRegressor(random_state=config.SEED)
    model.fit(train[FEATURES], train["Target"])
    mae = float(mean_absolute_error(test["Target"], model.predict(test[FEATURES])))
    baseline_mae = float((test["Target"] - test["Position"]).abs().mean())
    return Evaluation(model, mae, baseline_mae, test, len(train), len(test), cut, mae < baseline_mae)


def predict(
    evaluation: Evaluation,
    lap: int,
    tyre_life: int,
    current_position: int,
    compound: str,
    field_size: int,
) -> float:
    """Predict the position ``PREDICTION_HORIZON`` laps ahead, kept within the field."""
    row = pd.DataFrame(
        [[lap, tyre_life, current_position, config.COMPOUND_CODES[compound]]],
        columns=FEATURES,
    )
    return float(np.clip(evaluation.model.predict(row)[0], 1, field_size))
