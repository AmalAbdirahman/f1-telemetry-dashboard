import numpy as np
import pandas as pd

from f1dash import config
from f1dash.models import position


def race_laps(n_drivers=10, n_laps=50, seed=3):
    """Laps listed driver by driver, as the data layer sorts them."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_drivers):
        for lap in range(1, n_laps + 1):
            rows.append(
                {
                    "Driver": f"D{d}",
                    "LapNumber": float(lap),
                    "TyreLife": float(lap),
                    "Position": float(d + 1),
                    "Compound": "HARD" if d % 2 else "MEDIUM",
                    "TrackStatus": "1",
                    "Noise": rng.normal(),
                }
            )
    return pd.DataFrame(rows)


def test_regression_targets_never_come_from_another_driver():
    laps = race_laps()
    dataset = position.build_dataset(laps, horizon=3)
    # Each driver holds their position all race, so any leak shows as a mismatch.
    assert (dataset["Target"] == dataset["Position"]).all()


def test_regression_target_is_the_drivers_position_three_laps_later():
    laps = race_laps(n_drivers=2, n_laps=10)
    laps.loc[(laps["Driver"] == "D0") & (laps["LapNumber"] == 7), "Position"] = 9.0
    dataset = position.build_dataset(laps, horizon=3)
    row = dataset[(dataset["Driver"] == "D0") & (dataset["LapNumber"] == 4)].iloc[0]
    assert row["Target"] == 9


def test_regression_laps_with_no_known_future_are_dropped_not_invented():
    laps = race_laps(n_drivers=3, n_laps=10)
    dataset = position.build_dataset(laps, horizon=3)
    assert dataset["LapNumber"].max() == 7
    assert len(dataset) == 3 * 7
    # A retirement leaves no target, so it must not gain one.
    laps = laps[~((laps["Driver"] == "D2") & (laps["LapNumber"] > 5))]
    dataset = position.build_dataset(laps, horizon=3)
    assert dataset[dataset["Driver"] == "D2"]["LapNumber"].max() == 2


def test_regression_split_is_by_lap_and_targets_stay_out_of_the_test_period():
    dataset = position.build_dataset(race_laps(), horizon=3)
    train, test, cut = position.temporal_split(dataset, test_fraction=0.2, horizon=3)
    assert train["LapNumber"].max() + 3 <= cut
    assert test["LapNumber"].min() > cut
    # Every driver appears on both sides: the test set is later laps, not the last drivers.
    assert set(train["Driver"]) == set(test["Driver"])


def test_features_are_known_at_prediction_time_and_vary_by_driver():
    assert position.FEATURES == ["LapNumber", "TyreLife", "Position", "CompoundCode"]
    result = position.train_and_evaluate(position.build_dataset(race_laps(), horizon=3))
    front = position.predict(result, lap=30, tyre_life=20, current_position=1, compound="HARD", field_size=10)
    back = position.predict(result, lap=30, tyre_life=20, current_position=10, compound="HARD", field_size=10)
    assert front < back


def test_baseline_is_position_stays_the_same():
    laps = race_laps()
    # Half the field drops one place late on, so the naive baseline has a known error.
    mover = laps["Driver"].isin(["D0", "D2", "D4", "D6", "D8"]) & (laps["LapNumber"] >= 45)
    laps.loc[mover, "Position"] += 1
    result = position.train_and_evaluate(position.build_dataset(laps, horizon=3))
    test = result.test
    assert result.baseline_mae == (test["Target"] - test["Position"]).abs().mean()
    assert result.n_test == len(test)
    assert result.beats_baseline == (result.mae < result.baseline_mae)


def test_prediction_is_clipped_to_the_field():
    result = position.train_and_evaluate(position.build_dataset(race_laps(), horizon=3))
    assert 1 <= position.predict(result, 30, 20, 1, "HARD", 10) <= 10


def test_too_little_data_gives_no_model():
    laps = race_laps(n_drivers=2, n_laps=10)
    assert len(laps) < config.MIN_DATASET_ROWS
    assert position.train_and_evaluate(position.build_dataset(laps, horizon=3)) is None


def test_laps_without_a_compound_or_position_are_excluded():
    laps = race_laps(n_drivers=3, n_laps=10)
    laps.loc[0, "Compound"] = "UNKNOWN"
    laps.loc[1, "Position"] = np.nan
    dataset = position.build_dataset(laps, horizon=3)
    assert dataset[["CompoundCode", "Position", "Target"]].notna().all().all()
    assert dataset[dataset["Driver"] == "D0"]["LapNumber"].min() > 2


# ── Fix 2: at lap N the model only learns from outcomes known by lap N ───

def test_regression_model_at_lap_n_never_sees_later_outcomes():
    laps = race_laps()
    lap_n = 40
    first = position.train_and_evaluate(position.build_dataset(laps), known_by=lap_n)
    shuffled = laps.copy()
    later = shuffled["LapNumber"] > lap_n
    shuffled.loc[later, "Position"] = 11 - shuffled.loc[later, "Position"]
    second = position.train_and_evaluate(position.build_dataset(shuffled), known_by=lap_n)
    assert (first.mae, first.baseline_mae, first.n_train, first.n_test) == (
        second.mae, second.baseline_mae, second.n_train, second.n_test
    )
    assert first.cut_lap + config.PREDICTION_HORIZON <= lap_n
    assert (first.test["LapNumber"] + config.PREDICTION_HORIZON).max() <= lap_n


def test_too_early_in_the_race_gives_no_model():
    assert position.train_and_evaluate(position.build_dataset(race_laps()), known_by=6) is None
