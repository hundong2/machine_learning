"""Train a stacked ensemble for short-horizon mosquito trajectory forecasting.

This version uses a stronger strategy than a fixed weighted blend:

1. Build fold-bagged base predictors.
   - A recent-window random forest residual model.
   - A compact alpha-gating model between constant-velocity and
     constant-acceleration priors.
2. Convert their out-of-fold train predictions into meta-features.
3. Train a second-stage gate that decides, sample by sample, how far to move
   from the recent residual prediction toward the alpha-gated physics
   prediction.

The submission uses averaged fold-model test predictions at the base level,
then applies the learned second-stage gate.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
)

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
TRAIN_LABELS_PATH = DATA_DIR / "train_labels.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
SCENARIO_RESULTS_PATH = ROOT / "scenario_results.csv"
SUBMISSION_PATH = ROOT / "submission.csv"

R_HIT_RADIUS = 0.01
FORECAST_STEPS = 2
RANDOM_STATE = 42
N_SPLITS = 5

PHYSICS_BLEND_ACCEL_WEIGHT = 0.30
RECENT_RESIDUAL_BLEND_WEIGHT = 0.70
ALPHA_CLIP_LOW = -0.30
ALPHA_CLIP_HIGH = 1.30

FIXED_WEIGHT_RECENT = 0.34
FIXED_WEIGHT_ALPHA = 0.48
FIXED_WEIGHT_PHYSICS = 0.18


def load_series(sample_ids: list[str], folder: Path) -> np.ndarray:
    """Load each trajectory CSV into one (11, 3) tensor row."""

    trajectories = []
    for sample_id in sample_ids:
        frame = pd.read_csv(folder / f"{sample_id}.csv")
        trajectories.append(frame[["x", "y", "z"]].to_numpy(dtype=np.float32))
    return np.stack(trajectories)


def load_datasets() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Read train trajectories, train labels, test trajectories, and test ids."""

    train_labels = pd.read_csv(TRAIN_LABELS_PATH).sort_values("id").reset_index(drop=True)
    test_template = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    train_ids = train_labels["id"].tolist()
    test_ids = test_template["id"].tolist()

    train_series = load_series(train_ids, TRAIN_DIR)
    test_series = load_series(test_ids, TEST_DIR)
    train_targets = train_labels[["x", "y", "z"]].to_numpy(dtype=np.float32)
    return train_series, train_targets, test_series, test_ids


def build_motion_priors(trajectories: np.ndarray) -> dict[str, np.ndarray]:
    """Build analytical priors used by every learned component."""

    last_position = trajectories[:, -1, :]
    velocities = np.diff(trajectories, axis=1)
    last_velocity = velocities[:, -1, :]
    previous_velocity = velocities[:, -2, :]
    last_acceleration = last_velocity - previous_velocity

    constant_velocity = last_position + FORECAST_STEPS * last_velocity
    constant_acceleration = constant_velocity + (FORECAST_STEPS * last_acceleration)
    physics_blend = (
        (1.0 - PHYSICS_BLEND_ACCEL_WEIGHT) * constant_velocity
        + PHYSICS_BLEND_ACCEL_WEIGHT * constant_acceleration
    )

    return {
        "last_position": last_position,
        "velocities": velocities,
        "last_velocity": last_velocity,
        "previous_velocity": previous_velocity,
        "last_acceleration": last_acceleration,
        "constant_velocity": constant_velocity,
        "constant_acceleration": constant_acceleration,
        "physics_blend": physics_blend,
    }


def build_recent_residual_features(
    trajectories: np.ndarray,
    priors: dict[str, np.ndarray],
) -> np.ndarray:
    """Feature set for the recent-window residual model."""

    velocities = priors["velocities"]
    accelerations = np.diff(trajectories, n=2, axis=1)
    return np.concatenate(
        [
            trajectories[:, -5:, :].reshape(len(trajectories), -1),
            velocities[:, -4:, :].reshape(len(trajectories), -1),
            accelerations[:, -3:, :].reshape(len(trajectories), -1),
            priors["constant_velocity"],
            priors["constant_acceleration"],
        ],
        axis=1,
    ).astype(np.float32)


def build_alpha_summary_features(
    trajectories: np.ndarray,
    priors: dict[str, np.ndarray],
) -> np.ndarray:
    """Compact motion summary features for the alpha model and meta-gate."""

    velocities = priors["velocities"]
    jerks = np.diff(trajectories, n=3, axis=1)

    last_velocity = priors["last_velocity"]
    previous_velocity = priors["previous_velocity"]
    previous_previous_velocity = velocities[:, -3, :]
    last_acceleration = priors["last_acceleration"]
    previous_acceleration = previous_velocity - previous_previous_velocity
    last_position = priors["last_position"]

    def cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        denominator = (
            np.linalg.norm(lhs, axis=1) * np.linalg.norm(rhs, axis=1) + 1e-6
        )
        return np.sum(lhs * rhs, axis=1) / denominator

    return np.column_stack(
        [
            np.linalg.norm(last_velocity, axis=1),
            np.linalg.norm(previous_velocity, axis=1),
            np.linalg.norm(previous_previous_velocity, axis=1),
            np.linalg.norm(last_acceleration, axis=1),
            np.linalg.norm(previous_acceleration, axis=1),
            np.linalg.norm(jerks[:, -1, :], axis=1),
            cosine_similarity(last_velocity, previous_velocity),
            cosine_similarity(last_acceleration, previous_acceleration),
            cosine_similarity(last_acceleration, last_velocity),
            cosine_similarity(last_acceleration, previous_velocity),
            np.linalg.norm(last_velocity, axis=1)
            / (np.linalg.norm(previous_velocity, axis=1) + 1e-6),
            np.linalg.norm(previous_velocity, axis=1)
            / (np.linalg.norm(previous_previous_velocity, axis=1) + 1e-6),
            np.linalg.norm(last_acceleration, axis=1)
            / (np.linalg.norm(previous_acceleration, axis=1) + 1e-6),
            last_velocity[:, 0],
            last_velocity[:, 1],
            last_velocity[:, 2],
            last_acceleration[:, 0],
            last_acceleration[:, 1],
            last_acceleration[:, 2],
            previous_velocity[:, 0],
            previous_velocity[:, 1],
            previous_velocity[:, 2],
            previous_acceleration[:, 0],
            previous_acceleration[:, 1],
            previous_acceleration[:, 2],
            last_position[:, 0],
            last_position[:, 1],
            last_position[:, 2],
        ]
    ).astype(np.float32)


def build_feature_sets(trajectories: np.ndarray) -> dict[str, np.ndarray]:
    """Assemble the priors and derived feature matrices."""

    priors = build_motion_priors(trajectories)
    return {
        "recent_residual": build_recent_residual_features(trajectories, priors),
        "alpha_summary": build_alpha_summary_features(trajectories, priors),
        **priors,
    }


def slice_feature_sets(
    feature_sets: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Slice every feature tensor with the same row indices."""

    return {key: value[indices] for key, value in feature_sets.items()}


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return both distance errors and the competition hit metric."""

    distance = np.linalg.norm(y_pred - y_true, axis=1)
    return {
        "mean_distance": float(distance.mean()),
        "median_distance": float(np.median(distance)),
        "r_hit_at_1cm": float((distance <= R_HIT_RADIUS).mean()),
    }


def build_recent_residual_model() -> RandomForestRegressor:
    """Model for residual corrections around the constant-velocity prior."""

    return RandomForestRegressor(
        n_estimators=800,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_alpha_model() -> HistGradientBoostingRegressor:
    """Model for sample-wise acceleration gating."""

    return HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=40,
        random_state=RANDOM_STATE,
    )


def build_gate_model() -> HistGradientBoostingRegressor:
    """Second-stage model that chooses between the two strong base predictors."""

    return HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=50,
        random_state=RANDOM_STATE,
    )


def compute_alpha_targets(
    targets: np.ndarray,
    feature_sets: dict[str, np.ndarray],
) -> np.ndarray:
    """Project the true target onto the line between CV and CA priors."""

    constant_velocity = feature_sets["constant_velocity"]
    delta = feature_sets["constant_acceleration"] - constant_velocity
    numerator = np.sum((targets - constant_velocity) * delta, axis=1)
    denominator = np.sum(delta * delta, axis=1) + 1e-12
    return (numerator / denominator).astype(np.float32)


def compute_gate_targets(
    targets: np.ndarray,
    recent_component: np.ndarray,
    alpha_component: np.ndarray,
) -> np.ndarray:
    """Project the true target onto the line between the two strong base models."""

    delta = alpha_component - recent_component
    numerator = np.sum((targets - recent_component) * delta, axis=1)
    denominator = np.sum(delta * delta, axis=1) + 1e-12
    weights = numerator / denominator
    return np.clip(weights, ALPHA_CLIP_LOW, ALPHA_CLIP_HIGH).astype(np.float32)


def predict_recent_component(
    model: RandomForestRegressor,
    feature_sets: dict[str, np.ndarray],
) -> np.ndarray:
    """Predict the residual-model component on one dataset split."""

    constant_velocity = feature_sets["constant_velocity"]
    corrected = constant_velocity + model.predict(feature_sets["recent_residual"])
    return (
        (1.0 - RECENT_RESIDUAL_BLEND_WEIGHT) * constant_velocity
        + RECENT_RESIDUAL_BLEND_WEIGHT * corrected
    )


def predict_alpha_component(
    model: HistGradientBoostingRegressor,
    feature_sets: dict[str, np.ndarray],
) -> np.ndarray:
    """Predict the alpha-gating component on one dataset split."""

    constant_velocity = feature_sets["constant_velocity"]
    delta = feature_sets["constant_acceleration"] - constant_velocity
    alpha = np.clip(
        model.predict(feature_sets["alpha_summary"]),
        ALPHA_CLIP_LOW,
        ALPHA_CLIP_HIGH,
    )
    return constant_velocity + alpha[:, None] * delta


def build_fixed_ensemble(
    recent_component: np.ndarray,
    alpha_component: np.ndarray,
    physics_blend: np.ndarray,
) -> np.ndarray:
    """Keep the previous fixed blend as a comparison baseline."""

    return (
        FIXED_WEIGHT_RECENT * recent_component
        + FIXED_WEIGHT_ALPHA * alpha_component
        + FIXED_WEIGHT_PHYSICS * physics_blend
    )


def build_gate_meta_features(
    summary_features: np.ndarray,
    physics_blend: np.ndarray,
    recent_component: np.ndarray,
    alpha_component: np.ndarray,
) -> np.ndarray:
    """Create second-stage features from base predictions and motion summaries."""

    delta = alpha_component - recent_component
    return np.concatenate(
        [
            summary_features,
            physics_blend,
            recent_component,
            alpha_component,
            delta,
            np.linalg.norm(delta, axis=1, keepdims=True),
        ],
        axis=1,
    ).astype(np.float32)


def predict_gated_ensemble(
    gate_weights: np.ndarray,
    recent_component: np.ndarray,
    alpha_component: np.ndarray,
) -> np.ndarray:
    """Move from the recent predictor toward the alpha predictor by a learned amount."""

    clipped_weights = np.clip(gate_weights, ALPHA_CLIP_LOW, ALPHA_CLIP_HIGH)
    return recent_component + clipped_weights[:, None] * (alpha_component - recent_component)


def generate_base_predictions(
    train_feature_sets: dict[str, np.ndarray],
    train_targets: np.ndarray,
    test_feature_sets: dict[str, np.ndarray],
    n_splits: int = N_SPLITS,
) -> tuple[dict[str, np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """Create OOF train predictions and fold-averaged test predictions."""

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_indices = list(splitter.split(train_targets))
    alpha_targets = compute_alpha_targets(train_targets, train_feature_sets)

    train_recent = np.zeros_like(train_targets)
    train_alpha = np.zeros_like(train_targets)
    test_recent_folds = []
    test_alpha_folds = []

    for train_idx, valid_idx in fold_indices:
        train_features = slice_feature_sets(train_feature_sets, train_idx)
        valid_features = slice_feature_sets(train_feature_sets, valid_idx)

        recent_model = build_recent_residual_model()
        recent_model.fit(
            train_features["recent_residual"],
            train_targets[train_idx] - train_features["constant_velocity"],
        )
        train_recent[valid_idx] = predict_recent_component(recent_model, valid_features)
        test_recent_folds.append(predict_recent_component(recent_model, test_feature_sets))

        alpha_model = build_alpha_model()
        alpha_model.fit(train_features["alpha_summary"], alpha_targets[train_idx])
        train_alpha[valid_idx] = predict_alpha_component(alpha_model, valid_features)
        test_alpha_folds.append(predict_alpha_component(alpha_model, test_feature_sets))

    return (
        {
            "train_recent": train_recent,
            "train_alpha": train_alpha,
            "train_physics": train_feature_sets["physics_blend"],
            "test_recent": np.mean(test_recent_folds, axis=0),
            "test_alpha": np.mean(test_alpha_folds, axis=0),
            "test_physics": test_feature_sets["physics_blend"],
        },
        fold_indices,
    )


def evaluate_scenarios(
    train_targets: np.ndarray,
    train_feature_sets: dict[str, np.ndarray],
    base_predictions: dict[str, np.ndarray],
    fold_indices: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Evaluate base predictors, fixed blend, and stacked gate ensemble."""

    gate_meta = build_gate_meta_features(
        summary_features=train_feature_sets["alpha_summary"],
        physics_blend=base_predictions["train_physics"],
        recent_component=base_predictions["train_recent"],
        alpha_component=base_predictions["train_alpha"],
    )
    gate_targets = compute_gate_targets(
        targets=train_targets,
        recent_component=base_predictions["train_recent"],
        alpha_component=base_predictions["train_alpha"],
    )

    gate_weights_oof = np.zeros(len(train_targets), dtype=np.float32)
    for train_idx, valid_idx in fold_indices:
        gate_model = build_gate_model()
        gate_model.fit(gate_meta[train_idx], gate_targets[train_idx])
        gate_weights_oof[valid_idx] = gate_model.predict(gate_meta[valid_idx])

    fixed_ensemble = build_fixed_ensemble(
        recent_component=base_predictions["train_recent"],
        alpha_component=base_predictions["train_alpha"],
        physics_blend=base_predictions["train_physics"],
    )
    stacked_gate_ensemble = predict_gated_ensemble(
        gate_weights=gate_weights_oof,
        recent_component=base_predictions["train_recent"],
        alpha_component=base_predictions["train_alpha"],
    )

    predictions_by_name = {
        "physics_blend_prior": base_predictions["train_physics"],
        "recent_residual_component": base_predictions["train_recent"],
        "alpha_gating_component": base_predictions["train_alpha"],
        "fixed_blend_ensemble": fixed_ensemble,
        "stacked_gate_ensemble": stacked_gate_ensemble,
    }
    descriptions = {
        "physics_blend_prior": "70% constant-velocity + 30% constant-acceleration prior.",
        "recent_residual_component": (
            "Recent-window random forest residuals blended 70% with the "
            "constant-velocity prior."
        ),
        "alpha_gating_component": (
            "Sample-wise acceleration gating predicted from compact motion "
            "summary statistics."
        ),
        "fixed_blend_ensemble": (
            "Previous fixed blend: 34% recent residual + 48% alpha gating "
            "+ 18% physics prior."
        ),
        "stacked_gate_ensemble": (
            "Second-stage gate between the recent residual and alpha-gating "
            "base predictors, trained on OOF meta-features."
        ),
    }

    rows = []
    for scenario_name, predictions in predictions_by_name.items():
        fold_metrics = []
        for fold, (_, valid_idx) in enumerate(fold_indices, start=1):
            metrics = score_predictions(train_targets[valid_idx], predictions[valid_idx])
            metrics["fold"] = float(fold)
            fold_metrics.append(metrics)
        rows.append(
            {
                "scenario": scenario_name,
                "description": descriptions[scenario_name],
                "cv_mean_distance": float(np.mean([m["mean_distance"] for m in fold_metrics])),
                "cv_median_distance": float(
                    np.mean([m["median_distance"] for m in fold_metrics])
                ),
                "cv_r_hit_at_1cm": float(np.mean([m["r_hit_at_1cm"] for m in fold_metrics])),
                "cv_r_hit_std": float(np.std([m["r_hit_at_1cm"] for m in fold_metrics])),
            }
        )

    results = pd.DataFrame(rows).sort_values(
        by=["cv_r_hit_at_1cm", "cv_mean_distance"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return results, gate_targets


def build_final_submission_predictions(
    train_targets: np.ndarray,
    train_feature_sets: dict[str, np.ndarray],
    test_feature_sets: dict[str, np.ndarray],
    base_predictions: dict[str, np.ndarray],
) -> np.ndarray:
    """Train the second-stage gate on train OOF meta-features and predict test."""

    gate_meta_train = build_gate_meta_features(
        summary_features=train_feature_sets["alpha_summary"],
        physics_blend=base_predictions["train_physics"],
        recent_component=base_predictions["train_recent"],
        alpha_component=base_predictions["train_alpha"],
    )
    gate_meta_test = build_gate_meta_features(
        summary_features=test_feature_sets["alpha_summary"],
        physics_blend=base_predictions["test_physics"],
        recent_component=base_predictions["test_recent"],
        alpha_component=base_predictions["test_alpha"],
    )
    gate_targets = compute_gate_targets(
        targets=train_targets,
        recent_component=base_predictions["train_recent"],
        alpha_component=base_predictions["train_alpha"],
    )

    gate_model = build_gate_model()
    gate_model.fit(gate_meta_train, gate_targets)
    gate_weights_test = gate_model.predict(gate_meta_test)
    return predict_gated_ensemble(
        gate_weights=gate_weights_test,
        recent_component=base_predictions["test_recent"],
        alpha_component=base_predictions["test_alpha"],
    )


def save_submission(test_ids: list[str], predictions: np.ndarray) -> None:
    """Write the exact competition submission format."""

    submission = pd.DataFrame(predictions, columns=["x", "y", "z"])
    submission.insert(0, "id", test_ids)
    submission.to_csv(SUBMISSION_PATH, index=False)


def main() -> None:
    """Run validation, stacked training, and submission export."""

    train_series, train_targets, test_series, test_ids = load_datasets()
    train_feature_sets = build_feature_sets(train_series)
    test_feature_sets = build_feature_sets(test_series)

    base_predictions, fold_indices = generate_base_predictions(
        train_feature_sets=train_feature_sets,
        train_targets=train_targets,
        test_feature_sets=test_feature_sets,
        n_splits=N_SPLITS,
    )
    scenario_results, _ = evaluate_scenarios(
        train_targets=train_targets,
        train_feature_sets=train_feature_sets,
        base_predictions=base_predictions,
        fold_indices=fold_indices,
    )
    scenario_results.to_csv(SCENARIO_RESULTS_PATH, index=False)

    test_predictions = build_final_submission_predictions(
        train_targets=train_targets,
        train_feature_sets=train_feature_sets,
        test_feature_sets=test_feature_sets,
        base_predictions=base_predictions,
    )
    save_submission(test_ids, test_predictions)

    print("Scenario ranking:")
    print(scenario_results.to_string(index=False))
    print()
    print("Saved scenario metrics to:", SCENARIO_RESULTS_PATH)
    print("Saved submission to:", SUBMISSION_PATH)


if __name__ == "__main__":
    main()
