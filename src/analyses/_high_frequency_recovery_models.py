"""Shared estimators for the high-frequency boundary recovery outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


TREATMENT = "Higher-Repression Southwest Zone"
DISTANCE = "Signed Distance to Historical Repression Boundary km"
COMMUNE = "Linked Climate Commune Code"
EVENT_TIMES = tuple(range(-5, 9))
REFERENCE_TIME = -1


@dataclass
class ModelFit:
    coefficients: np.ndarray
    covariance: np.ndarray
    columns: tuple[str, ...]
    observations: int
    events: int
    villages: int
    groups: int
    shock_family: str
    outcome: str
    specification: str


def _cluster_meat(scores: np.ndarray, identifiers: pd.Series) -> tuple[np.ndarray, int]:
    frame = pd.DataFrame(scores)
    frame["_cluster"] = identifiers.astype("string").to_numpy()
    sums = frame.groupby("_cluster", observed=True, sort=False).sum(numeric_only=True).to_numpy(float)
    return sums.T @ sums, len(sums)


def _two_way_covariance(
    x: np.ndarray,
    residual: np.ndarray,
    cluster_a: pd.Series,
    cluster_b: pd.Series,
) -> np.ndarray:
    n, k = x.shape
    bread = np.linalg.pinv(x.T @ x)
    scores = x * residual[:, None]
    meat_a, groups_a = _cluster_meat(scores, cluster_a)
    meat_b, groups_b = _cluster_meat(scores, cluster_b)
    intersection = cluster_a.astype("string") + "|" + cluster_b.astype("string")
    meat_ab, groups_ab = _cluster_meat(scores, intersection)

    def correction(groups: int) -> float:
        return (groups / max(groups - 1, 1)) * ((n - 1) / max(n - k, 1))

    covariance = bread @ (
        correction(groups_a) * meat_a
        + correction(groups_b) * meat_b
        - correction(groups_ab) * meat_ab
    ) @ bread
    return (covariance + covariance.T) / 2.0


def _select_full_rank(frame: pd.DataFrame, mandatory: list[str]) -> pd.DataFrame:
    matrix = frame.to_numpy(float)
    mandatory_indices = [frame.columns.get_loc(column) for column in mandatory]
    mandatory_matrix = matrix[:, mandatory_indices]
    if np.linalg.matrix_rank(mandatory_matrix) != len(mandatory_indices):
        raise ValueError("Mandatory treatment-by-event coefficients are not identified")
    selected = mandatory_indices.copy()
    current_rank = len(selected)
    for candidate in range(matrix.shape[1]):
        if candidate in selected:
            continue
        trial = selected + [candidate]
        rank = np.linalg.matrix_rank(matrix[:, trial])
        if rank > current_rank:
            selected.append(candidate)
            current_rank = rank
    return frame.iloc[:, selected]


def _eligible_event_villages(data: pd.DataFrame, outcome: str) -> pd.DataFrame:
    support = (
        data.groupby(["Event ID", "Village Code"], observed=True)
        .agg(
            valid=(outcome, lambda value: int(value.notna().sum())),
            reference=("Event Time", lambda value: int(REFERENCE_TIME in set(value[data.loc[value.index, outcome].notna()]))),
            pre=("Event Time", lambda value: int(np.sum(value[data.loc[value.index, outcome].notna()].between(-5, -1)))),
            post=("Event Time", lambda value: int(np.sum(value[data.loc[value.index, outcome].notna()].between(0, 8)))),
        )
    )
    valid_pairs = support.loc[
        support["valid"].ge(8)
        & support["reference"].eq(1)
        & support["pre"].ge(2)
        & support["post"].ge(3)
    ].reset_index()
    return data.merge(valid_pairs[["Event ID", "Village Code"]], on=["Event ID", "Village Code"], how="inner", validate="many_to_one")


def _absorb_fixed_effects(values: pd.DataFrame, identifiers: list[pd.Series]) -> pd.DataFrame:
    residual = values.astype(float).copy()
    for _ in range(500):
        largest_adjustment = 0.0
        for identifier in identifiers:
            adjustment = residual.groupby(identifier, observed=True).transform("mean")
            largest_adjustment = max(largest_adjustment, float(np.nanmax(np.abs(adjustment.to_numpy(float)))))
            residual -= adjustment
        if largest_adjustment < 1e-10:
            return residual
    raise RuntimeError("Fixed-effect absorption did not converge")


def fit_dynamic_path(
    stack: pd.DataFrame,
    shock_family: str,
    outcome: str,
    specification: str,
) -> ModelFit:
    data = stack.loc[stack["Shock Family"].eq(shock_family)].copy()
    data = _eligible_event_villages(data, outcome)
    if specification == "Primary":
        data["Analysis Group"] = data["Event ID"].astype("string")
    elif specification == "Within commune":
        data[COMMUNE] = data[COMMUNE].astype("string")
        data["Analysis Group"] = data["Event ID"].astype("string") + "|" + data[COMMUNE]
        cross_side = data.groupby("Analysis Group", observed=True)[TREATMENT].nunique()
        data = data.loc[data["Analysis Group"].isin(cross_side.loc[cross_side.eq(2)].index)].copy()
        data = _eligible_event_villages(data, outcome)
    else:
        raise ValueError(f"Unknown specification: {specification}")
    data = data.dropna(subset=[outcome]).copy()
    data["Event Village"] = data["Event ID"].astype("string") + "|" + data["Village Code"].astype("string")
    data["Group Time"] = data["Analysis Group"] + "|" + data["Event Time"].astype(str)

    regressors: dict[str, np.ndarray] = {}
    for event_time in EVENT_TIMES:
        if event_time == REFERENCE_TIME:
            continue
        indicator = data["Event Time"].eq(event_time).to_numpy(float)
        treatment = data[TREATMENT].to_numpy(float)
        distance = data[DISTANCE].to_numpy(float)
        regressors[f"Southwest x k={event_time}"] = treatment * indicator
        regressors[f"Distance x k={event_time}"] = distance * indicator
        regressors[f"Southwest x distance x k={event_time}"] = treatment * distance * indicator
    x_raw = pd.DataFrame(regressors, index=data.index)
    y_raw = data[outcome].astype(float)

    absorbed = _absorb_fixed_effects(
        pd.concat([y_raw.rename("_outcome"), x_raw], axis=1),
        [data["Event Village"], data["Group Time"]],
    )
    y = absorbed.pop("_outcome").to_numpy(float)
    x = absorbed
    mandatory = [f"Southwest x k={event_time}" for event_time in EVENT_TIMES if event_time != REFERENCE_TIME]
    x = _select_full_rank(x, mandatory)
    matrix = x.to_numpy(float)
    coefficients = np.linalg.lstsq(matrix, y, rcond=None)[0]
    residual = y - matrix @ coefficients
    covariance = _two_way_covariance(
        matrix,
        residual,
        data["Event ID"],
        data["Village Code"],
    )
    return ModelFit(
        coefficients=coefficients,
        covariance=covariance,
        columns=tuple(x.columns.astype(str)),
        observations=len(data),
        events=int(data["Event ID"].nunique()),
        villages=int(data["Village Code"].nunique()),
        groups=int(data["Analysis Group"].nunique()),
        shock_family=shock_family,
        outcome=outcome,
        specification=specification,
    )


def fit_scalar_outcome(
    event_data: pd.DataFrame,
    shock_family: str,
    outcome: str,
    specification: str,
) -> ModelFit:
    data = event_data.loc[event_data["Shock Family"].eq(shock_family)].dropna(
        subset=[outcome, TREATMENT, DISTANCE, "Event ID", "Village Code", COMMUNE]
    ).copy()
    if specification == "Primary":
        data["Analysis Group"] = data["Event ID"].astype("string")
    elif specification == "Within commune":
        data[COMMUNE] = data[COMMUNE].astype("string")
        data["Analysis Group"] = data["Event ID"].astype("string") + "|" + data[COMMUNE]
        cross_side = data.groupby("Analysis Group", observed=True)[TREATMENT].nunique()
        data = data.loc[data["Analysis Group"].isin(cross_side.loc[cross_side.eq(2)].index)].copy()
    else:
        raise ValueError(f"Unknown specification: {specification}")
    frame = pd.DataFrame({
        "Southwest": data[TREATMENT].astype(float),
        "Distance": data[DISTANCE].astype(float),
        "Southwest x distance": data[TREATMENT].astype(float) * data[DISTANCE].astype(float),
    }, index=data.index)
    y_raw = data[outcome].astype(float)
    y = (y_raw - y_raw.groupby(data["Analysis Group"], observed=True).transform("mean")).to_numpy(float)
    x = frame - frame.groupby(data["Analysis Group"], observed=True).transform("mean")
    x = _select_full_rank(x, ["Southwest"])
    matrix = x.to_numpy(float)
    coefficients = np.linalg.lstsq(matrix, y, rcond=None)[0]
    residual = y - matrix @ coefficients
    covariance = _two_way_covariance(matrix, residual, data["Event ID"], data["Village Code"])
    return ModelFit(
        coefficients=coefficients,
        covariance=covariance,
        columns=tuple(x.columns.astype(str)),
        observations=len(data),
        events=int(data["Event ID"].nunique()),
        villages=int(data["Village Code"].nunique()),
        groups=int(data["Analysis Group"].nunique()),
        shock_family=shock_family,
        outcome=outcome,
        specification=specification,
    )


def fit_distributed_lag(
    panel: pd.DataFrame,
    outcome: str,
    specification: str,
) -> ModelFit:
    data = panel.loc[panel["Cross-Side CHIRPS Cell"].eq(1)].copy()
    data = data.sort_values(["CHIRPS Cell ID", "Composite Date"]).reset_index(drop=True)
    climate = data[["CHIRPS Cell ID", "Composite Date", "Interval-Aligned Rainfall Anomaly Z"]].drop_duplicates(
        ["CHIRPS Cell ID", "Composite Date"]
    ).sort_values(["CHIRPS Cell ID", "Composite Date"]).reset_index(drop=True)
    rainfall = climate["Interval-Aligned Rainfall Anomaly Z"].astype(float)
    climate["Dry intensity"] = np.maximum(-rainfall, 0.0)
    climate["Wet intensity"] = np.maximum(rainfall, 0.0)
    for shock in ("Dry", "Wet"):
        for lag in range(-3, 9):
            climate[f"{shock} intensity k={lag}"] = (
                climate.groupby("CHIRPS Cell ID", observed=True)[f"{shock} intensity"].shift(lag)
            )
    data = data.merge(
        climate.drop(columns=["Interval-Aligned Rainfall Anomaly Z"]),
        on=["CHIRPS Cell ID", "Composite Date"],
        how="left",
        validate="many_to_one",
    )
    if specification == "Primary":
        data["Analysis Group"] = data["CHIRPS Cell ID"].astype("string")
    elif specification == "Within commune":
        data[COMMUNE] = data[COMMUNE].astype("string")
        data["Analysis Group"] = data["CHIRPS Cell ID"].astype("string") + "|" + data[COMMUNE]
        cross_side = data.groupby("Analysis Group", observed=True)[TREATMENT].nunique()
        data = data.loc[data["Analysis Group"].isin(cross_side.loc[cross_side.eq(2)].index)].copy()
    else:
        raise ValueError(f"Unknown specification: {specification}")
    data["Group Date"] = data["Analysis Group"] + "|" + data["Composite Date"].astype("string")
    lag_columns = [f"{shock} intensity k={lag}" for shock in ("Dry", "Wet") for lag in range(-3, 9)]
    data = data.dropna(subset=[outcome, TREATMENT, DISTANCE, "Village Code", "Composite Date", *lag_columns]).copy()

    regressors: dict[str, np.ndarray] = {}
    treatment = data[TREATMENT].to_numpy(float)
    distance = data[DISTANCE].to_numpy(float)
    for shock in ("Dry", "Wet"):
        for lag in range(-3, 9):
            intensity = data[f"{shock} intensity k={lag}"].to_numpy(float)
            regressors[f"Southwest x {shock} k={lag}"] = treatment * intensity
            regressors[f"Distance x {shock} k={lag}"] = distance * intensity
            regressors[f"Southwest x distance x {shock} k={lag}"] = treatment * distance * intensity
    x_raw = pd.DataFrame(regressors, index=data.index)
    absorbed = _absorb_fixed_effects(
        pd.concat([data[outcome].astype(float).rename("_outcome"), x_raw], axis=1),
        [data["Village Code"].astype("string"), data["Group Date"]],
    )
    y = absorbed.pop("_outcome").to_numpy(float)
    mandatory = [f"Southwest x {shock} k={lag}" for shock in ("Dry", "Wet") for lag in range(-3, 9)]
    x = _select_full_rank(absorbed, mandatory)
    matrix = x.to_numpy(float)
    coefficients = np.linalg.lstsq(matrix, y, rcond=None)[0]
    residual = y - matrix @ coefficients
    covariance = _two_way_covariance(
        matrix,
        residual,
        data["Village Code"],
        data["Composite Date"].astype("string"),
    )
    return ModelFit(
        coefficients=coefficients,
        covariance=covariance,
        columns=tuple(x.columns.astype(str)),
        observations=len(data),
        events=int(data["Composite Date"].nunique()),
        villages=int(data["Village Code"].nunique()),
        groups=int(data["Analysis Group"].nunique()),
        shock_family="Joint dry/wet distributed lag",
        outcome=outcome,
        specification=specification,
    )


def distributed_lag_rows(fit: ModelFit) -> pd.DataFrame:
    rows = []
    for shock in ("Dry", "Wet"):
        for lag in range(-3, 9):
            result = linear_contrast(fit, {f"Southwest x {shock} k={lag}": 1.0})
            rows.append({
                "shock_family": shock,
                "outcome": fit.outcome,
                "specification": fit.specification,
                "event_time": lag,
                **result,
                "observations": fit.observations,
                "dates": fit.events,
                "villages": fit.villages,
                "groups": fit.groups,
            })
    return pd.DataFrame(rows)


def distributed_pretrend_test(fit: ModelFit, shock: str) -> tuple[float, float, int]:
    names = [f"Southwest x {shock} k={lag}" for lag in (-3, -2, -1)]
    indices = [fit.columns.index(name) for name in names]
    beta = fit.coefficients[indices]
    covariance = fit.covariance[np.ix_(indices, indices)]
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
    covariance_psd = eigenvectors @ np.diag(np.maximum(eigenvalues, 1e-12)) @ eigenvectors.T
    statistic = float(beta @ np.linalg.pinv(covariance_psd) @ beta)
    degrees = int(np.linalg.matrix_rank(covariance_psd))
    return statistic, float(chi2.sf(statistic, degrees)), degrees


def linear_contrast(fit: ModelFit, weights: dict[str, float]) -> dict[str, float]:
    vector = np.zeros(len(fit.columns), dtype=float)
    for column, weight in weights.items():
        if column in fit.columns:
            vector[fit.columns.index(column)] = weight
        elif weight != 0:
            raise ValueError(f"Required coefficient missing: {column}")
    estimate = float(vector @ fit.coefficients)
    variance = float(vector @ fit.covariance @ vector)
    standard_error = float(np.sqrt(max(variance, 0.0)))
    ci_low = estimate - 1.96 * standard_error
    ci_high = estimate + 1.96 * standard_error
    p_value = float(2.0 * norm.sf(abs(estimate / standard_error))) if standard_error > 0 else np.nan
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "compatibility_frontier": max(abs(ci_low), abs(ci_high)),
    }


def event_time_rows(fit: ModelFit) -> pd.DataFrame:
    rows = []
    for event_time in EVENT_TIMES:
        if event_time == REFERENCE_TIME:
            estimate = standard_error = ci_low = ci_high = 0.0
        else:
            result = linear_contrast(fit, {f"Southwest x k={event_time}": 1.0})
            estimate = result["estimate"]
            standard_error = result["standard_error"]
            ci_low = result["ci_low"]
            ci_high = result["ci_high"]
        rows.append({
            "shock_family": fit.shock_family,
            "outcome": fit.outcome,
            "specification": fit.specification,
            "event_time": event_time,
            "estimate": estimate,
            "standard_error": standard_error,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "observations": fit.observations,
            "events": fit.events,
            "villages": fit.villages,
            "groups": fit.groups,
        })
    return pd.DataFrame(rows)


def pretrend_test(fit: ModelFit) -> tuple[float, float, int]:
    names = [f"Southwest x k={event_time}" for event_time in (-5, -4, -3, -2)]
    indices = [fit.columns.index(name) for name in names]
    beta = fit.coefficients[indices]
    covariance = fit.covariance[np.ix_(indices, indices)]
    statistic = float(beta @ np.linalg.pinv(covariance) @ beta)
    degrees = int(np.linalg.matrix_rank(covariance))
    p_value = float(chi2.sf(statistic, degrees)) if degrees > 0 else np.nan
    return statistic, p_value, degrees


def scalar_result_row(fit: ModelFit, estimand: str, scale: str) -> dict[str, object]:
    result = linear_contrast(fit, {"Southwest": 1.0})
    return {
        "shock_family": fit.shock_family,
        "outcome": fit.outcome,
        "estimand": estimand,
        "specification": fit.specification,
        "scale": scale,
        **result,
        "observations": fit.observations,
        "events": fit.events,
        "villages": fit.villages,
        "groups": fit.groups,
        "inference": "Two-way clustered by rainfall-cell event and village",
    }
