"""
Vegetation succession and land-cover evolution model for middle Yangtze River floodplains.

This module implements three submodules described in the specification:
1. Initial succession driven by lagged mean inundation duration.
2. Community transitions driven by current-year inundation intensity.
3. Human land-cover expansion with neighborhood and suitability constraints.

It also defines a lightweight calibration/validation helper to repeat simulations
and evaluate area errors and pixel-level accuracy.

All calculations are vectorised over numpy arrays. Land-cover classes are
encoded as integers using the :class:`LandCover` constants. Arrays must share
identical shapes and represent the same pixel grid.
"""

import os
from dataclasses import dataclass
import math
from typing import Dict, Iterable, Optional, Tuple, Sequence
from osgeo import gdal
from tqdm import tqdm
import basic_function as bf
import numpy as np
import rasterio


class LandCover:
    """Integer codes for land-cover and vegetation classes."""

    SEASONAL_WATER = 0  # R
    HYDROPHYTE = 1  # H
    EMERGENT = 2  # E
    WOODLAND = 3  # F
    AGRICULTURE = 4  # A
    URBAN = 5  # U
    BARE = 6  # exposed bar or other natural non-vegetated cover

    HUMAN_CLASSES = {AGRICULTURE, URBAN, WOODLAND}
    VEGETATION_CLASSES = {HYDROPHYTE, EMERGENT, WOODLAND}


LAND_COVER_LABELS = {
    LandCover.SEASONAL_WATER: "Seasonal water (R)",
    LandCover.HYDROPHYTE: "Hydrophyte (H)",
    LandCover.EMERGENT: "Emergent (E)",
    LandCover.WOODLAND: "Woodland (F)",
    LandCover.AGRICULTURE: "Agriculture (A)",
    LandCover.URBAN: "Urban (U)",  # natural non-vegetated
}

# Evaluation metrics should exclude seasonal water per user instruction.
EVAL_CLASSES = (
    LandCover.HYDROPHYTE,
    LandCover.EMERGENT,
    LandCover.WOODLAND,
    LandCover.AGRICULTURE,
    LandCover.URBAN,
)


@dataclass
class InitialSuccessionParams:
    """Parameters for the lagged inundation duration driven succession."""

    means: Tuple[float, float, float] = (152.7, 115.5, 72.8)  # R-R, R-H, R-E
    stds: Tuple[float, float, float] = (58.3, 49.8, 41.1)
    weights: Tuple[float, float, float] = (9.7, 1.5, 2)  # w_RR, w_RH, w_RE


@dataclass
class CommunitySuccessionParams:
    """Parameters for inundation-intensity driven community transitions."""

    log_means: Tuple[float, float, float, float] = (5.16, 4.85, 4.13, 3.26)  # R, H, E, F
    log_stds: Tuple[float, float, float, float] = (0.85, 1.50, 1.66, 1.43)
    weights: Tuple[float, float, float, float] = (47.35, 79.3, 120.0, 1.0)  # R, H, E, F


@dataclass
class HumanExpansionParams:
    """Parameters controlling human land-cover expansion."""

    annual_rates: Dict[str, float]
    rate_correction: float = 1.0
    inundation_threshold: float = 26.0
    pixel_area_km2: float = 0.0009  # example for 30 m grid
    adjacency_weight: float = 0.7
    suitability_weight: float = 0.3
    base_suitability: Dict[int, float] = None

    def __post_init__(self) -> None:
        if self.base_suitability is None:
            self.base_suitability = {
                LandCover.BARE: 1.0,
                LandCover.HYDROPHYTE: 1.0,
                LandCover.EMERGENT: 0.5,
                LandCover.SEASONAL_WATER: 0.2,
            }
        # ensure weights sum to one when used
        total = self.adjacency_weight + self.suitability_weight
        self.adjacency_weight /= total
        self.suitability_weight /= total


def _normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    coeff = 1.0 / (std * math.sqrt(2 * math.pi))
    z = (x - mean) / std
    return coeff * np.exp(-0.5 * z * z)


def _lognormal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    safe_x = np.clip(x, a_min=1e-6, a_max=None)
    coeff = 1.0 / (safe_x * sigma * math.sqrt(2 * math.pi))
    z = (np.log(safe_x) - mu) / sigma
    return coeff * np.exp(-0.5 * z * z)


def _eight_neighbor_count(mask: np.ndarray, target_value: int) -> np.ndarray:
    padded = np.pad(mask == target_value, 1, mode="edge")
    counts = np.zeros_like(mask, dtype=int)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                    continue
            counts += padded[1 + dx : 1 + dx + mask.shape[0], 1 + dy : 1 + dy + mask.shape[1]]
    return counts


def simulate_initial_succession(
    land_cover: np.ndarray,
    id_t: np.ndarray,
    id_t_minus1: np.ndarray,
    id_t_minus2: np.ndarray,
    params: InitialSuccessionParams,
) -> np.ndarray:
    """Simulate the initial succession for pixels currently in seasonal water.

    The probability of following each path (R-R, R-H, R-E) depends on the
    three-year mean inundation duration. Instead of stochastic sampling, the
    model deterministically selects the path with the highest likelihood.
    """

    output = land_cover.copy()
    seasonal_mask = land_cover == LandCover.SEASONAL_WATER
    if not np.any(seasonal_mask):
        return output

    id_mean = (id_t + id_t_minus1 + id_t_minus2) / 3.0

    likelihoods = []
    for mean, std, weight in zip(params.means, params.stds, params.weights):
        likelihoods.append(_normal_pdf(id_mean, mean, std) * weight)
    likelihoods = np.stack(likelihoods, axis=-1)

    target_classes = [
        LandCover.SEASONAL_WATER,
        LandCover.HYDROPHYTE,
        LandCover.EMERGENT,
    ]

    target_idx = np.argmax(likelihoods, axis=-1)
    mapping = np.array(target_classes, dtype=int)
    output[seasonal_mask] = mapping[target_idx][seasonal_mask]

    # # enforce 8-neighbour connectivity for new vegetated pixels
    # for target in (LandCover.HYDROPHYTE, LandCover.EMERGENT):
    #     new_mask = (land_cover == LandCover.SEASONAL_WATER) & (output == target)
    #     neighbor_counts = _eight_neighbor_count(output, target)
    #     isolated = new_mask & (neighbor_counts == 0)
    #     output[isolated] = LandCover.SEASONAL_WATER

    return output


def simulate_community_succession(
    land_cover: np.ndarray,
    inundation_intensity: np.ndarray,
    params: CommunitySuccessionParams,
) -> np.ndarray:
    """Simulate transitions among vegetation communities driven by current flood intensity.

    The target class is picked deterministically by maximising the habitat
    suitability (log-normal likelihood multiplied by weight).
    """

    output = land_cover.copy()
    veg_mask = np.isin(land_cover, [LandCover.HYDROPHYTE, LandCover.EMERGENT])
    inun_mask = ~np.isnan(inundation_intensity)
    veg_mask = np.logical_and(veg_mask, inun_mask)
    if not np.any(veg_mask):
        return output

    habitat = []
    for mu, sigma, weight in zip(params.log_means, params.log_stds, params.weights):
        habitat.append(_lognormal_pdf(inundation_intensity, mu, sigma) * weight)
    habitat = np.stack(habitat, axis=-1)

    target_classes = [
        LandCover.SEASONAL_WATER,
        LandCover.HYDROPHYTE,
        LandCover.EMERGENT,
        LandCover.WOODLAND,
    ]

    # ds = gdal.Open('G:\A_Landsat_Floodplain_veg\Water_level_python\Post_TGD\\ele_DT_inundation_frequency_posttgd.TIF')
    # for layer in range(4):
    #     bf.write_raster(ds, habitat[:, :, layer],'G:\A_GEDI_Floodplain_vegh\Veg_map\Succession\Calibration\\', f'layer_{layer}.tif', )
    target_idx = np.argmax(habitat, axis=-1)
    mapping = np.array(target_classes, dtype=int)
    output[veg_mask] = mapping[target_idx][veg_mask]
    print(str(np.nansum(output == 0) * 0.03 * 0.03))
    return output


def _adjacency_fraction(mask: np.ndarray, target_value: int, window: int = 5) -> np.ndarray:
    pad = window // 2
    padded = np.pad(mask == target_value, pad, mode="edge")
    counts = np.zeros_like(mask, dtype=float)
    window_area = window * window
    for dx in range(-pad, pad + 1):
        for dy in range(-pad, pad + 1):
            counts += padded[pad + dx : pad + dx + mask.shape[0], pad + dy : pad + dy + mask.shape[1]]
    return counts / window_area


def apply_human_expansion(
    land_cover: np.ndarray,
    inundation_intensity: np.ndarray,
    params: HumanExpansionParams,
) -> np.ndarray:
    """Expand human land-cover classes deterministically using suitability ranking."""

    output = land_cover.copy()
    non_human_mask = ~np.isin(output, list(LandCover.HUMAN_CLASSES))
    if not np.any(non_human_mask):
        return output

    for label, rate in params.annual_rates.items():
        target_class = {
            "A": LandCover.AGRICULTURE,
            "F": LandCover.WOODLAND,
            "U": LandCover.URBAN,
        }.get(label)
        if target_class is None:
            continue

        target_area_km2 = rate * params.rate_correction
        pixel_count = max(int(target_area_km2 / params.pixel_area_km2), 0)
        if pixel_count == 0:
            continue

        candidate_mask = (
            non_human_mask
            & (inundation_intensity < params.inundation_threshold)
            & (_eight_neighbor_count(output, target_class) > 0)
        )
        if not np.any(candidate_mask):
            continue

        adjacency_score = _adjacency_fraction(output, target_class)
        suitability_base = np.zeros_like(output, dtype=float)
        for lc_value, base_score in params.base_suitability.items():
            suitability_base[output == lc_value] = base_score

        suitability = (
            params.adjacency_weight * adjacency_score
            + params.suitability_weight * suitability_base
        )
        candidate_scores = suitability[candidate_mask]
        candidate_indices = np.flatnonzero(candidate_mask)
        if candidate_indices.size == 0:
            continue

        chosen_count = min(pixel_count, candidate_indices.size)
        order = np.argsort(candidate_scores)[::-1]
        chosen = candidate_indices[order[:chosen_count]]
        flat_output = output.ravel()
        flat_output[chosen] = target_class
        output = flat_output.reshape(output.shape)
        non_human_mask = ~np.isin(output, list(LandCover.HUMAN_CLASSES))

    return output


@dataclass
class CalibrationConfig:
    initial_params: InitialSuccessionParams
    community_params: CommunitySuccessionParams
    area_weight: float = 0.7
    accuracy_weight: float = 0.3
    learning_rate: float = 0.5
    max_steps: int = 10000
    clip_min: float = 1e-6


def _overall_accuracy(truth: np.ndarray, prediction: np.ndarray, classes: Optional[Iterable[int]] = None) -> float:
    valid = (~np.isnan(truth)) & (~np.isnan(prediction))
    if classes is not None:
        mask = valid & np.isin(truth, list(classes))
        total = np.sum(mask)
        if total == 0:
            return 0.0
        correct = np.sum((truth == prediction) & mask)
        return correct / total
    correct = np.sum((truth == prediction) & valid)
    total = np.sum(valid)
    return 0.0 if total == 0 else correct / total


def _area_error(truth: np.ndarray, prediction: np.ndarray, classes: Iterable[int]) -> float:
    errors = []
    valid = (~np.isnan(truth)) & (~np.isnan(prediction))
    total_pixels = np.sum(valid)
    if total_pixels == 0:
        return 0.0
    for cls in classes:
        truth_area = np.sum(valid & (truth == cls)) / total_pixels
        pred_area = np.sum(valid & (prediction == cls)) / total_pixels
        errors.append(abs(truth_area - pred_area))
    return float(np.mean(errors))


def _evaluate_objective(
    area_weight: float,
    accuracy_weight: float,
    reference_land_cover: np.ndarray,
    aggregated_prediction: np.ndarray,
    classes: Iterable[int] = EVAL_CLASSES,
) -> Dict[str, float]:
    area_err = _area_error(reference_land_cover, aggregated_prediction, classes=classes)
    oa = _overall_accuracy(reference_land_cover, aggregated_prediction, classes=classes)
    objective = area_weight * area_err + accuracy_weight * (1 - oa)
    return {"area_error": area_err, "overall_accuracy": oa, "objective": objective}


def _per_class_f1(truth: np.ndarray, prediction: np.ndarray, classes: Iterable[int]) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    valid = (~np.isnan(truth)) & (~np.isnan(prediction))
    for cls in classes:
        mask = valid & ((truth == cls) | (prediction == cls))
        if not np.any(mask):
            scores[cls] = 0.0
            continue
        tp = np.sum((truth == cls) & (prediction == cls) & valid)
        fp = np.sum((truth != cls) & (prediction == cls) & valid)
        fn = np.sum((truth == cls) & (prediction != cls) & valid)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            scores[cls] = 0.0
        else:
            scores[cls] = 2 * precision * recall / (precision + recall)
    return scores


def _confusion_matrix(truth: np.ndarray, prediction: np.ndarray, classes: Iterable[int]) -> np.ndarray:
    classes = list(classes)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    valid = (~np.isnan(truth)) & (~np.isnan(prediction))
    for i, truth_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            matrix[i, j] = int(np.sum(valid & (truth == truth_cls) & (prediction == pred_cls)))
    return matrix


def _area_comparison(truth: np.ndarray, prediction: np.ndarray, classes: Iterable[int]) -> Dict[int, Tuple[float, float]]:
    comparison: Dict[int, Tuple[float, float]] = {}
    valid = (~np.isnan(truth)) & (~np.isnan(prediction))
    total_pixels = np.sum(valid)
    if total_pixels == 0:
        return {cls: (0.0, 0.0) for cls in classes}
    for cls in classes:
        truth_area = np.sum(valid & (truth == cls)) / total_pixels
        pred_area = np.sum(valid & (prediction == cls)) / total_pixels
        comparison[cls] = (truth_area, pred_area)
    return comparison


def run_single_year(
    land_cover: np.ndarray,
    id_t: np.ndarray,
    id_t_minus1: np.ndarray,
    id_t_minus2: np.ndarray,
    inundation_intensity: np.ndarray,
    human_params: HumanExpansionParams,
    init_params: InitialSuccessionParams,
    community_params: CommunitySuccessionParams,
    output_dir: Optional[str] = None,
    profile: Optional[dict] = None,
    label: Optional[str] = None,
) -> np.ndarray:
    """Run the three deterministic submodules for one year.

    The typical usage pattern is::

        next_lc = run_single_year(
            land_cover=current_lc,
            id_t=id_series[2][year_index],
            id_t_minus1=id_series[1][year_index],
            id_t_minus2=id_series[0][year_index],
            inundation_intensity=intensity_series[year_index],
            human_params=human_params_by_year[year_index],
            init_params=init_params,
            community_params=community_params,
        )

    """
    lc = simulate_initial_succession(land_cover, id_t, id_t_minus1, id_t_minus2, init_params)
    lc = simulate_community_succession(lc, inundation_intensity, community_params)
    lc = apply_human_expansion(lc, inundation_intensity, human_params)

    if profile is not None:
        target_dir = output_dir or r"G:\\A_GEDI_Floodplain_vegh\\Veg_map\\Succession\\Calibration"
        _ensure_dir(target_dir)
        fname = label or "run_single_year.tif"
        tif_path = os.path.join(target_dir, fname)
        _save_raster(tif_path, lc.astype(np.int16), profile)

    print("Single-year simulation complete.")
    return lc


def simulate_period(
    initial_land_cover: np.ndarray,
    id_series: Tuple[np.ndarray, np.ndarray, np.ndarray],
    intensity_series: np.ndarray,
    human_params_by_year: Tuple[HumanExpansionParams, ...],
    init_params: InitialSuccessionParams,
    community_params: CommunitySuccessionParams,
    output_dir: Optional[str] = None,
    reference_series: Optional[Sequence[np.ndarray]] = None,
    profile: Optional[dict] = None,
    years: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Run the full succession model for a period of years."""
    land_cover = initial_land_cover.copy()

    if output_dir:
        _ensure_dir(output_dir)
        if profile is None:
            raise ValueError("profile is required when output_dir is provided")

    for idx, year in enumerate(tqdm(range(len(intensity_series)), desc="Simulating years")):
        id_t_minus2, id_t_minus1, id_t = (arr[year] for arr in id_series)
        land_cover = run_single_year(
            land_cover,
            id_t=id_t,
            id_t_minus1=id_t_minus1,
            id_t_minus2=id_t_minus2,
            inundation_intensity=intensity_series[idx],
            human_params=human_params_by_year[idx],
            init_params=init_params,
            community_params=community_params,
            profile=profile
        )
        if output_dir:
            year_label = years[idx] if years is not None else idx
            tif_path = os.path.join(output_dir, f"predict_{year_label}.tif")
            _save_raster(tif_path, land_cover.astype(np.int16), profile)

            if reference_series is not None:
                reference = reference_series[idx]
                area_comp = _area_comparison(reference, land_cover, classes=EVAL_CLASSES)
                f1_scores = _per_class_f1(reference, land_cover, classes=EVAL_CLASSES)
                confusion = _confusion_matrix(reference, land_cover, classes=EVAL_CLASSES)

                metrics_path = os.path.join(output_dir, f"metrics_{year_label}.csv")
                with open(metrics_path, "w", encoding="utf-8") as fh:
                    fh.write("class,truth_area,pred_area,f1\n")
                    for cls in EVAL_CLASSES:
                        truth_area, pred_area = area_comp[cls]
                        fh.write(
                            f"{cls},{truth_area:.6f},{pred_area:.6f},{f1_scores.get(cls, 0.0):.6f}\n"
                        )

                conf_path = os.path.join(output_dir, f"confusion_{year_label}.csv")
                np.savetxt(conf_path, confusion, fmt="%d", delimiter=",")
    return land_cover


def calibrate(
    config: CalibrationConfig,
    initial_land_cover: np.ndarray,
    id_series: Tuple[np.ndarray, np.ndarray, np.ndarray],
    intensity_series: np.ndarray,
    human_params_by_year: Tuple[HumanExpansionParams, ...],
    reference_land_cover: np.ndarray,
    output_dir: Optional[str] = None,
    output_profile: Optional[dict] = None,
    output_year: Optional[int] = None,
) -> Dict[str, float]:
    """Calibrate parameters via deterministic simulations.

    The calibration loop relies on the supplied parameters and deterministically
    adjusts community habitat suitability weights during validation. For each
    iteration, if a habitat type is over-represented relative to the reference
    map, its weight is reduced proportionally. No stochastic sampling or
    gradient descent is used.
    """

    init_params = config.initial_params
    community_params = config.community_params
    human_params = human_params_by_year

    def _run_and_score(cur_init: InitialSuccessionParams, cur_comm: CommunitySuccessionParams, output, prof, years,):
        aggregated = simulate_period(
            initial_land_cover=initial_land_cover,
            id_series=id_series,
            intensity_series=intensity_series,
            human_params_by_year=human_params,
            init_params=cur_init,
            community_params=cur_comm,
            output_dir = output,
            profile = prof,
            years=years
        )
        metrics = _evaluate_objective(
            area_weight=config.area_weight,
            accuracy_weight=config.accuracy_weight,
            reference_land_cover=reference_land_cover,
            aggregated_prediction=aggregated,
        )
        area_comp = _area_comparison(reference_land_cover, aggregated, classes=EVAL_CLASSES)
        f1_scores = _per_class_f1(reference_land_cover, aggregated, classes=EVAL_CLASSES)
        return aggregated, metrics, area_comp, f1_scores

    def _maybe_save_iteration(step: int, raster: np.ndarray) -> None:
        if output_dir is None or output_profile is None:
            return
        _ensure_dir(output_dir)
        label = output_year if output_year is not None else "calibration"
        tif_path = os.path.join(output_dir, f"calibration_step_{step:05d}_{label}.tif")
        _save_raster(tif_path, raster.astype(np.int16), output_profile)

    def _print_iteration(step: int, metrics: Dict[str, float], area_comp: Dict[int, Tuple[float, float]], f1_scores: Dict[int, float]):
        header = (
            "Calibration iteration {step}: objective={objective:.6f}, area_error={area_error:.6f}, "
            "overall_accuracy={overall_accuracy:.6f}, weights={weights}"
        ).format(
            step=step,
            objective=metrics["objective"],
            area_error=metrics["area_error"],
            overall_accuracy=metrics["overall_accuracy"],
            weights=community_params.weights,
        )
        print(header)
        for cls in EVAL_CLASSES:
            truth_area, pred_area = area_comp.get(cls, (0.0, 0.0))
            print(
                "  class {cls} ({label}): truth_area={ta:.6f}, pred_area={pa:.6f}, f1={f1:.6f}".format(
                    cls=cls,
                    label=LAND_COVER_LABELS.get(cls, ""),
                    ta=truth_area,
                    pa=pred_area,
                    f1=f1_scores.get(cls, 0.0),
                )
            )

    aggregated, metrics, area_comp, f1_scores = _run_and_score(init_params, community_params, output_dir, output_profile, output_year)
    _print_iteration(0, metrics, area_comp, f1_scores)
    _maybe_save_iteration(0, aggregated)

    return metrics


def _parse_rate(value: Optional[str]) -> float:
    return float(value) if value is not None else 0.0


def _build_human_params(
    years: int,
    rate_a: float,
    rate_f: float,
    rate_u: float,
    pixel_area: float,
    inundation_threshold: float,
    rate_correction: float,
) -> Tuple[HumanExpansionParams, ...]:
    return tuple(
        HumanExpansionParams(
            annual_rates={"A": rate_a, "F": rate_f, "U": rate_u},
            pixel_area_km2=pixel_area,
            inundation_threshold=inundation_threshold,
            rate_correction=rate_correction,
        )
        for _ in range(years)
    )


def _print_comparison(area_comp: Dict[int, Tuple[float, float]], f1_scores: Dict[int, float]) -> None:
    print("Class-wise area proportion (truth vs prediction) and F1-score:")
    for cls, (truth_area, pred_area) in area_comp.items():
        label = LAND_COVER_LABELS.get(cls, f"Class {cls}")
        f1 = f1_scores.get(cls, 0.0)
        print(
            f"- {label}: truth={truth_area:.4f}, prediction={pred_area:.4f}, F1={f1:.4f}"
        )


def _load_array(path: str) -> np.ndarray:
    arr = np.load(path)
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Expected ndarray at {path}")
    return arr


def _load_raster_with_profile(path: str) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        return src.read(1), src.profile


def _load_raster(path: str) -> np.ndarray:
    arr, _ = _load_raster_with_profile(path)
    return arr


def _reclass_land_cover(a: np.ndarray) -> np.ndarray:
    """Reclass raw land-cover codes and handle nodata values.

    The raw rasters encode nodata as ``-32768``; these are mapped to ``np.nan``
    so downstream evaluations ignore empty pixels. Original codes are
    normalised following the user-provided rules:

    * ``0`` and ``5`` → ``0``
    * ``6`` → ``5``
    """

    a = a.astype("float32", copy=True)
    a[a == -32768] = np.nan
    valid_mask = ~np.isnan(a)
    a[np.logical_and(valid_mask, np.isin(a, [0, 5]))] = 0
    a[np.logical_and(valid_mask, a == 6)] = 5
    return a

def _reclass_land_cover2(a: np.ndarray) -> np.ndarray:
    """Reclass raw land-cover codes and handle nodata values.

    The raw rasters encode nodata as ``-32768``; these are mapped to ``np.nan``
    so downstream evaluations ignore empty pixels. Original codes are
    normalised following the user-provided rules:

    * ``0`` and ``5`` → ``0``
    * ``6`` → ``5``
    """

    a = a.astype("float32", copy=True)
    a[a == -32768] = np.nan
    valid_mask = ~np.isnan(a)
    return a


def _load_land_cover(year: int, base_dir: str) -> np.ndarray:
    path = f"{base_dir}/predict_{year}.tif"
    return _reclass_land_cover(_load_raster(path))


def _load_land_cover_with_profile(year: int, base_dir: str) -> Tuple[np.ndarray, dict]:
    path = f"{base_dir}/predict_{year}.tif"
    arr, profile = _load_raster_with_profile(path)
    return _reclass_land_cover(arr), profile

def _load_land_cover_with_profile2(year: int, base_dir: str) -> Tuple[np.ndarray, dict]:
    path = f"{base_dir}/predict_{year}.tif"
    arr, profile = _load_raster_with_profile(path)
    return _reclass_land_cover2(arr), profile


def _load_inundation_duration(year: int, base_dir: str) -> np.ndarray:
    path = f"{base_dir}/{year}/inun_duration.tif"
    return _load_raster(path)


def _load_inundation_mean_level(year: int, base_dir: str) -> np.ndarray:
    path = f"{base_dir}/{year}/inun_mean_wl.tif"
    return _load_raster(path)


def _build_id_and_intensity_series(
    years: Sequence[int],
    duration_dir: str,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    required_duration_years = set(years)
    for y in years:
        required_duration_years.update({y - 1, y - 2})

    duration_cache: Dict[int, np.ndarray] = {}
    intensity_cache: Dict[int, np.ndarray] = {}

    for y in sorted(required_duration_years):
        duration = _load_inundation_duration(y, duration_dir)
        mean_wl = _load_inundation_mean_level(y, duration_dir)
        duration_cache[y] = duration
        intensity_cache[y] = duration * mean_wl
        intensity_cache[y][duration == 0] = 0

    id_tminus2_list = []
    id_tminus1_list = []
    id_t_list = []
    intensity_list = []
    for y in years:
        id_tminus2_list.append(duration_cache[y - 2])
        id_tminus1_list.append(duration_cache[y - 1])
        id_t_list.append(duration_cache[y])
        intensity_list.append(intensity_cache[y])

    return (
        np.stack(id_tminus2_list, axis=0),
        np.stack(id_tminus1_list, axis=0),
        np.stack(id_t_list, axis=0),
    ), np.stack(intensity_list, axis=0)


def _parse_weights(text: Optional[str], expected: int) -> Tuple[float, ...]:
    if text is None:
        return tuple(1.0 for _ in range(expected))
    parts = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(parts) != expected:
        raise ValueError(f"Expected {expected} weights, got {len(parts)}")
    return tuple(parts)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_raster(path: str, array: np.ndarray, profile: dict) -> None:
    profile = profile.copy()
    data = array.astype("float32", copy=False)
    profile.update({"count": 1, "dtype": data.dtype, "nodata": np.nan})
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _run_validation(
    initial_land_cover: np.ndarray,
    id_series: Tuple[np.ndarray, np.ndarray, np.ndarray],
    intensity_series: np.ndarray,
    human_params_by_year: Tuple[HumanExpansionParams, ...],
    init_params: InitialSuccessionParams,
    community_params: CommunitySuccessionParams,
    reference_land_cover: np.ndarray,
):
    prediction = simulate_period(
        initial_land_cover=initial_land_cover,
        id_series=id_series,
        intensity_series=intensity_series,
        human_params_by_year=human_params_by_year,
        init_params=init_params,
        community_params=community_params,
    )
    metrics = _evaluate_objective(
        area_weight=0.7,
        accuracy_weight=0.3,
        reference_land_cover=reference_land_cover,
        aggregated_prediction=prediction,
    )
    area_comp = _area_comparison(reference_land_cover, prediction, classes=EVAL_CLASSES)
    f1_scores = _per_class_f1(reference_land_cover, prediction, classes=EVAL_CLASSES)
    return metrics, area_comp, f1_scores


if __name__ == "__main__":

    # for inun, human, out in zip(['H1', 'H1', 'H2', 'H2'], [(0.5, 4.07, 0.61),  (0.0, 0.67, 0.05), (0.5, 4.07, 0.61), (0.0, 0.67, 0.05)], ['S1', 'S2', 'S3', 'S4']):
    #     # File locations provided by the user
    #     veg_map_dir = r"G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_tif"
    #     inundation_dir = f"G:\A_Veg_Model\TGP_current_influence\\{inun}"
    #
    #     # Simulate period
    #     calibration_years = list(range(2002, 2014))
    #     init_lc_cal, profile_cal = _load_land_cover_with_profile(2002, veg_map_dir)
    #     ref_cal = _load_land_cover(2013, veg_map_dir)
    #     ref_series_cal = [_load_land_cover(year, veg_map_dir) for year in calibration_years]
    #     id_series_cal, intensity_cal = _build_id_and_intensity_series(calibration_years, inundation_dir)
    #     human_params_cal = _build_human_params(
    #         years=len(calibration_years),
    #         rate_a=human[0],
    #         rate_f=human[1],
    #         rate_u=human[2],
    #         pixel_area=0.0009,
    #         inundation_threshold=26.0,
    #         rate_correction=1.0,
    #     )
    #
    #     init_params = InitialSuccessionParams()
    #     base_comm_params = CommunitySuccessionParams()
    #     calib_config = CalibrationConfig(
    #         initial_params=init_params,
    #         community_params=base_comm_params,
    #         area_weight=0.7,
    #         accuracy_weight=0.3,
    #         learning_rate=0.5,
    #         max_steps=1,
    #         clip_min=1e-6,
    #     )
    #     calibration_output_dir = f"G:\A_Veg_Model\TGP_current_influence\\{out}\\tif"
    #     _ensure_dir(calibration_output_dir)
    #
    #     calib_result = calibrate(
    #         config=calib_config,
    #         initial_land_cover=init_lc_cal,
    #         id_series=id_series_cal,
    #         intensity_series=intensity_cal,
    #         human_params_by_year=human_params_cal,
    #         reference_land_cover=ref_cal,
    #         output_dir=calibration_output_dir,
    #         output_profile=profile_cal,
    #         output_year=calibration_years,
    #     )
    #
    # for inun, human, out in zip(['H1', 'H1', 'H2', 'H2'], [(0.5, 4.07, 0.61),  (0.0, 0.67, 0.05), (0.5, 4.07, 0.61), (0.0, 0.67, 0.05)], ['S1', 'S2', 'S3', 'S4']):
    #     # File locations provided by the user
    #     veg_map_dir = r"G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_tif"
    #     inundation_dir = f"G:\A_Veg_Model\TGP_current_influence\\{inun}"
    #
    #     # Simulate period
    #     calibration_years = list(range(2002, 2014))
    #     init_lc_cal, profile_cal = _load_land_cover_with_profile(2002, veg_map_dir)
    #     ref_cal = _load_land_cover(2013, veg_map_dir)
    #     ref_series_cal = [_load_land_cover(year, veg_map_dir) for year in calibration_years]
    #     id_series_cal, intensity_cal = _build_id_and_intensity_series(calibration_years, inundation_dir)
    #     human_params_cal = _build_human_params(
    #         years=len(calibration_years),
    #         rate_a=human[0],
    #         rate_f=human[1],
    #         rate_u=human[2],
    #         pixel_area=0.0009,
    #         inundation_threshold=26.0,
    #         rate_correction=1.0,
    #     )
    #
    #     init_params = InitialSuccessionParams()
    #     base_comm_params = CommunitySuccessionParams()
    #     calib_config = CalibrationConfig(
    #         initial_params=init_params,
    #         community_params=base_comm_params,
    #         area_weight=0.7,
    #         accuracy_weight=0.3,
    #         learning_rate=0.5,
    #         max_steps=1,
    #         clip_min=1e-6,
    #     )
    #     calibration_output_dir = f"G:\A_Veg_Model\TGP_current_influence\\{out}\\tif"
    #     _ensure_dir(calibration_output_dir)
    #
    #     calib_result = calibrate(
    #         config=calib_config,
    #         initial_land_cover=init_lc_cal,
    #         id_series=id_series_cal,
    #         intensity_series=intensity_cal,
    #         human_params_by_year=human_params_cal,
    #         reference_land_cover=ref_cal,
    #         output_dir=calibration_output_dir,
    #         output_profile=profile_cal,
    #         output_year=calibration_years,
    #     )
    #

    # for inun, human, out in zip(['H1', 'H1', 'H2', 'H2'], [(0.5, 4.07, 0.61),  (0.0, 0.67, 0.05), (0.5, 4.07, 0.61), (0.0, 0.67, 0.05)], ['S1', 'S2', 'S3', 'S4']):
    #     # File locations provided by the user
    #     veg_map_dir = r"G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_tif"
    #     inundation_dir = f"G:\A_Veg_Model\TGP_current_influence\\{inun}"
    #
    #     # Simulate period
    #     calibration_years = list(range(2002, 2014))
    #     init_lc_cal, profile_cal = _load_land_cover_with_profile(2002, veg_map_dir)
    #     ref_cal = _load_land_cover(2013, veg_map_dir)
    #     ref_series_cal = [_load_land_cover(year, veg_map_dir) for year in calibration_years]
    #     id_series_cal, intensity_cal = _build_id_and_intensity_series(calibration_years, inundation_dir)
    #     human_params_cal = _build_human_params(
    #         years=len(calibration_years),
    #         rate_a=human[0],
    #         rate_f=human[1],
    #         rate_u=human[2],
    #         pixel_area=0.0009,
    #         inundation_threshold=26.0,
    #         rate_correction=1.0,
    #     )
    #
    #     init_params = InitialSuccessionParams()
    #     base_comm_params = CommunitySuccessionParams()
    #     calib_config = CalibrationConfig(
    #         initial_params=init_params,
    #         community_params=base_comm_params,
    #         area_weight=0.7,
    #         accuracy_weight=0.3,
    #         learning_rate=0.5,
    #         max_steps=1,
    #         clip_min=1e-6,
    #     )
    #     calibration_output_dir = f"G:\A_Veg_Model\TGP_current_influence\\{out}\\tif"
    #     _ensure_dir(calibration_output_dir)
    #
    #     calib_result = calibrate(
    #         config=calib_config,
    #         initial_land_cover=init_lc_cal,
    #         id_series=id_series_cal,
    #         intensity_series=intensity_cal,
    #         human_params_by_year=human_params_cal,
    #         reference_land_cover=ref_cal,
    #         output_dir=calibration_output_dir,
    #         output_profile=profile_cal,
    #         output_year=calibration_years,
    #     )
    #

    # for inun, human, out in zip(['H1', 'H1', 'H2', 'H2'], [(0.7, 0.7, 0.6),  (0.0, 0.67, 0.05), (0.7, 0.7, 0.6), (0.0, 0.67, 0.05)], ['S1', 'S2', 'S3', 'S4']):
    #     # File locations provided by the user
    #     veg_in_dir = f"G:\A_Veg_Model\TGP_current_influence\\{out}\\tif"
    #     veg_map_dir = r"G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_tif"
    #     inundation_dir = f"G:\A_Veg_Model\TGP_current_influence\\{inun}"
    #
    #     # Simulate period
    #     calibration_years = list(range(2013, 2021))
    #     # init_lc_cal, profile_cal = _load_land_cover_with_profile(2013, veg_in_dir)
    #     init_lc_cal, profile_cal = _load_land_cover_with_profile2(2013, veg_in_dir)
    #     ref_cal = _load_land_cover(2020, veg_map_dir)
    #     ref_series_cal = [_load_land_cover(year, veg_map_dir) for year in calibration_years]
    #     id_series_cal, intensity_cal = _build_id_and_intensity_series(calibration_years, inundation_dir)
    #     human_params_cal = _build_human_params(
    #         years=len(calibration_years),
    #         rate_a=human[0],
    #         rate_f=human[1],
    #         rate_u=human[2],
    #         pixel_area=0.0009,
    #         inundation_threshold=26.0,
    #         rate_correction=1.0,
    #     )
    #
    #     init_params = InitialSuccessionParams()
    #     base_comm_params = CommunitySuccessionParams()
    #     calib_config = CalibrationConfig(
    #         initial_params=init_params,
    #         community_params=base_comm_params,
    #         area_weight=0.7,
    #         accuracy_weight=0.3,
    #         learning_rate=0.5,
    #         max_steps=1,
    #         clip_min=1e-6,
    #     )
    #     calibration_output_dir = f"G:\A_Veg_Model\TGP_current_influence\\{out}\\tif"
    #     _ensure_dir(calibration_output_dir)
    #
    #     calib_result = calibrate(
    #         config=calib_config,
    #         initial_land_cover=init_lc_cal,
    #         id_series=id_series_cal,
    #         intensity_series=intensity_cal,
    #         human_params_by_year=human_params_cal,
    #         reference_land_cover=ref_cal,
    #         output_dir=calibration_output_dir,
    #         output_profile=profile_cal,
    #         output_year=calibration_years,
    #     )
    #

    for inun, human, out in zip(['H2', 'H1', 'H2', 'H1'], [ (0.0, 0.0, 0.0), (0.0, 0, 0.0), (0.7, 0.7, 0.6), (0.7, 0.7, 0.6),  ], ['S4', 'S2', 'S3', 'S1']):
        # File locations provided by the user
        veg_in_dir = f"G:\A_Veg_Model\TGP_future_influence\\{out}\\tif"
        veg_map_dir = r"G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_tif"
        inundation_dir = f"G:\A_Veg_Model\TGP_future_influence\\{inun}"

        # Simulate period
        calibration_years = list(range(2021, 2041))
        # init_lc_cal, profile_cal = _load_land_cover_with_profile(2013, veg_in_dir)
        init_lc_cal, profile_cal = _load_land_cover_with_profile(2020, veg_map_dir)
        ref_cal = _load_land_cover(2020, veg_map_dir)
        ref_series_cal = [_load_land_cover(year, veg_map_dir) for year in range(2000, 2020)]
        id_series_cal, intensity_cal = _build_id_and_intensity_series(calibration_years, inundation_dir)
        human_params_cal = _build_human_params(
            years=len(calibration_years),
            rate_a=human[0],
            rate_f=human[1],
            rate_u=human[2],
            pixel_area=0.0009,
            inundation_threshold=26.0,
            rate_correction=1.0,
        )

        init_params = InitialSuccessionParams()
        base_comm_params = CommunitySuccessionParams()
        calib_config = CalibrationConfig(
            initial_params=init_params,
            community_params=base_comm_params,
            area_weight=0.7,
            accuracy_weight=0.3,
            learning_rate=0.5,
            max_steps=1,
            clip_min=1e-6,
        )
        calibration_output_dir = f"G:\A_Veg_Model\TGP_future_influence\\{out}\\tif"
        _ensure_dir(calibration_output_dir)

        calib_result = calibrate(
            config=calib_config,
            initial_land_cover=init_lc_cal,
            id_series=id_series_cal,
            intensity_series=intensity_cal,
            human_params_by_year=human_params_cal,
            reference_land_cover=ref_cal,
            output_dir=calibration_output_dir,
            output_profile=profile_cal,
            output_year=calibration_years,
        )



    #
    #
    # # Calibration period: drive 2000 → 2013
    # calibration_years = list(range(2001, 2014))
    # init_lc_cal, profile_cal = _load_land_cover_with_profile(2000, veg_map_dir)
    # ref_cal = _load_land_cover(2013, veg_map_dir)
    # ref_series_cal = [_load_land_cover(year, veg_map_dir) for year in calibration_years]
    # id_series_cal, intensity_cal = _build_id_and_intensity_series(calibration_years, inundation_dir)
    # human_params_cal = _build_human_params(
    #     years=len(calibration_years),
    #     rate_a=0.5,
    #     rate_f=4.07,
    #     rate_u=0.61,
    #     pixel_area=0.0009,
    #     inundation_threshold=26.0,
    #     rate_correction=1.0,
    # )
    # # human_params_cal = _build_human_params(
    # #     years=len(calibration_years),
    # #     rate_a=0,
    # #     rate_f=0,
    # #     rate_u=0,
    # #     pixel_area=0.0009,
    # #     inundation_threshold=26.0,
    # #     rate_correction=1.0,
    # # )
    #
    # init_params = InitialSuccessionParams()
    # base_comm_params = CommunitySuccessionParams()
    #
    # calib_config = CalibrationConfig(
    #     initial_params=init_params,
    #     community_params=base_comm_params,
    #     area_weight=0.7,
    #     accuracy_weight=0.3,
    #     learning_rate=0.5,
    #     max_steps=1,
    #     clip_min=1e-6,
    # )
    #
    # calibration_output_dir = r"G:\\A_GEDI_Floodplain_vegh\\Veg_map\\Succession\\Calibration"
    # _ensure_dir(calibration_output_dir)
    #
    # calib_result = calibrate(
    #     config=calib_config,
    #     initial_land_cover=init_lc_cal,
    #     id_series=id_series_cal,
    #     intensity_series=intensity_cal,
    #     human_params_by_year=human_params_cal,
    #     reference_land_cover=ref_cal,
    #     output_dir=calibration_output_dir,
    #     output_profile=profile_cal,
    #     output_year=calibration_years[-1],
    # )
    #
    # # simulate period: drive 1990 → 2000
    # calibration_years = list(range(1990, 2001))
    # init_lc_cal, profile_cal = _load_land_cover_with_profile(1990, veg_map_dir)
    # ref_cal = _load_land_cover(2000, veg_map_dir)
    # ref_series_cal = [_load_land_cover(year, veg_map_dir) for year in calibration_years]
    # id_series_cal, intensity_cal = _build_id_and_intensity_series(calibration_years, inundation_dir)
    # human_params_cal = _build_human_params(
    #     years=len(calibration_years),
    #     rate_a=0.0,
    #     rate_f=0.67,
    #     rate_u=0.05,
    #     pixel_area=0.0009,
    #     inundation_threshold=26.0,
    #     rate_correction=1.0,
    # )
    # # human_params_cal = _build_human_params(
    # #     years=len(calibration_years),
    # #     rate_a=0,
    # #     rate_f=0,
    # #     rate_u=0,
    # #     pixel_area=0.0009,
    # #     inundation_threshold=26.0,
    # #     rate_correction=1.0,
    # # )
    #
    # init_params = InitialSuccessionParams()
    # base_comm_params = CommunitySuccessionParams()
    #
    # calib_config = CalibrationConfig(
    #     initial_params=init_params,
    #     community_params=base_comm_params,
    #     area_weight=0.7,
    #     accuracy_weight=0.3,
    #     learning_rate=0.5,
    #     max_steps=1,
    #     clip_min=1e-6,
    # )
    #
    # calibration_output_dir = r"G:\\A_GEDI_Floodplain_vegh\\Veg_map\\Succession\\Calibration"
    # _ensure_dir(calibration_output_dir)
    #
    # calib_result = calibrate(
    #     config=calib_config,
    #     initial_land_cover=init_lc_cal,
    #     id_series=id_series_cal,
    #     intensity_series=intensity_cal,
    #     human_params_by_year=human_params_cal,
    #     reference_land_cover=ref_cal,
    #     output_dir=calibration_output_dir,
    #     output_profile=profile_cal,
    #     output_year=calibration_years[-1],
    # )
    #
    # # simulate period: drive 1990 → 2000
    # calibration_years = list(range(2013, 2021))
    # init_lc_cal, profile_cal = _load_land_cover_with_profile(2013, veg_map_dir)
    # ref_cal = _load_land_cover(2020, veg_map_dir)
    # ref_series_cal = [_load_land_cover(year, veg_map_dir) for year in calibration_years]
    # id_series_cal, intensity_cal = _build_id_and_intensity_series(calibration_years, inundation_dir)
    # human_params_cal = _build_human_params(
    #     years=len(calibration_years),
    #     rate_a=0.7,
    #     rate_f=0.7,
    #     rate_u=0.6,
    #     pixel_area=0.0009,
    #     inundation_threshold=26.0,
    #     rate_correction=1.0,
    # )
    # # human_params_cal = _build_human_params(
    # #     years=len(calibration_years),
    # #     rate_a=0,
    # #     rate_f=0,
    # #     rate_u=0,
    # #     pixel_area=0.0009,
    # #     inundation_threshold=26.0,
    # #     rate_correction=1.0,
    # # )
    #
    # init_params = InitialSuccessionParams()
    # base_comm_params = CommunitySuccessionParams()
    #
    # calib_config = CalibrationConfig(
    #     initial_params=init_params,
    #     community_params=base_comm_params,
    #     area_weight=0.7,
    #     accuracy_weight=0.3,
    #     learning_rate=0.5,
    #     max_steps=1,
    #     clip_min=1e-6,
    # )
    #
    # calibration_output_dir = r"G:\\A_GEDI_Floodplain_vegh\\Veg_map\\Succession\\Calibration"
    # _ensure_dir(calibration_output_dir)
    #
    # calib_result = calibrate(
    #     config=calib_config,
    #     initial_land_cover=init_lc_cal,
    #     id_series=id_series_cal,
    #     intensity_series=intensity_cal,
    #     human_params_by_year=human_params_cal,
    #     reference_land_cover=ref_cal,
    #     output_dir=calibration_output_dir,
    #     output_profile=profile_cal,
    #     output_year=calibration_years[-1],
    # )