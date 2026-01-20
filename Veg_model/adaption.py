"""
MAVI simulation module driven by changing hydrology and geomorphology.

This script trains a single pixel-scale LSTM on the full multi-year time
series (1988–2020) and evaluates it with an 80/20 pixel split. The LSTM
consumes complete per-pixel sequences of yearly drivers and predicts MAVI at
every timestep, saving per-year rasters and metrics without any sliding-window
variants.
"""
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm


def _import_torch():
    """Import torch with a clearer error if DLL loading fails.

    A Windows "DLL initialization routine failed" typically indicates a missing
    or mismatched CUDA runtime. This wrapper raises a descriptive error so users
    know to install a GPU-enabled PyTorch build with the correct NVIDIA drivers
    (or enable CPU fallback explicitly if desired).
    """

    try:
        import torch  # type: ignore
        from torch import nn  # type: ignore
        from torch.utils.data import DataLoader, Dataset  # type: ignore
    except OSError as exc:
        raise RuntimeError(
            "PyTorch failed to load (likely missing CUDA DLLs). "
            "Install a GPU-enabled torch matching your CUDA toolkit, or set "
            "USE_CPU_TORCH=1 to allow CPU-only fallback if you accept the "
            "slower runtime."
        ) from exc

    return torch, nn, DataLoader, Dataset


torch, nn, DataLoader, Dataset = _import_torch()


# ------------------------------
# Path configuration
# ------------------------------
BASE_MAVI_TEMPLATE = (
    r"G:/A_Landsat_Floodplain_veg/Landsat_floodplain_2020_datacube/"
    r"OSAVI_noninun_curfit_datacube/Phemetric_tif/{year}/{year}_MAVI.TIF"
)
INUNDATION_DURATION_TEMPLATE = (
    r"G:/A_Landsat_Floodplain_veg/Water_level_python/Inundation_indicator/"
    r"inundation_factor/{year}/inun_duration.tif"
)
INUNDATION_MEAN_WL_TEMPLATE = (
    r"G:/A_Landsat_Floodplain_veg/Water_level_python/Inundation_indicator/"
    r"inundation_factor/{year}/inun_mean_wl.tif"
)
ELEVATION_PRE_PATH = (
    r"G:/A_Landsat_Floodplain_veg/Water_level_python/Post_TGD/ele_pretgd4model.TIF"
)
ELEVATION_POST_PATH = (
    r"G:/A_Landsat_Floodplain_veg/Water_level_python/Post_TGD/ele_posttgd4model.TIF"
)
CLIMATE_ROOT = (
    r"G:/A_Climatology_dataset/station_dataset/CMA_dataset/"
    r"2400_all_station_1950_2023/CMA_OUTPUT/annual_mean"
)
CLIMATE_VARIABLES = ["AGB", "TEM", "RHU", "PRE", "PRS", "GST", "WIN"]
CLIMATE_TEMPLATE = r"{root}/{var}/{year}_annual_mean_{var}.tif"
TOP_K_FEATURES = 6
SEGMENT_COL_RANGES: List[Tuple[int, int]] = [
    (0, 950),
    (950, 6100),
    (6100, 10210),
    (10210, 16537),
]

OUTPUT_ROOT = Path(r"G:\A_Veg_Model\TGP_current_influence\A1")
FEATURE_CSV_DIR = OUTPUT_ROOT / "features"
MODEL_PATH = OUTPUT_ROOT / "lstm_model.pth"
PREDICTED_MAVI_DIR = OUTPUT_ROOT / "predicted_mavi"


# ------------------------------
# Utility functions
# ------------------------------

def collect_valid_coords(
    years: List[int],
    reference_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect valid pixel coordinates (row, col) that have observed MAVI in any year."""

    rows, cols = reference_shape
    total_pixels = rows * cols

    valid_coords = set()
    for year in years:
        csv_path = FEATURE_CSV_DIR / f"mavi_features_{year}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Feature CSV for year {year} not found at {csv_path}. Run export_features_to_csv first."
            )
        df = pd.read_csv(csv_path, usecols=["row", "col", "MAVI"])
        df = df.dropna(subset=["MAVI"])
        valid_coords.update(zip(df["row"].astype(int), df["col"].astype(int)))

    if not valid_coords:
        raise ValueError("No valid pixel sequences found.")

    coord_list = sorted(valid_coords)
    pixel_mask = np.zeros(total_pixels, dtype=bool)
    for r, c in coord_list:
        pixel_mask[r * cols + c] = True

    return np.array(coord_list, dtype=np.int32), pixel_mask

def load_raster(path: Path) -> Tuple[np.ndarray, Affine, Optional[str]]:
    """Read a raster file using rasterio and return array, transform, and projection."""

    if not path.exists():
        raise FileNotFoundError(f"Cannot open raster: {path}")

    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float32)
        transform = src.transform
        projection = src.crs.to_wkt() if src.crs is not None else None
    return array, transform, projection


def write_raster(
    path: Path,
    array: np.ndarray,
    transform: Affine,
    projection: Optional[str],
    nodata: float = -9999.0,
) -> None:
    """Save a numpy array to GeoTIFF using rasterio."""

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = array.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=np.float32,
        transform=transform,
        crs=projection,
        nodata=nodata,
    ) as dst:
        dst.write(array.astype(np.float32), 1)


def _load_elevation_pair() -> Tuple[np.ndarray, np.ndarray]:
    """Load and cache the pre/post-TGD elevation rasters once."""

    pre_array, _, _ = load_raster(Path(ELEVATION_PRE_PATH))
    post_array, _, _ = load_raster(Path(ELEVATION_POST_PATH))
    return pre_array, post_array


def interpolate_elevation(
    year: int,
    pre_mid_year: float = 1995.0,
    post_mid_year: float = 2011.5,
) -> np.ndarray:
    """
    Interpolate yearly elevation using pre- and post-TGD DEMs.

    The pre-TGD DEM represents the mid-point of 1987-2002, and the post-TGD
    DEM represents the mid-point of 2003-2020. Linear interpolation is used
    for intermediate years; values are extrapolated for years outside the
    anchor range.
    """
    pre_array, post_array = _load_elevation_pair()

    if year <= pre_mid_year:
        return pre_array
    if year >= post_mid_year:
        return post_array

    weight = (year - pre_mid_year) / (post_mid_year - pre_mid_year)
    return pre_array + (post_array - pre_array) * weight


def compute_inundation_frequency(duration_array: np.ndarray, days_per_year: float = 365.0) -> np.ndarray:
    """Compute inundation frequency from inundation duration (days per year)."""
    return np.clip(duration_array / days_per_year, 0, 1)

def build_sequences_for_coords(
    years: List[int],
    coords: np.ndarray,
    feature_keys: List[str],
    reference_shape: Tuple[int, int],
    inun_duration_template: str,
    inun_mean_wl_template: str,
    climate_root: str = CLIMATE_ROOT,
) -> np.ndarray:
    """Build feature sequences for provided coordinates using custom templates."""

    num_pixels = len(coords)
    timesteps = len(years)
    feat_dim = len(feature_keys)
    sequences = np.zeros((num_pixels, timesteps, feat_dim), dtype=np.float32)

    for t, year in enumerate(years):
        feature_arrays = load_feature_arrays(
            year,
            feature_keys,
            inun_duration_template=inun_duration_template,
            inun_mean_wl_template=inun_mean_wl_template,
            climate_root=climate_root,
        )
        for f_idx, key in enumerate(feature_keys):
            arr = np.nan_to_num(feature_arrays[key], nan=0.0)
            sequences[:, t, f_idx] = arr[coords[:, 0], coords[:, 1]]

    return sequences


def load_feature_arrays(year: int, feature_keys: Iterable[str]) -> Dict[str, np.ndarray]:
    """Load feature rasters for a single year (excluding MAVI)."""
    duration_array, _, _ = load_raster(Path(INUNDATION_DURATION_TEMPLATE.format(year=year)))
    mean_wl_array, _, _ = load_raster(Path(INUNDATION_MEAN_WL_TEMPLATE.format(year=year)))
    elevation_array = interpolate_elevation(year)
    inundation_frequency = compute_inundation_frequency(duration_array)

    data = {
        "IF": inundation_frequency,
        "ID": duration_array,
        "IH": mean_wl_array,
        "ELE": elevation_array,
    }

    for var in CLIMATE_VARIABLES:
        if var not in feature_keys:
            continue
        climate_path = Path(
            CLIMATE_TEMPLATE.format(root=CLIMATE_ROOT, var=var, year=year)
        )
        data[var], _, _ = load_raster(climate_path)

    return {k: v for k, v in data.items() if k in feature_keys}


def gather_year_data(year: int) -> Tuple[Dict[str, np.ndarray], Tuple, str]:
    """Load rasters for a single year and assemble feature arrays."""
    mavi_array, geotransform, projection = load_raster(
        Path(BASE_MAVI_TEMPLATE.format(year=year))
    )
    features = load_feature_arrays(
        year, ["IF", "ID", "IH", "ELE", *CLIMATE_VARIABLES]
    )
    features["MAVI"] = mavi_array
    return features, geotransform, projection


def export_features_to_csv(years: List[int]) -> None:
    """Flatten yearly feature rasters into per-year CSV files.

    Each output row corresponds to a single pixel (row/column indexed) for a
    given year so downstream usage remains strictly at the pixel scale. Only
    missing years are processed to avoid regenerating already exported data.
    """
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FEATURE_CSV_DIR.mkdir(parents=True, exist_ok=True)

    missing_years = [y for y in years if not (FEATURE_CSV_DIR / f"mavi_features_{y}.csv").exists()]
    if not missing_years:
        print("All feature CSVs already exist; skipping extraction step entirely.")
        return

    for year in tqdm(missing_years, desc="Exporting rasters to CSV"):
        year_csv_path = FEATURE_CSV_DIR / f"mavi_features_{year}.csv"

        data, _, _ = gather_year_data(year)
        valid_mask = ~np.isnan(data["MAVI"])
        for key, array in data.items():
            valid_mask &= ~np.isnan(array)

        if valid_mask.sum() == 0:
            continue

        row_indices, col_indices = np.nonzero(valid_mask)
        row = {
            "year": np.repeat(year, valid_mask.sum()),
            "row": row_indices,
            "col": col_indices,
        }
        for key, array in data.items():
            row[key] = array[valid_mask]
        df = pd.DataFrame(row)

        df.to_csv(year_csv_path, index=False)
        print(f"Feature CSV saved to {year_csv_path}")


def _load_array_from_csv(year: int, column: str, shape: Tuple[int, int]) -> np.ndarray:
    """Reconstruct a full-size array for a column from the per-year CSV file."""

    csv_path = FEATURE_CSV_DIR / f"mavi_features_{year}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Feature CSV for year {year} not found at {csv_path}. Run export_features_to_csv first."
        )

    df = pd.read_csv(csv_path, usecols=["row", "col", column])
    arr = np.full(shape, np.nan, dtype=np.float32)
    rows = df["row"].to_numpy(dtype=int)
    cols = df["col"].to_numpy(dtype=int)
    values = df[column].to_numpy(dtype=np.float32)
    valid = ~np.isnan(values)
    if valid.any():
        arr[rows[valid], cols[valid]] = values[valid]
    return arr


def compute_window_mean_from_csv(
    years: List[int], keys: List[str], shape: Tuple[int, int]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute multi-year means using the per-year CSV exports instead of rasters.

    Values are accumulated with zero-filled missing entries while per-pixel valid counts
    are tracked so averages reflect the available observations for each pixel.
    """

    sum_dict: Dict[str, np.ndarray] = {
        key: np.zeros(shape, dtype=np.float32) for key in keys
    }
    count_dict: Dict[str, np.ndarray] = {
        key: np.zeros(shape, dtype=np.float32) for key in keys
    }

    for year in years:
        csv_path = FEATURE_CSV_DIR / f"mavi_features_{year}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Feature CSV for year {year} not found at {csv_path}. Run export_features_to_csv first."
            )

        usecols = ["row", "col", *keys]
        df = pd.read_csv(csv_path, usecols=usecols)
        rows = df["row"].to_numpy(dtype=int)
        cols = df["col"].to_numpy(dtype=int)

        for key in keys:
            values = df[key].to_numpy(dtype=np.float32)
            valid = ~np.isnan(values)
            if not valid.any():
                continue
            sum_dict[key][rows[valid], cols[valid]] += values[valid]
            count_dict[key][rows[valid], cols[valid]] += 1.0

    mean_dict: Dict[str, np.ndarray] = {}
    for key in keys:
        counts = count_dict[key]
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = sum_dict[key] / np.maximum(counts, 1)
        mean[counts == 0] = 0.0
        mean_dict[key] = mean.astype(np.float32)

    return mean_dict, count_dict


def load_year_arrays_from_csv(
    year: int, keys: List[str], shape: Tuple[int, int], fill_value: float = 0.0
) -> Dict[str, np.ndarray]:
    """Reconstruct full-sized arrays for requested columns from a year CSV."""

    csv_path = FEATURE_CSV_DIR / f"mavi_features_{year}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Feature CSV for year {year} not found at {csv_path}. Run export_features_to_csv first."
        )

    usecols = ["row", "col", *keys]
    df = pd.read_csv(csv_path, usecols=usecols)
    arrays: Dict[str, np.ndarray] = {
        key: np.full(shape, fill_value, dtype=np.float32) for key in keys
    }

    rows = df["row"].to_numpy(dtype=int)
    cols = df["col"].to_numpy(dtype=int)
    for key in keys:
        values = df[key].to_numpy(dtype=np.float32)
        valid = ~np.isnan(values)
        if not valid.any():
            continue
        arrays[key][rows[valid], cols[valid]] = values[valid]

    return arrays


def compute_period_mean_from_csv(years: List[int], feature_keys: List[str]) -> pd.DataFrame:
    """Compute per-pixel multi-year means from existing per-year feature CSVs."""

    required_cols = ["row", "col", "MAVI", *feature_keys]
    sum_df: Optional[pd.DataFrame] = None
    count_df: Optional[pd.DataFrame] = None

    for year in years:
        csv_path = FEATURE_CSV_DIR / f"mavi_features_{year}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Feature CSV for year {year} not found at {csv_path}. Run export_features_to_csv first."
            )

        df = pd.read_csv(csv_path, usecols=required_cols)
        df = df.set_index(["row", "col"]).astype(np.float32)

        # Fill missing values with zeros for accumulation and track valid counts per pixel
        df_filled = df.fillna(0.0)
        df_count = df.notna().astype(np.float32)

        if sum_df is None:
            sum_df = df_filled
            count_df = df_count
        else:
            sum_df = sum_df.add(df_filled, fill_value=0.0)
            count_df = count_df.add(df_count, fill_value=0.0)

    if sum_df is None or count_df is None:
        raise ValueError("No data loaded to compute period means.")

    # Divide per pixel by the number of valid years for that pixel; avoid divide-by-zero warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_df = sum_df / count_df
    mean_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    mean_df.reset_index(inplace=True)
    return mean_df


def build_training_samples_from_period_means(
    pre_years: List[int],
    post_years: List[int],
    feature_keys: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build training samples from multi-year means of pre- and post-TGD periods.

    A single training set is constructed by contrasting the pre-TGD mean with
    the post-TGD mean for every pixel, yielding one delta feature vector and
    one delta MAVI value per pixel. This satisfies the requirement to train a
    single model using only the long-term averages of 1988–2002 and
    2003–2020.
    """

    pre_means_df = compute_period_mean_from_csv(pre_years, feature_keys)
    post_means_df = compute_period_mean_from_csv(post_years, feature_keys)

    merged = pre_means_df.merge(
        post_means_df,
        on=["row", "col"],
        suffixes=("_pre", "_post"),
        how="inner",
    )

    delta_columns = {}
    for key in feature_keys:
        delta_columns[key] = merged[f"{key}_post"] - merged[f"{key}_pre"]
    X = pd.DataFrame(delta_columns).values

    y = (merged["MAVI_post"] - merged["MAVI_pre"]).values

    valid_mask = ~np.isnan(y)
    for i in range(X.shape[1]):
        valid_mask &= ~np.isnan(X[:, i])

    X = X[valid_mask]
    y = y[valid_mask]
    return X, y


def build_full_sequences(
    years: List[int],
    feature_keys: List[str],
    reference_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct pixel-level sequences (all years) for LSTM training.

    Returns sequences (N, T, F), targets (N, T), masks (N, T) indicating valid
    MAVI observations, a pixel mask identifying which flattened indices
    participate in the sequences, and an array of pixel coordinates (row, col)
    aligned to the first dimension of the tensors.
    """

    feature_columns = [*feature_keys]
    rows, cols = reference_shape
    total_pixels = rows * cols

    # Identify pixels with any observed MAVI across all years without building
    # full-sized dense tensors.
    valid_coords = set()
    for year in years:
        csv_path = FEATURE_CSV_DIR / f"mavi_features_{year}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Feature CSV for year {year} not found at {csv_path}. Run export_features_to_csv first."
            )

        df = pd.read_csv(csv_path, usecols=["row", "col", "MAVI"])
        df = df.dropna(subset=["MAVI"])
        valid_coords.update(zip(df["row"].astype(int), df["col"].astype(int)))

    if not valid_coords:
        raise ValueError("No valid pixel sequences found for LSTM training.")

    coord_list = sorted(valid_coords)
    coord_to_idx = {coord: idx for idx, coord in enumerate(coord_list)}

    pixel_mask = np.zeros(total_pixels, dtype=bool)
    for r, c in coord_list:
        pixel_mask[r * cols + c] = True

    num_valid = len(coord_list)
    timesteps = len(years)
    feat_dim = len(feature_columns)

    sequences = np.zeros((num_valid, timesteps, feat_dim), dtype=np.float32)
    targets = np.zeros((num_valid, timesteps), dtype=np.float32)
    masks = np.zeros((num_valid, timesteps), dtype=bool)

    for t, year in enumerate(years):
        csv_path = FEATURE_CSV_DIR / f"mavi_features_{year}.csv"
        df = pd.read_csv(csv_path, usecols=["row", "col", "MAVI", *feature_columns])

        rows_arr = df["row"].to_numpy(dtype=int)
        cols_arr = df["col"].to_numpy(dtype=int)
        coord_indices = np.fromiter(
            (coord_to_idx.get((r, c), -1) for r, c in zip(rows_arr, cols_arr)),
            dtype=np.int64,
        )

        valid = coord_indices >= 0
        if not np.any(valid):
            continue

        idxs = coord_indices[valid]

        mavi_vals = df.loc[valid, "MAVI"].to_numpy(dtype=np.float32)
        masks[idxs, t] = ~np.isnan(mavi_vals)
        targets[idxs, t] = np.nan_to_num(mavi_vals, nan=0.0)

        for f_idx, key in enumerate(feature_columns):
            feat_vals = df.loc[valid, key].to_numpy(dtype=np.float32)
            sequences[idxs, t, f_idx] = np.nan_to_num(feat_vals, nan=0.0)

    coords_array = np.array(coord_list, dtype=np.int32)

    return sequences, targets, masks, pixel_mask, coords_array


def compute_feature_stats(
    sequences: np.ndarray, masks: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean/std per feature using only timesteps with valid MAVI."""

    feat_dim = sequences.shape[2]
    means = np.zeros(feat_dim, dtype=np.float32)
    stds = np.ones(feat_dim, dtype=np.float32)

    for f in range(feat_dim):
        feat_vals = sequences[:, :, f][masks]
        if feat_vals.size == 0:
            continue
        means[f] = feat_vals.mean()
        std_val = feat_vals.std()
        stds[f] = std_val if std_val > 1e-6 else 1.0

    return means, stds


def normalize_features(
    sequences: np.ndarray, feature_means: np.ndarray, feature_stds: np.ndarray
) -> np.ndarray:
    """Return a normalized copy of sequences using provided stats."""

    norm = sequences.astype(np.float32, copy=True)
    for f in range(norm.shape[2]):
        norm[:, :, f] = (norm[:, :, f] - feature_means[f]) / feature_stds[f]
    return norm


def compute_feature_correlations(
    sequences: np.ndarray, targets: np.ndarray, masks: np.ndarray, feature_keys: List[str]
) -> pd.DataFrame:
    """Compute per-feature Pearson correlation with MAVI over valid pixels/timesteps."""

    correlations: List[Dict[str, float]] = []
    valid_targets = targets[masks]
    for f_idx, key in enumerate(feature_keys):
        feat_vals = sequences[:, :, f_idx][masks]
        if feat_vals.size < 2 or valid_targets.size < 2:
            corr = 0.0
        else:
            with np.errstate(invalid="ignore"):
                corr = float(np.corrcoef(feat_vals, valid_targets)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        correlations.append({"feature": key, "correlation": corr, "abs_correlation": abs(corr)})

    corr_df = pd.DataFrame(correlations)
    corr_df.sort_values(by="abs_correlation", ascending=False, inplace=True)
    return corr_df


def select_top_features(corr_df: pd.DataFrame, top_k: int) -> List[str]:
    """Return the top-k features ranked by absolute correlation."""

    top = corr_df.head(top_k)
    return top["feature"].tolist()


def _to_numpy(array: np.ndarray) -> np.ndarray:
    """Convert numpy/cupy array to numpy on CPU."""
    try:  # cupy array
        import cupy as cp

        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
    except Exception:
        pass
    return np.asarray(array)


class PixelSequenceDataset(Dataset):
    """PyTorch dataset for pixel-level yearly sequences."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray, masks: np.ndarray):
        self.sequences = torch.from_numpy(sequences.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))
        self.masks = torch.from_numpy(masks.astype(np.float32))

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx], self.masks[idx]


class PixelLSTM(nn.Module):
    """Two-layer LSTM that predicts MAVI at every timestep."""

    def __init__(
        self, feature_dim: int, hidden_size: int = 128, dropout: float = 0.2, num_layers: int = 2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        preds = self.head(out).squeeze(-1)
        return preds


def _resolve_device(requested: str = "cuda") -> str:
    """Select device, preferring CUDA but gracefully falling back to CPU."""

    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print(
            "CUDA requested but not available; falling back to CPU. "
            "Install a CUDA-enabled PyTorch build to enable GPU acceleration."
        )
        return "cpu"
    return requested


def _masked_metrics(preds: np.ndarray, targets: np.ndarray, masks: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, and R² using only valid mask positions."""

    valid = masks.astype(bool)
    if not valid.any():
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

    pred_valid = preds[valid]
    target_valid = targets[valid]
    mae = float(mean_absolute_error(target_valid, pred_valid))
    rmse = float(np.sqrt(np.mean((pred_valid - target_valid) ** 2)))
    r2 = float(r2_score(target_valid, pred_valid))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _segment_masks(coords: np.ndarray, segment_ranges: List[Tuple[int, int]]) -> List[np.ndarray]:
    """Create boolean masks for each segment based on column ranges."""

    col_vals = coords[:, 1]
    return [
        (col_vals >= start) & (col_vals < end)
        for start, end in segment_ranges
    ]


def compute_overall_and_segment_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    coords: np.ndarray,
    segment_ranges: List[Tuple[int, int]],
) -> Dict[str, float]:
    """Compute overall and per-segment MAE/RMSE/R²."""

    metrics = {k: v for k, v in _masked_metrics(preds, targets, masks).items()}
    seg_masks = _segment_masks(coords, segment_ranges)
    for idx, seg_mask in enumerate(seg_masks, start=1):
        seg_valid = masks & seg_mask[:, None]
        seg_metrics = _masked_metrics(preds, targets, seg_valid)
        metrics.update({
            f"segment{idx}_mae": seg_metrics["mae"],
            f"segment{idx}_rmse": seg_metrics["rmse"],
            f"segment{idx}_r2": seg_metrics["r2"],
        })
    return metrics


def predict_dataset(
    model: nn.Module,
    sequences: np.ndarray,
    device: str = "cuda",
    batch_size: int = 256,
) -> np.ndarray:
    """Run batched forward passes and return predictions."""

    device = _resolve_device(device)
    model.eval()
    preds = np.zeros((len(sequences), sequences.shape[1]), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            end = min(start + batch_size, len(sequences))
            batch = torch.from_numpy(sequences[start:end]).to(device)
            preds[start:end] = model(batch).cpu().numpy()
    return preds


def train_lstm(
    train_sequences: np.ndarray,
    train_targets: np.ndarray,
    train_masks: np.ndarray,
    val_sequences: np.ndarray,
    val_targets: np.ndarray,
    val_masks: np.ndarray,
    device: str = "cuda",
    batch_size: int = 256,
    epochs: int = 90,
    learning_rate: float = 5e-4,
    patience: int = 8,
    min_delta: float = 5e-5,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
) -> Tuple[PixelLSTM, Dict[str, float]]:
    """Train a GPU-first LSTM model with normalization and regularization."""

    device = _resolve_device(device)
    model = PixelLSTM(
        feature_dim=train_sequences.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )
    criterion = nn.L1Loss(reduction="none")

    train_loader = DataLoader(
        PixelSequenceDataset(train_sequences, train_targets, train_masks),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        PixelSequenceDataset(val_sequences, val_targets, val_masks),
        batch_size=batch_size,
        shuffle=False,
    )

    def _evaluate_loader(loader: DataLoader) -> Dict[str, float]:
        preds: List[np.ndarray] = []
        target_list: List[np.ndarray] = []
        mask_list: List[np.ndarray] = []
        with torch.no_grad():
            for batch_x, batch_y, batch_mask in loader:
                batch_x = batch_x.to(device)
                preds.append(model(batch_x).cpu().numpy())
                target_list.append(batch_y.numpy())
                mask_list.append(batch_mask.numpy())
        pred_arr = np.concatenate(preds)
        target_arr = np.concatenate(target_list)
        mask_arr = np.concatenate(mask_list)
        return _masked_metrics(pred_arr, target_arr, mask_arr)

    best_val_mae = float("inf")
    best_val_r2 = float("nan")
    best_state: Optional[OrderedDict[str, torch.Tensor]] = None
    epochs_without_improve = 0
    best_train_metrics: Dict[str, float] = {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y, batch_mask in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            masked_loss = (loss * batch_mask).sum() / torch.clamp(batch_mask.sum(), min=1.0)
            masked_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += masked_loss.item() * batch_x.size(0)
        epoch_loss /= len(train_loader.dataset)

        model.eval()
        train_metrics = _evaluate_loader(train_loader)
        val_metrics = _evaluate_loader(val_loader)
        val_mae = val_metrics["mae"]
        val_r2 = val_metrics["r2"]
        print(
            "Epoch {}/{}: train_L1={:.4f}, train_MAE={:.4f}, train_RMSE={:.4f}, train_R2={:.4f}, "
            "val_MAE={:.4f}, val_R2={:.4f}".format(
                epoch + 1,
                epochs,
                epoch_loss,
                train_metrics["mae"],
                train_metrics["rmse"],
                train_metrics["r2"],
                val_mae,
                val_r2,
            )
        )

        scheduler.step(val_mae if not np.isnan(val_mae) else 0.0)

        if val_mae < (best_val_mae - min_delta):
            best_val_mae = val_mae
            best_val_r2 = val_r2
            best_state = OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())
            epochs_without_improve = 0
            best_train_metrics = train_metrics
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(
                    f"Early stopping after epoch {epoch + 1}: val_MAE did not improve for {patience} epochs."
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "val_mae": best_val_mae,
        "val_r2": best_val_r2,
        "train_mae": best_train_metrics["mae"],
        "train_rmse": best_train_metrics["rmse"],
        "train_r2": best_train_metrics["r2"],
    }


def hyperparameter_search(
    train_sequences: np.ndarray,
    train_targets: np.ndarray,
    train_masks: np.ndarray,
    val_sequences: np.ndarray,
    val_targets: np.ndarray,
    val_masks: np.ndarray,
    device: str = "cuda",
) -> Tuple[PixelLSTM, Dict[str, float], Dict[str, float]]:
    """Train multiple LSTM configs and pick the best validation R².

    Stores all trial results to ``hyperparameter_results.csv`` for auditing and
    returns the best-performing model, metrics, and configuration.
    """

    configs = [
        {"hidden_size": 160, "num_layers": 2, "dropout": 0.4, "learning_rate": 6e-4, "weight_decay": 2e-4},
    ]

    results: List[Dict[str, float]] = []
    best_model: Optional[PixelLSTM] = None
    best_metrics: Dict[str, float] = {"val_mae": float("inf"), "val_r2": -np.inf}
    best_config: Dict[str, float] = {}

    for idx, cfg in enumerate(configs, start=1):
        print(f"\n=== Hyperparameter set {idx}/{len(configs)}: {cfg} ===")
        model, metrics = train_lstm(
            train_sequences,
            train_targets,
            train_masks,
            val_sequences,
            val_targets,
            val_masks,
            device=device,
            learning_rate=float(cfg["learning_rate"]),
            weight_decay=float(cfg["weight_decay"]),
            hidden_size=int(cfg["hidden_size"]),
            num_layers=int(cfg["num_layers"]),
            dropout=float(cfg["dropout"]),
        )

        cfg_result = {**cfg, "val_mae": metrics.get("val_mae", np.nan), "val_r2": metrics.get("val_r2", np.nan), "train_mae": metrics.get("train_mae", np.nan), "train_rmse": metrics.get("train_rmse", np.nan), "train_r2": metrics.get("train_r2", np.nan)}
        results.append(cfg_result)

        if metrics.get("val_r2", -np.inf) > best_metrics.get("val_r2", -np.inf):
            best_model = model
            best_metrics = metrics
            best_config = cfg

    results_df = pd.DataFrame(results)
    results_csv = OUTPUT_ROOT / "hyperparameter_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"Hyperparameter search results saved to {results_csv}")

    if best_model is None:
        raise RuntimeError("Hyperparameter search failed to produce a model.")

    print(f"Best configuration: {best_config} with val_R2={best_metrics.get('val_r2'):.4f}")
    return best_model, best_metrics, best_config


def predict_full_sequence(
    model: PixelLSTM,
    sequences: np.ndarray,
    years: List[int],
    coords: np.ndarray,
    pixel_mask: np.ndarray,
    geotransform: Tuple,
    projection: str,
    reference_shape: Tuple[int, int],
    device: str = "cuda",
    overwrite_predictions: bool = False,
    output_dir: Optional[Path] = None,
    use_observed: bool = True,
) -> List[Dict[str, float]]:
    """Predict MAVI for all years using the trained LSTM and save rasters.

    Existing prediction rasters are left untouched unless ``overwrite_predictions``
    is ``True`` to avoid producing redundant outputs on reruns.
    """

    device = _resolve_device(device)
    if output_dir is None:
        output_dir = PREDICTED_MAVI_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    n_pixels, timesteps, _ = sequences.shape
    preds = np.zeros((n_pixels, timesteps), dtype=np.float32)
    batch_size = 50000
    with torch.no_grad():
        for start in tqdm(range(0, n_pixels, batch_size), desc="Predicting MAVI sequences"):
            end = min(start + batch_size, n_pixels)
            batch = torch.from_numpy(sequences[start:end]).to(device)
            batch_preds = model(batch).cpu().numpy()
            preds[start:end] = batch_preds

    yearly_metrics: List[Dict[str, float]] = []
    total_pixels = reference_shape[0] * reference_shape[1]
    cols = reference_shape[1]

    for t_idx, year in enumerate(years):
        flat_pred = np.full(total_pixels, np.nan, dtype=np.float32)
        flat_pred[pixel_mask] = preds[:, t_idx]
        predicted_grid = flat_pred.reshape(reference_shape)

        if use_observed:
            observed_grid = _load_array_from_csv(year, "MAVI", reference_shape)
            valid_mask = ~np.isnan(observed_grid)
            if np.any(valid_mask):
                mae = float(mean_absolute_error(observed_grid[valid_mask], predicted_grid[valid_mask]))
                rmse = float(np.sqrt(np.mean((observed_grid[valid_mask] - predicted_grid[valid_mask]) ** 2)))
                r2 = float(r2_score(observed_grid[valid_mask], predicted_grid[valid_mask]))
            else:
                mae = float("nan")
                rmse = float("nan")
                r2 = float("nan")
        else:
            mae = float("nan")
            rmse = float("nan")
            r2 = float("nan")

        seg_metrics: Dict[str, float] = {}
        for idx, (start_col, end_col) in enumerate(SEGMENT_COL_RANGES, start=1):
            seg_mask = (
                (coords[:, 1] >= start_col)
                & (coords[:, 1] < end_col)
            )
            if not np.any(seg_mask):
                seg_metrics.update({
                    f"segment{idx}_mae": float("nan"),
                    f"segment{idx}_rmse": float("nan"),
                    f"segment{idx}_r2": float("nan"),
                })
                continue
            if not use_observed:
                seg_mae = float("nan")
                seg_rmse = float("nan")
                seg_r2 = float("nan")
            else:
                obs_flat = _load_array_from_csv(year, "MAVI", reference_shape).reshape(-1)
                obs_vals = obs_flat[coords[seg_mask, 0] * cols + coords[seg_mask, 1]]
                pred_vals = preds[seg_mask, t_idx]
                valid_seg = ~np.isnan(obs_vals)
                if np.any(valid_seg):
                    seg_mae = float(mean_absolute_error(obs_vals[valid_seg], pred_vals[valid_seg]))
                    seg_rmse = float(np.sqrt(np.mean((obs_vals[valid_seg] - pred_vals[valid_seg]) ** 2)))
                    seg_r2 = float(r2_score(obs_vals[valid_seg], pred_vals[valid_seg]))
                else:
                    seg_mae = float("nan")
                    seg_rmse = float("nan")
                    seg_r2 = float("nan")
            seg_metrics.update({
                f"segment{idx}_mae": seg_mae,
                f"segment{idx}_rmse": seg_rmse,
                f"segment{idx}_r2": seg_r2,
            })

        yearly_metrics.append({"year": year, "mae": mae, "rmse": rmse, "r2": r2, **seg_metrics})

        out_path = output_dir / f"{year}_predicted_MAVI.tif"
        if overwrite_predictions or not out_path.exists():
            write_raster(out_path, predicted_grid, geotransform, projection)
            print(
                f"Saved predicted MAVI for {year} to {out_path} (MAE={mae:.4f}, R2={r2:.4f})"
            )
        else:
            print(
                f"Prediction already exists for {year}, skipping write: {out_path} "
                f"(MAE={mae:.4f}, R2={r2:.4f})"
            )

    return yearly_metrics


def main() -> None:
    pre_years = list(range(1988, 2003))
    post_years = list(range(2003, 2021))
    all_years = pre_years + post_years

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PREDICTED_MAVI_DIR.mkdir(parents=True, exist_ok=True)

    first_year_data, geotransform, projection = gather_year_data(pre_years[0])
    reference_shape = first_year_data["MAVI"].shape

    export_features_to_csv(all_years)

    feature_keys = ["IF", "ID", "IH", "ELE", *CLIMATE_VARIABLES]

    sequences, targets, masks, pixel_mask, coords = build_full_sequences(
        all_years, feature_keys, reference_shape
    )

    # Use the provided feature subset directly to mirror the reference version.
    selected_features = ["ID", "IF", "ELE", "IH", "TEM", "AGB"]
    print(f"Using fixed feature set: {selected_features}")

    selected_indices = [feature_keys.index(f) for f in selected_features]
    sequences = sequences[:, :, selected_indices]
    feature_keys = selected_features

    feature_means, feature_stds = compute_feature_stats(sequences, masks)
    sequences_norm = normalize_features(sequences, feature_means, feature_stds)

    n_samples = len(sequences)
    permutation = np.random.permutation(n_samples)
    split_idx = int(0.8 * n_samples)
    train_idx, val_idx = permutation[:split_idx], permutation[split_idx:]

    model, training_metrics, best_config = hyperparameter_search(
        sequences_norm[train_idx],
        targets[train_idx],
        masks[train_idx],
        sequences_norm[val_idx],
        targets[val_idx],
        masks[val_idx],
        device="cuda",
    )

    # Evaluate per-segment metrics on train/val splits
    train_preds = predict_dataset(model, sequences_norm[train_idx], device="cuda")
    val_preds = predict_dataset(model, sequences_norm[val_idx], device="cuda")
    train_metrics_segments = compute_overall_and_segment_metrics(
        train_preds, targets[train_idx], masks[train_idx], coords[train_idx], SEGMENT_COL_RANGES
    )
    val_metrics_segments = compute_overall_and_segment_metrics(
        val_preds, targets[val_idx], masks[val_idx], coords[val_idx], SEGMENT_COL_RANGES
    )

    segment_metrics_df = pd.DataFrame([
        {"dataset": "train", **train_metrics_segments},
        {"dataset": "val", **val_metrics_segments},
    ])
    segment_metrics_csv = OUTPUT_ROOT / "segment_metrics.csv"
    segment_metrics_df.to_csv(segment_metrics_csv, index=False)
    print(f"Segmented train/val metrics saved to {segment_metrics_csv}")

    torch.save({
        "state_dict": model.state_dict(),
        "feature_keys": feature_keys,
        "feature_means": feature_means.tolist(),
        "feature_stds": feature_stds.tolist(),
        "training_mae": training_metrics.get("train_mae"),
        "training_rmse": training_metrics.get("train_rmse"),
        "training_r2": training_metrics.get("train_r2"),
        "validation_mae": training_metrics.get("val_mae"),
        "validation_r2": training_metrics.get("val_r2"),
        "best_config": best_config,
    }, MODEL_PATH)
    print(f"LSTM model saved to {MODEL_PATH}")

    print("\n=== Predicting full MAVI sequence with trained LSTM ===")
    yearly_metrics = predict_full_sequence(
        model,
        sequences_norm,
        all_years,
        coords,
        pixel_mask,
        geotransform,
        projection,
        reference_shape,
        device="cuda",
    )

    yearly_metrics_df = pd.DataFrame(yearly_metrics)
    yearly_metrics_csv = OUTPUT_ROOT / "yearly_metrics.csv"
    yearly_metrics_df.to_csv(yearly_metrics_csv, index=False)
    print(f"All per-year metrics saved to {yearly_metrics_csv}")

    mean_mae = float(np.nanmean([m.get("mae") for m in yearly_metrics]))
    mean_rmse = float(np.nanmean([m.get("rmse") for m in yearly_metrics]))
    mean_r2 = float(np.nanmean([m.get("r2") for m in yearly_metrics]))
    summary_df = pd.DataFrame(
        [
            {
                "mean_mae": mean_mae,
                "mean_rmse": mean_rmse,
                "mean_r2": mean_r2,
                "training_mae": training_metrics.get("train_mae"),
                "training_rmse": training_metrics.get("train_rmse"),
                "training_r2": training_metrics.get("train_r2"),
                "validation_mae": training_metrics.get("val_mae"),
                "validation_r2": training_metrics.get("val_r2"),
            }
        ]
    )
    summary_csv = OUTPUT_ROOT / "summary_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary metrics saved to {summary_csv}")

def main2() -> None:
    """Simulate post-2003 MAVI under alternate hydrologic scenarios."""

    post_years = list(range(2003, 2021))
    all_years = list(range(1988, 2021))

    scenario_configs = [
        {
            "name": "H1_case1",
            "inun_root": r"G:/A_Veg_Model/TGP_current_influence/H1",
        },
        {
            "name": "H1_case2",
            "inun_root": r"G:/A_Veg_Model/TGP_current_influence/H1",
        },
    ]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    first_year_data, geotransform, projection = gather_year_data(all_years[0])
    reference_shape = first_year_data["MAVI"].shape

    # Ensure base CSVs exist so coordinates align with the trained model.
    export_features_to_csv(all_years)
    coords, pixel_mask = collect_valid_coords(all_years, reference_shape)

    # Load trained model and normalization stats.
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    feature_keys: List[str] = checkpoint["feature_keys"]
    feature_means = np.array(checkpoint["feature_means"], dtype=np.float32)
    feature_stds = np.array(checkpoint["feature_stds"], dtype=np.float32)
    best_config = checkpoint.get("best_config", {})
    hidden_size = int(best_config.get("hidden_size", 192))
    num_layers = int(best_config.get("num_layers", 2))
    dropout = float(best_config.get("dropout", 0.4))

    device = _resolve_device("cuda")
    model = PixelLSTM(
        feature_dim=len(feature_keys),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    for scenario in scenario_configs:
        name = scenario["name"]
        inun_root = scenario["inun_root"]
        duration_template = fr"{inun_root}/{{year}}/inun_duration.tif"
        mean_wl_template = fr"{inun_root}/{{year}}/inun_mean_wl.tif"

        print(f"\n=== Running scenario {name} ===")
        seqs = build_sequences_for_coords(
            post_years,
            coords,
            feature_keys,
            reference_shape,
            duration_template,
            mean_wl_template,
            climate_root=CLIMATE_ROOT,
        )
        seqs_norm = normalize_features(seqs, feature_means, feature_stds)

        scenario_output_dir = OUTPUT_ROOT / f"predicted_mavi_{name}"
        yearly_metrics = predict_full_sequence(
            model,
            seqs_norm,
            post_years,
            coords,
            pixel_mask,
            geotransform,
            projection,
            reference_shape,
            device=device,
            output_dir=scenario_output_dir,
            use_observed=False,
        )
        for m in yearly_metrics:
            m["scenario"] = name

        yearly_metrics_df = pd.DataFrame(yearly_metrics)
        yearly_metrics_csv = OUTPUT_ROOT / f"yearly_metrics_{name}.csv"
        yearly_metrics_df.to_csv(yearly_metrics_csv, index=False)
        print(f"Scenario {name} per-year metrics saved to {yearly_metrics_csv}")

        mean_mae = float(np.nanmean([m.get("mae") for m in yearly_metrics]))
        mean_rmse = float(np.nanmean([m.get("rmse") for m in yearly_metrics]))
        mean_r2 = float(np.nanmean([m.get("r2") for m in yearly_metrics]))
        summary_df = pd.DataFrame(
            [
                {
                    "scenario": name,
                    "mean_mae": mean_mae,
                    "mean_rmse": mean_rmse,
                    "mean_r2": mean_r2,
                }
            ]
        )
        summary_csv = OUTPUT_ROOT / f"summary_metrics_{name}.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"Scenario {name} summary metrics saved to {summary_csv}")
if __name__ == "__main__":
    main2()