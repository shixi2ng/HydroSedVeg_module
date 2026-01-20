"""
Generate Above-Ground Biomass Density (AGBD) rasters using vegetation community
classification rasters and MAVI rasters.

For each year, this script pairs a vegetation type GeoTIFF with the
corresponding MAVI GeoTIFF (matching a 4-digit year in the filename), then
applies vegetation-specific AGBD formulas:

AGBD_F = 0.13 * exp(3.62 * MAVI_F + 1.67) - 0.13 * exp(1.67)
AGBD_A = 0.07 * exp(4.66 * MAVI_A + 1.12) - 0.07 * exp(1.12)
AGBD_H = 0.16 * exp(1.51 * MAVI_H + 2.92) - 0.16 * exp(2.92)
AGBD_E = (AGBD_H + AGBD_F) / 2

Vegetation type 0 is always set to 0. Types 1–4 use the above formulas based on
the fixed mapping below (defaults: 1->H, 2->E, 3->F, 4->A).

Paths requested for batch processing (edit the constants below if needed):
VEG_DIR = r\"G:\\A_Veg_Model\\TGP_current_influence\\S1\\tif\"
MAVI_DIR = r\"G:\\A_Veg_Model\\TGP_current_influence\\A1\\predicted_mavi_H1_case1\"
OUTPUT_DIR = r\"G:\\A_Veg_Model\\TGP_current_influence\\A1\\AGBD\"
"""
import os
import re
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from osgeo import gdal


AGBDFormulas = Dict[int, Callable[[np.ndarray], np.ndarray]]


def agbd_f(mavi: np.ndarray) -> np.ndarray:
    return 0.13 * np.exp(3.62 * mavi + 1.67) - 0.13 * np.exp(1.67)


def agbd_a(mavi: np.ndarray) -> np.ndarray:
    return 0.07 * np.exp(4.66 * mavi + 1.12) - 0.07 * np.exp(1.12)


def agbd_h(mavi: np.ndarray) -> np.ndarray:
    return 0.16 * np.exp(1.51 * mavi + 2.92) - 0.16 * np.exp(2.92)


def agbd_e(mavi: np.ndarray) -> np.ndarray:
    return (0.16 * np.exp(1.51 * mavi + 2.92) - 0.16 * np.exp(2.92) + 0.13 * np.exp(3.62 * mavi + 1.67) - 0.13 * np.exp(1.67)) / 2


FORMULA_ALIAS = {
    "F": agbd_f,
    "A": agbd_a,
    "H": agbd_h,
    "E": agbd_e,
}


def parse_type_mapping(raw_mapping: Optional[list]) -> AGBDFormulas:
    """Parse vegetation-type mapping strings like ["1:F", "2:A", "3:H", "4:H"]."""
    if not raw_mapping:
        return {1: agbd_h, 2: agbd_e, 3: agbd_f, 4: agbd_a}

    mapping: AGBDFormulas = {}
    for item in raw_mapping:
        if ":" not in item:
            raise ValueError(f"Invalid mapping '{item}'. Use the form type:code (e.g., 1:F)")
        key_str, formula_key = item.split(":", 1)
        veg_type = int(key_str)
        code = formula_key.strip().upper()
        if code not in FORMULA_ALIAS:
            raise ValueError(f"Unknown formula code '{code}' in mapping '{item}'. Use F, A, H, or E.")
        mapping[veg_type] = FORMULA_ALIAS[code]

    for required in (1, 2, 3, 4):
        if required not in mapping:
            raise ValueError("Please provide mappings for vegetation types 1, 2, 3, and 4.")

    return mapping


def extract_year_from_name(name: str) -> Optional[str]:
    match = re.search(r"(19|20)\d{2}", name)
    return match.group(0) if match else None


def read_single_band(path: str) -> Tuple[np.ndarray, gdal.Dataset, Optional[float]]:
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f"Unable to open raster: {path}")
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(np.float32)
    nodata = band.GetNoDataValue()
    return arr, ds, nodata


def write_raster(reference: gdal.Dataset, data: np.ndarray, output_path: str, nodata_value: float) -> None:
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        output_path,
        reference.RasterXSize,
        reference.RasterYSize,
        1,
        gdal.GDT_Float32,
        ["COMPRESS=LZW"],
    )
    out_ds.SetGeoTransform(reference.GetGeoTransform())
    out_ds.SetProjection(reference.GetProjection())

    band = out_ds.GetRasterBand(1)
    band.SetNoDataValue(nodata_value)
    band.WriteArray(data)
    band.FlushCache()
    out_ds = None


def compute_agbd(
    veg_arr: np.ndarray,
    mavi_arr: np.ndarray,
    mapping: AGBDFormulas,
    veg_nodata: Optional[float],
    mavi_nodata: Optional[float],
    nodata_value: float,
) -> np.ndarray:
    if veg_arr.shape != mavi_arr.shape:
        raise ValueError("Vegetation type raster and MAVI raster must have the same shape.")

    invalid_mask = np.zeros_like(veg_arr, dtype=bool)
    if veg_nodata is not None:
        invalid_mask |= veg_arr == veg_nodata
    if mavi_nodata is not None:
        invalid_mask |= mavi_arr == mavi_nodata

    agbd = np.full(veg_arr.shape, np.nan, dtype=np.float32)
    agbd[invalid_mask] = np.nan

    zero_mask = (veg_arr == 0) & ~invalid_mask
    agbd[zero_mask] = 0.0

    for veg_type, func in mapping.items():
        type_mask = (veg_arr == veg_type) & ~invalid_mask
        if np.any(type_mask):
            agbd[type_mask] = func(mavi_arr[type_mask])

    agbd_filled = np.where(np.isnan(agbd), nodata_value, agbd)
    return agbd_filled.astype(np.float32)


def process_year(
    veg_path: str,
    mavi_path: str,
    output_dir: str,
    year: str,
    mapping: AGBDFormulas,
    nodata_value: float,
) -> str:
    veg_arr, veg_ds, veg_nodata = read_single_band(veg_path)
    mavi_arr, mavi_ds, mavi_nodata = read_single_band(mavi_path)

    agbd = compute_agbd(veg_arr, mavi_arr, mapping, veg_nodata, mavi_nodata, nodata_value)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"AGBD_{year}.tif")
    write_raster(veg_ds, agbd, output_path, nodata_value)
    print(f"Saved AGBD raster for {year} -> {output_path}")
    return output_path


def find_yearly_pairs(veg_dir: str, mavi_dir: str) -> Dict[str, Tuple[str, str]]:
    pairs: Dict[str, Tuple[str, str]] = {}
    mavi_by_year: Dict[str, str] = {}

    for fname in os.listdir(mavi_dir):
        if fname.lower().endswith(".tif"):
            year = extract_year_from_name(fname)
            if year:
                mavi_by_year[year] = os.path.join(mavi_dir, fname)

    for fname in os.listdir(veg_dir):
        if fname.lower().endswith(".tif"):
            year = extract_year_from_name(fname)
            if year and year in mavi_by_year:
                pairs[year] = (os.path.join(veg_dir, fname), mavi_by_year[year])
            elif year:
                print(f"Warning: No MAVI file found for vegetation year {year}, skipping.")
    return pairs


def main() -> None:
    veg_dir = r"G:\A_Veg_Model\TGP_future_influence\S4\tif"
    mavi_dir = r"G:\A_Veg_Model\TGP_future_influence\A0\predicted_mavi_H1_case2"
    output_dir = r"G:\A_Veg_Model\TGP_future_influence\A4\AGBD"
    nodata_value = -9999.0

    if not os.path.isdir(veg_dir):
        raise NotADirectoryError(f"Vegetation directory not found: {veg_dir}")
    if not os.path.isdir(mavi_dir):
        raise NotADirectoryError(f"MAVI directory not found: {mavi_dir}")

    mapping = {1: agbd_h, 2: agbd_e, 3: agbd_f, 4: agbd_a}
    yearly_pairs = find_yearly_pairs(veg_dir, mavi_dir)

    if not yearly_pairs:
        raise RuntimeError("No matching vegetation/MAVI pairs were found. Check filenames and directories.")

    for year, (veg_path, mavi_path) in sorted(yearly_pairs.items()):
        process_year(veg_path, mavi_path, output_dir, year, mapping, nodata_value)


if __name__ == "__main__":
    main()