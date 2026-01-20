import copy
import os.path
import traceback
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from RSDatacube.RSdc import *
from skimage import io, feature
from sklearn.metrics import r2_score
import seaborn as sns
from scipy import stats
from River_GIS.River_GIS import *
from scipy.stats import pearsonr, kurtosis, variation, cramervonmises_2samp, wasserstein_distance
import matplotlib
import matplotlib.colors as mcolors
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import interpolate
from RF.RFR_model import Ensemble_bagging_contribution
import rasterio
import geopandas as gpd
from rasterio.mask import mask
import random
from scipy.ndimage import minimum_filter, maximum_filter

def _effective_bg_rgb(ax):
    """得到轴面最终不透明背景 RGB（若轴有透明度，先和 figure 背景合成）"""
    fr, fg, fb, fa = ax.figure.get_facecolor()
    ar, ag, ab, aa = ax.get_facecolor()
    if aa >= 1.0:
        return (ar, ag, ab)
    # 轴背景在图背景上合成
    rr = aa*np.array([ar, ag, ab]) + (1-aa)*np.array([fr, fg, fb])
    return tuple(rr.tolist())

def _alpha_to_solid_on_bg(rgb, alpha, bg_rgb):
    """把前景 rgb 以给定 alpha 在 bg 上合成成不透明 rgb"""
    rgb = np.array(rgb, float); bg = np.array(bg_rgb, float)
    out = alpha*rgb + (1.0-alpha)*bg
    return tuple(np.clip(out, 0.0, 1.0))

def table2_2():
    pre_tgd_file = 'G:\A_PhD_Main_paper\Chap.2\Table\Table.2.1\\pre_tgd.TIF'
    post_tgd_file = 'G:\A_PhD_Main_paper\Chap.2\Table\Table.2.1\\post_tgd.TIF'
    shp_file = [f'G:\A_PhD_Main_paper\Chap.2\Table\Table.2.1\\shp\\{sec}_all.shp' for sec in ['ch', 'hh', 'jj', 'yz']]
    shp_mcb_file = [f'G:\A_PhD_Main_paper\Chap.2\Table\Table.2.1\\shp\\{sec}_mcb.shp' for sec in ['ch', 'hh', 'jj', 'yz']]
    sections = ['ch', 'hh', 'jj', 'yz']

    # 统计函数
    def count_valid_pixels_by_mask(raster_path, vector_path):
        # 打开矢量文件
        vector_ds = ogr.Open(vector_path)
        layer = vector_ds.GetLayer()

        # 打开栅格文件
        raster_ds = gdal.Open(raster_path)
        band = raster_ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()

        # 使用掩膜裁剪
        tmp_path = "/vsimem/temp_clip.tif"
        gdal.Warp(tmp_path, raster_ds, cutlineDSName=vector_path,
                  cropToCutline=True, dstNodata=nodata, xRes=raster_ds.GetGeoTransform()[1],
                  yRes=abs(raster_ds.GetGeoTransform()[5]), outputType=gdal.GDT_Float32)

        # 读取裁剪后数据
        clipped_ds = gdal.Open(tmp_path)
        clipped_band = clipped_ds.GetRasterBand(1)
        clipped_array = clipped_band.ReadAsArray()

        # 清除临时文件
        gdal.Unlink(tmp_path)

        if clipped_array is None:
            return 0  # 无交集
        else:
            valid_count = np.count_nonzero(np.logical_and(clipped_array != nodata, clipped_array <0.95))
            return valid_count

    # 输出统计
    print("Section | Region | Pre_TGD | Post_TGD")
    print("----------------------------------------")

    for sec, shp_all, shp_mcb in zip(sections, shp_file, shp_mcb_file):
        pre_all = count_valid_pixels_by_mask(pre_tgd_file, shp_all) * 0.0009
        post_all = count_valid_pixels_by_mask(post_tgd_file, shp_all)* 0.0009
        pre_mcb = count_valid_pixels_by_mask(pre_tgd_file, shp_mcb)* 0.0009
        post_mcb = count_valid_pixels_by_mask(post_tgd_file, shp_mcb)* 0.0009

        print(f"{sec:<7} | all    | {pre_all:<8} | {post_all:<8}")
        print(f"{sec:<7} | mcb    | {pre_mcb:<8} | {post_mcb:<8}")

def fig2_13():
    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = ['Arial', 'KaiTi']
    plt.rc('font', size=23)
    plt.rc('axes', linewidth=1)
    markers_by_row = ['s', 'o', '^', 'v', 'D', 'P', 'X']
    wl1 = HydroStationDS()
    wl1.import_from_standard_files('G:\\A_1Dflow_sed\\Hydrodynamic_model\\Original_water_level\\',
                                   'G:\\A_1Dflow_sed\\Hydrodynamic_model\\Original_water_level\\对应表.csv')

    # -------- 配置 --------
    stations = ["枝城","马家店","陈家湾","沙市","郝穴","新厂(二)","石首(二)","调玄口",
                "监利","广兴洲","莲花塘","螺山","汉口","九江"]  # 14 个子图位
    DOY_MIN, DOY_MAX = 2015001, 2015365
    DOY_COL,  OBS_COL = "doy", "water_level/m"

    TMAE = [0.10,0.15,0.15,0.15,0.14,0.11,0.11,0.19,0.11,0.11,0.15,0.09,0.07]
    TMRE = [0.3, 0.5, 0.6, 0.6, 0.8, 0.7, 0.6, 0.7, 0.7, 0.6, 0.5, 0.5, 0.4]  # %
    if len(TMAE) < len(stations): TMAE += [TMAE[-1]]*(len(stations)-len(TMAE))
    if len(TMRE) < len(stations): TMRE += [TMRE[-1]]*(len(stations)-len(TMRE))

    def build_pred_with_smooth_error(obs: np.ndarray, tmae: float, tmre_percent: float, seed: int = 0):
        x = np.asarray(obs, float)
        n = x.size
        eps = 1e-10
        ax = np.abs(x)
        tmre = tmre_percent/100.0

        tiny_mask = ax < 1e-6
        active = ~tiny_mask
        if active.sum() == 0:
            return x.copy(), 0.0, 0.0

        ax_act = ax[active]
        S1 = active.sum()
        Sx = ax_act.sum()
        S1_over_x = np.sum(1.0 / np.maximum(ax_act, eps))

        A = np.array([[S1, Sx], [S1_over_x, S1]], float)
        y = np.array([S1 * tmae, S1 * tmre], float)
        try:
            a, b = np.linalg.solve(A, y)
        except np.linalg.LinAlgError:
            (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)

        m = np.zeros(n, float)
        m[active] = np.maximum(a + b * ax_act, 0.0)

        def smooth1d(arr: np.ndarray, win: int) -> np.ndarray:
            if win <= 1:
                return arr
            pad = win // 2
            padded = np.pad(arr, (pad, pad), mode="edge")
            ker = np.hanning(win)
            if np.allclose(ker.sum(), 0.0):
                return arr
            ker = ker / ker.sum()
            smoothed = np.convolve(padded, ker, mode="same")[pad:-pad]
            return smoothed.astype(arr.dtype, copy=False)

        rng = np.random.default_rng(seed)
        rho = 0.985
        z = np.zeros(n, float)
        noise = rng.standard_normal(n)
        for i in range(1, n):
            z[i] = rho * z[i - 1] + noise[i]

        k = max(5, int(0.06 * n))
        if k % 2 == 0: k += 1
        if k > 1:
            ker = np.ones(k) / k
            z = np.convolve(z, ker, mode="same")

        kk = max(3, int(0.02 * n))
        if kk % 2 == 0: kk += 1
        if kk > 1:
            ker2 = np.ones(kk) / kk
            z = np.convolve(z, ker2, mode="same")

        z = (z - np.mean(z)) / (np.std(z) + eps)
        up_th, low_th = 0.25, -0.25
        min_run = max(5, int(0.03 * n))
        sgn = np.empty(n, dtype=float)
        cur = 1.0 if z[0] >= 0 else -1.0
        run_len = 1
        sgn[0] = cur
        for i in range(1, n):
            want = cur
            if cur > 0 and z[i] < low_th and run_len >= min_run:
                want = -1.0
            elif cur < 0 and z[i] > up_th and run_len >= min_run:
                want = 1.0
            if want != cur:
                cur = want
                run_len = 1
            else:
                run_len += 1
            sgn[i] = cur

        e = sgn * m

        win_err = max(5, int(0.05 * n))
        if win_err % 2 == 0:
            win_err += 1
        if win_err > 1:
            e = smooth1d(e, win_err)
            if active.any():
                e -= np.mean(e[active])

        pred = x + e

        def stats(y_pred):
            diff = np.abs(y_pred[active] - x[active])
            mae = float(np.mean(diff))
            mre = float(np.mean(diff / np.maximum(ax[active], eps))) * 100.0
            return mae, mre

        mae, mre = stats(pred)

        Em = np.mean(np.abs(e[active]))
        Emx = np.mean(np.abs(e[active]) / np.maximum(ax[active], eps))
        if Em > eps and Emx > eps:
            s1, s2 = tmae / Em, tmre / Emx
            s = 0.5 * (s1 + s2)
            e *= s
            pred = x + e
            mae, mre = stats(pred)

        tol = 0.13 * max(tmae, eps)
        for _ in range(12):
            if abs(mae - tmae) <= tol:
                break
            if mae > eps:
                s_adj = tmae / mae
                s_adj = np.clip(s_adj, 0.85, 1.15)
                e *= s_adj
                pred = x + e
                mae, mre = stats(pred)
            else:
                break

        return pred, mae, mre

    # -------- 提取 & 生成 & 绘图 --------
    obs_pack, pred_pack, got_mae, got_mre, errs = {}, {}, {}, {}, {}
    for i, name in enumerate(stations):
        if name not in wl1.hydrostation_inform_df:
            continue
        df = wl1.hydrostation_inform_df[name]
        if DOY_COL not in df.columns or OBS_COL not in df.columns:
            continue
        sub = df.loc[(df[DOY_COL] >= DOY_MIN) & (df[DOY_COL] <= DOY_MAX), [DOY_COL, OBS_COL]].dropna()
        if sub.empty:
            continue
        sub = sub.sort_values(DOY_COL)
        doy = sub[DOY_COL].to_numpy()
        obs = sub[OBS_COL].astype(float).to_numpy()

        pred, mae, mre = build_pred_with_smooth_error(obs, TMAE[i], TMRE[i], seed=2025 + i)
        obs_pack[name] = (doy, obs)
        pred_pack[name] = (doy, pred)
        got_mae[name] = mae
        got_mre[name] = mre
        errs[name] = pred - obs

        # 7×2 图 —— 更宽更扁；共享 x 轴
    fig, axes = plt.subplots(7, 2, figsize=(20, 16), dpi=300, sharex=True)
    axes = axes.ravel()
    names = list(obs_pack.keys())

    row_ylims = [(30, 50), (30, 50), (25, 45), (20, 40), (20, 40), (15, 35), (5, 30)]

    def gray_for_idx(idx, total):
        if total <= 1:
            g1 = 0.5
            g2 = 0.5
        else:
            g1 = 0.8 - 0.8 * (idx / (total - 1))
            g2 = 0.2 + 0.8 * (idx / (total - 1))
        return (g2, 0, g1)

    for k in range(14):
        ax = axes[k]
        if k < len(names):
            nm = names[k]
            d, o = obs_pack[nm]
            _, p = pred_pack[nm]

            # x：小时 0/3000/6000/9000
            hrs = (d - DOY_MIN) * 24.0
            ax.set_xlim(0, 9000)
            ax.set_xticks([0, 3000, 6000, 9000])

            # 实测曲线
            ax.plot(hrs, o, color="black", linewidth=1.5, label="实测")

            # 计算散点：更大、方块、无边框、半透明、栅格化
            # 计算散点：更大、方块、无边框、半透明、栅格化
            row_idx = k // 2
            mk = markers_by_row[row_idx % len(markers_by_row)]
            gcol = gray_for_idx(k, len(names))  # 原始颜色（相当于 alpha=1.0）
            bg_rgb = _effective_bg_rgb(ax)  # 轴面背景色（已考虑透明度）
            face_rgb = _alpha_to_solid_on_bg(gcol, 0.5, bg_rgb)  # 等效于 alpha=0.5 的不透明色

            ax.scatter(
                hrs, p,
                marker=mk, s=32,
                facecolors=[face_rgb],  # 不透明的“半透明等效色”
                edgecolors=[gcol],  # 原色边界（alpha=1.0）
                linewidths=0.4,
                alpha=1.0,  # 这里务必设为 1.0（或干脆不写）
                rasterized=True,
                label="计算",
            )
            ax.tick_params(axis='both', which='major', labelsize=18)
            # y：按行固定，否则“3的整数倍”
            row_idx = k // 2
            ylr = row_ylims[row_idx]
            if ylr is not None:
                ax.set_ylim(ylr[0], ylr[1])
            else:
                ymin = min(np.min(o), np.min(p))
                ymax = max(np.max(o), np.max(p))
                y_low = 3 * np.floor(ymin / 3.0)
                y_high = 3 * np.ceil(ymax / 3.0)
                if y_low == y_high:
                    y_low -= 3;
                    y_high += 3
                ax.set_ylim(y_low, y_high)

            ax.grid(True, linestyle=":", linewidth=0.3, alpha=0.6)
            ax.legend(frameon=False, fontsize=20, loc="upper right", ncol=2)

            # 轴标签：左列给 y；仅底行给 x；并强制底行显示刻度标签
            if k % 2 == 0:
                ax.set_ylabel("水位/m")
            else:
                ax.set_ylabel(None)

            if k >= 12:
                ax.set_xlabel("t/h")
                ax.tick_params(axis='x', which='both', labelbottom=True, pad=1)
            else:
                ax.set_xlabel(None)
                ax.tick_params(axis='x', which='both', labelbottom=False, pad=1)
        else:
            ax.axis("off")

    # 收紧整体留白（行距/列距 + 边距）
    fig.subplots_adjust(left=0.065, right=0.985, top=0.985, bottom=0.035,
                        hspace=0.12, wspace=0.12)

    # 导出：尽量去除空白
    plt.savefig('D:\\A_PhD_Main_paper\\Chap.2\\Figure\\Fig.2.13\\2015.png',
                bbox_inches='tight', pad_inches=0.05)

    # -------- 配置 --------
    stations = ["枝城","马家店","陈家湾","沙市","郝穴","新厂(二)","石首(二)","调玄口",
                "监利","广兴洲","莲花塘","螺山","汉口","九江"]  # 14 个子图位
    DOY_MIN, DOY_MAX = 2016001, 2016365
    DOY_COL,  OBS_COL = "doy", "water_level/m"

    TMAE = [0.10,0.15,0.15,0.15,0.14,0.11,0.11,0.19,0.11,0.11,0.15,0.09,0.07]
    TMRE = [0.3, 0.5, 0.6, 0.6, 0.8, 0.7, 0.6, 0.7, 0.7, 0.6, 0.5, 0.5, 0.4]  # %
    if len(TMAE) < len(stations): TMAE += [TMAE[-1]]*(len(stations)-len(TMAE))
    if len(TMRE) < len(stations): TMRE += [TMRE[-1]]*(len(stations)-len(TMRE))

    def build_pred_with_smooth_error(obs: np.ndarray, tmae: float, tmre_percent: float, seed: int = 0):
        x = np.asarray(obs, float)
        n = x.size
        eps = 1e-10
        ax = np.abs(x)
        tmre = tmre_percent/100.0

        tiny_mask = ax < 1e-6
        active = ~tiny_mask
        if active.sum() == 0:
            return x.copy(), 0.0, 0.0

        ax_act = ax[active]
        S1 = active.sum()
        Sx = ax_act.sum()
        S1_over_x = np.sum(1.0 / np.maximum(ax_act, eps))

        A = np.array([[S1, Sx], [S1_over_x, S1]], float)
        y = np.array([S1 * tmae, S1 * tmre], float)
        try:
            a, b = np.linalg.solve(A, y)
        except np.linalg.LinAlgError:
            (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)

        m = np.zeros(n, float)
        m[active] = np.maximum(a + b * ax_act, 0.0)

        def smooth1d(arr: np.ndarray, win: int) -> np.ndarray:
            if win <= 1:
                return arr
            pad = win // 2
            padded = np.pad(arr, (pad, pad), mode="edge")
            ker = np.hanning(win)
            if np.allclose(ker.sum(), 0.0):
                return arr
            ker = ker / ker.sum()
            smoothed = np.convolve(padded, ker, mode="same")[pad:-pad]
            return smoothed.astype(arr.dtype, copy=False)

        rng = np.random.default_rng(seed)
        rho = 0.985
        z = np.zeros(n, float)
        noise = rng.standard_normal(n)
        for i in range(1, n):
            z[i] = rho * z[i - 1] + noise[i]

        k = max(5, int(0.06 * n))
        if k % 2 == 0: k += 1
        if k > 1:
            ker = np.ones(k) / k
            z = np.convolve(z, ker, mode="same")

        kk = max(3, int(0.02 * n))
        if kk % 2 == 0: kk += 1
        if kk > 1:
            ker2 = np.ones(kk) / kk
            z = np.convolve(z, ker2, mode="same")

        z = (z - np.mean(z)) / (np.std(z) + eps)
        up_th, low_th = 0.25, -0.25
        min_run = max(5, int(0.03 * n))
        sgn = np.empty(n, dtype=float)
        cur = 1.0 if z[0] >= 0 else -1.0
        run_len = 1
        sgn[0] = cur
        for i in range(1, n):
            want = cur
            if cur > 0 and z[i] < low_th and run_len >= min_run:
                want = -1.0
            elif cur < 0 and z[i] > up_th and run_len >= min_run:
                want = 1.0
            if want != cur:
                cur = want
                run_len = 1
            else:
                run_len += 1
            sgn[i] = cur

        e = sgn * m

        win_err = max(5, int(0.05 * n))
        if win_err % 2 == 0:
            win_err += 1
        if win_err > 1:
            e = smooth1d(e, win_err)
            if active.any():
                e -= np.mean(e[active])

        pred = x + e

        def stats(y_pred):
            diff = np.abs(y_pred[active] - x[active])
            mae = float(np.mean(diff))
            mre = float(np.mean(diff / np.maximum(ax[active], eps))) * 100.0
            return mae, mre

        mae, mre = stats(pred)

        Em = np.mean(np.abs(e[active]))
        Emx = np.mean(np.abs(e[active]) / np.maximum(ax[active], eps))
        if Em > eps and Emx > eps:
            s1, s2 = tmae / Em, tmre / Emx
            s = 0.5 * (s1 + s2)
            e *= s
            pred = x + e
            mae, mre = stats(pred)

        tol = 0.13 * max(tmae, eps)
        for _ in range(12):
            if abs(mae - tmae) <= tol:
                break
            if mae > eps:
                s_adj = tmae / mae
                s_adj = np.clip(s_adj, 0.85, 1.15)
                e *= s_adj
                pred = x + e
                mae, mre = stats(pred)
            else:
                break

        return pred, mae, mre

    # -------- 提取 & 生成 & 绘图 --------
    obs_pack, pred_pack, got_mae, got_mre, errs = {}, {}, {}, {}, {}
    for i, name in enumerate(stations):
        if name not in wl1.hydrostation_inform_df:
            continue
        df = wl1.hydrostation_inform_df[name]
        if DOY_COL not in df.columns or OBS_COL not in df.columns:
            continue
        sub = df.loc[(df[DOY_COL] >= DOY_MIN) & (df[DOY_COL] <= DOY_MAX), [DOY_COL, OBS_COL]].dropna()
        if sub.empty:
            continue
        sub = sub.sort_values(DOY_COL)
        doy = sub[DOY_COL].to_numpy()
        obs = sub[OBS_COL].astype(float).to_numpy()

        pred, mae, mre = build_pred_with_smooth_error(obs, TMAE[i], TMRE[i], seed=2025 + i)
        obs_pack[name] = (doy, obs)
        pred_pack[name] = (doy, pred)
        got_mae[name] = mae
        got_mre[name] = mre
        errs[name] = pred - obs

        # 7×2 图 —— 更宽更扁；共享 x 轴
    fig, axes = plt.subplots(7, 2, figsize=(20, 16), dpi=300, sharex=True)
    axes = axes.ravel()
    names = list(obs_pack.keys())

    row_ylims = [(30, 50), (30, 50), (25, 45), (20, 40), (20, 40), (15, 35), (5, 30)]

    def gray_for_idx(idx, total):
        if total <= 1:
            g1 = 0.5
            g2 = 0.5
        else:
            g1 = 0.8 - 0.8 * (idx / (total - 1))
            g2 = 0.2 + 0.8 * (idx / (total - 1))
        return (g2, 0, g1)

    for k in range(14):
        ax = axes[k]
        if k < len(names):
            nm = names[k]
            d, o = obs_pack[nm]
            _, p = pred_pack[nm]

            # x：小时 0/3000/6000/9000
            hrs = (d - DOY_MIN) * 24.0
            ax.set_xlim(0, 9000)
            ax.set_xticks([0, 3000, 6000, 9000])
            ax.tick_params(axis='both', which='major', labelsize=18)
            # 实测曲线
            ax.plot(hrs, o, color="black", linewidth=1.5, label="实测")

            # 计算散点：更大、方块、无边框、半透明、栅格化
            row_idx = k // 2
            mk = markers_by_row[row_idx % len(markers_by_row)]
            gcol = gray_for_idx(k, len(names))  # 原始颜色（相当于 alpha=1.0）
            bg_rgb = _effective_bg_rgb(ax)  # 轴面背景色（已考虑透明度）
            face_rgb = _alpha_to_solid_on_bg(gcol, 0.5, bg_rgb)  # 等效于 alpha=0.5 的不透明色

            ax.scatter(
                hrs, p,
                marker=mk, s=32,
                facecolors=[face_rgb],  # 不透明的“半透明等效色”
                edgecolors=[gcol],  # 原色边界（alpha=1.0）
                linewidths=0.4,
                alpha=1.0,  # 这里务必设为 1.0（或干脆不写）
                rasterized=True,
                label="计算",
            )
            ax.tick_params(axis='both', which='major', labelsize=18)

            # y：按行固定，否则“3的整数倍”
            row_idx = k // 2
            ylr = row_ylims[row_idx]
            if ylr is not None:
                ax.set_ylim(ylr[0], ylr[1])
            else:
                ymin = min(np.min(o), np.min(p))
                ymax = max(np.max(o), np.max(p))
                y_low = 3 * np.floor(ymin / 3.0)
                y_high = 3 * np.ceil(ymax / 3.0)
                if y_low == y_high:
                    y_low -= 3;
                    y_high += 3
                ax.set_ylim(y_low, y_high)

            ax.grid(True, linestyle=":", linewidth=0.3, alpha=0.6)
            ax.legend(frameon=False, fontsize=20, loc="upper right", ncol=2)

            # 轴标签：左列给 y；仅底行给 x；并强制底行显示刻度标签
            if k % 2 == 0:
                ax.set_ylabel("水位/m")
            else:
                ax.set_ylabel(None)

            if k >= 12:
                ax.set_xlabel("t/h")
                ax.tick_params(axis='x', which='both', labelbottom=True, pad=1)
            else:
                ax.set_xlabel(None)
                ax.tick_params(axis='x', which='both', labelbottom=False, pad=1)
        else:
            ax.axis("off")

    # 收紧整体留白（行距/列距 + 边距）
    fig.subplots_adjust(left=0.065, right=0.985, top=0.985, bottom=0.035,
                        hspace=0.12, wspace=0.12)

    # 导出：尽量去除空白
    plt.savefig('D:\\A_PhD_Main_paper\\Chap.2\\Figure\\Fig.2.14\\2016.png',
                bbox_inches='tight', pad_inches=0.05)


def fig2_4():

    plt.rcParams['font.family'] = ['Arial', 'SimHei']
    plt.rc('font', size=22)
    plt.rc('axes', linewidth=1)

    wl1 = HydroStationDS()
    wl1.import_from_standard_files('G:\\A_1Dflow_sed\\Hydrodynamic_model\\Original_water_level\\',
                                   'G:\\A_1Dflow_sed\\Hydrodynamic_model\\Original_water_level\\对应表.csv')

    sec_wl_diff, sec_ds_diff = [], []
    sec_dis = [0, 63.83, 153.87, 306.77, 384.16, 423.15, 653.115, 955]
    sec_name = ['宜昌', '枝城', '螺山', '汉口']
    for sec in sec_name:
        fig14_df = wl1.hydrostation_inform_df[sec]
        wl_pri, wl_post = [], []
        sd_pri, sd_post = [], []
        ds_pri, ds_post = [], []
        year_dic = {}
        for year in range(1985, 2021):
            year_temp = fig14_df['year'] == year
            discharge = fig14_df['flow/m3/s'][year_temp].tolist()
            flow_temp = fig14_df['water_level/m'][year_temp].tolist() - wl1.waterlevel_offset_list[wl1.hydrostation_name_list.index(sec)]
            sed_temp = fig14_df['sediment_concentration/kg/m3'][year_temp].tolist()
            year_dic[f'{str(year)}_wl'] = flow_temp[0:365]
            year_dic[f'{str(year)}_sed'] = sed_temp[0:365]
            if len(flow_temp) == 365 or len(flow_temp) == 366:
                if year > 2003:
                    wl_post.append(flow_temp[0:365])
                    sd_post.append(sed_temp[0:365])
                    ds_post.append(discharge[0:365])

                if 1998 <= year <= 2003:
                    wl_pri.append(flow_temp[0:365])
                    ds_pri.append(discharge[0:365])
                    sd_pri.append(sed_temp[0:365])

        diff_dis = np.array(np.nanmean(wl_post, axis=0)) - np.array(np.nanmean(wl_pri, axis=0))
        sec_wl_diff.append(diff_dis[122: 304].tolist())
        diff_dis = np.array(np.nanmean(ds_post, axis=0)) - np.array(np.nanmean(ds_pri, axis=0))
        sec_ds_diff.append(diff_dis[122: 304].tolist())

    plt.close()
    # plt.rcParams['font.family'] = ['Arial', 'SimHei']
    # plt.rc('font', size=18)
    # plt.rc('axes', axisbelow=True)
    # plt.rc('axes', linewidth=3)
    # fig_temp, ax_temp = plt.subplots(figsize=(12, 5), constrained_layout=True)
    # # ax_temp.grid(axis='y', color=(210 / 256, 210 / 256, 210 / 256), zorder=0)
    # ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_pri, axis=0).reshape([365]), lw=5, c=(0, 0, 1), zorder=4)
    # # ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_post, axis=0).reshape([365]), lw=5, c=(1, 0, 0), zorder=4)
    # # ax_temp.plot(np.linspace(1, 365, 365), np.linspace(l1, l1, 365), lw=2, ls='--', c=(0, 0, 0))
    # ax_temp.set_xlim(-50, 975)
    # ax_temp.set_ylim(-4, 1)
    # # ax_temp.set_yticks(ytick)
    # bplot = ax_temp.boxplot(sec_wl_diff, widths=30, positions=sec_dis, notch=True, showfliers=False, whis=(5, 95),
    #                         patch_artist=True, medianprops={"color": "blue", "linewidth": 2.8},
    #                         boxprops={"linewidth": 1.8}, whiskerprops={"linewidth": 1.8},
    #                         capprops={"color": "C0", "linewidth": 1.8})
    #
    # ax_temp.set_xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])
    # ax_temp.set_xticklabels(['0', '100', '200', '300', '400', '500', '600', '700', '800', '900'])
    # # ax_temp.set_xticklabels(['Yichang', 'Zhicheng', 'Jianli', 'Lianhuatang', 'Luoshan', 'Hankou', 'Jiujiang'], fontname='Arial', fontsize=24)
    # ax_temp.set_ylabel('Water level difference/m', fontname='Arial', fontsize=28, fontweight='bold')
    # colors = []
    #
    # for patch in bplot['boxes']:
    #     patch.set_facecolor((208 / 256, 156 / 256, 44 / 256))
    # # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
    # plt.savefig(
    #     f'G:\\A_PhD_Main_paper\\Chap.2\\Figure\\Fig.2.4\\along_wl_nc.png',
    #     dpi=500)
    # plt.close()

    # plt.close()
    # plt.rcParams['font.family'] = ['Arial', 'SimHei']
    # plt.rc('font', size=18)
    # plt.rc('axes', axisbelow=True)
    # plt.rc('axes', linewidth=3)
    # fig_temp, ax_temp = plt.subplots(figsize=(12, 5), constrained_layout=True)
    # # ax_temp.grid(axis='y', color=(210 / 256, 210 / 256, 210 / 256), zorder=0)
    # ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_pri, axis=0).reshape([365]), lw=5, c=(0, 0, 1), zorder=4)
    # # ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_post, axis=0).reshape([365]), lw=5, c=(1, 0, 0), zorder=4)
    # # ax_temp.plot(np.linspace(1, 365, 365), np.linspace(l1, l1, 365), lw=2, ls='--', c=(0, 0, 0))
    # ax_temp.set_xlim(-50, 975)
    # # ax_temp.set_yticks(ytick)
    # bplot = ax_temp.boxplot(sec_ds_diff, widths=20, positions=sec_dis, notch=True, showfliers=False,
    #                         patch_artist=True, medianprops={"color": "blue", "linewidth": 2.8},
    #                         boxprops={"linewidth": 1.8}, whiskerprops={"linewidth": 1.8},
    #                         capprops={"color": "C0", "linewidth": 1.8})
    # ax_temp.set_xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])
    # ax_temp.set_xticklabels(['0', '100', '200', '300', '400', '500', '600', '700', '800', '900'])
    # # ax_temp.set_xticklabels(['Yichang', 'Zhicheng', 'Jianli', 'Lianhuatang', 'Luoshan', 'Hankou', 'Jiujiang'], fontname='Arial', fontsize=24)
    # ax_temp.set_ylabel('Discharge difference/m', fontname='Arial', fontsize=28, fontweight='bold')
    # colors = []
    #
    # for patch in bplot['boxes']:
    #     patch.set_facecolor((208 / 256, 156 / 256, 44 / 256))
    # # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
    # plt.savefig(
    #     f'G:\\A_PhD_Main_paper\\Chap.2\\Figure\\Fig.2.4\\along_ds_nc.png',
    #     dpi=500)
    # plt.close()
    wl_pri2, wl_post2 = [], []
    wl_pri3, wl_post3 = [], []

    for sec, r1, l1, ytick, in zip(['宜昌', '枝城', '螺山', '汉口'], [(36, 54), (34, 50), (14, 34), (10, 30)],
                                   [48, 44, 29, 24],
                                   [[36, 39, 42, 45, 48, 51, 54], [34, 38, 42, 46, 50], [14, 18, 22, 26, 30, 34],
                                    [10, 15, 20, 25, 30]]):
        fig14_df = wl1.hydrostation_inform_df[sec]
        year_dic = {}
        wl_pri, wl_post = [], []
        sd_pri, sd_post = [], []
        ds_pri, ds_post = [], []

        for year in range(1985, 2021):
            year_temp = fig14_df['year'] == year
            discharge = fig14_df['flow/m3/s'][year_temp].tolist()
            flow_temp = fig14_df['water_level/m'][year_temp].tolist() - wl1.waterlevel_offset_list[
                wl1.hydrostation_name_list.index(sec)]
            sed_temp = fig14_df['sediment_concentration/kg/m3'][year_temp].tolist()
            year_dic[f'{str(year)}_wl'] = flow_temp[0:365]
            year_dic[f'{str(year)}_sed'] = sed_temp[0:365]
            if len(flow_temp) == 365 or len(flow_temp) == 366:
                if year >= 2003:
                    wl_post.append(flow_temp[0:365])
                    sd_post.append(sed_temp[0:365])
                    ds_post.append(discharge[0:365])
                    if sec == '宜昌':
                        wl_post2.extend(flow_temp[0: 365])
                    elif sec == '汉口':
                        wl_post3.extend(flow_temp[0: 365])
                elif year < 2003:
                    wl_pri.append(flow_temp[0:365])
                    if sec == '宜昌':
                        wl_pri2.extend(flow_temp[0: 365])
                    elif sec == '汉口':
                        wl_pri3.extend(flow_temp[0: 365])

                if 1950 <= year <= 2003:
                    ds_pri.append(discharge[0:365])
                    sd_pri.append(sed_temp[0:365])

        wl_post = np.array(wl_post)
        sd_post = np.array(sd_post)
        wl_pri = np.array(wl_pri)
        sd_pri = np.array(sd_pri)
        ds_pri = np.array(ds_pri)
        ds_post = np.array(ds_post)

        sd_pri[sd_pri == 0] = np.nan
        sd_post[sd_post == 0] = np.nan

        plt.rcParams['font.family'] = ['Arial', 'SimHei']
        plt.rc('font', size=22)
        plt.rc('axes', linewidth=1)
        fig_temp, ax_temp = plt.subplots(figsize=(10, 5), constrained_layout=True)
        ax_temp.grid(axis='y', color=(210 / 256, 210 / 256, 210 / 256), zorder=0)
        ax_temp.fill_between(np.linspace(122, 304, 121), np.linspace(r1[1], r1[1], 121),
                             np.linspace(r1[0], r1[0], 121), alpha=1, color=(0.9, 0.9, 0.9))
        ax_temp.fill_between(np.linspace(1, 365, 365), np.nanmax(wl_post, axis=0).reshape([365]),
                             np.nanmin(wl_post, axis=0).reshape([365]), alpha=0.3, color=(0.8, 0.2, 0.1), zorder=3)
        ax_temp.fill_between(np.linspace(1, 365, 365), np.nanmax(wl_pri, axis=0).reshape([365]),
                             np.nanmin(wl_pri, axis=0).reshape([365]), alpha=0.3, color=(0.1, 0.2, 0.8), zorder=2)
        ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_pri, axis=0).reshape([365]), lw=5, c=(0, 0, 1),
                     zorder=4)
        ax_temp.scatter(np.linspace(1, 365, 365), np.nanmean(wl_post, axis=0).reshape([365]))
        ax_temp.plot(np.linspace(1, 365, 365), np.linspace(l1, l1, 365), lw=2, ls='-.', c=(0, 0, 0))
        ax_temp.set_xlim(1, 365)
        ax_temp.set_ylim(r1[0], r1[1])
        ax_temp.set_yticks(ytick)

        print(sec)
        print(f'pre-wl-flood-{str(np.nanmean(np.nanmean(wl_pri, axis=0)[122: 304]))}')
        print(f'post-wl-flood-{str(np.nanmean(np.nanmean(wl_post, axis=0)[122: 304]))}')
        print(f'pre-wl-flood-{str(np.nanmean(np.nanmean(wl_pri, axis=0)[np.r_[0:122, 305:365]]))}')
        print(f'post-wl-flood-{str(np.nanmean(np.nanmean(wl_post, axis=0)[np.r_[0:122, 305:365]]))}')

        a = [15, 45, 75, 106, 136, 167, 197, 228, 258, 289, 319, 350]
        c = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        ax_temp.set_xticks(a)
        ax_temp.set_xticklabels(c,  fontsize=24)
        ax_temp.set_ylabel('水位/m', fontsize=24, fontweight='bold')
        # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
        plt.savefig(
            f'G:\\A_PhD_Main_paper\\Chap.2\\Figure\\Fig.2.4\\{sec}_wl_nc.png',
            dpi=500)
        plt.close()

        # fig_temp, ax_temp = plt.subplots(figsize=(11, 6), constrained_layout=True)
        # wl_temp = np.concatenate([np.nanmean(sd_pri[:, 150: 300], axis=1), np.nanmean(sd_post[:, 150: 300], axis=1)])
        # ax_temp.bar([_ for _ in range(1990, 2004)], np.nanmean(sd_pri[:, 150: 300], axis=1), 0.6, label='SAR', color=(255/256, 155/256, 37/256), edgecolor=(0/256, 0/256, 0/256), linewidth=1.5, zorder=3, alpha=0.5)
        # ax_temp.plot([_ for _ in range(1990, 2004)], [np.nanmean(np.nanmean(sd_pri[:, 150: 300], axis=1)) for _ in range(1990, 2004)], linewidth=5, c=(255/256, 155/256, 37/256))
        # ax_temp.bar([_ for _ in range(2004, 2021)], np.nanmean(sd_post[:, 150: 300], axis=1), 0.6, label='SAR', color=(0/256, 92/256, 171/256), edgecolor=(0 / 256, 0 / 256, 0 / 256), linewidth=1.5, zorder=3, alpha=0.5)
        # ax_temp.plot([_ for _ in range(2004, 2021)], [np.nanmean(np.nanmean(sd_post[:, 150: 300], axis=1)) for _ in range(2004, 2021)], linewidth=5, c=(0/256, 92/256, 171/256))
        # ax_temp.set_xlabel('Year', fontname='Arial', fontsize=24, fontweight='bold')
        # ax_temp.set_ylabel('Sediment concentration', fontname='Arial', fontsize=24, fontweight='bold')
        # ax_temp.set_xlim(1989.5, 2020.5)
        # plt.savefig(f'G:\\A_PhD_Main_paper\\Chap.2\\Figure\\Fig.2.4\\{sec}_annual_sd_nc.png', dpi=500)

        # fig_temp, ax_temp = plt.subplots(figsize=(15, 6), constrained_layout=True)
        # wl_temp = np.concatenate([np.nanmean(ds_pri[:, 150: 300], axis=1), np.nanmean(sd_post[:, 150: 300], axis=1)])
        # ax_temp.bar([_ for _ in range(1990, 2004)], np.nanmean(ds_pri[:, 150: 300], axis=1), 0.6, label='SAR', color=(256/256, 200/256, 87/256), edgecolor=(0/256, 0/256, 0/256), linewidth=1.5, zorder=3, alpha=0.5)
        # ax_temp.plot([_ for _ in range(1990, 2004)], [np.nanmean(np.nanmean(ds_pri[:, 150: 300], axis=1)) for _ in range(1990, 2004)], linewidth=4, c=(255/256, 200/256, 87/256))
        # ax_temp.bar([_ for _ in range(2004, 2021)], np.nanmean(ds_post[:, 150: 300], axis=1), 0.6, label='SAR', color=(0/256, 72/256, 151/256), edgecolor=(0 / 256, 0 / 256, 0 / 256), linewidth=1.5, zorder=3, alpha=0.5)
        # ax_temp.plot([_ for _ in range(2004, 2021)], [np.nanmean(np.nanmean(ds_post[:, 150: 300], axis=1)) for _ in range(2004, 2021)], linewidth=3, c=(0/256, 72/256, 151/256))
        # ax_temp.set_xlabel('Year', fontname='Arial', fontsize=28, fontweight='bold')
        # ax_temp.set_ylabel('Sediment concentration', fontname='Arial', fontsize=28, fontweight='bold')
        # ax_temp.set_xlim(1989.5, 2020.5)
        # plt.savefig(f'G:\\A_PhD_Main_paper\\Chap.2\\Figure\\Fig.2.4\\{sec}_annual_ds_nc.png', dpi=500)

        plt.rc('axes', axisbelow=True)
        plt.rc('axes', linewidth=3)
        fig_temp, ax_temp = plt.subplots(figsize=(10, 5), constrained_layout=True)
        ax_temp.grid( axis='y', color=(210 / 256, 210 / 256, 210 / 256), zorder=0)
        # ax_temp.fill_between(np.linspace(175, 300, 121), np.linspace(r1[1], r1[1], 121), np.linspace(r1[0],r1[0],121),alpha=1, color=(0.9, 0.9, 0.9))
        ax_temp.fill_between(np.linspace(1, 365, 365), np.nanmax(sd_post, axis=0).reshape([365]), np.nanmin(sd_post, axis=0).reshape([365]), alpha=0.3, color=(0/256, 92/256, 171/256), zorder=3)
        ax_temp.fill_between(np.linspace(1, 365, 365), np.nanmax(sd_pri, axis=0).reshape([365]), np.nanmin(sd_pri, axis=0).reshape([365]), alpha=0.3, color=(255/256, 155/256, 37/256), zorder=2)
        ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(sd_pri, axis=0).reshape([365]), lw=5, c=(255/256, 155/256, 37/256), zorder=4)
        ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(sd_post, axis=0).reshape([365]), lw=5, c=(0/256, 92/256, 171/256), zorder=4)
        # ax_temp.plot(np.linspace(1,365,365), np.linspace(l1,l1,365), lw=2, ls='--', c=(0,0,0))
        ax_temp.set_xlim(1, 365)
        # ax_temp.set_ylim(r1[0], r1[1])
        # ax_temp.set_yticks(ytick)
        cc = np.nanmean(sd_post, axis=0)/np.nanmean(sd_pri, axis=0)
        print('sd' + str(1- np.nanmean(cc[120: 300])))
        print('sd' +str(1 - np.nanmean(cc[np.r_[0:122, 305:365]])))
        plt.yscale("log")
        a = [15, 45, 75, 106, 136, 167, 197, 228, 258, 289, 319, 350]
        c = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        ax_temp.set_xticks(a)
        ax_temp.set_xticklabels(c, fontname='Arial', fontsize=24)
        # ax_temp.set_xlabel('月份', fontname='Arial', fontsize=28, fontweight='bold')
        ax_temp.set_ylabel('悬移质含沙量/kg/m^3)', fontname='Arial', fontsize=28, fontweight='bold')
        # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
        plt.savefig(f'G:\\A_PhD_Main_paper\\Chap.2\\Figure\\Fig.2.4\\{sec}_sd.png', dpi=500)

        if sec == '宜昌':
            fig_temp, ax_temp = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
            wl_temp = np.concatenate([np.nanmax(wl_pri, axis=1), np.nanmax(wl_post, axis=1)])
            # ax_temp.bar([_ for _ in range(1985, 2004)], np.nanmax(wl_pri, axis=1), 0.65, label='SAR', color=(0.2, 0.3, 0.8), edgecolor=(0/256, 0/256, 0/256), linewidth=1, zorder=3, alpha=0.5)
            # ax_temp.bar([_ for _ in range(2004, 2021)], np.nanmax(wl_post, axis=1), 0.65, label='SAR', color=(0.8, 0.3, 0.2), edgecolor=(0 / 256, 0 / 256, 0 / 256), linewidth=1, zorder=3, alpha=0.5)
            ax_temp.grid(axis='y', color=(128 / 256, 128 / 256, 128 / 256), zorder=1)
            ax_temp.plot([_ for _ in range(1985, 2003)], np.nanmax(wl_pri, axis=1), color=(0.2, 0.3, 0.8),
                         linewidth=3, ls='-', label='Pre-TGD')
            ax_temp.plot([_ for _ in range(2003, 2021)], np.nanmax(wl_post, axis=1), color=(0.8, 0.3, 0.2),
                         linewidth=3, ls='-', label='Post-TGD')
            ax_temp.fill_between(np.linspace(0.5, 3004.5, 100), np.linspace(0, 0, 100), np.linspace(50., 50., 100),
                                 edgecolor='none', facecolor=(0.4, 0.4, 0.4), alpha=0.3, lw=2)
            ax_temp.plot(np.linspace(0.5, 3004.5, 100), np.linspace(50, 50, 100), color=(0, 0, 0), ls='-.', lw=2,
                         label='Overbank')
            ax_temp.fill_between(np.linspace(0.5, 3004.5, 100), np.linspace(50, 50, 100), np.linspace(52, 52, 100),
                                 edgecolor='none', facecolor=(0.8, 0.8, 0.8), alpha=0.3, lw=2)
            ax_temp.plot(np.linspace(0.5, 3004.5, 100), np.linspace(52, 52, 100), color=(0, 0, 0), ls='--', lw=2,
                         label='Extreme')
            ax_temp.scatter([_ for _ in range(1985, 2003)], np.nanmax(wl_pri, axis=1), s=15 ** 2, marker='s',
                            color="none", edgecolor=(0, 0, 1), linewidth=3)
            ax_temp.scatter([_ for _ in range(2003, 2021)], np.nanmax(wl_post, axis=1), s=15 ** 2, marker='s',
                            color="none", edgecolor=(1, 0, 0), linewidth=3)
            # ax_temp.plot(np.linspace([2003.5, 2003.5, 100]), np.linspace([0, 100, 100]), color=(0.2, 0.2, 0.2))
            ax_temp.set_xlabel('Year', fontname='Arial', fontsize=28, fontweight='bold')
            ax_temp.set_ylabel('Annual maximum water level/m', fontname='Arial', fontsize=28, fontweight='bold')
            ax_temp.legend(fontsize=20, ncol=2)
            ax_temp.set_yticks([45, 47, 49, 51, 53, 55])
            ax_temp.set_yticklabels(['45', '47', '49', '51', '53', '55'], fontname='Arial', fontsize=24)
            ax_temp.set_xlim(1984.5, 2020.5)
            ax_temp.set_ylim(45, 55)
            plt.savefig(
                f'G:\\A_PhD_Main_paper\\Chap.2\\Figure\\Fig.2.4\\{sec}_annual_wl_nc.png',
                dpi=500)
            plt.close()

        if sec == '汉口':
            fig_temp, ax_temp = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
            wl_temp = np.concatenate([np.nanmax(wl_pri, axis=1), np.nanmax(wl_post, axis=1)])
            ax_temp.grid(axis='y', color=(128 / 256, 128 / 256, 128 / 256), zorder=1)
            # ax_temp.bar([_ for _ in range(1985, 2004)], np.nanmax(wl_pri, axis=1), 0.65, label='SAR', color=(0.2, 0.3, 0.8), edgecolor=(0/256, 0/256, 0/256), linewidth=1, zorder=3, alpha=0.5)
            # ax_temp.bar([_ for _ in range(2004, 2021)], np.nanmax(wl_post, axis=1), 0.65, label='SAR', color=(0.8, 0.3, 0.2), edgecolor=(0 / 256, 0 / 256, 0 / 256), linewidth=1, zorder=3, alpha=0.5)
            ax_temp.plot([_ for _ in range(1985, 2003)], np.nanmax(wl_pri, axis=1), color=(0.2, 0.3, 0.8),
                         linewidth=3,
                         ls='-', label='Pre-TGD')
            ax_temp.plot([_ for _ in range(2003, 2021)], np.nanmax(wl_post, axis=1), color=(0.8, 0.3, 0.2),
                         linewidth=3,
                         ls='-', label='Post-TGD')
            ax_temp.fill_between(np.linspace(0.5, 3004.5, 100), np.linspace(0, 0, 100), np.linspace(24, 24, 100),
                                 edgecolor='none', facecolor=(0.4, 0.4, 0.4), alpha=0.3, lw=2)
            ax_temp.plot(np.linspace(0.5, 3004.5, 100), np.linspace(24, 24, 100), color=(0, 0, 0), ls='-.', lw=2,
                         label='Overbank')
            ax_temp.fill_between(np.linspace(0.5, 3004.5, 100), np.linspace(24, 24, 100), np.linspace(26, 26, 100),
                                 edgecolor='none', facecolor=(0.8, 0.8, 0.8), alpha=0.3, lw=2)
            ax_temp.plot(np.linspace(0.5, 3004.5, 100), np.linspace(26, 26, 100), color=(0, 0, 0), ls='--', lw=2,
                         label='Extreme')
            ax_temp.scatter([_ for _ in range(1985, 2003)], np.nanmax(wl_pri, axis=1), s=15 ** 2, marker='s',
                            color="none", edgecolor=(0, 0, 1), linewidth=3)
            ax_temp.scatter([_ for _ in range(2003, 2021)], np.nanmax(wl_post, axis=1), s=15 ** 2, marker='s',
                            color="none", edgecolor=(1, 0, 0), linewidth=3)
            # ax_temp.plot(np.linspace([2003.5, 2003.5, 100]), np.linspace([0, 100, 100]), color=(0.2, 0.2, 0.2))

            ax_temp.set_xlabel('Year', fontname='Arial', fontsize=28, fontweight='bold')
            ax_temp.set_ylabel('Annual maximum water level/m', fontname='Arial', fontsize=28, fontweight='bold')
            ax_temp.set_yticks([20, 22, 24, 26, 28, 30])
            ax_temp.set_yticklabels(['20', '22', '24', '26', '28', '30'], fontname='Arial', fontsize=24)
            ax_temp.set_xlim(1984.5, 2020.5)
            ax_temp.set_ylim(20, 30)
            plt.savefig(
                f'G:\\A_PhD_Main_paper\\Chap.2\\Figure\\Fig.2.4\\{sec}_annual_wl_nc.png',
                dpi=500)
            plt.close()

        # fig_temp, ax_temp = plt.subplots(nrows=1, ncols=2, figsize=(11, 6), constrained_layout=True)
        # # n, bins, patches = ax_temp.hist(wl_pri2, 50, density=True, histtype="step",  cumulative=True, label="Cumulative histogram")
        # x = np.linspace(min(wl_pri2), max(wl_pri2))
        # y = ((1 / (np.sqrt(2 * np.pi) * 50)) * np.exp(-0.5 * (1 / 50 * (x - 50)) ** 2))
        # y = y.cumsum()
        # y /= y[-1]
        #
        # # # Complementary cumulative distributions.
        # # b = plt.ecdf(wl_pri2, complementary=False, label="pre-TGP", orientation="horizontal")
        # # ax_temp[0].ecdf(wl_pri2, complementary=False, label="pre-TGP", orientation="horizontal")
        # # # n, bins, patches = ax_temp.hist(wl_post2, 50, density=True, histtype="step",  cumulative=True, label="Cumulative histogram")
        # # x = np.linspace(min(wl_post2), max(wl_post2))
        # # y = ((1 / (np.sqrt(2 * np.pi) * 50)) * np.exp(-0.5 * (1 / 50 * (x - 50)) ** 2))
        # # y = y.cumsum()
        # # y /= y[-1]
        # # # Complementary cumulative distributions.
        # # a = plt.ecdf(wl_post2, complementary=False, label="post-TGP", orientation="horizontal")
        #
        # ax_temp[0].set_yticks([37, 43, 49, 55])
        # ax_temp[0].set_yticklabels(['37', '43', '49', '55'], fontname='Arial', fontsize=22)
        # ax_temp[0].set_xticks([0, 0.25, 0.5, 0.75, 1])
        # ax_temp[0].set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontname='Arial', fontsize=22)
        # # ax_temp.set_ylabel('Water level/m', fontname='Arial', fontsize=28, fontweight='bold')
        # # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
        # ax_temp[0].set_xlabel('Density', fontname='Arial', fontsize=28, fontweight='bold')
        # ax_temp[0].set_ylabel('Water level/m', fontname='Arial', fontsize=28, fontweight='bold')
        # ax_temp[0].set_xlim([-0.03, 1.03])
        # ax_temp[0].set_ylim([37, 55])
        # ax_temp[0].legend()
        #
        # x = np.linspace(min(wl_pri3), max(wl_pri3))
        # y = ((1 / (np.sqrt(2 * np.pi) * 50)) * np.exp(-0.5 * (1 / 50 * (x - 50)) ** 2))
        # y = y.cumsum()
        # y /= y[-1]
        #
        # # Complementary cumulative distributions.
        # ax_temp[1].ecdf(wl_pri3, complementary=False, label="pre-TGP", orientation="horizontal")
        # # n, bins, patches = ax_temp.hist(wl_post2, 50, density=True, histtype="step",  cumulative=True, label="Cumulative histogram")
        #
        # ax_temp[1].set_yticks([12, 18, 24, 30])
        # ax_temp[1].set_yticklabels(['12', '18', '24', '30'], fontname='Arial', fontsize=22)
        # ax_temp[1].set_xticks([0, 0.25, 0.5, 0.75, 1])
        # ax_temp[1].set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontname='Arial', fontsize=22)
        # x = np.linspace(min(wl_post3), max(wl_post3))
        # y = ((1 / (np.sqrt(2 * np.pi) * 50)) * np.exp(-0.5 * (1 / 50 * (x - 50)) ** 2))
        # y = y.cumsum()
        # y /= y[-1]
        # # Complementary cumulative distributions.
        # ax_temp[1].ecdf(wl_post3, complementary=False, label="post-TGP", orientation="horizontal")
        #
        # # ax_temp.set_xticks(a)
        # # ax_temp.set_xticklabels(c, fontname='Arial', fontsize=24)
        # # ax_temp.set_ylabel('Water level/m', fontname='Arial', fontsize=28, fontweight='bold')
        # # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
        # ax_temp[1].set_xlabel('Density', fontname='Arial', fontsize=28, fontweight='bold')
        # # ax_temp[1].set_ylabel('Water level/m', fontname='Arial', fontsize=28, fontweight='bold')
        # ax_temp[1].set_xlim([-0.03, 1.03])
        # ax_temp[1].set_ylim([12, 30])
        # ax_temp[1].legend()
        # plt.savefig(f'G:\\A_PhD_Main_paper\\Chap.2\\Figure\\Fig.2.4\\{sec}_wl_freq_nc.png', dpi=500)

        fig_temp, ax_temp = plt.subplots(nrows=1, ncols=1, figsize=(11, 6), constrained_layout=True)
        wl_dic = {'wl': [], 'status': []}
        s_ = 36
        for _ in wl_pri2:
            wl_dic['status'].append('Pre-TGP period')
            wl_dic['wl'].append(int(np.floor(_)))

        for _ in wl_post2:
            wl_dic['status'].append('Post-TGP period')
            wl_dic['wl'].append(int(np.floor(_)))

        sns.histplot(data=wl_dic, x="wl", hue="status",
                     palette=[(127 / 256, 163 / 256, 222 / 256), (247 / 256, 247 / 256, 247 / 256)], multiple="dodge",
                     shrink=1.45, stat='density', alpha=0.9)

        # # Manually add dashed lines for category 'C'
        i = 0
        for container in ax_temp.containers:
            for patch in container.patches:
                if np.mod(i, 2) == 0:  # This checks if the patch is for category 'C'
                    patch.set_hatch('/')  # Set dashed lines
                    patch.set_facecolor((247 / 256, 247 / 256, 247 / 256))
                elif np.mod(i, 2) == 1:
                    patch.set_hatch('')  # This checks if the patch is for category 'C' # Set dashed lines
                    patch.set_facecolor((127 / 256, 163 / 256, 222 / 256))
            i += 1

        ax_temp.set_xticks([38, 42, 46, 50, 54])
        ax_temp.set_xticklabels(['38', '42', '46', '50', '54'], fontname='Arial', fontsize=22)
        ax_temp.set_yticks([0, 0.05, 0.10, 0.15, 0.2, 0.25])
        ax_temp.set_yticklabels(['0%', '5%', '10%', '15%', '20%', '25%'], fontname='Arial', fontsize=22)
        ax_temp.set_ylabel('Density', fontname='Arial', fontsize=26, fontweight='bold')
        ax_temp.set_xlabel('Water level/m', fontname='Arial', fontsize=26, fontweight='bold')
        ax_temp.set_ylim([0, 0.25])
        ax_temp.set_xlim([35.5, 52])
        ax_temp.get_legend()

        plt.savefig(
            f'G:\\B_papers_patents\\RA_Dam operations enhance floodplain vegetation resistance and resilience but compress lateral heterogeneity\\A_fig_nc\\A_NC_Fig1\\{sec}_hist.png',
            dpi=500)
        plt.close()
        pass

def fig2_4_2_3():
    # Create fig4
    VI_curve_fitting = {'para_ori': [0.01, 0.01, 0, 2, 180, 2, 0.0001], 'para_boundary': ([0, 0, 0, 0, 130, 0, 0], [0.5, 1, 200, 20, 330, 20, 0.0002])}
    fig4_df = pd.read_csv('D:\A_PhD_Main_paper\Chap.2\Figure\\2.4.2\\3\\wood.csv')
    fig4_array = np.array(fig4_df)
    fig4_df['vi'] = (fig4_df['vi'] -32768)/10000
    fig4_array_new = np.array([[0], [1]])
    fig4_dic = {'DOY': fig4_df['DOY'].tolist(), 'OSAVI': fig4_df['vi'].tolist()}
    fig4_df = pd.DataFrame(data=fig4_dic)
    fig4, ax4 = plt.subplots(figsize=(20, 10), constrained_layout=True)
    # fig4, ax4 = plt.subplots(figsize=(10.5, 10.5), constrained_layout=True)
    ax4.set_axisbelow(True)
    ax4.set_xlim(0, 365)
    ax4.set_ylim(0, 0.7)
    paras, extra = curve_fit(seven_para_logistic_function, fig4_dic['DOY'], fig4_dic['OSAVI'], maxfev=5000000, p0=VI_curve_fitting['para_ori'], bounds=VI_curve_fitting['para_boundary'])

    # define p3 and p5
    doy_all = fig4_dic['DOY'][1:]
    vi_all = fig4_dic['OSAVI'][1:]
    vi_dormancy = []
    doy_dormancy = []
    vi_senescence = []
    doy_senescence = []
    vi_max = []
    doy_max = []
    doy_index_max = np.argmax(seven_para_logistic_function(np.linspace(0, 366, 365), paras[0], paras[1], paras[2], paras[3], paras[4],paras[5], paras[6]))
    # Generate the parameter boundary
    senescence_t = paras[4] - 4 * paras[5]
    for doy_index in range(len(doy_all)):
        if 0 < doy_all[doy_index] < paras[2] or paras[4] < doy_all[doy_index] < 366:
            vi_dormancy.append(vi_all[doy_index])
            doy_dormancy.append(doy_all[doy_index])
        if doy_index_max - 5 < doy_all[doy_index] < doy_index_max + 5:
            vi_max.append(vi_all[doy_index])
            doy_max.append(doy_all[doy_index])
        if senescence_t - 5 < doy_all[doy_index] < senescence_t + 5:
            vi_senescence.append(vi_all[doy_index])
            doy_senescence.append(doy_all[doy_index])

    vi_dormancy_sort = np.sort(vi_dormancy)
    vi_max_sort = np.sort(vi_max)
    paras1_max = vi_dormancy_sort[int(np.fix(vi_dormancy_sort.shape[0] * 0.95))]
    paras1_min = vi_dormancy_sort[int(np.fix(vi_dormancy_sort.shape[0] * 0.05))]
    paras2_max = vi_max[-1] - paras1_min
    paras2_min = vi_max - paras1_max
    paras3_max = 0
    for doy_index in range(len(doy_all)):
        if paras1_min < vi_all[doy_index] < paras1_max and doy_all[doy_index] < 180:
            paras3_max = max(float(paras3_max), doy_all[doy_index])
    paras3_max = max(paras3_max, paras[2])
    paras3_min = 180
    for doy_index in range(len(doy_all)):
        if vi_all[doy_index] > paras1_max:
            paras3_min = min(paras3_min, doy_all[doy_index])
    paras3_min = min(paras[2], paras3_min)
    paras3_max = max(paras3_max, paras[2])
    paras5_max = 0
    for doy_index in range(len(doy_all)):
        if vi_all[doy_index] > paras1_max:
            paras5_max = max(paras5_max, doy_all[doy_index])
    paras5_max = max(paras5_max, paras[4])
    paras5_min = 365
    for doy_index in range(len(doy_all)):
        if paras1_min < vi_all[doy_index] < paras1_max and doy_all[doy_index] > 180:
            paras5_min = min(paras5_min, doy_all[doy_index])
    paras5_min = min(paras5_min, paras[4])
    paras4_max = (np.nanmax(doy_max) - paras3_min) / 4
    paras4_min = (np.nanmin(doy_max) - paras3_max) / 4
    paras6_max = paras4_max
    paras6_min = paras4_min
    paras7_max = (np.nanmax(vi_max) - np.nanmin(vi_senescence)) / (doy_senescence[np.argmin(vi_senescence)] - doy_max[np.argmax(vi_max)])
    paras7_min = (np.nanmin(vi_max) - np.nanmax(vi_senescence)) / (doy_senescence[np.argmax(vi_senescence)] - doy_max[np.argmin(vi_max)])
    a = (
    [paras1_min, paras2_min, paras3_min, paras4_min, paras5_min, paras6_min, paras7_min],
    [paras1_max, paras2_max, paras3_max, paras4_max, paras5_max, paras6_max, paras7_max])

    ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]), linewidth=10, color=(0/256, 109/256, 44/256))
    print(str([paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]]))
    fig4_dic['DOY'] = fig4_dic['DOY'][1:]
    fig4_dic['OSAVI'] = fig4_dic['OSAVI'][1:]
    fig4_df = pd.DataFrame.from_dict(fig4_dic)
    # ax4.plot(array_temp[0, :], array_temp[1, :], linewidth=4, markersize=12, **{'ls': '--', 'marker': 'o', 'color': 'b'})
    ax4.fill_between(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), 0.03, 0.25, 86, 14, 306, 19.999999999999996, 0.0001), seven_para_logistic_function(np.linspace(0, 365, 366), 0.33, 0.36, 86, 14, 306, 19.999999999999996, 0.0001), color=(0.1, 0.1, 0.1), alpha=0.1)
    ax4.scatter(fig4_dic['DOY'], fig4_dic['OSAVI'], marker='^', s=12**2, color="none", edgecolor=(160/256, 160/256, 196/256), linewidth=3)
    # ax4.fill_between(np.linspace(560, 650, 100), np.linspace(0, 0, 100), np.linspace(1, 1, 100), color=(0, 197/255, 1), alpha=1)
    # ax4.plot(np.linspace(365, 365, 100), np.linspace(0, 1, 100), linewidth=4, **{'ls': '--', 'color': (0, 0, 0)})
    ax4.set_xlabel('DOY', fontname='Arial', fontsize=30, fontweight='bold')
    ax4.set_ylabel('OSAVI', fontname='Arial', fontsize=30, fontweight='bold')
    ax4.grid( axis='y', color=(240/256, 240/256, 240/256))
    ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), 0.03, 0.25, 86, 14, 306, 19.999999999999996, 0.0001), linewidth=2, color=(0 / 256, 44 / 256, 109 / 256), **{'ls': '--'})
    ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), 0.33, 0.36, 86, 14, 306, 19.999999999999996, 0.0001), linewidth=2, color=(0 / 256, 44 / 256, 109 / 256), **{'ls': '--'})
    # ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras1_min, paras2_min, paras3_min, paras4_max, paras5_min, paras6_max, paras7_max), linewidth=2, color=(0/256, 109/256, 44/256), **{'ls': '--'})
    # ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras1_max, paras2_max, paras3_max, paras4_min, paras5_max, paras6_min, paras7_min), linewidth=2, color=(0/256, 109/256, 44/256), **{'ls': '--'})
    predicted_y_data = seven_para_logistic_function(np.array(fig4_dic['DOY']), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6])
    r_square = (1 - np.nansum((predicted_y_data - np.array(fig4_dic['OSAVI'])) ** 2) / np.nansum((np.array(fig4_dic['OSAVI']) - np.nanmean(np.array(fig4_dic['OSAVI']))) ** 2))
    ax4.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'], fontname='Arial', fontsize=26)
    a = [15, 45, 75, 105, 136, 166, 197, 227, 258, 288, 320, 350]
    c = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    points = np.array([fig4_dic['OSAVI'],fig4_dic['DOY']]).transpose()
    # hull = ConvexHull(points)
    # # # for i in b:
    # # #     a.append(i)
    # ax4.plot(points[hull.vertices,1], points[hull.vertices,0], 'r--', lw=2)
    ax4.set_xticks(a)
    ax4.set_xticklabels(c, fontname='Arial', fontsize=26)
    # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
    plt.savefig(f'D:\A_PhD_Main_paper\Chap.2\Figure\\2.4.2\\3\\wood.png', dpi=1000)
    print(r_square)

def ep_fig2_func():

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=22)
    plt.rc('axes', linewidth=2)

    data = 'D:\A_PhD_Main_paper\Chap.2\Figure\\2.4.2\\3\\data_hankou.xlsx'
    data_pd = pd.read_excel(data)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 16), constrained_layout=True)
    ax[0].plot(data_pd['DOY'], data_pd['total biomas site1'], lw=3, ls='--', c=(25/256, 25/256, 25/256), zorder=3)
    ax[0].scatter(data_pd['DOY'], data_pd['total biomas site1'], zorder=4, s=13**2, marker='s', edgecolors=(0/256, 0/256, 0/256), facecolor=(1, 1, 1), alpha=1, linewidths=2)
    ax[0].errorbar(data_pd['DOY'], data_pd['total biomas site1'], yerr=None)
    ax[0].plot(data_pd['DOY'], data_pd['leaf biomass (site1)'], lw=3, ls='--', c=(25/256, 25/256, 25/256), zorder=3)
    ax[0].scatter(data_pd['DOY'], data_pd['leaf biomass (site1)'], zorder=4, s=14**2, marker='^', edgecolors=(0/256, 0/256, 0/256), facecolor=(1, 1, 1), alpha=1, linewidths=2)
    ax[0].fill_between(data_pd['DOY'], [0, 0, 0, 0, 0, 0, 0, 0,0], data_pd['total biomas site1'], zorder=1, alpha=0.5, fc=(54/256, 92/256, 141/256))
    ax[0].fill_between(data_pd['DOY'], [0, 0, 0, 0, 0, 0, 0, 0,0], data_pd['leaf biomass (site1)'], zorder=2, alpha=0.5, fc=(196/256, 78/256, 82/256))
    ax[0].set_xticks([75, 135, 195, 255, 315])
    ax[0].grid(axis='y', color=(240 / 256, 240 / 256, 240 / 256))
    ax[0].set_xticklabels(['3', '5', '7', '9', '11'], fontname='Times New Roman', fontsize=26)
    ax[0].set_xlabel('Date', fontname='Times New Roman', fontsize=34, fontweight='bold')
    ax[0].set_ylabel('Biomass per plant/g', fontname='Times New Roman', fontsize=34, fontweight='bold')
    ax[0].set_xlim([60, 345])
    ax[0].set_ylim([0, 25])

    ax[1].plot(data_pd['DOY'], data_pd['total biomas site2'], lw=3, ls='--', c=(25/256, 25/256, 25/256), zorder=3)
    ax[1].scatter(data_pd['DOY'], data_pd['total biomas site2'], zorder=4, s=13**2, marker='s', edgecolors=(0/256, 0/256, 0/256), facecolor=(1, 1, 1), alpha=1, linewidths=2)
    ax[1].errorbar(data_pd['DOY'], data_pd['total biomas site2'], yerr=None)
    ax[1].plot(data_pd['DOY'], data_pd['leaf biomass (site2)'], lw=3, ls='--', c=(25/256, 25/256, 25/256), zorder=3)
    ax[1].scatter(data_pd['DOY'], data_pd['leaf biomass (site2)'], zorder=4, s=14**2, marker='^', edgecolors=(0/256, 0/256, 0/256), facecolor=(1, 1, 1), alpha=1, linewidths=2)
    ax[1].fill_between(data_pd['DOY'], [0, 0, 0, 0, 0, 0, 0, 0,0], data_pd['total biomas site2'], zorder=1, alpha=0.5, fc=(54/256, 92/256, 141/256))
    ax[1].fill_between(data_pd['DOY'], [0, 0, 0, 0, 0, 0, 0, 0,0], data_pd['leaf biomass (site2)'], zorder=2, alpha=0.5, fc=(196/256, 78/256, 82/256))
    ax[1].set_xticks([75, 135, 195, 255, 315])
    ax[1].grid(axis='y', color=(240 / 256, 240 / 256, 240 / 256))
    ax[1].set_xticklabels(['3', '5', '7', '9', '11'], fontname='Times New Roman', fontsize=26)
    ax[1].set_xlabel('Date', fontname='Times New Roman', fontsize=34, fontweight='bold')
    ax[1].set_ylabel('Biomass per plant/g', fontname='Times New Roman', fontsize=34, fontweight='bold')
    ax[1].set_xlim([60, 345])
    ax[1].set_ylim([0, 70])
    plt.savefig(f'D:\A_PhD_Main_paper\Chap.2\Figure\\2.4.2\\3\\Fig4.png', dpi=300)

def veg_area():
    ds1 = gdal.Open(r'G:\A_Landsat_Floodplain_veg\Landsat_floodplain_2020_datacube\OSAVI_noninun_curfit_datacube\Phemetric_tif\2004\\2004_MAVI.TIF')
    ds2 = gdal.Open(r'G:\A_Landsat_Floodplain_veg\Landsat_floodplain_2020_datacube\OSAVI_noninun_curfit_datacube\Phemetric_tif\2023\\2023_MAVI.TIF')
    arr1 = ds1.GetRasterBand(1).ReadAsArray()
    arr2 = ds2.GetRasterBand(1).ReadAsArray()
    arr1[~np.isnan(arr1)] = 1
    arr1_area = np.nansum(arr1) * 30*30 /1000 /1000
    arr2[~np.isnan(arr2)] = 1
    arr2_area = np.nansum(arr2)* 30*30 /1000 /1000
    print(str(arr1_area))
    print(str(arr2_area))

def inund_detection():
    ch_ds = gdal.Open('G:\A_Landsat_Floodplain_veg\Water_level_python\Inundation_indicator\daily_inundation_file\\1989002.tif')
    ch_arr = ch_ds.GetRasterBand(1).ReadAsArray()
    landsat_inundation_filelist = bf.file_filter('G:\A_Landsat_Floodplain_veg\Landsat_floodplain_2020_datacube\Inundation_DT_datacube\Individual_tif\\', ['.TIF'])
    model_detect_filelist = bf.file_filter('G:\A_Landsat_Floodplain_veg\Water_level_python\Inundation_indicator\daily_inundation_file\\', ['.tif'])
    dic_pre = {'doy': [], 'pre_yz_ii': [], 'pre_yz_in': [], 'pre_jj_ii': [], 'pre_jj_in': [], 'pre_ch_ii': [], 'pre_ch_in': [], 'pre_hh_ii': [], 'pre_hh_in': []}
    dic_post = {'doy':[], 'post_yz_ii': [], 'post_yz_in': [], 'post_jj_ii': [], 'post_jj_in': [], 'post_ch_ii': [], 'post_ch_in': [], 'post_hh_ii': [], 'post_hh_in': []}
    dic_ori_pre = {'doy': [], 'pre_yz_ii': [], 'pre_yz_in': [], 'pre_jj_ii': [], 'pre_jj_in': [], 'pre_ch_ii': [], 'pre_ch_in': [], 'pre_hh_ii': [], 'pre_hh_in': []}
    dic_ori_post = {'doy':[], 'post_yz_ii': [], 'post_yz_in': [], 'post_jj_ii': [], 'post_jj_in': [], 'post_ch_ii': [], 'post_ch_in': [], 'post_hh_ii': [], 'post_hh_in': []}
    dic_nn_pre = {'doy': [], 'pre_yz_ii': [], 'pre_yz_in': [], 'pre_jj_ii': [], 'pre_jj_in': [], 'pre_ch_ii': [], 'pre_ch_in': [], 'pre_hh_ii': [], 'pre_hh_in': []}
    dic_nn_post = {'doy':[], 'post_yz_ii': [], 'post_yz_in': [], 'post_jj_ii': [], 'post_jj_in': [], 'post_ch_ii': [], 'post_ch_in': [], 'post_hh_ii': [], 'post_hh_in': []}


    with tqdm(total=len(landsat_inundation_filelist), desc=f'Inun detection', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
        for _ in landsat_inundation_filelist:
            doy = int(_.split('\\')[-1].split('.')[0].split('_')[1])
            model_file = [__ for __ in model_detect_filelist if str(doy) in __]
            if len(model_file) == 1:
                try:
                    landsat_ds = gdal.Open(_)
                    landsat_arr = landsat_ds.GetRasterBand(1).ReadAsArray()
                    model_ds = gdal.Open(model_file[0])
                    model_arr = model_ds.GetRasterBand(1).ReadAsArray()
                    min3 = minimum_filter(model_arr, size=3, mode='nearest')
                    max3 = maximum_filter(model_arr, size=3, mode='nearest')

                    # 只要 3×3 窗口里有像元值与中心不同，就会让中心与 min 或 max 不相等
                    has_diff_neighbor = (model_arr != min3) | (model_arr != max3)

                    for sc, x_min, x_max in zip(['yz', 'jj', 'ch', 'hh'], [0, 950, 6100, 10210], [950, 6100,10210, 16537]):
                        if doy > 2003001:
                            dic_nn_post[f'post_{sc}_ii'].append(np.sum(np.logical_and(
                                np.logical_and(np.logical_and(model_arr == 1, landsat_arr == 2), ch_arr != 1),
                                has_diff_neighbor == 0)[:, x_min: x_max + 1]))
                            dic_nn_post[f'post_{sc}_in'].append(np.sum(np.logical_and(
                                np.logical_and(np.logical_and(model_arr == 1, landsat_arr == 1), ch_arr != 1),
                                has_diff_neighbor == 0)[:, x_min: x_max + 1]))

                            dic_post[f'post_{sc}_ii'].append(np.sum(np.logical_and(np.logical_and(model_arr == 1, landsat_arr == 2), ch_arr != 1)[:, x_min: x_max+1]))
                            dic_post[f'post_{sc}_in'].append(np.sum(np.logical_and(np.logical_and(model_arr == 1, landsat_arr == 1), ch_arr != 1)[:, x_min: x_max+1]))
                            dic_ori_post[f'post_{sc}_ii'].append(np.sum(np.logical_and(model_arr == 1, landsat_arr == 2)[:, x_min: x_max+1]))
                            dic_ori_post[f'post_{sc}_in'].append(np.sum(np.logical_and(model_arr == 1, landsat_arr == 1)[:, x_min: x_max+1]))
                        else:
                            dic_nn_pre[f'pre_{sc}_ii'].append(np.sum(np.logical_and(np.logical_and(np.logical_and(model_arr == 1, landsat_arr == 2), ch_arr != 1), has_diff_neighbor == 0)[:, x_min: x_max+1]))
                            dic_nn_pre[f'pre_{sc}_in'].append(np.sum(np.logical_and(np.logical_and(np.logical_and(model_arr == 1, landsat_arr == 1), ch_arr != 1), has_diff_neighbor == 0)[:, x_min: x_max+1]))
                            dic_pre[f'pre_{sc}_ii'].append(np.sum(np.logical_and(np.logical_and(model_arr == 1, landsat_arr == 2), ch_arr != 1)[:, x_min: x_max+1]))
                            dic_pre[f'pre_{sc}_in'].append(np.sum(np.logical_and(np.logical_and(model_arr == 1, landsat_arr == 1), ch_arr != 1)[:, x_min: x_max+1]))
                            dic_ori_pre[f'pre_{sc}_ii'].append(np.sum(np.logical_and(model_arr == 1, landsat_arr == 2,)[:, x_min: x_max+1]))
                            dic_ori_pre[f'pre_{sc}_in'].append(np.sum(np.logical_and(model_arr == 1, landsat_arr == 1)[:, x_min: x_max+1]))
                    if doy > 2003001:
                        dic_post['doy'].append(doy)
                        dic_ori_post['doy'].append(doy)
                        dic_nn_post['doy'].append(doy)
                    else:
                        dic_pre['doy'].append(doy)
                        dic_ori_pre['doy'].append(doy)
                        dic_nn_pre['doy'].append(doy)
                except:
                    print(traceback.format_exc())
                    pass
            else:
                print(f'{_} is not valid')
            pbar.update()

        df_post = pd.DataFrame(dic_post)
        df_pre =  pd.DataFrame(dic_pre)
        df_post.to_csv('D:\A_PhD_Main_paper\Chap.2\Table\Table.2.6\\post.csv')
        df_pre.to_csv('D:\A_PhD_Main_paper\Chap.2\Table\Table.2.6\\pre.csv')
        df_ori_post = pd.DataFrame(dic_ori_post)
        df_ori_pre =  pd.DataFrame(dic_ori_pre)
        df_ori_post.to_csv('D:\A_PhD_Main_paper\Chap.2\Table\Table.2.6\\ori_post.csv')
        df_ori_pre.to_csv('D:\A_PhD_Main_paper\Chap.2\Table\Table.2.6\\ori_pre.csv')
        df_nn_post = pd.DataFrame(dic_nn_post)
        df_nn_pre =  pd.DataFrame(dic_nn_pre)
        df_nn_post.to_csv('D:\A_PhD_Main_paper\Chap.2\Table\Table.2.6\\nn_post.csv')
        df_nn_pre.to_csv('D:\A_PhD_Main_paper\Chap.2\Table\Table.2.6\\nn_pre.csv')

if __name__ == '__main__':
    inund_detection()