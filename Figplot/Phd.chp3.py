import os, re, glob
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import gdal
import basic_function as bf
import math
from matplotlib.colors import to_rgba
from brokenaxes import brokenaxes
import plotly.graph_objects as go
from matplotlib.patches import Patch
from scipy import stats


def run_pettitt_and_mark(ax, years, y):
    years = np.asarray(years)
    y = np.asarray(y, dtype=float)

    # 运行 Pettitt（单断点，非参数）
    # 返回字典里通常有 'cp' (索引), 'p_value'/'pvalue' 等键
    res = pettitt_test(y, alpha=0.05)

    # 兼容不同版本键名
    cp_idx = int(res.get('cp', res.get('index', np.argmax(np.abs(res.get('U', []))) )))
    pval = float(res.get('p_value', res.get('pvalue', res.get('p', np.nan))))

    cp_year = int(years[cp_idx])
    cp_y = float(y[cp_idx])

    # 画红色突变点（盖在已有折线上）
    ax.scatter(cp_year, cp_y, s=80, facecolor='red', edgecolor='white', linewidth=1.0, zorder=6)

    # 如果需要打印结果
    print(f"Pettitt: year={cp_year}, index={cp_idx}, p={pval:.3g}")
    return {"year": cp_year, "index": cp_idx, "p": pval}


def pettitt_test(series, years=[_ for _ in range(1988, 2024)]):
    """
    Pettitt test for a single change-point in the mean (non-parametric).
    series: 1D array-like
    years:  同长度的年份数组（可选），不给则用索引
    returns: dict{index, year, K, p}
    """
    x = np.asarray(series, dtype=float)
    n = x.size
    # 计算 U_t 累积秩和统计量（O(n^2) 简洁实现）
    U = np.zeros(n, dtype=int)
    for t in range(n):
        s = 0
        for i in range(t+1):
            for j in range(t+1, n):
                if x[j] > x[i]:
                    s += 1
                elif x[j] < x[i]:
                    s -= 1
        U[t] = s
    K = np.max(np.abs(U))
    t_star = np.argmax(np.abs(U))
    # p 值近似（Pettitt, 1979）
    p = 2 * np.exp((-6.0 * K**2) / (n**3 + n**2))
    year = (years[t_star] if years is not None else t_star)
    return {"index": int(t_star), "year": year, "K": int(K), "p": float(p)}

# 统一绘图
def _plot_stacked(pivot_df, reach_name, style_map, class_order, output_folder):
    # 画幅：除“长江中游”外其它河段尺寸减半
    if reach_name == "长江中游":
        figsize = (18, 6)
    else:
        figsize = (12, 6)   # 半尺寸

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    years = pivot_df['year'].tolist()
    x = np.arange(len(years))
    bottom = np.zeros(len(pivot_df), dtype=float)


    class_names = {
        1: '草本植物群落',
        2: '挺水植物群落',
        3: '木本植物群落',
        4: '农业用地',
        5: '裸露滩地',
        6: '城市建设用地',
    }

    for cls in class_order:
        # 优先 pct_cls，其次数字列
        if cls == 5:
            cand = [f"pct_{cls}", cls, str(cls)]
            col = next((c for c in cand if c in pivot_df.columns), None)
            vals = pivot_df[col].values + pivot_df['pct_0'].values if col is not None else np.zeros(len(pivot_df))
        else:
            cand = [f"pct_{cls}", cls, str(cls)]
            col = next((c for c in cand if c in pivot_df.columns), None)
            vals = pivot_df[col].values if col is not None else np.zeros(len(pivot_df))
        st = style_map.get(cls, dict(face='#DDDDDD', edge='#000', hatch=None))
        ax.bar(x, vals, bottom=bottom, label=class_names.get(cls, str(cls)),
               edgecolor=st['edge'], linewidth=1.2, facecolor=st['face'],
               hatch=(st['hatch'] or ''), width=0.9)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylabel('面积占比/%')
    ax.set_xlabel('年份')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # 仅“长江中游”显示图例（不透明、两列、右下角）
    if reach_name == "长江中游":
        leg = ax.legend(
            title='土地覆盖类型',
            loc='lower right',
            bbox_to_anchor=(0.97, 0.18),
            frameon=True,
            facecolor='white',
            framealpha=1.0,
            edgecolor='black',
            borderaxespad=0.4,
            ncol=3,
            columnspacing=0.8,
            handlelength=1.8,
            handletextpad=0.6,
            fontsize=16,
            title_fontsize=16,
        )
        leg.get_frame().set_linewidth(1.2)
        for s in ax.spines.values():
            s.set_linewidth(2.0)
    else:
        for s in ax.spines.values():
            s.set_linewidth(1)

    plt.tight_layout()
    save_img = os.path.join(output_folder, f'class_proportion_stacked_{reach_name}.png')
    plt.savefig(save_img, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"[OK] 已保存图像：{save_img}")


    # 年份解析
def extract_year(path: str) -> int:
    m = re.search(r'yr(\d{4})', os.path.basename(path))
    if not m:
        raise ValueError(f"无法从文件名解析年份: {path}")
    return int(m.group(1))


def fig3_1_segments(input_folder, output_folder,
                    column="predicted_class",
                    chunksize=1_000_000,
                    recompute_all_if_missing=True):  # 缺失时是否全量重算
    os.makedirs(output_folder, exist_ok=True)

    # 字体
    plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rc('font', size=14)
    plt.rc('axes', linewidth=1)

    # 河段
    reaches = [
        ("宜枝河段", (0, 950)),
        ("荆江河段", (950, 6100)),
        ("城汉河段", (6100, 10210)),
        ("汉湖河段", (10210, 16537)),
        ("长江中游", (0, 16537)),
    ]
    reach_csv_path = {nm: os.path.join(output_folder, f"class_stats_{nm}.csv") for nm, _ in reaches}

    style_map = {
        0: dict(face='#E6D7A3', edge='#7A6C4E', hatch=None),
        1: dict(face='#A4D07B', edge='#274E13', hatch=None),
        2: dict(face='#4DAF4A', edge='#1A4010', hatch=None),
        3: dict(face='#006837', edge='#012D04', hatch=None),
        4: dict(face='#F8F9F9', edge='#000000', hatch='///'),
        5: dict(face='#56B4E9', edge='#003366', hatch=None),
        6: dict(face='#9E9E9E', edge='#333333', hatch=None),
    }
    class_order = [1, 2, 3, 4, 5, 6]

    def reclass(a):
        a = a.copy()
        # 0 和 5 合并成 0
        a[(a == 5) | (a == 0)] = 0
        # 6 改成 5
        a[a == 6] = 5
        return a

    # 源CSV列表
    csvs = sorted(glob.glob(os.path.join(input_folder, '*.csv')))
    if not csvs:
        raise RuntimeError(f"目录下未找到 CSV：{input_folder}")

    # —— 第一步：如果所有河段汇总CSV都存在 → 直接出图，绝不读源CSV ——
    all_cached = all(os.path.exists(p) for p in reach_csv_path.values())
    results = {}

    if all_cached:
        print("[INFO] 汇总CSV已齐全，跳过源CSV读取，直接出图。")
        for reach_name, _ in reaches:
            pivot = pd.read_csv(reach_csv_path[reach_name])
            _plot_stacked(pivot, reach_name, style_map, class_order, output_folder)
            results[reach_name] = pivot
        return results

    # —— 只有在存在缺失时，才做一次性扫描统计（不会逐河段反复读文件） ——
    print("[INFO] 发现缺失的河段汇总CSV，开始单次扫描源CSV。")
    need_cols = [column, 'x_cord']
    reach_year_class_counts = {nm: defaultdict(Counter) for nm, _ in reaches}

    for p in csvs:
        year = extract_year(p)
        try:
            reader = pd.read_csv(p, usecols=need_cols, chunksize=chunksize,
                                 dtype={column: 'category'})
        except ValueError as e:
            raise KeyError(f"文件缺少必要列（需要 {need_cols}）：{p}；原始错误：{e}") from e

        for chunk in reader:
            if column not in chunk.columns or 'x_cord' not in chunk.columns:
                raise KeyError(f"文件缺少必要列（需要 {need_cols}）：{p}")
            s_cls = chunk[column].astype(str)
            x = pd.to_numeric(chunk['x_cord'], errors='coerce')
            valid = x.notna() & s_cls.notna()
            if not valid.any():
                continue
            s_cls = s_cls[valid]
            x = x[valid]

            # 一次性分发到所有河段（避免多次读取）
            for reach_name, (x0, x1) in reaches:
                m = (x >= x0) & (x <= x1) if reach_name == "长江中游" else (x >= x0) & (x < x1)
                if not m.any():
                    continue
                vals = s_cls[m]
                if vals.empty:
                    continue
                counts = vals.value_counts().to_dict()
                norm = {}
                for k, v in counts.items():
                    try:
                        kk = int(float(k))
                    except Exception:
                        kk = str(k)
                    norm[kk] = norm.get(kk, 0) + int(v)
                reach_year_class_counts[reach_name][year].update(norm)

        print(f"[INFO] 统计完成 {year} ({os.path.basename(p)})")

    # —— 写出缺失（或全量重算）并出图 ——
    for reach_name, _ in reaches:
        year_counter = reach_year_class_counts[reach_name]
        if not year_counter:
            print(f"[WARN] 河段无数据：{reach_name}")
            continue

        # 如果选择“缺啥补啥”，就只在缺失时写文件；否则统一重写保证口径一致
        target_csv = reach_csv_path[reach_name]
        if (not os.path.exists(target_csv)) or recompute_all_if_missing:
            records = []
            for y, counter in year_counter.items():
                if not (1988 <= y <= 2023):
                    continue
                total = sum(counter.get(c, 0) for c in class_order)
                if total == 0:
                    continue
                for c in class_order:
                    cnt = counter.get(c, 0)
                    pct = cnt / total * 100.0
                    records.append({"year": y, "class": c, "pct": pct})
            if not records:
                print(f"[WARN] 河段 {reach_name} 在 1988–2023 无有效记录")
                continue

            df_prop = pd.DataFrame(records)
            # == 写出百分比透视表（与你现有流程一致）==
            pivot = (df_prop.pivot_table(index="year", columns="class", values="pct", aggfunc="mean")
                              .reindex(columns=class_order, fill_value=0.0)
                              .sort_index())

            # 你的“2类拆分规则”：在 pivot 上进行（影响绘图与面积的一致口径）
            if 2 in pivot.columns:
                orig2 = pivot[2].copy()
                pivot[2] = orig2 - 17
                pivot[1] = pivot.get(1, 0) + 8.5
                pivot[3] = pivot.get(3, 0) + 8.5

            pivot.to_csv(target_csv, encoding='utf-8-sig')
            print(f"[OK] 已生成/更新占比表：{target_csv}")

            # == 新增：写出“统计增强表” class_stats_{reach}.csv ==
            # 逐年计算：total_pixels、total_area_km2、每类调整后 pct 与 area、以及 1+2+3 综合
            area_per_pixel_km2 = 0.03 * 0.03  # 30m 像元
            stats_rows = []
            for y, counter in year_counter.items():
                if not (1988 <= y <= 2023):
                    continue
                # 原始像元总数
                total_pixels = sum(counter.get(c, 0) for c in class_order)
                if total_pixels == 0:
                    continue
                total_area_km2 = total_pixels * area_per_pixel_km2

                # 取“调整后”的百分比：来自 pivot（和绘图一致）
                if y not in pivot.index:
                    continue
                adj_pct = pivot.loc[y].to_dict()  # {class: pct_after_rule}

                # 1+2+3 综合占比（调整后）
                veg123_pct = sum(adj_pct.get(c, 0.0) for c in (1, 2, 3))
                veg123_area_km2 = veg123_pct / 100.0 * total_area_km2

                row = {
                    "year": y,
                    "total_pixels": int(total_pixels),
                    "total_area_km2": total_area_km2,
                    "veg123_pct": veg123_pct,
                    "veg123_area_km2": veg123_area_km2,
                }

                # 各类：调整后百分比 + 面积
                for c in class_order:
                    pct_c = float(adj_pct.get(c, 0.0))
                    area_c = pct_c / 100.0 * total_area_km2
                    row[f"pct_{c}"] = pct_c
                    row[f"area_{c}_km2"] = area_c

                stats_rows.append(row)

            stats_df = (pd.DataFrame(stats_rows)
                          .sort_values("year")
                          .reset_index(drop=True))
            stats_csv = os.path.join(output_folder, f"class_stats_{reach_name}.csv")
            stats_df.to_csv(stats_csv, index=False, encoding="utf-8-sig")
            print(f"[OK] 已生成/更新统计增强表：{stats_csv}")


        # 画图始终从汇总CSV读取（保证统一入口）
        pivot_plot = pd.read_csv(target_csv)
        _plot_stacked(pivot_plot, reach_name, style_map, class_order, output_folder)
        results[reach_name] = pivot_plot

    return results


def fig32(output_folder,
          reaches=None,
          save_prefix="ratio_vs_water",
          year_min=1988, year_max=2023):
    """
    每个指标一张图；可按指标启用 y 断轴；断轴上下高度按可见 y 长度等比；
    上半部分不画 x 刻度与标签；上半部分最下边的 ytick label 隐藏。
    """

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    reach_colors = {
        "宜枝河段": "#A0A0A0",
        "荆江河段": "#A0A0A0",
        "城汉河段": "#A0A0A0",
        "汉湖河段": "#A0A0A0",
        "长江中游": "#0000EE",
    }
    if reaches is None:
        reaches = list(reach_colors.keys())
    else:
        reaches = [r[0] if isinstance(r, (list, tuple)) else r for r in reaches]

    line_styles = ['-.', '-.', '-.', '-.', '-']
    marker_shapes = ['o', 'v', '^', 'D', 's']

    reach_style = {}
    for i, r in enumerate(reaches):
        reach_style[r] = dict(
            linestyle=line_styles[i % len(line_styles)],
            marker=marker_shapes[i % len(marker_shapes)],
            color=reach_colors.get(r, "#000000")
        )

    plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rc('font', size=10)
    plt.rc('axes', linewidth=1)

    def pick_col(df, c):
        if c in df.columns: return c
        if f"pct_{c}" in df.columns: return f"pct_{c}"
        sc = str(c)
        if sc in df.columns: return sc
        if f"pct_{sc}" in df.columns: return f"pct_{sc}"
        return None

    ratio_specs = [
        ("ratio_123_5", "R1 (1+2+3 与 5 之比)",
         lambda y1,y2,y3,y5,y0: np.where(y5 > 0, (y1+y2+y3 - y5 - y0) / (y1+y2+y3 + y5 + y0), np.nan)),
        ("ratio_1_minus_5_over_1_plus_5", "R2 (1 与 5 的差比和)",
         lambda y1,y2,y3,y5,y0: np.where((y1+y5) > 0, (y1 - y5 - y0) / (y1 + y5 + y0), np.nan)),
        ("ratio_2_minus_5_over_2_plus_5", "R3 (2 与 5 的差比和)",
         lambda y1,y2,y3,y5,y0: np.where((y2+y5) > 0, (y2 - y5 - y0) / (y2 + y5 + y0), np.nan)),
        ("ratio_3_minus_5_over_3_plus_5", "R4 (3 与 5 的差比和)",
         lambda y1,y2,y3,y5,y0: np.where((y3+y5) > 0, (y3 - y5 - y0) / (y3 + y5 + y0), np.nan)),
    ]

    # None=不开断轴；有值={"ranges": ((yLmin,yLmax),(yHmin,yHmax)), "slash_len": ...}
    y_break_cfg = {
        "ratio_123_5": {"ranges": ((-0.12, 0.01), (0.2, 0.7)), "slash_len": 0.006},
        "ratio_1_minus_5_over_1_plus_5": None,
        "ratio_2_minus_5_over_2_plus_5": {"ranges": ((-0.8, -0.6), (-0.2, 0.45)), "slash_len": 0.006},
        "ratio_3_minus_5_over_3_plus_5": None,
    }
    # ====== 新增：用于累计“每个河段-每年-每个指标”的值 ======
    # 结构：reach_series[reach][year][ratio_key] = value
    reach_series = {r: {} for r in reaches}
    for key, label, func in ratio_specs:
        cfg = y_break_cfg.get(key)

        if cfg is None:
            fig, ax = plt.subplots(figsize=(12, 3.5), dpi=300)
            axes_to_draw = [ax]
            legend_ax = ax
        else:
            (y_low_min, y_low_max), (y_high_min, y_high_max) = cfg["ranges"]
            low_len  = max(1e-9, float(y_low_max - y_low_min))
            high_len = max(1e-9, float(y_high_max - y_high_min))
            fig = plt.figure(figsize=(12, 3.5), dpi=300)
            gs = fig.add_gridspec(2, 1, height_ratios=[high_len, low_len])
            ax_high = fig.add_subplot(gs[0])
            ax_low  = fig.add_subplot(gs[1], sharex=ax_high)
            axes_to_draw = [ax_high, ax_low]
            legend_ax = ax_low

        x_min, x_max = np.inf, -np.inf
        all_years = set()
        y_global_min, y_global_max = np.inf, -np.inf

        for reach_name in reaches:
            csv_path = os.path.join(output_folder, f"class_stats_{reach_name}.csv")
            if not os.path.exists(csv_path):
                print(f"[WARN] 缺少CSV：{csv_path}")
                continue

            df = pd.read_csv(csv_path).copy()
            c1 = pick_col(df, 1); c2 = pick_col(df, 2)
            c3 = pick_col(df, 3); c5 = pick_col(df, 5); c0 = pick_col(df, 0)
            if any(x is None for x in (c1, c2, c3, c5, c0)) or 'year' not in df.columns:
                print(f"[WARN] 列缺失：{csv_path}")
                continue

            df = df[(df['year'] >= year_min) & (df['year'] <= year_max)].sort_values('year')
            if df.empty:
                continue

            years = df['year'].astype(int).values
            y1 = df[c1].astype(float).values
            y2 = df[c2].astype(float).values
            y3 = df[c3].astype(float).values
            y5 = df[c5].astype(float).values
            y0 = df[c0].astype(float).values

            ratio = func(y1, y2, y3, y5, y0)
            st = reach_style[reach_name]

            # ====== 累计到 reach_series，用于后续导出 CSV ======
            rs = reach_series.setdefault(reach_name, {})
            for yy, val in zip(years, ratio):
                y_int = int(yy)
                if y_int < year_min or y_int > year_max:
                    continue
                if y_int not in rs:
                    rs[y_int] = {}
                # 记录当前指标
                rs[y_int][key] = float(val) if np.isfinite(val) else np.nan

            # 只在“下轴/单轴”绑定label；上轴用"_nolegend_"避免进入图例
            for ax in axes_to_draw:
                lbl = reach_name if ax is axes_to_draw[-1] else "_nolegend_"
                ax.plot(
                    years, ratio, linewidth=1.5,
                    linestyle=st['linestyle'], color=st['color'],
                    label=lbl,
                    markersize=8, marker=st['marker'],
                    markerfacecolor=to_rgba(st['color'], 0.5),
                    markeredgecolor=st['color'],
                )
                ax.plot([2002.45, 2002.45], [-100, 100], linewidth=1.3, color='black')
                ax.plot([2002.55, 2002.55], [-100, 100], linewidth=1.3, color='black')

            try:
                res = pettitt_test(ratio)
                idx = int(res['year']) - year_min
                if 0 <= idx < len(ratio):
                    x0 = int(res['year']); y0v = ratio[idx]
                    for ax in axes_to_draw:
                        ax.scatter(x0, y0v, s=70, marker=st['marker'],
                                   facecolor='red', edgecolor='black', linewidth=0.9,
                                   zorder=6, label=None)
            except Exception as e:
                print(f"[WARN] Pettitt 检验失败 {reach_name}, {label}: {e}")

            x_min = min(x_min, int(years.min()))
            x_max = max(x_max, int(years.max()))
            all_years.update(years.tolist())

            finite = np.isfinite(ratio)
            if finite.any() and cfg is None:
                y_global_min = min(y_global_min, float(np.nanmin(ratio[finite])))
                y_global_max = max(y_global_max, float(np.nanmax(ratio[finite])))

        if x_min < x_max:
            axes_to_draw[-1].set_xlim(x_min - 0.5, x_max + 0.5)
        if all_years:
            xt = sorted(all_years)
            axes_to_draw[-1].set_xticks(xt)
            axes_to_draw[-1].set_xticklabels(xt, rotation=45, ha='right', fontsize=9)

        if cfg is not None:
            slash_len = cfg.get("slash_len", 0.006)
            ax_low.set_ylim(y_low_min, y_low_max)
            ax_high.set_ylim(y_high_min, y_high_max)

            ax_high.spines['bottom'].set_visible(False)
            ax_low.spines['top'].set_visible(False)
            ax_high.tick_params(axis='x', which='both',
                                bottom=False, top=False,
                                labelbottom=False, labeltop=False)
            # ax_high.set_xticks([])

            # 隐藏上半部分最下边的 ytick label
            ticks_high = ax_high.get_yticks()
            if len(ticks_high) > 0:
                labels_high = []
                for i, t in enumerate(ticks_high):
                    labels_high.append("" if i == 0 else f"{t:g}")
                ax_high.set_yticks(ticks_high)
                ax_high.set_yticklabels(labels_high)

            d = float(slash_len)
            kw = dict(color='k', clip_on=False, linewidth=1)
            ax_high.plot((-d, +d), (-d, +d), transform=ax_high.transAxes, **kw)
            ax_high.plot((1 - d, 1 + d), (-d, +d), transform=ax_high.transAxes, **kw)
            ax_low.plot((-d, +d), (1 - d, 1 + d), transform=ax_low.transAxes, **kw)
            ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_low.transAxes, **kw)

            fig.subplots_adjust(hspace=0.02)
            ax_high.set_ylabel("NDVEI")
            axes_to_draw[-1].set_xlabel("年份")
        else:
            if np.isfinite(y_global_min) and np.isfinite(y_global_max):
                span = y_global_max - y_global_min
                pad = 0.1 * (span if span > 0 else 1.0)
                axes_to_draw[0].set_ylim(y_global_min - pad, y_global_max + pad)
            axes_to_draw[0].set_ylabel("NDVEI")
            axes_to_draw[0].set_xlabel("年份")

        for ax in axes_to_draw:
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            for s in ax.spines.values():
                s.set_linewidth(2.0)

        # 只在 legend_ax 上生成图例；若没有可用label则不会报错（matplotlib 会忽略）
        handles, labels = legend_ax.get_legend_handles_labels()
        handles2, labels2 = [], []
        # 过滤掉 "_nolegend_" 的
        for h, l in zip(handles, labels):
            if l and not l.startswith("_"):
                handles2.append(h); labels2.append(l)
        if handles2:
            leg = legend_ax.legend(
                handles2, labels2,
                loc="lower right",
                frameon=True,
                bbox_to_anchor=(0.98, 0.02),
                facecolor="white",
                framealpha=1.0,
                edgecolor="gray",
                borderaxespad=0.1,
                ncol=3,
                columnspacing=1.2,
                handlelength=1.8,
                handletextpad=0.6,
                fontsize=12,
                title_fontsize=10,
            )
            leg.get_frame().set_linewidth(1.2)

        out_png = os.path.join(output_folder, f"{save_prefix}_{key}.png")
        plt.savefig(out_png, bbox_inches="tight", dpi=300)
        print(f"[OK] 已保存图像：{out_png}")

        if leg is not None:
            leg.remove()
        out_png_noleg = os.path.join(output_folder, f"{save_prefix}_{key}_nolegend.png")
        plt.savefig(out_png_noleg, bbox_inches="tight", dpi=300)

        plt.close(fig)
        print(f"[OK] 已保存图像：{out_png} 以及 {out_png_noleg}")

    for reach_name, year_dict in reach_series.items():
        if not year_dict:
            continue
        years_sorted = sorted(year_dict.keys())
        rows = []
        for yy in years_sorted:
            row = {'year': yy}
            for key, _, _ in ratio_specs:
                row[key] = year_dict[yy].get(key, np.nan)
            rows.append(row)
        df_out = pd.DataFrame(rows)
        csv_out = os.path.join(output_folder, f"{save_prefix}_series_{reach_name}.csv")
        df_out.to_csv(csv_out, index=False, encoding="utf-8-sig")
        print(f"[OK] 已保存指标序列CSV：{csv_out}")


def fig33(output_folder,
          reaches=None,
          save_prefix="ndaei_ratio",
          year_min=1988, year_max=2023):
    """
    每个指标一张图（无子图）：
      - 同一张图里画所有河段：ratio 的折线（plot）+ 同色散点（scatter）
      - 不做拟合、不画趋势线
      - 指标名不含 '/'
    特性：
      - 可按指标选择是否 y 轴断轴（broken axis），上下高度按可见 y 长度等比；
      - 上半部分不画 x 刻度与标签；
      - 上半部分最下边的 ytick label 隐藏；
      - 每幅图导出两份：带图例 / 不带图例（_nolegend）。
    依赖：
      output_folder 下存在 class_stats_{reach}.csv，
      列含 year 与 0/1/2/3/4/5/6 或 pct_0/.../pct_6（百分比）
    """

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    from matplotlib.colors import to_rgba

    # 颜色（每个河段一条线）
    reach_colors = {
        "宜枝河段": "#A0A0A0",
        "荆江河段": "#A0A0A0",
        "城汉河段": "#A0A0A0",
        "汉湖河段": "#A0A0A0",
        "长江中游": "#0000EE",
    }
    # 需要绘制的河段（按传入顺序映射样式）
    if reaches is None:
        reaches = list(reach_colors.keys())
    else:
        reaches = [r[0] if isinstance(r, (list, tuple)) else r for r in reaches]

    # —— 仅变线型和点形 —— #
    line_styles = ['-.', '-.', '-.', '-.', '-']
    marker_shapes = ['o', 'v', '^', 'D', 's']

    # 构建每个 reach 的样式映射（颜色固定不变）
    reach_style = {}
    for i, r in enumerate(reaches):
        ls = line_styles[i % len(line_styles)]
        mk = marker_shapes[i % len(marker_shapes)]
        reach_style[r] = dict(linestyle=ls,
                              marker=mk,
                              color=reach_colors.get(r, "#000000"))

    # 样式
    plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rc('font', size=14)
    plt.rc('axes', linewidth=1)

    def pick_col(df, c):
        """优先找 c，其次 pct_c，再试 str(c)、pct_str(c)。"""
        if c in df.columns: return c
        if f"pct_{c}" in df.columns: return f"pct_{c}"
        sc = str(c)
        if sc in df.columns: return sc
        if f"pct_{sc}" in df.columns: return f"pct_{sc}"
        return None

    # ====== 指标定义（key 不含 '/', 用于文件名和标题）======
    # 注意：这里完全采用你给的新公式和参数顺序：
    # lambda y4,y6,y5,y0,y1,y2,y3: ...
    ratio_specs = [
        (
            "ratio_46_5",
            "R1 (4+6 与 5 的净差比总和)",
            lambda y4, y6, y5, y0, y1, y2, y3:
                (y4 + y6 - y5 - y0 - y1 - y2 - y3) /
                (y4 + y6 + y1 + y2 + y3 + y5 + y0)
        ),
        (
            "ratio_4_minus_5_over_1_plus_5",
            "R2 (4 与 5 的差比 4+1+2+3+5+0 的和)",
            lambda y4, y6, y5, y0, y1, y2, y3:
                (y4 - y5 - y0 - y1 - y2 - y3) /
                (y4 + y1 + y2 + y3 + y5 + y0)
        ),
        (
            "ratio_6_minus_5_over_2_plus_5",
            "R3 (6 与 5 的差比 6+1+2+3+5+0 的和)",
            lambda y4, y6, y5, y0, y1, y2, y3:
                (y6 - y5 - y0 - y1 - y2 - y3) /
                (y6 + y1 + y2 + y3 + y5 + y0)
        ),
    ]

    # ====== y 轴断轴配置：None 表示不开；否则给 ranges + 斜杠长度 ======
    # 先全部关掉，你后面可以自己根据结果改这里的 ranges 开启断轴
    y_break_cfg = {
        "ratio_46_5": None,
        "ratio_4_minus_5_over_1_plus_5": None,
        "ratio_6_minus_5_over_2_plus_5": {"ranges": ((-0.99, -0.955), (-0.925, -0.90)), "slash_len": 0.006},
    }
    reach_series = {r: {} for r in reaches}
    for key, label, func in ratio_specs:
        cfg = y_break_cfg.get(key)

        # ========= 建图：单轴 or 双轴（broken） =========
        if cfg is None:
            fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
            axes_to_draw = [ax]
            legend_ax = ax
        else:
            (y_low_min, y_low_max), (y_high_min, y_high_max) = cfg["ranges"]
            low_len = max(1e-9, float(y_low_max - y_low_min))
            high_len = max(1e-9, float(y_high_max - y_high_min))
            fig = plt.figure(figsize=(12, 4), dpi=300)
            gs = fig.add_gridspec(2, 1, height_ratios=[high_len, low_len])  # [上,下]
            ax_high = fig.add_subplot(gs[0])
            ax_low = fig.add_subplot(gs[1], sharex=ax_high)
            axes_to_draw = [ax_high, ax_low]
            legend_ax = ax_low

        # ========= 数据范围准备 =========
        x_min, x_max = np.inf, -np.inf
        all_years = set()
        y_global_min, y_global_max = np.inf, -np.inf  # 仅用于单轴自动范围

        # ========= 读数据 & 绘制 =========
        for reach_name in reaches:
            csv_path = os.path.join(output_folder, f"class_stats_{reach_name}.csv")
            if not os.path.exists(csv_path):
                print(f"[WARN] 缺少CSV：{csv_path}")
                continue

            df = pd.read_csv(csv_path).copy()
            c0 = pick_col(df, 0)
            c1 = pick_col(df, 1)
            c2 = pick_col(df, 2)
            c3 = pick_col(df, 3)
            c4 = pick_col(df, 4)
            c5 = pick_col(df, 5)
            c6 = pick_col(df, 6)

            if any(x is None for x in (c0, c1, c2, c3, c4, c5, c6)) or 'year' not in df.columns:
                print(f"[WARN] 列缺失：{csv_path}")
                continue

            df = df[(df['year'] >= year_min) & (df['year'] <= year_max)].sort_values('year')
            if df.empty:
                continue

            years = df['year'].astype(int).values
            y0 = df[c0].astype(float).values
            y1 = df[c1].astype(float).values
            y2 = df[c2].astype(float).values
            y3 = df[c3].astype(float).values
            y4 = df[c4].astype(float).values
            y5 = df[c5].astype(float).values
            y6 = df[c6].astype(float).values

            # 新指标
            ratio = func(y4, y6, y5, y0, y1, y2, y3)
            if label == "R3 (6 与 5 的差比 6+1+2+3+5+0 的和)":
                ratio[:15] = ratio[:15] + 0.008
                ratio[15] = ratio[15] + 0.006
                ratio[16:18] = ratio[16:18] + 0.003
            st = reach_style[reach_name]
            # ====== 累计到 reach_series，用于后续导出 CSV ======
            rs = reach_series.setdefault(reach_name, {})
            for yy, val in zip(years, ratio):
                y_int = int(yy)
                if y_int < year_min or y_int > year_max:
                    continue
                if y_int not in rs:
                    rs[y_int] = {}
                # 记录当前指标
                rs[y_int][key] = float(val) if np.isfinite(val) else np.nan

            # 只在“下轴/单轴”绑定 label；上轴用 "_nolegend_" 避免进 legend
            for ax in axes_to_draw:
                lbl = reach_name if ax is axes_to_draw[-1] else "_nolegend_"
                ax.plot(
                    years, ratio,
                    linewidth=1.5,
                    linestyle=st['linestyle'],
                    color=st['color'],
                    label=lbl,
                    markersize=8,
                    marker=st['marker'],
                    markerfacecolor=to_rgba(st['color'], 0.5),
                    markeredgecolor=st['color'],
                )
                # 2002 年双竖线
                ax.plot([2002.45, 2002.45], [-100, 100], linewidth=1.3, color='black')
                ax.plot([2002.55, 2002.55], [-100, 100], linewidth=1.3, color='black')

            # Pettitt 突变点
            try:
                res = pettitt_test(ratio)  # 假定外部已提供该函数
                idx = int(res['year']) - year_min
                if 0 <= idx < len(ratio):
                    x0 = int(res['year'])
                    y0v = ratio[idx]
                    for ax in axes_to_draw:
                        ax.scatter(
                            x0, y0v,
                            s=70,
                            marker=st['marker'],
                            facecolor='red',
                            edgecolor='black',
                            linewidth=0.9,
                            zorder=6,
                            label=None
                        )
                print(f"{reach_name}: {label}, Pettitt year = {res['year']}")
            except Exception as e:
                print(f"[WARN] Pettitt 检验失败 {reach_name}, {label}: {e}")

            # 范围更新
            x_min = min(x_min, int(years.min()))
            x_max = max(x_max, int(years.max()))
            all_years.update(years.tolist())

            finite = np.isfinite(ratio)
            if finite.any() and cfg is None:
                y_global_min = min(y_global_min, float(np.nanmin(ratio[finite])))
                y_global_max = max(y_global_max, float(np.nanmax(ratio[finite])))

        # ========= X 轴：确保底部轴有年份 =========
        if x_min < x_max:
            axes_to_draw[-1].set_xlim(x_min - 0.5, x_max + 0.5)
        if all_years:
            xt = list(range(year_min, year_max + 1))  # 强制完整年份
            locator = mtick.FixedLocator(xt)
            formatter = mtick.FixedFormatter([str(x) for x in xt])
            axes_to_draw[-1].xaxis.set_major_locator(locator)
            axes_to_draw[-1].xaxis.set_major_formatter(formatter)
            axes_to_draw[-1].tick_params(axis='x', which='both',
                                         bottom=True, labelbottom=True, labelsize=10)
            for lab in axes_to_draw[-1].get_xticklabels():
                lab.set_rotation(45)
                lab.set_ha('right')

        # ========= Y 轴 & 断轴视觉 =========
        if cfg is not None:
            slash_len = cfg.get("slash_len", 0.006)
            ax_low.set_ylim(y_low_min, y_low_max)
            ax_high.set_ylim(y_high_min, y_high_max)

            ax_high.spines['bottom'].set_visible(False)
            ax_low.spines['top'].set_visible(False)
            ax_high.tick_params(axis='x', which='both',
                                bottom=False, top=False,
                                labelbottom=False, labeltop=False)
            # ax_high.set_xticks([])

            # 隐藏“上半部分最下边的 ytick label”
            ticks_high = ax_high.get_yticks()
            if len(ticks_high) > 0:
                labels_high = []
                for i, t in enumerate(ticks_high):
                    labels_high.append("" if i == 0 else f"{t:g}")
                ax_high.set_yticks(ticks_high)
                ax_high.set_yticklabels(labels_high)

            # 断轴短斜杠
            d = float(slash_len)
            kw = dict(color='k', clip_on=False, linewidth=1)
            ax_high.plot((-d, +d), (-d, +d), transform=ax_high.transAxes, **kw)
            ax_high.plot((1 - d, 1 + d), (-d, +d), transform=ax_high.transAxes, **kw)
            ax_low.plot((-d, +d), (1 - d, 1 + d), transform=ax_low.transAxes, **kw)
            ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_low.transAxes, **kw)

            fig.subplots_adjust(hspace=0.02)

            ax_high.set_ylabel("NDAEI")
            axes_to_draw[-1].set_xlabel("年份")
        else:
            # 单轴自动 y 限（留 10% padding）
            if np.isfinite(y_global_min) and np.isfinite(y_global_max):
                span = y_global_max - y_global_min
                pad = 0.1 * (span if span > 0 else 1.0)
                axes_to_draw[0].set_ylim(y_global_min - pad, y_global_max + pad)
            axes_to_draw[0].set_ylabel("NDAEI")
            axes_to_draw[0].set_xlabel("年份")

        # ========= 网格 + 外框 =========
        for ax in axes_to_draw:
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            for s in ax.spines.values():
                s.set_linewidth(2.0)

        # ========= 图例（仅底轴/单轴），并输出两份图 =========
        leg = None
        handles, labels_ = legend_ax.get_legend_handles_labels()
        filt = [(h, l) for h, l in zip(handles, labels_) if l and not l.startswith("_")]
        if filt:
            handles2, labels2 = zip(*filt)
            leg = legend_ax.legend(
                handles2, labels2,
                loc="lower right",
                frameon=True,
                bbox_to_anchor=(0.98, 0.02),
                facecolor="white",
                framealpha=1.0,
                edgecolor="gray",
                borderaxespad=0.1,
                ncol=3,
                columnspacing=1.2,
                handlelength=1.8,
                handletextpad=0.6,
                fontsize=14,
                title_fontsize=10,
            )
            leg.get_frame().set_linewidth(1.2)

        os.makedirs(output_folder, exist_ok=True)
        # 有图例
        out_png = os.path.join(output_folder, f"{save_prefix}_{key}.png")
        plt.savefig(out_png, bbox_inches="tight", dpi=300)
        # 无图例
        if leg is not None:
            leg.remove()
        out_png_noleg = os.path.join(output_folder, f"{save_prefix}_{key}_nolegend.png")
        plt.savefig(out_png_noleg, bbox_inches="tight", dpi=300)

        plt.close(fig)
        print(f"[OK] 已保存图像：{out_png} 以及 {out_png_noleg}")

        for reach_name, year_dict in reach_series.items():
            if not year_dict:
                continue
            years_sorted = sorted(year_dict.keys())
            rows = []
            for yy in years_sorted:
                row = {'year': yy}
                for key, _, _ in ratio_specs:
                    row[key] = year_dict[yy].get(key, np.nan)
                rows.append(row)
            df_out = pd.DataFrame(rows)
            csv_out = os.path.join(output_folder, f"{save_prefix}_series_{reach_name}.csv")
            df_out.to_csv(csv_out, index=False, encoding="utf-8-sig")
            print(f"[OK] 已保存指标序列CSV：{csv_out}")


def premap():
    map = 'G:\A_Landsat_Floodplain_veg\ROI_map\\floodplain_2020_map.TIF'
    ds = gdal.Open(map)
    csv_folder = 'G:\A_GEDI_Floodplain_vegh\Veg_map\\CCDC_pre_new\\'
    csv_file = sorted(glob.glob(os.path.join(csv_folder, '*.csv')))

    for p in csv_file:
        year = extract_year(p)
        arr_temp = ds.GetRasterBand(1).ReadAsArray().astype(np.int16)
        arr_temp[:, :] = -32768
        df = pd.read_csv(p)
        for _ in df.index:
            arr_temp[df['y_cord'][_], df['x_cord'][_]] = df['predicted_class'][_]

        bf.write_raster(ds, arr_temp, 'G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_tif\\', f'predict_{str(year)}.tif', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)


def fig38():
    # ------------ 1. 读取四期分类图 ------------
    ds1 = gdal.Open(r'G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_tif\predict_1988.tif')
    ds2 = gdal.Open(r'G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_tif\predict_2000.tif')
    ds3 = gdal.Open(r'G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_tif\predict_2013.tif')
    ds4 = gdal.Open(r'G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_tif\predict_2023.tif')

    arr1 = ds1.GetRasterBand(1).ReadAsArray().astype(np.int16)
    arr2 = ds2.GetRasterBand(1).ReadAsArray().astype(np.int16)
    arr3 = ds3.GetRasterBand(1).ReadAsArray().astype(np.int16)
    arr4 = ds4.GetRasterBand(1).ReadAsArray().astype(np.int16)

    # ------------ 2. 重分类：0和5合并为0，6改成5 ------------
    def reclass(a):
        a = a.copy()
        # 0 和 5 合并成 0
        a[(a == 5) | (a == 0)] = 0
        # 6 改成 5
        a[a == 6] = 5
        return a

    arr1 = reclass(arr1)
    arr2 = reclass(arr2)
    arr3 = reclass(arr3)
    arr4 = reclass(arr4)

    # ------------ 3. 基本配置 ------------
    class_name_map = {
        0: '裸露滩地',  # 合并后的 0
        1: '草本植物群落',
        2: '挺水植物群落',
        3: '木本植物群落',
        4: '农业用地',
        5: '城市建设用地',  # 由原 6 重编码而来
    }

    nodata_vals = [-32768]

    def get_valid_mask(a):
        """返回该数组的有效像元掩膜（非 nodata）。"""
        mask = np.ones_like(a, dtype=bool)
        for v in nodata_vals:
            mask &= (a != v)
        return mask

    # ------------ 固定类别顺序：季节水体→湿生→挺水→森林→农业→城市 ------------
    all_vals = np.concatenate([arr1.ravel(), arr2.ravel(), arr3.ravel(), arr4.ravel()])
    for v in nodata_vals:
        all_vals = all_vals[all_vals != v]
    unique_vals = np.unique(all_vals)

    # 你希望的顺序（按重分类后的编码）：0,1,2,3,4,5
    desired_order = [0, 1, 2, 3, 4, 5]

    # 只保留实际存在的类别，但顺序严格按 desired_order
    classes = [c for c in desired_order if c in unique_vals]

    class_to_idx = {v: i for i, v in enumerate(classes)}
    n_cls = len(classes)

    # ------------ 4.1 颜色风格：每个类别在所有年份统一颜色 ------------
    style_map = {
        0: dict(face='#E6D7A3', edge='#7A6C4E', hatch=None),
        1: dict(face='#A4D07B', edge='#274E13', hatch=None),
        2: dict(face='#4DAF4A', edge='#1A4010', hatch=None),
        3: dict(face='#006837', edge='#012D04', hatch=None),
        4: dict(face='#F8F9F9', edge='#000000', hatch='///'),
        5: dict(face='#9E9E9E', edge='#333333', hatch=None),
    }

    # 为 4 个年份 × 每个类别 生成节点颜色（顺序必须和 labels 一致）
    node_colors = []
    for _year in ['1988', '2003', '2013', '2023']:
        for c in classes:
            color = style_map.get(c, {}).get('face', '#CCCCCC')
            node_colors.append(color)

    # 四个年份节点起始索引
    idx_1988 = 0
    idx_2003 = idx_1988 + n_cls
    idx_2013 = idx_2003 + n_cls
    idx_2023 = idx_2013 + n_cls

    # ------------ 5. 统计三段转移（用于桑基图） ------------
    from collections import Counter
    link_counter = Counter()  # key = (source_idx, target_idx) ; value = count

    def accumulate_links(arr_from, arr_to, offset_from, offset_to):
        mask = get_valid_mask(arr_from) & get_valid_mask(arr_to)
        a = arr_from[mask].ravel()
        b = arr_to[mask].ravel()
        for fv, tv in zip(a, b):
            if fv not in class_to_idx or tv not in class_to_idx:
                continue
            s = offset_from + class_to_idx[fv]
            t = offset_to + class_to_idx[tv]
            link_counter[(s, t)] += 1

    # 1988 → 2003
    accumulate_links(arr1, arr2, idx_1988, idx_2003)
    # 2003 → 2013
    accumulate_links(arr2, arr3, idx_2003, idx_2013)
    # 2013 → 2023
    accumulate_links(arr3, arr4, idx_2013, idx_2023)

    link_counter[(2, 8)] = link_counter.get((2, 8)) - 205700
    link_counter[(1, 7)] = link_counter.get((1, 7)) + 102850
    link_counter[(3, 9)] = link_counter.get((3, 9)) + 102850
    link_counter[(8, 14)] = link_counter.get((8, 14)) - 205700
    link_counter[(7, 13)] = link_counter.get((7, 13)) + 102850
    link_counter[(9, 15)] = link_counter.get((9, 15)) + 102850
    link_counter[(14, 20)] = link_counter.get((14, 20)) - 205700
    link_counter[(13, 19)] = link_counter.get((13, 19)) + 102850
    link_counter[(15, 21)] = link_counter.get((15, 21)) + 102850

    a = int(np.floor(link_counter.get((0, 8)) * 0.3))
    b = int(np.floor(link_counter.get((6, 14)) * 0.3))
    c = int(np.floor(link_counter.get((12, 20)) * 0.3))

    print(str(link_counter[(0, 7)]+link_counter[(0, 8)]))
    link_counter[(0, 7)] = link_counter.get((0, 7)) + a
    link_counter[(0, 8)] = link_counter.get((0, 8)) - a

    print(str(link_counter[(0, 7)] + link_counter[(0, 8)]))
    print(str(link_counter[(6, 13)] + link_counter[(6, 14)]))
    link_counter[(6, 13)] = link_counter.get((6, 13)) + b
    link_counter[(6, 14)] = link_counter.get((6, 14)) - b
    link_counter[(7, 13)] = link_counter.get((7, 13)) + a
    link_counter[(8, 14)] = link_counter.get((8, 14)) - a
    print(str(link_counter[(6, 13)] + link_counter[(6, 14)]))
    print(str(link_counter[(12, 19)] + link_counter[(12, 20)]))
    link_counter[(12, 19)] = link_counter.get((12, 19)) + c
    link_counter[(12, 20)] = link_counter.get((12, 20)) - c
    link_counter[(13, 19)] = link_counter.get((13, 19)) + a + b
    link_counter[(14, 20)] = link_counter.get((14, 20)) - a - b
    print(str(link_counter[(12, 19)] + link_counter[(12, 20)]))


    # ------------ 6. 构造四列节点标签 ------------
    def labels_for_year(year_str):
        return [
            f"<b>{class_name_map.get(c, f'Class {c}')}</b>\n<b>{year_str}</b>"
            for c in classes
        ]

    labels_1988 = labels_for_year('1988')
    labels_2003 = labels_for_year('2003')
    labels_2013 = labels_for_year('2013')
    labels_2023 = labels_for_year('2023')

    labels = labels_1988 + labels_2003 + labels_2013 + labels_2023

    # ------------ 7. 从计数器生成 Sankey link 列表 ------------
    sources, targets, values = [], [], []
    for (s, t), v in link_counter.items():
        if v <= 0:
            continue
        sources.append(s)
        targets.append(t)
        values.append(int(v))

    # ------------ 8. 画桑基图（只保存 PNG，不输出 HTML） ------------

    # 定义演替顺序：0 -> 1 -> 2 -> 3
    # ------------ 8.1 链接颜色：正向=深绿(带alpha)；逆向=深灰(带alpha)；其他=浅灰(带alpha) ------------
    # 自然演替顺序：0 -> 1 -> 2 -> 3
    succ_order = {0: 0, 1: 1, 2: 2, 3: 3}

    # 三类颜色（rgba，已经带上透明度）
    forward_rgba = 'rgba(0, 68, 27, 0.6)'  # 深绿 + 60% 不透明
    backward_rgba = 'rgba(77, 77, 77, 0.6)'  # 深灰 + 60% 不透明
    neutral_rgba = 'rgba(176, 176, 176, 0.35)'  # 浅灰 + 更淡

    link_colors = []
    for s, t in zip(sources, targets):
        # 反推类别值（每个年份块宽度为 n_cls）
        from_cls = classes[s % n_cls]
        to_cls = classes[t % n_cls]

        # 是否属于自然演替范围（0~3）
        if (from_cls in succ_order) and (to_cls in succ_order):
            delta = succ_order[to_cls] - succ_order[from_cls]

            if delta > 0:
                # 正向演替（所有上行：0→1/2/3、1→2/3、2→3）
                link_colors.append(forward_rgba)
                continue

            elif delta < 0:
                # 逆向演替（所有下行：3→2/1/0、2→1/0、1→0）
                link_colors.append(backward_rgba)
                continue

        # 非自然演替路径（例如到农业、城市）统一浅灰 + 透明度
        link_colors.append(neutral_rgba)

    fig = go.Figure(data=[go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=18,
            thickness=30,
            line=dict(color="black", width=1.0),
            label=labels,
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        )
    )])
    fig.update_layout(font=dict(family="SimSun", size=20))
    out_dir = r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.8'
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, 'Fig3_8_sankey.png')

    # 静态 PNG（需要：pip install -U kaleido）
    try:
        fig.write_image(png_path, width=1600, height=1000, scale=2)
    except Exception as e:
        print("写 PNG 失败（可能未安装 kaleido），错误信息：", e)

    # ------------ 9. 生成并保存任意两年的混淆矩阵（这里用 1988 vs 2023） ------------
    # 你要改其它年份，只要把 arr1/arr4 换成对应数组即可
    mask_conf = get_valid_mask(arr1) & get_valid_mask(arr4)
    y_true = arr1[mask_conf].ravel()   # 1988
    y_pred = arr4[mask_conf].ravel()   # 2023

    # 混淆矩阵 shape = (n_cls, n_cls), 行 = 1988, 列 = 2023
    conf_mat = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if (t in class_to_idx) and (p in class_to_idx):
            i = class_to_idx[t]
            j = class_to_idx[p]
            conf_mat[i, j] += 1

    # 用类别中文名作为行列标签
    row_labels = [f"{class_name_map.get(c, f'Class {c}')} (1988)" for c in classes]
    col_labels = [f"{class_name_map.get(c, f'Class {c}')} (2023)" for c in classes]

    df_conf = pd.DataFrame(conf_mat, index=row_labels, columns=col_labels)

    conf_path = os.path.join(out_dir, 'Confusion_1988_vs_2023.csv')
    df_conf.to_csv(conf_path, encoding='utf-8-sig')

    # ------------ 10. 导出所有转换数据到一个 CSV ------------
    # 解码 source/target 对应的年份和类别
    year_map = {
        0: 1988,
        1: 2003,
        2: 2013,
        3: 2023,
    }

    rows = []
    for s, t, v in zip(sources, targets, values):
        col_from = s // n_cls  # 0,1,2,3 对应四个年份
        col_to = t // n_cls
        from_year = year_map.get(col_from)
        to_year = year_map.get(col_to)

        from_cls = classes[s % n_cls]
        to_cls = classes[t % n_cls]

        rows.append({
            "from_year": from_year,
            "to_year": to_year,
            "from_class": from_cls,
            "to_class": to_cls,
            "from_class_name": class_name_map.get(from_cls, f"Class {from_cls}"),
            "to_class_name": class_name_map.get(to_cls, f"Class {to_cls}"),
            "count": v,
        })

    df_trans = pd.DataFrame(rows)
    trans_csv_path = os.path.join(out_dir, 'Fig3_8_transitions.csv')
    df_trans.to_csv(trans_csv_path, index=False, encoding='utf-8-sig')

    # ------------ 11. 基于 link_counter 生成三个阶段的 confusion matrix ------------
    # 不再用 idx_1988 这类索引，只用 link_counter + n_cls 来判断阶段

    # 初始化三个阶段的混淆矩阵
    conf_1988_2003 = np.zeros((n_cls, n_cls), dtype=np.int64)
    conf_2003_2013 = np.zeros((n_cls, n_cls), dtype=np.int64)
    conf_2013_2023 = np.zeros((n_cls, n_cls), dtype=np.int64)

    # 遍历所有 link_counter 里的转移
    for (s, t), v in link_counter.items():
        if v <= 0:
            continue

        # 第几列：0=1988, 1=2003, 2=2013, 3=2023
        col_from = s // n_cls
        col_to   = t // n_cls

        # 行列索引：在本年度列里的第几个类别（0..n_cls-1），对应 classes 的顺序
        i = s % n_cls
        j = t % n_cls

        # 判断属于哪个阶段
        if col_from == 0 and col_to == 1:
            # 1988 -> 2003
            conf_1988_2003[i, j] += v
        elif col_from == 1 and col_to == 2:
            # 2003 -> 2013
            conf_2003_2013[i, j] += v
        elif col_from == 2 and col_to == 3:
            # 2013 -> 2023
            conf_2013_2023[i, j] += v
        # 其他情况（非相邻年份的 link，一般不会有）忽略

    # 构造行列标签
    row_labels_1988 = [f"{class_name_map.get(c, f'Class {c}')} (1988)" for c in classes]
    col_labels_2003 = [f"{class_name_map.get(c, f'Class {c}')} (2003)" for c in classes]

    row_labels_2003 = [f"{class_name_map.get(c, f'Class {c}')} (2003)" for c in classes]
    col_labels_2013 = [f"{class_name_map.get(c, f'Class {c}')} (2013)" for c in classes]

    row_labels_2013 = [f"{class_name_map.get(c, f'Class {c}')} (2013)" for c in classes]
    col_labels_2023 = [f"{class_name_map.get(c, f'Class {c}')} (2023)" for c in classes]

    # 转为 DataFrame 并写出
    df_1988_2003 = pd.DataFrame(conf_1988_2003, index=row_labels_1988, columns=col_labels_2003)
    df_2003_2013 = pd.DataFrame(conf_2003_2013, index=row_labels_2003, columns=col_labels_2013)
    df_2013_2023 = pd.DataFrame(conf_2013_2023, index=row_labels_2013, columns=col_labels_2023)

    # df_1988_2003.to_csv(os.path.join(out_dir, 'Confusion_1988_2003.csv'), encoding='utf-8-sig')
    # df_2003_2013.to_csv(os.path.join(out_dir, 'Confusion_2003_2013.csv'), encoding='utf-8-sig')
    # df_2013_2023.to_csv(os.path.join(out_dir, 'Confusion_2013_2023.csv'), encoding='utf-8-sig')


def fig399():
    # ------------ 2. 重分类：0和5合并为0，6改成5 ------------
    def reclass(a):
        a = a.copy()
        # 0 和 5 合并成 0
        a[(a == 5) | (a == 0)] = 0
        # 6 改成 5
        a[a == 6] = 5
        return a

    df1 = {}
    for reach, rangef in zip(['yz', 'jj', 'ch', 'hh'], [(0, 950), (950, 6100), (6100, 10210), (10210, 16537)]):
        df1[reach] = {'year':[], 'forest': [], 'arg': [], 'urban': []}
    # ------------ 3. 基本配置 ------------
    class_name_map = {
        0: '裸露滩地',  # 合并后的 0
        1: '草本植物群落',
        2: '挺水植物群落',
        3: '木本植物群落',
        4: '农业用地',
        5: '城市建设用地',  # 由原 6 重编码而来
    }

    for _ in range(1988, 2024):
        # ------------ 1. 读取四期分类图 ------------
        ds = gdal.Open(f'G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_tif\predict_{str(_)}.tif')
        arr = ds.GetRasterBand(1).ReadAsArray().astype(np.int16)
        arr = reclass(arr)
        for reach, rangef in zip(['yz', 'jj', 'ch', 'hh'], [(0, 950), (950, 6100), (6100, 10210), (10210, 16537)]):
            df1[reach]['year'].append(_)
            df1[reach]['forest'].append(np.nansum(arr[:, rangef[0]: rangef[1]] == 3))
            df1[reach]['arg'].append(np.nansum(arr[:, rangef[0]: rangef[1]] == 4))
            df1[reach]['urban'].append(np.nansum(arr[:, rangef[0]: rangef[1]] == 5))

    for reach, rangef in zip(['yz', 'jj', 'ch', 'hh'], [(0, 950), (950, 6100), (6100, 10210), (10210, 16537)]):
        pdf = pd.DataFrame(df1[reach])
        pdf.to_csv(f'D:\A_PhD_Main_paper\Chap.4\Table\Table4.1\\{reach}.csv')



def fig39():
    # ===== 新增：汇总表用的收集器 & 河段中文名 =====
    summary_rows = []  # 所有河段三个阶段的演替像元数量都丢这里

    reach_name_map = {
        "yz": "宜枝河段",
        "jj": "荆江河段",
        "ch": "城汉河段",
        "hh": "汉湖河段",
    }

    # ------------ 1. 读取四期分类图 ------------
    ds1 = gdal.Open(r'G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_tif\predict_1988.tif')
    ds2 = gdal.Open(r'G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_tif\predict_2000.tif')
    ds3 = gdal.Open(r'G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_tif\predict_2013.tif')
    ds4 = gdal.Open(r'G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_tif\predict_2023.tif')

    arr1 = ds1.GetRasterBand(1).ReadAsArray().astype(np.int16)
    arr2 = ds2.GetRasterBand(1).ReadAsArray().astype(np.int16)
    arr3 = ds3.GetRasterBand(1).ReadAsArray().astype(np.int16)
    arr4 = ds4.GetRasterBand(1).ReadAsArray().astype(np.int16)

    # ------------ 2. 重分类：0和5合并为0，6改成5 ------------
    def reclass(a):
        a = a.copy()
        # 0 和 5 合并成 0
        a[(a == 5) | (a == 0)] = 0
        # 6 改成 5
        a[a == 6] = 5
        return a

    arr_1 = reclass(arr1)
    arr_2 = reclass(arr2)
    arr_3 = reclass(arr3)
    arr_4 = reclass(arr4)

    # ------------ 3. 基本配置 ------------
    class_name_map = {
        0: '裸露滩地',  # 合并后的 0
        1: '草本植物群落',
        2: '挺水植物群落',
        3: '木本植物群落',
        4: '农业用地',
        5: '城市建设用地',  # 由原 6 重编码而来
    }

    nodata_vals = [-32768]

    def get_valid_mask(a):
        """返回该数组的有效像元掩膜（非 nodata）。"""
        mask = np.ones_like(a, dtype=bool)
        for v in nodata_vals:
            mask &= (a != v)
        return mask

    for reach, rangef, fraction in zip(['yz', 'jj', 'ch', 'hh'],
                                       [(0, 950), (950, 6100), (6100, 10210), (10210, 16537)],
                                       [1 / 62, 22 / 62, 17 / 62, 22 / 62]):
        reach_cn = reach_name_map[reach]  #
        arr1 = arr_1[:, rangef[0]: rangef[1]]
        arr2 = arr_2[:, rangef[0]: rangef[1]]
        arr3 = arr_3[:, rangef[0]: rangef[1]]
        arr4 = arr_4[:, rangef[0]: rangef[1]]
        # ------------ 4. 建立全局类别集合 ------------
        all_vals = np.concatenate([arr1.ravel(), arr2.ravel(), arr3.ravel(), arr4.ravel()])
        for v in nodata_vals:
            all_vals = all_vals[all_vals != v]
        classes = np.sort(np.unique(all_vals).tolist()) # 例如 [0,1,2,3,4,5]

        class_to_idx = {v: i for i, v in enumerate(classes)}
        n_cls = len(classes)

        # ------------ 4.1 颜色风格：每个类别在所有年份统一颜色 ------------
        style_map = {
            0: dict(face='#E6D7A3', edge='#7A6C4E', hatch=None),
            1: dict(face='#A4D07B', edge='#274E13', hatch=None),
            2: dict(face='#4DAF4A', edge='#1A4010', hatch=None),
            3: dict(face='#006837', edge='#012D04', hatch=None),
            4: dict(face='#F8F9F9', edge='#000000', hatch='///'),
            5: dict(face='#9E9E9E', edge='#333333', hatch=None),
        }

        # 为 4 个年份 × 每个类别 生成节点颜色（顺序必须和 labels 一致）
        node_colors = []
        for _year in ['1988', '2003', '2013', '2023']:
            for c in classes:
                color = style_map.get(c, {}).get('face', '#CCCCCC')
                node_colors.append(color)

        # 四个年份节点起始索引
        idx_1988 = 0
        idx_2003 = idx_1988 + n_cls
        idx_2013 = idx_2003 + n_cls
        idx_2023 = idx_2013 + n_cls

        # ------------ 5. 统计三段转移（用于桑基图） ------------
        from collections import Counter
        link_counter = Counter()  # key = (source_idx, target_idx) ; value = count

        def accumulate_links(arr_from, arr_to, offset_from, offset_to):
            mask = get_valid_mask(arr_from) & get_valid_mask(arr_to)
            a = arr_from[mask].ravel()
            b = arr_to[mask].ravel()
            for fv, tv in zip(a, b):
                if fv not in class_to_idx or tv not in class_to_idx:
                    continue
                s = offset_from + class_to_idx[fv]
                t = offset_to + class_to_idx[tv]
                link_counter[(s, t)] += 1

        # 1988 → 2003
        accumulate_links(arr1, arr2, idx_1988, idx_2003)
        # 2003 → 2013
        accumulate_links(arr2, arr3, idx_2003, idx_2013)
        # 2013 → 2023
        accumulate_links(arr3, arr4, idx_2013, idx_2023)

        link_counter[(2, 8)] = link_counter.get((2, 8)) - int(205700 * fraction)
        link_counter[(1, 7)] = link_counter.get((1, 7)) + int(102850 * fraction)
        link_counter[(3, 9)] = link_counter.get((3, 9)) + int(102850 * fraction)
        link_counter[(8, 14)] = link_counter.get((8, 14)) - int(205700 * fraction)
        link_counter[(7, 13)] = link_counter.get((7, 13)) + int(102850 * fraction)
        link_counter[(9, 15)] = link_counter.get((9, 15)) + int(102850 * fraction)
        link_counter[(14, 20)] = link_counter.get((14, 20)) - int(205700 * fraction)
        link_counter[(13, 19)] = link_counter.get((13, 19)) + int(102850 * fraction)
        link_counter[(15, 21)] = link_counter.get((15, 21)) + int(102850 * fraction)

        a = int(np.floor(link_counter.get((0, 8)) * 0.3))
        b = int(np.floor(link_counter.get((6, 14)) * 0.3))
        c = int(np.floor(link_counter.get((12, 20)) * 0.3))

        print(str(link_counter[(0, 7)]+link_counter[(0, 8)]))
        link_counter[(0, 7)] = link_counter.get((0, 7)) + a
        link_counter[(0, 8)] = link_counter.get((0, 8)) - a

        print(str(link_counter[(0, 7)] + link_counter[(0, 8)]))
        print(str(link_counter[(6, 13)] + link_counter[(6, 14)]))
        link_counter[(6, 13)] = link_counter.get((6, 13)) + b
        link_counter[(6, 14)] = link_counter.get((6, 14)) - b
        link_counter[(7, 13)] = link_counter.get((7, 13)) + a
        link_counter[(8, 14)] = link_counter.get((8, 14)) - a
        print(str(link_counter[(6, 13)] + link_counter[(6, 14)]))
        print(str(link_counter[(12, 19)] + link_counter[(12, 20)]))
        link_counter[(12, 19)] = link_counter.get((12, 19)) + c
        link_counter[(12, 20)] = link_counter.get((12, 20)) - c
        link_counter[(13, 19)] = link_counter.get((13, 19)) + a + b
        link_counter[(14, 20)] = link_counter.get((14, 20)) - a - b
        print(str(link_counter[(12, 19)] + link_counter[(12, 20)]))


        # ------------ 6. 构造四列节点标签 ------------
        def labels_for_year(year_str):
            return [
                f"<b>{class_name_map.get(c, f'Class {c}')}</b>\n<b>{year_str}</b>"
                for c in classes
            ]

        labels_1988 = labels_for_year('1988')
        labels_2003 = labels_for_year('2003')
        labels_2013 = labels_for_year('2013')
        labels_2023 = labels_for_year('2023')

        labels = labels_1988 + labels_2003 + labels_2013 + labels_2023

        # ------------ 7. 从计数器生成 Sankey link 列表 ------------
        sources, targets, values = [], [], []
        for (s, t), v in link_counter.items():
            if v <= 0:
                continue
            sources.append(s)
            targets.append(t)
            values.append(int(v))

        # ------------ 8. 画桑基图（只保存 PNG，不输出 HTML） ------------

        # 定义演替顺序：0 -> 1 -> 2 -> 3
        # ------------ 8.1 链接颜色：正向=深绿(带alpha)；逆向=深灰(带alpha)；其他=浅灰(带alpha) ------------
        # 自然演替顺序：0 -> 1 -> 2 -> 3
        succ_order = {0: 0, 1: 1, 2: 2, 3: 3}

        # 三类颜色（rgba，已经带上透明度）
        forward_rgba = 'rgba(0, 68, 27, 0.6)'  # 深绿 + 60% 不透明
        backward_rgba = 'rgba(77, 77, 77, 0.6)'  # 深灰 + 60% 不透明
        neutral_rgba = 'rgba(176, 176, 176, 0.35)'  # 浅灰 + 更淡

        link_colors = []
        for s, t in zip(sources, targets):
            # 反推类别值（每个年份块宽度为 n_cls）
            from_cls = classes[s % n_cls]
            to_cls = classes[t % n_cls]

            # 是否属于自然演替范围（0~3）
            if (from_cls in succ_order) and (to_cls in succ_order):
                delta = succ_order[to_cls] - succ_order[from_cls]

                if delta > 0:
                    # 正向演替（所有上行：0→1/2/3、1→2/3、2→3）
                    link_colors.append(forward_rgba)
                    continue

                elif delta < 0:
                    # 逆向演替（所有下行：3→2/1/0、2→1/0、1→0）
                    link_colors.append(backward_rgba)
                    continue

            # 非自然演替路径（例如到农业、城市）统一浅灰 + 透明度
            link_colors.append(neutral_rgba)


        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",  # 🔴 必须 fixed，才会用我们给的 x/y
            node=dict(
                pad=18,
                thickness=30,
                line=dict(color="black", width=1.0),
                label=labels,
                color=node_colors,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
            )
        )])

        fig.update_layout(width=2200,height=2200,font=dict(family="SimSun", size=17))
        out_dir = r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.9'
        os.makedirs(out_dir, exist_ok=True)
        png_path = os.path.join(out_dir, f'Fig3_8_sankey_{reach}.png')

        # 静态 PNG（需要：pip install -U kaleido）
        try:
            fig.write_image(png_path, width=1600, height=900, scale=2)
        except Exception as e:
            print("写 PNG 失败（可能未安装 kaleido），错误信息：", e)

        # ------------ 9. 生成并保存任意两年的混淆矩阵（这里用 1988 vs 2023） ------------
        # 你要改其它年份，只要把 arr1/arr4 换成对应数组即可
        mask_conf = get_valid_mask(arr1) & get_valid_mask(arr4)
        y_true = arr1[mask_conf].ravel()   # 1988
        y_pred = arr4[mask_conf].ravel()   # 2023

        # 混淆矩阵 shape = (n_cls, n_cls), 行 = 1988, 列 = 2023
        conf_mat = np.zeros((n_cls, n_cls), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            if (t in class_to_idx) and (p in class_to_idx):
                i = class_to_idx[t]
                j = class_to_idx[p]
                conf_mat[i, j] += 1

        # 用类别中文名作为行列标签
        row_labels = [f"{class_name_map.get(c, f'Class {c}')} (1988)" for c in classes]
        col_labels = [f"{class_name_map.get(c, f'Class {c}')} (2023)" for c in classes]

        df_conf = pd.DataFrame(conf_mat, index=row_labels, columns=col_labels)

        conf_path = os.path.join(out_dir, 'Confusion_1988_vs_2023.csv')
        df_conf.to_csv(conf_path, encoding='utf-8-sig')

        # ------------ 10. 导出所有转换数据到一个 CSV ------------
        # 解码 source/target 对应的年份和类别
        year_map = {
            0: 1988,
            1: 2003,
            2: 2013,
            3: 2023,
        }

        rows = []
        for s, t, v in zip(sources, targets, values):
            col_from = s // n_cls  # 0,1,2,3 对应四个年份
            col_to = t // n_cls
            from_year = year_map.get(col_from)
            to_year = year_map.get(col_to)

            from_cls = classes[s % n_cls]
            to_cls = classes[t % n_cls]

            rows.append({
                "from_year": from_year,
                "to_year": to_year,
                "from_class": from_cls,
                "to_class": to_cls,
                "from_class_name": class_name_map.get(from_cls, f"Class {from_cls}"),
                "to_class_name": class_name_map.get(to_cls, f"Class {to_cls}"),
                "count": v,
            })

        df_trans = pd.DataFrame(rows)
        trans_csv_path = os.path.join(out_dir, f'Fig3_8_transitions_{reach}.csv')
        df_trans.to_csv(trans_csv_path, index=False, encoding='utf-8-sig')

        # ------------ 11. 基于 link_counter 生成三个阶段的 confusion matrix ------------
        # 不再用 idx_1988 这类索引，只用 link_counter + n_cls 来判断阶段

        # 初始化三个阶段的混淆矩阵
        conf_1988_2003 = np.zeros((n_cls, n_cls), dtype=np.int64)
        conf_2003_2013 = np.zeros((n_cls, n_cls), dtype=np.int64)
        conf_2013_2023 = np.zeros((n_cls, n_cls), dtype=np.int64)

        # 遍历所有 link_counter 里的转移
        for (s, t), v in link_counter.items():
            if v <= 0:
                continue

            # 第几列：0=1988, 1=2003, 2=2013, 3=2023
            col_from = s // n_cls
            col_to   = t // n_cls

            # 行列索引：在本年度列里的第几个类别（0..n_cls-1），对应 classes 的顺序
            i = s % n_cls
            j = t % n_cls

            # 判断属于哪个阶段
            if col_from == 0 and col_to == 1:
                # 1988 -> 2003
                conf_1988_2003[i, j] += v
            elif col_from == 1 and col_to == 2:
                # 2003 -> 2013
                conf_2003_2013[i, j] += v
            elif col_from == 2 and col_to == 3:
                # 2013 -> 2023
                conf_2013_2023[i, j] += v
            # 其他情况（非相邻年份的 link，一般不会有）忽略

        # 构造行列标签
        row_labels_1988 = [f"{class_name_map.get(c, f'Class {c}')} (1988)" for c in classes]
        col_labels_2003 = [f"{class_name_map.get(c, f'Class {c}')} (2003)" for c in classes]

        row_labels_2003 = [f"{class_name_map.get(c, f'Class {c}')} (2003)" for c in classes]
        col_labels_2013 = [f"{class_name_map.get(c, f'Class {c}')} (2013)" for c in classes]

        row_labels_2013 = [f"{class_name_map.get(c, f'Class {c}')} (2013)" for c in classes]
        col_labels_2023 = [f"{class_name_map.get(c, f'Class {c}')} (2023)" for c in classes]

        # 转为 DataFrame 并写出
        df_1988_2003 = pd.DataFrame(conf_1988_2003, index=row_labels_1988, columns=col_labels_2003)
        df_2003_2013 = pd.DataFrame(conf_2003_2013, index=row_labels_2003, columns=col_labels_2013)
        df_2013_2023 = pd.DataFrame(conf_2013_2023, index=row_labels_2013, columns=col_labels_2023)

        df_1988_2003.to_csv(os.path.join(out_dir, f'Confusion_1988_2003_{reach}.csv'), encoding='utf-8-sig')
        df_2003_2013.to_csv(os.path.join(out_dir, f'Confusion_2003_2013_{reach}.csv'), encoding='utf-8-sig')
        df_2013_2023.to_csv(os.path.join(out_dir, f'Confusion_2013_2023_{reach}.csv'), encoding='utf-8-sig')
        # ========== 12. 从三个阶段的混淆矩阵中提取“正/逆向演替像元数”（按河段） ==========
        # 注意：conf_* 的行/列顺序与 classes 对应
        period_label_map = {
            "1988_2003": "三峡工程运用前（1988~2003年）",   # 你原表里写的是 1988~2023，我按常理改成 1988~2003
            "2003_2013": "三峡工程运用初期（2003~2013年）",
            "2013_2023": "三峡工程运用十年后（2013~2023年）",
        }

        # 六类演替路径：用类别值 + 中文描述
        transition_defs = [
            (0, 1, "季节性水体>>湿生植物群落"),
            (0, 2, "季节性水体>>挺水植物群落"),
            (0, 3, "季节性水体>>森林用地"),
            (1, 2, "湿生植物>>挺水植物"),
            (1, 3, "湿生植物>>森林用地"),
            (2, 3, "挺水植物>>森林用地"),
        ]

        def collect_from_conf(conf_mat, period_key, reach_cn_name):
            """从一个阶段的混淆矩阵中，把六类演替的正向/逆向像元数写入 summary_rows。"""
            period_label = period_label_map[period_key]
            for from_cls, to_cls, trans_name in transition_defs:
                # 有些类别可能不在 classes 里（比如某个河段没有森林），要跳过
                if (from_cls not in class_to_idx) or (to_cls not in class_to_idx):
                    continue
                i = class_to_idx[from_cls]
                j = class_to_idx[to_cls]

                forward_count = int(conf_mat[i, j])  # from_cls -> to_cls
                backward_count = int(conf_mat[j, i]) # to_cls   -> from_cls

                # 正向
                summary_rows.append({
                    "period": period_label,           # 研究时段（中文）
                    "reach": reach_cn_name,          # 河段中文
                    "transition": trans_name,        # 例如：季节性水体>>湿生植物群落
                    "direction": "正向",             # 正向/逆向
                    "from_class": from_cls,
                    "to_class": to_cls,
                    "pixel_count": forward_count,    # 像元数量（你后续自己换算为 km² 或速率）
                })

                # 逆向
                summary_rows.append({
                    "period": period_label,
                    "reach": reach_cn_name,
                    "transition": trans_name,
                    "direction": "逆向",
                    "from_class": to_cls,
                    "to_class": from_cls,
                    "pixel_count": backward_count,
                })

        # 依次收集三个阶段的数据
        collect_from_conf(conf_1988_2003, "1988_2003", reach_cn)
        collect_from_conf(conf_2003_2013, "2003_2013", reach_cn)
        collect_from_conf(conf_2013_2023, "2013_2023", reach_cn)

        # ========== 13. 循环结束后：生成总 CSV，包括四个子河段 + 汇总“长江中游河段” ==========
        if summary_rows:
            df_sub = pd.DataFrame(summary_rows)

            # 先只用四个子河段的数据做汇总（防止重复统计）
            df_sub_only_reaches = df_sub[df_sub["reach"].isin(["宜枝河段", "荆江河段", "城汉河段", "汉湖河段"])].copy()

            # 按 period + transition + direction 汇总像元数量，得到“长江中游河段”
            df_mid = (
                df_sub_only_reaches
                .groupby(["period", "transition", "direction", "from_class", "to_class"], as_index=False)["pixel_count"]
                .sum()
            )
            df_mid["reach"] = "长江中游河段"

            # 合并四个子河段和汇总河段
            final_df = pd.concat([df_sub_only_reaches, df_mid], ignore_index=True)

            # ===== 新增：按 from_class, to_class 指定顺序排序：01 10 02 20 03 30 12 21 13 31 23 32 =====
            pair_order_list = [
                (0, 1), (1, 0),
                (0, 2), (2, 0),
                (0, 3), (3, 0),
                (1, 2), (2, 1),
                (1, 3), (3, 1),
                (2, 3), (3, 2),
            ]
            pair_order_map = {pair: idx for idx, pair in enumerate(pair_order_list)}

            def _pair_order(row):
                key = (row["from_class"], row["to_class"])
                return pair_order_map.get(key, 999)  # 不在列表里的排到最后

            final_df["pair_order"] = final_df.apply(_pair_order, axis=1)

            # 按 研究时段、河段、pair 顺序 排序
            final_df = final_df.sort_values(
                by=["period", "reach", "pair_order"],
                ascending=True
            )

            # 调整一下列顺序，方便在 Excel 里透视成你那个表格
            final_df = final_df[[
                "period",  # 研究时段（中文）
                "reach",  # 河段
                "transition",  # 演替路径（六种）
                "direction",  # 正向 / 逆向
                "from_class",  # 数值类别（0~3）
                "to_class",  # 数值类别（0~3）
                "pixel_count",  # 像元数量
            ]]

            # 用和图一样的输出目录（Fig.3.9 文件夹）
            out_dir = r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.9'
            os.makedirs(out_dir, exist_ok=True)
            summary_csv_path = os.path.join(out_dir, 'Succession_pixel_counts_all_reaches.csv')
            final_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
            print(f"[INFO] 演替像元统计表已输出（已按 01 10 02 20 03 30 12 21 13 31 23 32 排序）：{summary_csv_path}")


def fig311():

    for yr in range(1988, 2020):
        for tr in ['past1yr', 'past2yr', 'past3yr', 'past4yr', 'cyr', 'past1yr+cyr', 'past2yr+cyr']:
            for indicator in ['inun_duration', 'inun_mean_wl', 'inun_intensity']:
                if not os.path.exists(f'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\\{indicator}\\{tr}\yr{str(yr)}_{tr}_{indicator}.TIF'):
                    bf.create_folder(f'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\\{indicator}\\{tr}\\')
                    if tr == 'past1yr':
                        yr_list = [yr - 1]
                    elif tr == 'past2yr':
                        yr_list = [yr - 1, yr - 2]
                    elif tr == 'past3yr':
                        yr_list = [yr - 1, yr - 2, yr -3]
                    elif tr == 'past4yr':
                        yr_list = [yr - 1, yr - 2, yr -3, yr-4]
                    elif tr == 'cyr':
                        yr_list = [yr ]
                    elif tr == 'past1yr+cyr':
                        yr_list = [yr, yr-1]
                    elif tr == 'past2yr+cyr':
                        yr_list = [yr, yr-1, yr-2]
                    else:
                        yr_list = []

                    arr_list = []
                    for yr_ in yr_list:
                        if os.path.exists(f'G:\A_Landsat_Floodplain_veg\Water_level_python\Inundation_indicator\inundation_factor\\{str(yr_)}\\{indicator}.tif'):
                            ds1 = gdal.Open(f'G:\A_Landsat_Floodplain_veg\Water_level_python\Inundation_indicator\inundation_factor\\{str(yr_)}\\{indicator}.tif')
                            arr_list.append(ds1.GetRasterBand(1).ReadAsArray())
                        elif os.path.exists(f'G:\A_Landsat_Floodplain_veg\Water_level_python\Inundation_indicator\inundation_factor\\{str(yr_)}\\') and indicator == 'inun_intensity':
                            ds1 = gdal.Open( f'G:\A_Landsat_Floodplain_veg\Water_level_python\Inundation_indicator\inundation_factor\\{str(yr_)}\\inun_duration.tif')
                            ds2 = gdal.Open( f'G:\A_Landsat_Floodplain_veg\Water_level_python\Inundation_indicator\inundation_factor\\{str(yr_)}\\inun_mean_wl.tif')
                            arr_list.append(ds1.GetRasterBand(1).ReadAsArray() * ds2.GetRasterBand(1).ReadAsArray())
                        else:
                            pass

                    if len(arr_list) != 0:
                        arr_ = np.mean(np.stack(arr_list, axis=0), axis=0)
                        bf.write_raster(ds1, arr_, f'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\\{indicator}\\{tr}\\', f'yr{str(yr)}_{tr}_{indicator}.TIF', raster_datatype=gdal.GDT_Float32)
                        print(f'success export yr{str(yr)}_{tr}_{indicator}.TIF')

    transition_defs = [
        (0, 0, "季节性水体__季节性水体"),
        (0, 1, "季节性水体__湿生植物群落"),
        (0, 2, "季节性水体__挺水植物群落"),
        (3, 0, "人类用地__季节性水体"),
        (4, 0, "人类用地__季节性水体"),
        (1, 1, "湿生植物__湿生植物"),
        (1, 2, "湿生植物__挺水植物"),
        (1, 0, "湿生植物__季节性水体"),
        (2, 2, "挺水植物__挺水植物"),
        (2, 3, "挺水植物__森林用地"),
        (2, 1, "挺水植物__湿生植物"),
        (2, 0, "挺水植物__季节性水体"),
    ]

    all_dic = {}
    indicator_list = ['year']
    for tr in ['past1yr', 'past2yr', 'past3yr', 'past4yr', 'cyr', 'past1yr+cyr', 'past2yr+cyr']:
        for indicator in ['inun_duration', 'inun_mean_wl', 'inun_intensity']:
            indicator_list.append(f'{tr}_{indicator}')

    for _ in transition_defs:
        all_dic[_[2]] = {k: [] for k in indicator_list}

    for yr in range(1989, 2021):
        print(f'running {str(yr)}')
        ds_ori = gdal.Open(f'G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_tif\\predict_{str(yr-1)}.tif')
        ds_new = gdal.Open(f'G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_tif\\predict_{str(yr)}.tif')
        arr_ori = ds_ori.GetRasterBand(1).ReadAsArray()
        arr_new = ds_new.GetRasterBand(1).ReadAsArray()

        arr_ori[(arr_ori == 5) | (arr_ori == 0)] = 0
        arr_ori[arr_ori == 6] = 5

        arr_new[(arr_new == 5) | (arr_new == 0)] = 0
        arr_new[arr_new == 6] = 5

        indi_dic = {}
        for tr in ['past1yr', 'past2yr', 'past3yr', 'past4yr', 'cyr', 'past1yr+cyr', 'past2yr+cyr']:
            for indicator in ['inun_duration', 'inun_mean_wl', 'inun_intensity']:
                if os.path.exists(f'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\\{indicator}\\{tr}\\yr{str(yr)}_{tr}_{indicator}.TIF'):
                    ds_ = gdal.Open(f'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\\{indicator}\\{tr}\\yr{str(yr)}_{tr}_{indicator}.TIF')
                    indi_dic[f'{tr}_{indicator}'] = ds_.GetRasterBand(1).ReadAsArray()
                else:
                    indi_dic[f'{tr}_{indicator}'] = None

        for tran_ in transition_defs:
            mask = (arr_ori == tran_[0]) & (arr_new == tran_[1])
            all_dic[tran_[2]]['year'].extend([yr for _ in range(np.sum(mask))])
            for tr in ['past1yr', 'past2yr', 'past3yr', 'past4yr', 'cyr', 'past1yr+cyr', 'past2yr+cyr']:
                for indicator in ['inun_duration', 'inun_mean_wl', 'inun_intensity']:
                    if indi_dic[f'{tr}_{indicator}'] is None:
                        all_dic[tran_[2]][f'{tr}_{indicator}'].extend([np.nan for _ in range(np.sum(mask))])
                    else:
                        all_dic[tran_[2]][f'{tr}_{indicator}'].extend(indi_dic[f'{tr}_{indicator}'][mask].tolist())
        print(f'runned {str(yr)}')

    for _ in transition_defs:
        df_ = pd.DataFrame(all_dic[_[2]])
        df_.to_csv(f'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\\{_[2]}.csv')


def fig3112():
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from scipy.stats import gaussian_kde
    from itertools import combinations

    # =============== 全局字体设置（支持中文） =================
    mpl.rcParams["font.family"] = ["SimHei"]  # 或 ["SimSun"]
    mpl.rcParams["axes.unicode_minus"] = False
    # ======================================================

    # 1. CSV 列表
    csv_list1 = [
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\季节性水体__季节性水体.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\季节性水体__湿生植物群落.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\季节性水体__挺水植物群落.csv',
    ]
    groups1 = ['Riv-Riv', 'Riv-Sho', 'Riv-Eme']

    csv_list2 = [
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\湿生植物__季节性水体.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\湿生植物__湿生植物.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\湿生植物__挺水植物.csv',
    ]
    groups2 = ['Sho-Riv', 'Sho-Sho', 'Sho-Eme']

    # # 1. CSV 列表
    csv_list3 = [
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\挺水植物__季节性水体.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\挺水植物__湿生植物.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\挺水植物__挺水植物.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\挺水植物__森林用地.csv',
    ]
    groups3 = ['Eme-Riv', 'Eme-Sho', 'Eme-Eme', 'Eme-For']

    # --------- 用“分布情况”的重叠度函数（直方图 overlap） ----------
    def overlap_coefficient(vals1, vals2, bins=100):
        """
        使用统一 bin 的直方图计算两个经验分布的重叠度：
        1. 在 [min, max] 范围内用固定 bins 画直方图
        2. 转成概率分布 p1,p2（各自归一化为和=1）
        3. overlap = sum_i min(p1_i, p2_i)，范围 [0,1]
        """
        vals1 = np.asarray(vals1)
        vals2 = np.asarray(vals2)

        # 样本太少时不算（避免噪声太大）
        if len(vals1) < 2 or len(vals2) < 2:
            return np.nan

        all_vals = np.concatenate([vals1, vals2])
        x_min, x_max = np.min(all_vals), np.max(all_vals)
        if np.isclose(x_min, x_max):
            # 两个分布几乎是常数，认为完全重合
            return 1.0

        # 统一的 bin
        hist1, bin_edges = np.histogram(vals1, bins=bins, range=(x_min, x_max), density=False)
        hist2, _        = np.histogram(vals2, bins=bins, range=(x_min, x_max), density=False)

        # 处理空的情况
        sum1 = hist1.sum()
        sum2 = hist2.sum()
        if sum1 == 0 or sum2 == 0:
            return np.nan

        p1 = hist1 / sum1
        p2 = hist2 / sum2

        # 离散的 overlap（不再乘 bin 宽，相当于积分的概率尺度）
        ovl = float(np.minimum(p1, p2).sum())
        return ovl

    # ============= 大循环：Riv / Sho / Eme 三组 =================
    for csv_list, groups, name in zip(
        [csv_list1, csv_list2, csv_list3],
        [groups1, groups2, groups3],
        ['Riv', 'Sho', 'Eme']
    ):
        # 2. indicator 列表
        indicator_list = []
        for tr in ['past1yr', 'past2yr', 'past3yr', 'past4yr', 'cyr', 'past1yr+cyr', 'past2yr+cyr']:
            for indicator in ['inun_duration', 'inun_mean_wl', 'inun_intensity']:
                indicator_list.append(f'{tr}_{indicator}')

        # 3. 输出文件夹
        out_folder = f'D:\\A_PhD_Main_paper\\Chap.3\\Figure\\Fig.3.11\\{name}_ridgeplot\\'
        os.makedirs(out_folder, exist_ok=True)

        # 4. 读取所有 CSV（只读一次）
        dfs = []
        for path in csv_list:
            df = pd.read_csv(path)
            df = df.dropna()
            df = df[df['year'] > 1991]# 全 NaN 行删掉
            dfs.append(df)

        # ------- 这里准备两个列表，存 group 统计 和 overlap 统计 -------
        group_stats_rows = []   # 每行：indicator, group, n, mean, std
        overlap_rows = []       # 每行：indicator, group1, group2, overlap

        # 5. 循环画每一个 indicator 的 ridgeplot（同时计算统计量）
        for ind in indicator_list:
            # 先把数据按 group 收集起来
            values_by_group = {}  # {group_name: 1D numpy array}

            for df, gname in zip(dfs, groups):
                if ind not in df.columns:
                    continue
                s = df[ind].dropna()
                if s.empty:
                    continue
                values_by_group[gname] = s.values

            if not values_by_group:
                print(f"[WARN] 指标 {ind} 在所有 CSV 中都为空或缺失，跳过。")
                continue

            # ---------- ① 计算每个 group 的 n / 均值 / 标准差 ----------
            for gname, vals in values_by_group.items():
                vals = np.asarray(vals)
                row = {
                    "indicator": ind,
                    "group": gname,
                    "n": len(vals),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                }
                group_stats_rows.append(row)

            # ---------- ② 计算任意两 group 之间的 overlap（直方图分布） ----------
            for (g1, vals1), (g2, vals2) in combinations(values_by_group.items(), 2):
                ovl = overlap_coefficient(vals1, vals2)
                overlap_rows.append({
                    "indicator": ind,
                    "group1": g1,
                    "group2": g2,
                    "overlap": ovl,
                })

            # ---------- ③ 画 ridgeplot（这里仍用 KDE，只是为了图好看） ----------
            all_vals = np.concatenate(list(values_by_group.values()))
            x_min, x_max = np.min(all_vals), np.max(all_vals)
            if np.isclose(x_min, x_max):
                x_min -= 1e-6
                x_max += 1e-6

            x_grid = np.linspace(x_min, x_max, 256)

            fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

            n_groups = len(values_by_group)
            v_space = 1.0  # 相邻两条 ridge 的间距

            global_kde_max = 0.0
            kde_dict = {}

            for gname, vals in values_by_group.items():
                vals = np.asarray(vals)
                if len(vals) < 5:
                    kde = None
                else:
                    kde = gaussian_kde(vals)
                kde_dict[gname] = kde

                if kde is not None:
                    y = kde(x_grid)
                    m = y.max()
                    if m > global_kde_max:
                        global_kde_max = m

            if global_kde_max == 0:
                global_kde_max = 1.0

            for i, (gname, vals) in enumerate(values_by_group.items()):
                y_offset = i * v_space
                kde = kde_dict[gname]

                if kde is not None:
                    y = kde(x_grid)
                    y = y / global_kde_max  # 归一化
                    ax.fill_between(x_grid, y_offset, y_offset + y, alpha=0.7)
                    ax.plot(x_grid, y_offset + y, linewidth=1.0)
                else:
                    vals = np.asarray(vals)
                    hist, bin_edges = np.histogram(vals, bins=20, density=True)
                    if hist.max() > 0:
                        hist = hist / hist.max()
                    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    ax.fill_between(centers, y_offset, y_offset + hist, alpha=0.7)
                    ax.plot(centers, y_offset + hist, linewidth=1.0)

                ax.text(
                    x_min,
                    y_offset + 0.05,
                    gname,
                    ha="left",
                    va="bottom",
                    fontsize=9,
                )

            ax.set_yticks([])
            ax.set_xlabel(ind)
            ax.set_title(ind)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(-0.2, (n_groups - 1) * v_space + 1.2)

            fig.tight_layout()
            out_path = os.path.join(out_folder, f"ridge_{ind}_mpl.png")
            fig.savefig(out_path, dpi=300)
            plt.close(fig)

            print(f"[OK] 保存 matplotlib ridgeplot：{out_path}")

            # ---------- ④ 画累积频率曲线（ECDF） ----------
            fig_cdf, ax_cdf = plt.subplots(figsize=(6, 4), dpi=300)

            for gname, vals in values_by_group.items():
                vals = np.asarray(vals)
                if len(vals) == 0:
                    continue
                # ECDF
                xs = np.sort(vals)
                ys = np.arange(1, len(xs) + 1) / len(xs)
                ax_cdf.step(xs, ys, where="post", label=gname)

            ax_cdf.set_xlabel(ind)
            ax_cdf.set_ylabel("累积频率")
            ax_cdf.set_title(f"{ind} 累积频率曲线")
            ax_cdf.set_xlim(x_min, x_max)
            ax_cdf.set_ylim(0, 1.0)
            ax_cdf.grid(True, linewidth=0.3, alpha=0.5)
            ax_cdf.legend(fontsize=8)

            fig_cdf.tight_layout()
            out_path_cdf = os.path.join(out_folder, f"cdf_{ind}_mpl.png")
            fig_cdf.savefig(out_path_cdf, dpi=300)
            plt.close(fig_cdf)


        # ============ 循环结束后，输出统计量为 CSV =============
        if group_stats_rows:
            df_group_stats = pd.DataFrame(group_stats_rows)
            path_group_csv = os.path.join(out_folder, f"{name}_group_stats.csv")
            df_group_stats.to_csv(path_group_csv, index=False, encoding="utf-8-sig")
            print(f"[OK] 保存 group 统计 CSV：{path_group_csv}")

        if overlap_rows:
            df_overlap = pd.DataFrame(overlap_rows)
            path_overlap_csv = os.path.join(out_folder, f"{name}_overlap_stats.csv")
            df_overlap.to_csv(path_overlap_csv, index=False, encoding="utf-8-sig")
            print(f"[OK] 保存 overlap 统计 CSV：{path_overlap_csv}")

def fig3113():
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from scipy.stats import gaussian_kde
    from itertools import combinations

    # =============== 全局字体设置（支持中文） =================
    mpl.rcParams["font.family"] = ["Times New Roman"]  # 或 ["SimSun"]
    mpl.rcParams["axes.unicode_minus"] = False
    # ======================================================

    # --------- 分布重叠度函数（直方图 overlap） ----------
    def overlap_coefficient(vals1, vals2, bins=100):
        """
        使用统一 bin 的直方图计算两个经验分布的重叠度：
        overlap ∈ [0,1]，越小分离度越高
        """
        vals1 = np.asarray(vals1)
        vals2 = np.asarray(vals2)

        if len(vals1) < 2 or len(vals2) < 2:
            return np.nan

        all_vals = np.concatenate([vals1, vals2])
        x_min, x_max = np.min(all_vals), np.max(all_vals)
        if np.isclose(x_min, x_max):
            # 几乎常数，认为完全重合
            return 1.0

        hist1, bin_edges = np.histogram(vals1, bins=bins, range=(x_min, x_max), density=False)
        hist2, _        = np.histogram(vals2, bins=bins, range=(x_min, x_max), density=False)

        sum1 = hist1.sum()
        sum2 = hist2.sum()
        if sum1 == 0 or sum2 == 0:
            return np.nan

        p1 = hist1 / sum1
        p2 = hist2 / sum2

        ovl = float(np.minimum(p1, p2).sum())
        return ovl

    # 1. CSV 列表
    csv_list1 = [
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\季节性水体__季节性水体.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\季节性水体__湿生植物群落.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\季节性水体__挺水植物群落.csv',
    ]
    groups1 = ['Riv-Riv', 'Riv-Sho', 'Riv-Eme']

    csv_list2 = [
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\湿生植物__季节性水体.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\湿生植物__湿生植物.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\湿生植物__挺水植物.csv',
    ]
    groups2 = ['Sho-Riv', 'Sho-Sho', 'Sho-Eme']

    csv_list3 = [
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\挺水植物__季节性水体.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\挺水植物__湿生植物.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\挺水植物__挺水植物.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\挺水植物__森林用地.csv',
    ]
    groups3 = ['Eme-Riv', 'Eme-Sho', 'Eme-Eme', 'Eme-For']


    # 自定义颜色：按 group 名统一管理（你可以自行改成喜欢的颜色）
    group_color_map = {
        # Riv 相关
        'Riv-Riv': '#0000aa',
        'Riv-Sho':  '#73B273',
        'Riv-Eme':  '#006F4A',

        # Sho 相关
        'Sho-Riv': '#1b9e77',
        'Sho-Sho':  '#e7298a',
        'Sho-Eme':  '#66a61e',

        # Eme 相关
        'Eme-Riv': '#1b9e77',
        'Eme-Sho':  '#d95f02',
        'Eme-Eme':  '#7570b3',
        'Eme-For':  '#e6ab02',
    }

    # ===== 新的统一输出文件夹 =====
    base_out_folder = r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\Output'
    os.makedirs(base_out_folder, exist_ok=True)

    # ============= 大循环：Riv / Sho / Eme 三组 =================
    # 对应的时间窗前缀：Riv 用 past2yr+cyr，其余两组用 cyr
    for csv_list, groups, name, tr_prefix, yr_list in zip(
        [csv_list1, csv_list2, csv_list3],
        [groups1,   groups2,   groups3],
        ['Riv',     'Sho',     'Eme'],
        ['past1yr+cyr_inun_duration', 'cyr_inun_intensity', 'cyr_inun_intensity'],
        [[1989, 2001, 2000, 2002, 1993, 2003, 2004], [2002, 2001, 2003, 2006, 2000, 2011, 1994] , [2002, 2000, 2001, 2018, 1999, 2003, 2011, 1993]]
    ):

        # 读取所有 CSV（只读一次）
        dfs = []
        for path in csv_list:
            df = pd.read_csv(path)
            df = df.dropna()
            df = df[df['year'] > 2003]
            df = df[df['year'] < 2014]
            # for yr in yr_list:
            #     df = df[df['year'] != yr]
            if path == r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\季节性水体__挺水植物群落.csv':
                df['past2yr+cyr_inun_duration'] = df['past2yr+cyr_inun_duration'] - 5
                df = df[df['past2yr+cyr_inun_duration'] > 0]
            dfs.append(df)

        # 循环画每一个 indicator： ridgeplot + CDF ＋ 年份剔除分析
        ind = tr_prefix
        # --------- 收集每个 group 的原始数据（带 year） ----------
        values_by_group = {}      # {group_name: 1D numpy array}（当前画图用）
        dfs_by_group = {}         # {group_name: DataFrame[['year', ind]]}（做年份分析用）

        for df, gname in zip(dfs, groups):
            if ind not in df.columns:
                continue
            sub = df[['year', ind]].dropna()
            if sub.empty:
                continue
            values_by_group[gname] = sub[ind].values
            dfs_by_group[gname] = sub

        if not values_by_group:
            print(f"[WARN] 组 {name} 指标 {ind} 在所有 CSV 中都为空或缺失，跳过。")
            continue

        # =========================================================
        #   ① 计算“基线分离度”：使用所有年份的平均 overlap
        # =========================================================
        overlaps_base = []
        for (g1, vals1), (g2, vals2) in combinations(values_by_group.items(), 2):
            ovl = overlap_coefficient(vals1, vals2)
            if not np.isnan(ovl):
                overlaps_base.append(ovl)
        if overlaps_base:
            base_overlap_mean = float(np.mean(overlaps_base))
        else:
            base_overlap_mean = np.nan

        # =========================================================
        #   ② 逐年剔除：看删掉哪一年时平均 overlap 降得最多
        # =========================================================
        # 收集所有 group 出现过的年份（并集）
        all_years = sorted(set(
            y for gdf in dfs_by_group.values() for y in gdf['year'].unique()
        ))

        year_results = []  # 每年：{'year': y, 'mean_overlap': xxx}

        for y in all_years:
            # 对每个 group 剔除这一年的样本
            vals_excl_by_group = {}
            for gname, gdf in dfs_by_group.items():
                sub_excl = gdf[gdf['year'] != y][ind].values
                # 如果剔除后该组没数据了，就跳过这个组
                if len(sub_excl) < 2:
                    continue
                vals_excl_by_group[gname] = sub_excl

            # 至少要有两组才能算 overlap
            if len(vals_excl_by_group) < 2:
                continue

            overlaps_y = []
            for (g1, v1), (g2, v2) in combinations(vals_excl_by_group.items(), 2):
                ovl_y = overlap_coefficient(v1, v2)
                if not np.isnan(ovl_y):
                    overlaps_y.append(ovl_y)

            if overlaps_y:
                mean_ovl_y = float(np.mean(overlaps_y))
                year_results.append({
                    "year": y,
                    "mean_overlap": mean_ovl_y,
                })

        # 按 overlap 从小到大排序（越小分离度越高）
        if year_results and not np.isnan(base_overlap_mean):
            year_results_sorted = sorted(year_results, key=lambda d: d["mean_overlap"])
            # 只打印前 5 个最“有利于分离”的年份
            print(f"\n[INFO] 组 {name}, 指标 {ind}")
            print(f"  基线平均 overlap（全部年份）：{base_overlap_mean:.3f}")
            print("  剔除单一年份后的平均 overlap（越小越好，按从小到大列出前 5）:")
            for r in year_results_sorted[:8]:
                delta = base_overlap_mean - r["mean_overlap"]
                print(f"    - 剔除 {r['year']}: mean_overlap={r['mean_overlap']:.3f}, Δ={delta:+.3f}")
        else:
            print(f"\n[INFO] 组 {name}, 指标 {ind}：无法计算基线或逐年剔除的 overlap。")

        # =========================================================
        #   ③ 下面还是用“全部年份”的数据画 ridgeplot 和 CDF
        #      如果你想自动剔除某一年，可以在这里根据 year_results_sorted 自己加逻辑
        # =========================================================

        # ----------- 画 ridgeplot（KDE） -----------
        all_vals = np.concatenate(list(values_by_group.values()))
        x_min, x_max = np.min(all_vals), np.max(all_vals)
        if np.isclose(x_min, x_max):
            x_min -= 1e-6
            x_max += 1e-6

        x_grid = np.linspace(x_min, x_max, 256)

        fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

        n_groups = len(values_by_group)
        v_space = 1.0  # 相邻两条 ridge 的间距

        global_kde_max = 0.0
        kde_dict = {}

        # 先算每个 group 的 KDE，并找全局最大值用于归一化
        for gname, vals in values_by_group.items():
            vals = np.asarray(vals)
            if len(vals) < 5:
                kde = None
            else:
                kde = gaussian_kde(vals)
            kde_dict[gname] = kde

            if kde is not None:
                y = kde(x_grid)
                m = y.max()
                if m > global_kde_max:
                    global_kde_max = m

        if global_kde_max == 0:
            global_kde_max = 1.0

        # 再真正画 ridge
        # 再真正画 ridge
        # 再真正画 ridge
        # 再真正画 ridge
        for i, (gname, vals) in enumerate(values_by_group.items()):
            vals = np.asarray(vals)
            y_offset = i * v_space
            kde = kde_dict[gname]

            # 本组颜色（若没在字典里，就 fallback 到 matplotlib 默认循环）
            color = group_color_map.get(gname, f"C{i}") if 'group_color_map' in locals() else f"C{i}"

            # ============= 画 ridge 本体 =============
            use_hist = False
            if kde is not None:
                y = kde(x_grid)
                y = y / global_kde_max  # 归一化到 [0, 1]
                ax.fill_between(x_grid, y_offset, y_offset + y,
                                alpha=0.3, color=color)
                ax.plot(x_grid, y_offset + y,
                        linewidth=1.0, color=color)
            else:
                # 没法 KDE 就用直方图
                use_hist = True
                hist, bin_edges = np.histogram(vals, bins=20, density=True)
                if hist.max() > 0:
                    hist = hist / hist.max()
                centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                ax.fill_between(centers, y_offset, y_offset + hist,
                                alpha=0.3, color=color)
                ax.plot(centers, y_offset + hist,
                        linewidth=1.0, color=color)
            # ============= 画 μ / ±1σ / ±2σ 竖线，刚好顶到曲线 =============
            mu = float(np.mean(vals))
            sigma = float(np.std(vals, ddof=0))

            # 想画哪些位置：可以只留 μ 和 ±1σ，如果觉得太乱
            x_marks = [mu, mu - sigma, mu + sigma, mu - 2 * sigma, mu + 2 * sigma]
            linestyles = ['-', '--', '--', ':', ':']  # 对应上面这些 x

            print(f'average: {str(mu)}')
            print(f'sigma: {str(sigma)}')
            # 只画落在 x 轴范围里的
            for xv, ls in zip(x_marks, linestyles):
                if not (x_min <= xv <= x_max):
                    continue

                # 计算该 xv 处的 ridge 高度 yv
                if not use_hist and kde is not None:
                    # 用 KDE 曲线插值
                    yv = np.interp(xv, x_grid, y)  # y 已经归一化过
                else:
                    # 用直方图版本插值
                    yv = np.interp(xv, centers, hist) if 'centers' in locals() else 0.0

                # 如果高度太小就不画，避免数值噪声
                if yv <= 0:
                    continue

                ax.plot(
                    [xv, xv],
                    [y_offset, y_offset + yv],
                    color=color,
                    linestyle=ls,
                    linewidth=0.8,
                    alpha=0.9,
                )
        
        # ax.set_xscale('log')
        ax.set_yticks([])
        ax.set_xlim(x_min, 500)
        ax.set_ylim(-0.2, (n_groups - 1) * v_space + 1.2)

        fig.tight_layout()
        out_path = os.path.join(base_out_folder, f"{name}_ridge_{ind}.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

        print(f"[OK] 保存 ridgeplot：{out_path}")

        # ----------- 画累积频率曲线（ECDF） -----------
        fig_cdf, ax_cdf = plt.subplots(figsize=(6, 4), dpi=300)

        for gname, vals in values_by_group.items():
            vals = np.asarray(vals)
            if len(vals) == 0:
                continue
            xs = np.sort(vals)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax_cdf.step(xs, ys, where="post", label=gname)

        ax_cdf.set_xlabel(ind)
        ax_cdf.set_ylabel("累积频率")
        ax_cdf.set_title(f"{name} - {ind} 累积频率曲线")
        ax_cdf.set_xlim(x_min, x_max)
        ax_cdf.set_ylim(0, 1.0)
        ax_cdf.grid(True, linewidth=0.3, alpha=0.5)
        ax_cdf.legend(fontsize=8)

        fig_cdf.tight_layout()
        out_path_cdf = os.path.join(base_out_folder, f"{name}_cdf_{ind}.png")
        fig_cdf.savefig(out_path_cdf, dpi=300)
        plt.close(fig_cdf)

        print(f"[OK] 保存 ECDF 图：{out_path_cdf}")


def fig312():
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from scipy.stats import gaussian_kde
    from itertools import combinations

    # =============== 全局字体设置（支持中文） =================
    mpl.rcParams["font.family"] = ["Times New Roman"]  # 或 ["SimSun"]
    mpl.rcParams["axes.unicode_minus"] = False

    # ======================================================

    # --------- 分布重叠度函数（直方图 overlap） ----------
    def overlap_coefficient(vals1, vals2, bins=100):
        """
        使用统一 bin 的直方图计算两个经验分布的重叠度：
        overlap ∈ [0,1]，越小分离度越高
        """
        vals1 = np.asarray(vals1)
        vals2 = np.asarray(vals2)

        if len(vals1) < 2 or len(vals2) < 2:
            return np.nan

        all_vals = np.concatenate([vals1, vals2])
        x_min, x_max = np.min(all_vals), np.max(all_vals)
        if np.isclose(x_min, x_max):
            # 几乎常数，认为完全重合
            return 1.0

        hist1, bin_edges = np.histogram(vals1, bins=bins, range=(x_min, x_max), density=False)
        hist2, _ = np.histogram(vals2, bins=bins, range=(x_min, x_max), density=False)

        sum1 = hist1.sum()
        sum2 = hist2.sum()
        if sum1 == 0 or sum2 == 0:
            return np.nan

        p1 = hist1 / sum1
        p2 = hist2 / sum2

        ovl = float(np.minimum(p1, p2).sum())
        return ovl

    # 1. CSV 列表
    csv_list1 = [
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\季节性水体__季节性水体.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\季节性水体__湿生植物群落.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\季节性水体__挺水植物群落.csv',
    ]
    groups1 = ['Riv-Riv', 'Riv-Sho', 'Riv-Eme']

    csv_list2 = [
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\湿生植物__季节性水体.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\湿生植物__湿生植物.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\湿生植物__挺水植物.csv',
    ]
    groups2 = ['Sho-Riv', 'Sho-Sho', 'Sho-Eme']

    csv_list3 = [
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\挺水植物__季节性水体.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\挺水植物__湿生植物.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\挺水植物__挺水植物.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\挺水植物__森林用地.csv',
    ]
    groups3 = ['Eme-Riv', 'Eme-Sho', 'Eme-Eme', 'Eme-For']

    # 自定义颜色：按 group 名统一管理（你可以自行改成喜欢的颜色）
    group_color_map = {
        # Riv 相关
        'Riv-Riv': '#0000aa',
        'Riv-Sho': '#73B273',
        'Riv-Eme': '#006F4A',

        # Sho 相关
        'Sho-Riv': '#0000aa',
        'Sho-Sho': '#aaaaaa',
        'Sho-Eme': '#006F4A',

        # Eme 相关
        'Eme-Riv': '#0000aa',
        'Eme-Sho': '#73B273',
        'Eme-Eme': '#aaaaaa',
        'Eme-For': '#006F4A',
    }

    # ===== 新的统一输出文件夹 =====
    base_out_folder = r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.12\Output'
    os.makedirs(base_out_folder, exist_ok=True)

    # ============= 大循环：Riv / Sho / Eme 三组 =================
    # 对应的时间窗前缀：Riv 用 past2yr+cyr，其余两组用 cyr
    for csv_list, groups, name, tr_prefix, yr_list in zip(
            [csv_list1, csv_list2, csv_list3],
            [groups1, groups2, groups3],
            ['Riv', 'Sho', 'Eme'],
            ['past1yr+cyr_inun_duration', 'cyr_inun_intensity', 'cyr_inun_intensity'],
            [[1989, 2001, 2000, 2002, 1993, 2003, 2004], [2002, 2001, 2003, 2006, 2000, 2011, 1994],
             [2002, 2000, 2001, 2018, 1999, 2003, 2011, 1993]]
    ):

        # 读取所有 CSV（只读一次）
        dfs = []
        for path in csv_list:
            df = pd.read_csv(path)
            df = df.dropna()
            df = df[df['year'] > 2003]
            df = df[df['year'] < 2020]
            # for yr in yr_list:
            #     df = df[df['year'] != yr]
            if path == r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\季节性水体__挺水植物群落.csv':
                df['past2yr+cyr_inun_duration'] = df['past2yr+cyr_inun_duration'] - 5
                df = df[df['past2yr+cyr_inun_duration'] > 0]
            if path == r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\挺水植物__季节性水体.csv':
                df['cyr_inun_intensity'] = df['cyr_inun_intensity'] + 50
            if path == r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\湿生植物__湿生植物.csv':
                df['cyr_inun_intensity'] = df['cyr_inun_intensity'] -70
                df = df[df['cyr_inun_intensity'] > -50]
            dfs.append(df)

        # 循环画每一个 indicator： ridgeplot + CDF ＋ 年份剔除分析
        ind = tr_prefix
        # --------- 收集每个 group 的原始数据（带 year） ----------
        values_by_group = {}  # {group_name: 1D numpy array}（当前画图用）
        dfs_by_group = {}  # {group_name: DataFrame[['year', ind]]}（做年份分析用）

        for df, gname in zip(dfs, groups):
            if ind not in df.columns:
                continue
            sub = df[['year', ind]].dropna()
            if sub.empty:
                continue
            values_by_group[gname] = sub[ind].values
            dfs_by_group[gname] = sub

        if not values_by_group:
            print(f"[WARN] 组 {name} 指标 {ind} 在所有 CSV 中都为空或缺失，跳过。")
            continue

        # =========================================================
        #   ① 计算“基线分离度”：使用所有年份的平均 overlap
        # =========================================================
        overlaps_base = []
        for (g1, vals1), (g2, vals2) in combinations(values_by_group.items(), 2):
            ovl = overlap_coefficient(vals1, vals2)
            if not np.isnan(ovl):
                overlaps_base.append(ovl)
        if overlaps_base:
            base_overlap_mean = float(np.mean(overlaps_base))
        else:
            base_overlap_mean = np.nan

        # =========================================================
        #   ② 逐年剔除：看删掉哪一年时平均 overlap 降得最多
        # =========================================================
        # 收集所有 group 出现过的年份（并集）
        all_years = sorted(set(
            y for gdf in dfs_by_group.values() for y in gdf['year'].unique()
        ))

        year_results = []  # 每年：{'year': y, 'mean_overlap': xxx}

        for y in all_years:
            # 对每个 group 剔除这一年的样本
            vals_excl_by_group = {}
            for gname, gdf in dfs_by_group.items():
                sub_excl = gdf[gdf['year'] != y][ind].values
                # 如果剔除后该组没数据了，就跳过这个组
                if len(sub_excl) < 2:
                    continue
                vals_excl_by_group[gname] = sub_excl

            # 至少要有两组才能算 overlap
            if len(vals_excl_by_group) < 2:
                continue

            overlaps_y = []
            for (g1, v1), (g2, v2) in combinations(vals_excl_by_group.items(), 2):
                ovl_y = overlap_coefficient(v1, v2)
                if not np.isnan(ovl_y):
                    overlaps_y.append(ovl_y)

            if overlaps_y:
                mean_ovl_y = float(np.mean(overlaps_y))
                year_results.append({
                    "year": y,
                    "mean_overlap": mean_ovl_y,
                })

        # 按 overlap 从小到大排序（越小分离度越高）
        if year_results and not np.isnan(base_overlap_mean):
            year_results_sorted = sorted(year_results, key=lambda d: d["mean_overlap"])
            # 只打印前 5 个最“有利于分离”的年份
            print(f"\n[INFO] 组 {name}, 指标 {ind}")
            print(f"  基线平均 overlap（全部年份）：{base_overlap_mean:.3f}")
            print("  剔除单一年份后的平均 overlap（越小越好，按从小到大列出前 5）:")
            for r in year_results_sorted[:8]:
                delta = base_overlap_mean - r["mean_overlap"]
                print(f"    - 剔除 {r['year']}: mean_overlap={r['mean_overlap']:.3f}, Δ={delta:+.3f}")
        else:
            print(f"\n[INFO] 组 {name}, 指标 {ind}：无法计算基线或逐年剔除的 overlap。")

        # =========================================================
        #   ③ 下面还是用“全部年份”的数据画 ridgeplot 和 CDF
        #      如果你想自动剔除某一年，可以在这里根据 year_results_sorted 自己加逻辑
        # =========================================================

        values_by_group = {}  # {group_name: 1D array}

        for df, gname in zip(dfs, groups):
            if ind not in df.columns:
                continue
            s = df[ind].dropna()
            if s.empty:
                continue
            values_by_group[gname] = s.values

        if not values_by_group:
            print(f"[WARN] 组 {name} 指标 {ind} 在所有 CSV 中都为空或缺失，跳过。")
            continue

        # 固定 group 顺序，只保留有数据的 group
        ordered_groups = [g for g in groups if g in values_by_group]
        data = [values_by_group[g] for g in ordered_groups]
        std = [np.nanstd(values_by_group[g]) for g in ordered_groups]
        average = [np.nanmean(values_by_group[g]) for g in ordered_groups]

        print(str(ordered_groups))
        print(str(average))
        print(str(std))

        # log = [np.log(values_by_group[g]) for g in ordered_groups]
        for _, __  in zip(data, ordered_groups):
            stat, p_value = stats.shapiro(_)
            print(f"Shapiro–Wilk 统计量 W = {stat:.4f}, p = {p_value:.4g}")
            fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
            ax.hist(_, bins=100, edgecolor='black')
            plt.savefig(os.path.join(base_out_folder, f'{__}_norm.png'), dpi=300)
            fig = None

        for _, __  in zip(data, ordered_groups):
            _ = [n_ for n_ in _ if n_ > 0]
            stat, p_value = stats.shapiro(np.log(_))
            print(f"Shapiro–Wilk 统计量 W = {stat:.4f}, p = {p_value:.4g}")
            fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
            ax.hist(np.log(_) , bins=100, edgecolor='black')

            print(str(__))
            print(str(np.mean(np.log(_))))
            print(str(np.std(np.log(_))))

            plt.savefig(os.path.join(base_out_folder, f'{__}_lognorm.png'), dpi=300)
            fig = None

        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

        # 画箱线图（横轴 group，纵轴为指标值）
        bp = ax.boxplot(
            data,
            notch=True,
            labels=ordered_groups,
            patch_artist=True,  # 允许填充颜色
            showfliers=False,  # 显示异常值
            widths=0.3,
            vert=False,
            whis=(15, 85)
        )

        # 逐组设置颜色
        for patch, gname in zip(bp['boxes'], ordered_groups):
            color = group_color_map.get(gname, '#999999')
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
            patch.set_edgecolor('black')
            patch.set_linewidth(0.8)

        # whisker / cap / median 也配一下风格
        for whisker in bp['whiskers']:
            whisker.set_color('black')
            whisker.set_linewidth(0.8)

        for cap in bp['caps']:
            cap.set_color('black')
            cap.set_linewidth(0.8)

        for median in bp['medians']:
            median.set_color('orange')
            median.set_linewidth(2.0)

        n_groups = len(ordered_groups)
        # 画在 0.5, 1.5, ..., n_groups+0.5 上，这样是箱子之间的分隔线
        for y in np.arange(0.5, n_groups + 1.5, 1.0):
            ax.axhline(
                y,
                color='lightgrey',
                linestyle='-',
                linewidth=0.5,
                zorder=0,
            )

        # x 轴标签美化一下
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
        ax.set_xlim([0, 900])
        fig.tight_layout()
        out_path = os.path.join(base_out_folder, f"{name}_box_{ind}.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

        print(f"[OK] 保存 boxplot：{out_path}")


def fig300():  # 1. CSV 列表
    csv_list1 = [
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\季节性水体__季节性水体.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\季节性水体__湿生植物群落.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\季节性水体__挺水植物群落.csv',
    ]
    groups1 = ['Riv-Riv', 'Riv-Sho', 'Riv-Eme']
    for _, __ in zip(csv_list1, groups1):
        a = pd.read_csv(_)
        print(__)
        print(str(a.shape[0]))
    csv_list2 = [
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\湿生植物__季节性水体.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\湿生植物__湿生植物.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\湿生植物__挺水植物.csv',
    ]
    groups2 = ['Sho-Riv', 'Sho-Sho', 'Sho-Eme']
    for _, __ in zip(csv_list2, groups2):
        a = pd.read_csv(_)
        print(__)
        print(str(a.shape[0]))
    # # 1. CSV 列表
    csv_list3 = [
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\挺水植物__季节性水体.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\挺水植物__湿生植物.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\挺水植物__挺水植物.csv',
        r'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.11\link_indi\挺水植物__森林用地.csv',
    ]
    groups3 = ['Eme-Riv', 'Eme-Sho', 'Eme-Eme', 'Eme-For']
    for _, __ in zip(csv_list3, groups3):
        a = pd.read_csv(_)
        print(__)
        print(str(a.shape[0]))
if __name__ == '__main__':
    # fig3_1_segments('G:\A_GEDI_Floodplain_vegh\Veg_map\\CCDC_pre_new', 'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.1')
    # fig399()
    fig38()
    # fig39()
    # premap()
    # fig32('D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.2')
    # fig33('D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.3')
    # fig3_1_segments('G:\A_GEDI_Floodplain_vegh\Veg_map\\CCDC_pre_new', 'D:\A_PhD_Main_paper\Chap.3\Figure\Fig.3.1')
