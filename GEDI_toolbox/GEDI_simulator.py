import subprocess
import os
import pandas as pd
from pyproj import Transformer
import basic_function
from pathlib import Path
import time
import numpy as np


def run_GEDI_simulator_pipeline(uav_las_file: str, output_folder: str, coord_list: list, GEDI_id, buffer=50):

    # This function is basically run the GEDI simulator developed by Dr.Steven Hancock in this paper
    base_dir = Path(__file__).resolve().parent

    # 构造相对路径（指向平级子文件夹）
    las2las_exe = base_dir / "lastoolbox" / "las2las64.exe"
    gedirat_exe = base_dir / "gedi_simulator_compile" / "gediRat.exe"
    gedimetric_exe = base_dir / "gedi_simulator_compile" / "gediMetric.exe"

    # Create the folder
    cache_folder = os.path.join(output_folder, 'cache\\')
    waveform_folder = os.path.join(output_folder, 'waveform\\')
    metric_folder = os.path.join(output_folder, 'metric\\')
    basic_function.create_folder(cache_folder)
    basic_function.create_folder(waveform_folder)
    basic_function.create_folder(metric_folder)

    start_total = time.time()
    base_name = f"GEDI{str(GEDI_id)}_"

    try:
        # 1 裁剪 LAS
        print(f"[1] Clipping LAS for {base_name} ...")
        start = time.time()
        min_x = coord_list[0] - buffer
        max_x = coord_list[0] + buffer
        min_y = coord_list[1] - buffer
        max_y = coord_list[1] + buffer

        cut_las = os.path.join(cache_folder, f"las_{base_name}.las")

        subprocess.run([
            str(las2las_exe), "-i", uav_las_file, "-o", cut_las,
            "-keep_xy", str(min_x), str(max_x), str(min_y), str(max_y)
        ], check=True)
        print(f" LAS clipped in {time.time() - start:.2f} seconds.")

        # 2  gediRat 提取波形
        if os.path.exists(cut_las):
            print(f"[2] Running gediRat for {base_name} ...")
            start = time.time()
            wave_out = os.path.join(waveform_folder, f"wave_{base_name}.txt")
            subprocess.run([
                str(gedirat_exe),
                "-input", cut_las,
                "-output", wave_out,
                "-coord", str(coord_list[0]), str(coord_list[1]),
                "-aEPSG", "32649"
            ], check=True)
            print(f" gediRat done in {time.time() - start:.2f} seconds.")
        else:
            raise FileNotFoundError(f"Clipped LAS not found: {cut_las}")

        #  3 gediMetric 提取指标
        if os.path.exists(wave_out):
            print(f"[3] Running gediMetric for {base_name} ...")
            start = time.time()
            subprocess.run([
                str(gedimetric_exe),
                "-input", wave_out,
                "-outRoot", f'{metric_folder}\\{base_name}',
                "-rhRes", "1",
            ], check=True)
            print(f" gediMetric done in {time.time() - start:.2f} seconds.")
        else:
            raise FileNotFoundError(f"Waveform output not found: {wave_out}")

        print(f" All steps completed in {time.time() - start_total:.2f} seconds.")

    except subprocess.CalledProcessError as e:
        print(f" Subprocess failed: {e}")
    except Exception as e:
        print(f" Error: {e}")


def extract_metric(filepath, metric_name):
    index = None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#'):
                    # 提取带有字段名的注释行
                    header_line = line.strip('# \n')
                    fields = header_line.split(', ')
                    col_index_map = {}
                    for i, field in enumerate(fields):
                        field = field.strip()
                        if ' ' in field:
                            _, name = field.split(' ', 1)
                            col_index_map[name.strip()] = i
                    if metric_name in col_index_map:
                        index = col_index_map[metric_name]
                    else:
                        raise ValueError(f"文件中未找到指标 {metric_name}。")
                    break  # 找到 header 就退出

            # 再次遍历，读取数据行
            for line in f:
                if line.strip() == '' or line.startswith('#'):
                    continue
                parts = line.strip().split()
                if index is not None and len(parts) > index:
                    return float(parts[index])

    except Exception as e:
        print(f"读取文件 {filepath} 失败：{e}")
    return None

if __name__ == '__main__':
    csv_file = pd.read_excel('G:\A_GEDI_Floodplain_vegh\GEDI_simulator\\gedi_simulator.xlsx')
    #
    # 多指标提取部分
    all_metrics = {
        "RHuav90": "rhGauss 90",
        "RHuav95": "rhGauss 95",
        "RHuav98": "rhGauss 98",
        "RHuav100": "rhGauss 100",
        "RHmax90": "rhMax 90",
        "RHmax95": "rhMax 95",
        "RHmax98": "rhMax 98",
        "RHmax100": "rhMax 100",
        "RHinf90": "rhInfl 90",
        "RHinf95": "rhInfl 95",
        "RHinf98": "rhInfl 98",
        "RHinf100": "rhInfl 100",
    }

    for new_col, metric_name in all_metrics.items():
        values = []
        for fid in csv_file['FID']:
            wave_path = os.path.join(r'G:\A_GEDI_Floodplain_vegh\Airborne_LiDAR\Output\metric\\', f"GEDI{fid}_.metric.txt")
            if os.path.exists(wave_path):
                value = extract_metric(wave_path, metric_name)
                values.append(value)
            else:
                values.append(np.nan)
        csv_file[new_col] = values

    # 保存为最终汇总结果
    csv_file.to_csv(r'G:\A_GEDI_Floodplain_vegh\Airborne_LiDAR\Output\rhuav_metrics_summary.csv', index=False)