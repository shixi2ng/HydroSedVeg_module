import rasterio.features
import numpy as np
import basic_function as bf
import copy
from shapely import LineString, Point
import pandas as pd
import traceback
import time
from NDsm import NDSparseMatrix
import geopandas as gp
import os
from osgeo import gdal, osr
from tqdm.auto import tqdm
from shapely.ops import nearest_points
from datetime import datetime
import scipy.sparse as sm
import json
from concurrent import futures
import psutil


def construct_inunfac_dc(input_dic: dict, output_path, year, metadata_dic):

    start_time = time.time()
    print(f"Start constructing the {str(year)}  Inunfac datacube of {metadata_dic['ROI_name']}.")

    # Create the output path
    yearly_output_path = output_path + str(int(year)) + '\\'
    bf.create_folder(yearly_output_path)

    if not os.path.exists(output_path + f'{str(year)}\\metadata.json') or not not os.path.exists(output_path + f'{str(year)}\\paraname.npy'):

        # Create the para list
        para_list = list(input_dic.keys())

        if metadata_dic['huge_matrix']:
            if metadata_dic['ROI_name']:
                i, data_cube = 0, NDSparseMatrix()
                while i < len(para_list):
                    try:
                        t1 = time.time()
                        array_temp = input_dic[para_list[i]]
                        if np.isnan(metadata_dic['Nodata_value']):
                            array_temp[np.isnan(array_temp)] = 0
                        else:
                            array_temp[array_temp == metadata_dic['Nodata_value']] = 0

                        sm_temp = sm.csr_matrix(array_temp.astype(float))
                        array_temp = None
                        data_cube.append(sm_temp, name=para_list[i])
                        print(f'Assemble the {str(para_list[i])} into the Inunfac dc using {str(time.time() - t1)[0:5]}s (layer {str(i)} of {str(len(para_list))})')
                        i += 1
                    except:
                        error_inf = traceback.format_exc()
                        print(error_inf)

                # Save the sdc
                np.save(f'{yearly_output_path}paraname.npy', para_list)
                data_cube.save(f'{yearly_output_path}Inunfac_datacube\\')
                nodata_value = 0
            else:
                pass
        else:
            i = 0
            data_cube = np.zeros([input_dic[para_list[0]].shape[0], input_dic[para_list[0]].shape[1], len(para_list)])
            while i < len(para_list):

                try:
                    t1 = time.time()
                    array_temp = input_dic[para_list[i]]
                    data_cube[:, :, i] = array_temp
                    print(f'Assemble the {str(para_list[i])} into the Phemetric_datacube using {str(time.time() - t1)[0:5]}s (layer {str(i)} of {str(len(para_list))})')
                    i += 1
                except:
                    error_inf = traceback.format_exc()
                    print(error_inf)

            np.save(f'{yearly_output_path}paraname.npy', para_list)
            np.save(f'{yearly_output_path}Phemetric_datacube.npy', data_cube)

        # Save the metadata dic
        metadata_dic['inunyear'] = year
        with open(f'{yearly_output_path}metadata.json', 'w') as js_temp:
            json.dump(metadata_dic, js_temp)

    print(f"Finish constructing the {str(year)} Inunfac datacube of {metadata_dic['ROI_name']} in \033[1;31m{str(time.time() - start_time)} s\033[0m.")


def generate_inundation_indicator(namelist, datacube_list, dem_arr, dem_nodata, output_path):

    inun_duration = np.zeros([dem_arr.shape[0], dem_arr.shape[1]])
    inun_mean_waterlevel = np.zeros([dem_arr.shape[0], dem_arr.shape[1]])
    inun_max_waterlevel = np.zeros([dem_arr.shape[0], dem_arr.shape[1]])

    for _ in namelist:
        ds = gdal.Open(f'{output_path}\\daily_inundation_file\\{str(_)}.tif')
        arr = ds.GetRasterBand(1).ReadAsArray()
        arr[arr == 255] = 0

        inun_duration = inun_duration + arr
        inun_mean_waterlevel = inun_mean_waterlevel + arr * (datacube_list[namelist.index(_)].toarray() - dem_arr)
        inun_max_waterlevel = np.nanmax(np.stack([inun_max_waterlevel, arr * (datacube_list[namelist.index(_)].toarray() - dem_arr)], axis=2), axis=2)

    if np.isnan(dem_nodata):
        inun_duration[np.isnan(dem_arr)] = np.nan
        inun_mean_waterlevel[np.isnan(dem_arr)] = np.nan
        inun_max_waterlevel[np.isnan(dem_arr)] = np.nan
    else:
        inun_duration[inun_duration == dem_nodata] = np.nan
        inun_mean_waterlevel[inun_mean_waterlevel == dem_nodata] = np.nan
        inun_max_waterlevel[inun_max_waterlevel == dem_nodata] = np.nan

    return [inun_duration, inun_mean_waterlevel, inun_max_waterlevel]


def concept_inundation_model(wl_nm, wl_sm, demfile, thalweg, output_filepath):

    # Import datacube
    if demfile.endswith('.tif') or demfile.endswith('.TIF'):
        dem_file_ds = gdal.Open(demfile)
        dem_file_arr = dem_file_ds.GetRasterBand(1).ReadAsArray()
    else:
        raise TypeError('Please input the dem file with right type')

    for _ in range(len(wl_nm)):
        try:
            wl_sm_, wl_nm_ = wl_sm[_].toarray(), wl_nm[_]
            dem_file_arr[np.logical_and(wl_sm_ != 0, np.isnan(dem_file_arr))] = -100
            if wl_sm_.shape[0] != dem_file_arr.shape[0]:
                wl_sm_ = np.row_stack((wl_sm_, np.zeros([dem_file_arr.shape[0] - wl_sm_.shape[0], wl_sm_.shape[1]])))

            if wl_sm_.shape[1] != dem_file_arr.shape[1]:
                wl_sm_ = np.column_stack((wl_sm_, np.zeros([wl_sm_.shape[0], dem_file_arr.shape[1] - wl_sm_.shape[1]])))

            inun_level = np.array(wl_sm_ - dem_file_arr).astype(np.float32)
            inun_level[dem_file_arr == -100] = np.nan
            inun_arr = np.array(wl_sm_ > dem_file_arr).astype(np.uint8)
            bf.create_folder(f'{output_filepath}\\inundation_temp\\')
            bf.write_raster(dem_file_ds, inun_arr, f'{output_filepath}\\inundation_temp\\', str(wl_nm_) + '.tif', raster_datatype=gdal.GDT_Byte)

            src_temp = rasterio.open(f'{output_filepath}\\inundation_temp\\{str(wl_nm_)}.tif')
            shp_dic = ({'properties': {'raster_val': int(v)}, 'geometry': s} for i, (s, v) in
                       enumerate(rasterio.features.shapes(inun_arr, connectivity=8, transform=src_temp.transform)) if
                       ~np.isnan(v))
            meta = src_temp.meta.copy()
            meta.update(compress='lzw')

            shp_list = list(shp_dic)
            nw_shp_list = []
            shp_file = gp.GeoDataFrame.from_features(shp_list)
            for __ in range(shp_file.shape[0]):
                if shp_file['raster_val'][__] == 1:
                    if thalweg.intersects(shp_file['geometry'][__]):
                        nw_shp_list.append(shp_list[__])
            nw_shp_file = gp.GeoDataFrame.from_features(nw_shp_list)

            bf.create_folder(f'{output_filepath}\\daily_inundation_file\\')
            if os.path.exists(f'{output_filepath}\\daily_inundation_file\\{str(wl_nm_)}.tif'):
                os.remove(f'{output_filepath}\\daily_inundation_file\\{str(wl_nm_)}.tif')

            with rasterio.open(f'{output_filepath}\\daily_inundation_file\\{str(wl_nm_)}.tif', 'w+', **meta) as out:
                out_arr = out.read(1)

                # this is where we create a generator of geom, value pairs to use in rasterizing
                shapes = ((geom, value) for geom, value in zip(nw_shp_file.geometry, nw_shp_file.raster_val))

                burned = rasterio.features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=src_temp.transform)
                out.write_band(1, burned)
            print(f'The {str(wl_nm_)} is generated!')

            bf.create_folder(f'{output_filepath}\\daily_inundation_level\\')
            if os.path.exists(f'{output_filepath}\\daily_inundation_level\\{str(wl_nm_)}.tif'):
                os.remove(f'{output_filepath}\\daily_inundation_level\\{str(wl_nm_)}.tif')

            inun_ds = gdal.Open(f'{output_filepath}\\daily_inundation_file\\{str(wl_nm_)}.tif')
            inun_arr = inun_ds.GetRasterBand(1).ReadAsArray()
            inun_arr[inun_arr == 255] = 0
            inun_level = inun_level * inun_arr.astype(np.float32)
            bf.write_raster(inun_ds, inun_level, f'{output_filepath}\\daily_inundation_level\\', f'{str(wl_nm_)}.tif', raster_datatype=gdal.GDT_Float32)

            print(f'The {str(wl_nm_)} is generated!')

        except:
            print(traceback.format_exc())
            print(f'The {str(wl_nm_)} is not generated!!!!!')


def process_hydroinform_df(hydro_inform):
    try:
        yearly_hydroinform_all = []
        with tqdm(total=len(hydro_inform), desc=f'Process hydroinform df', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
            for _ in hydro_inform:
                yearly_hydroinform = []
                wl_list = _.split('[')
                for __ in range(2, len(_.split('['))):
                    wl_temp = []
                    wl_ = wl_list[__].split(']')[0].split(', ')
                    for ___ in wl_:
                        if "'" in ___:
                            ___ = ___.split("'")[1]

                        try:
                            wl_temp.append(float(___))
                        except:
                            wl_temp.append(str(___))

                    yearly_hydroinform.append(wl_temp)
                yearly_hydroinform_all.append(yearly_hydroinform)
                pbar.update()
        return yearly_hydroinform_all
    except:
        print(traceback.format_exc())


def generate_hydrodatacube(year, arr_size, hydro_dic, hydro_inform, x_list, y_list, outputfolder):

    try:
        # Define the sparse matrix
        Ysize, Xsize = arr_size[0], arr_size[1]
        doy_list = [year * 1000 + _ for _ in range(1, datetime(year=year + 1, month=1, day=1).toordinal() - datetime(year=year, month=1, day=1).toordinal() + 1)]
        sm_list = [sm.lil_matrix((Ysize, Xsize)) for _ in range(len(doy_list))]

        with tqdm(total=len(hydro_inform), desc=f'Generate hydro datacube', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
            for _ in range(len(hydro_inform)):
                y, x = int(y_list[_]), int(x_list[_])
                # print(str(y) + str(x))
                wl_start_series = np.array(hydro_dic[hydro_inform[_][1]])
                wl_end_series = np.array(hydro_dic[hydro_inform[_][2]])
                # print(str(wl_start_series))
                wl_start_dis = hydro_inform[_][3]
                wl_end_dis = hydro_inform[_][4]
                wl_inter = wl_start_series + (wl_end_series - wl_start_series) * wl_start_dis / (wl_start_dis + wl_end_dis)
                # print(str(wl_inter))
                # print(str(wl_end_dis))
                for __ in range(len(wl_inter)):
                    sm_list[__][y, x] = wl_inter[__]
                pbar.update()

        return sm_list

    except:
        print(traceback.format_exc())
        raise Exception('Something error')


def cubic_interpolate(p, x):
    """
    Cubic interpolation using the cubic polynomial
    p: array of 4 pixel values
    x: the fractional part of the desired position
    """
    return p[1] + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])))


def bicubic_interpolate(image, new_width, new_height):
    """
    Perform bicubic interpolation on an image
    image: input image as a 2D numpy array
    new_width, new_height: dimensions of the output image
    """
    src_height, src_width = image.shape
    scaled_image = np.zeros((new_height, new_width))

    x_ratio = src_width / new_width
    y_ratio = src_height / new_height

    for i in range(new_height):
        for j in range(new_width):
            x, y = j * x_ratio, i * y_ratio
            x_int, y_int = int(x), int(y)
            x_frac, y_frac = x - x_int, y - y_int

            # Get pixels in four corners
            pixels = np.zeros((4, 4))
            for dy in range(-1, 3):
                for dx in range(-1, 3):
                    x_src, y_src = min(src_width - 1, max(0, x_int + dx)), min(src_height - 1, max(0, y_int + dy))
                    pixels[dy + 1, dx + 1] = image[y_src, x_src]

            # Interpolate rows
            rows = np.zeros(4)
            for k in range(4):
                rows[k] = cubic_interpolate(pixels[k, :], x_frac)

            # Interpolate column
            scaled_image[i, j] = cubic_interpolate(rows, y_frac)

    return scaled_image


def retrieve_srs(ds_temp):
    proj = osr.SpatialReference(wkt=ds_temp.GetProjection())
    srs_temp = proj.GetAttrValue('AUTHORITY', 1)
    srs_temp = 'EPSG:' + str(srs_temp)
    return srs_temp


def distance_between_2points(point1, point2):

    # Check the type
    if isinstance(point1, Point):
        point1_x = point1.coords[0][0]
        point1_y = point1.coords[0][1]
    elif isinstance(point1, list):
        point1_x = point1[0]
        point1_y = point1[1]
    elif isinstance(point1, tuple):
        point1_x = point1[0]
        point1_y = point1[1]
    else:
        raise TypeError('Wrong datatype!')

    # Check the type
    if isinstance(point2, Point):
        point2_x = point2.coords[0][0]
        point2_y = point2.coords[0][1]
    elif isinstance(point2, list):
        point2_x = point2[0]
        point2_y = point2[1]
    elif isinstance(point2, tuple):
        point2_x = point2[0]
        point2_y = point2[1]
    else:
        raise TypeError('Wrong datatype!')

    return np.sqrt((point1_x - point2_x) ** 2 + (point1_y - point2_y) ** 2)


def determin_start_vertex_of_point(point1, line: LineString):

    # Check the type
    if isinstance(point1, Point):
        point1_x = point1.coords[0][0]
        point1_y = point1.coords[0][1]
    elif isinstance(point1, list):
        point1_x = point1[0]
        point1_y = point1[1]
    elif isinstance(point1, tuple):
        point1_x = point1[0]
        point1_y = point1[1]
    else:
        raise TypeError('Wrong datatype!')

    start_vertex = None
    line_coords = list(line.coords)
    for _ in range(len(line_coords) - 1):
        if (line_coords[_][0] - point1_x) * (line_coords[_ + 1][0] - point1_x) <= 0:
            if (line_coords[_][1] - point1_y) * (line_coords[_ + 1][1] - point1_y) <= 0:
                if line_coords[_ + 1][0] - line_coords[_][0] == 0:
                    start_vertex = _
                    break
                else:
                    point1_y_temp = line_coords[_][1] + (line_coords[_ + 1][1] - line_coords[_][1]) * (point1_x - line_coords[_][0]) / (line_coords[_ + 1][0] - line_coords[_][0])

                if point1_y_temp - point1_y < 0.1:
                    start_vertex = _
                    break
    if start_vertex is None:
        raise Exception('The point is not on the line')
    return start_vertex


def dis2points_via_line(point1, point2, line_temp: LineString, line_distance=None):

    if line_distance is not None:
        if len(line_distance) != len(list(line_temp.coords)):
            raise Exception('Line distance is not consistent with the line string!')

    t1_all, t2_all, t3_all = 0, 0, 0
    t1 = time.time()
    # Check the type
    if isinstance(point1, Point):
        point1_x = point1.coords[0][0]
        point1_y = point1.coords[0][1]
    elif isinstance(point1, list):
        point1_x = point1[0]
        point1_y = point1[1]
    elif isinstance(point1, tuple):
        point1_x = point1[0]
        point1_y = point1[1]
    else:
        raise TypeError('Wrong datatype!')

    # Check the type
    if isinstance(point2, Point):
        point2_x = point2.coords[0][0]
        point2_y = point2.coords[0][1]
    elif isinstance(point2, list):
        point2_x = point2[0]
        point2_y = point2[1]
    elif isinstance(point2, tuple):
        point2_x = point2[0]
        point2_y = point2[1]
    else:
        raise TypeError('Wrong datatype!')

    if not isinstance(line_temp, LineString):
        raise TypeError('The Line should under the shapely.linestring type')
    else:
        line_arr = np.array(line_temp.coords)
    t1_all += time.time()-t1

    t2 = time.time()
    # Process line and points
    point1_index, point2_index = None, None
    # Solution 1
    for _ in range(line_arr.shape[0] - 1):
        if point1_index is None:
            if (line_arr[_, 0] - point1_x) * (line_arr[_ + 1, 0] - point1_x) <= 0:
                if (line_arr[_, 1] - point1_y) * (line_arr[_ + 1, 1] - point1_y) <= 0:
                    point1_y_temp = line_arr[_, 1] + (line_arr[_ + 1, 1] - line_arr[_, 1]) * (point1_x - line_arr[_, 0]) / (line_arr[_ + 1, 0] - line_arr[_, 0])
                    if point1_y_temp - point1_y < 0.1:
                        point1_index = _
        if point2_index is None:
            if (line_arr[_, 0] - point2_x) * (line_arr[_ + 1, 0] - point2_x) <= 0:
                if (line_arr[_, 1] - point2_y) * (line_arr[_ + 1, 1] - point2_y) <= 0:
                    point2_y_temp = line_arr[_, 1] + (line_arr[_ + 1, 1] - line_arr[_, 1]) * (point2_x - line_arr[_, 0]) / (line_arr[_ + 1, 0] - line_arr[_, 0])
                    if point2_y_temp - point2_y < 0.1:
                        point2_index = _
        if point1_index is not None and point2_index is not None:
            break
    # Solution 2
    # itr = abs(line_arr[0, 0] - line_arr[-1, 0]) / line_arr.shape[0]
    # np.argwhere(np.logical_and(point1_x - itr < line_arr[:, 0], line_arr[:, 0] < point1_x + 100))
    #
    t2_all += time.time() - t2

    t3 = time.time()
    # Check if two points both on the line
    if point1_index is None or point2_index is None:
        raise Exception('Make sure both points are on the line')
    else:
        if point1_index > point2_index:
            dis = 0
            for _ in range(point2_index + 1, point1_index + 2):
                if _ == point2_index + 1:
                    dis += distance_between_2points([line_arr[_, 0], line_arr[_, 1]], [point2_x, point2_y])
                elif _ == point1_index + 1:
                    dis += distance_between_2points([line_arr[_ - 1, 0], line_arr[_ - 1, 1]], [point1_x, point1_y])
                else:
                    if line_distance is None:
                        dis += distance_between_2points([line_arr[_, 0], line_arr[_, 1]], [line_arr[_ - 1, 0], line_arr[_ - 1, 1]])
                    else:
                        dis += line_distance[_]
            t3_all += time.time() - t3
            # print(f'{str(t1_all)}s, {str(t2_all)}s, {str(t3_all)}s')
            return dis
        elif point2_index > point1_index:
            dis = 0
            for _ in range(point1_index + 1, point2_index + 2):
                if _ == point2_index + 1:
                    dis += distance_between_2points([line_arr[_ - 1, 0], line_arr[_ - 1, 1]], [point2_x, point2_y])
                elif _ == point1_index + 1:
                    dis += distance_between_2points([line_arr[_, 0], line_arr[_, 1]], [point1_x, point1_y])
                else:
                    if line_distance is None:
                        dis += distance_between_2points([line_arr[_, 0], line_arr[_, 1]], [line_arr[_ - 1, 0], line_arr[_ - 1, 1]])
                    else:
                        dis += line_distance[_]
            t3_all += time.time() - t3
            # print(f'{str(t1_all)}s, {str(t2_all)}s, {str(t3_all)}s')
            return dis
        elif point1_index == point2_index:
            t3_all += time.time() - t3
            # print(f'{str(t1_all)}s, {str(t2_all)}s, {str(t3_all)}s')
            return distance_between_2points([point1_x, point1_y], [point2_x, point2_y])


def hydro_wl_interpolation(df: pd.DataFrame, thal, year_range, geotransform: list, cs_list: list, hydro_pos: list):

    try:
        # Drop unnes water level
        all_year_list, year_domain = [year_ for year_ in year_range], []
        for _ in cs_list:
            unne_series = None
            for year in year_range:
                if unne_series is None:
                    unne_series = (thal.hydro_inform_dic[_]['year'] != year)
                else:
                    unne_series = unne_series & (thal.hydro_inform_dic[_]['year'] != year)

            thal.hydro_inform_dic[_] = thal.hydro_inform_dic[_].drop(thal.hydro_inform_dic[_][unne_series].index).reset_index(drop=True)
            year_domain.append(np.unique(np.array(thal.hydro_inform_dic[_]['year'])).tolist())

        # Define the output data
        ul_x, x_res, ul_y, y_res = geotransform
        df = df.reset_index(drop=True)
        yearly_wl = {str(year) + '_wl': [] for year in year_range}

        with tqdm(total=df.shape[0], desc='Process hydrodatacube water level', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
            for len_t in range(df.shape[0]):
                y_temp, x_temp = int(df['y'][len_t]), int(df['x'][len_t])
                coord_x, coord_y = ul_x + (x_temp + 0.5) * x_res, ul_y + (y_temp + 0.5) * y_res
                hydro_pos_temp = copy.deepcopy(hydro_pos)

                if thal.smoothed_Thalweg is None:
                    nearest_p = nearest_points(Point([coord_x, coord_y]), thal.Thalweg_Linestring)[1]
                    start_vertex_index = determin_start_vertex_of_point(nearest_p, thal.Thalweg_Linestring)

                    factor = None
                    hydro_index_list, year_list = [], []
                    for year in year_range:
                        start_hydro_index = copy.deepcopy(start_vertex_index)
                        end_hydro_index = copy.deepcopy(start_vertex_index + 1)

                        if start_hydro_index < min(hydro_pos):
                            factor = 1
                            start_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] >= start_hydro_index]
                            if len(start_hydro_index_list) == 0:
                                raise Exception('Code Error!')
                            else:
                                start_hydro_index = min(start_hydro_index_list)

                            end_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] > start_hydro_index]
                            if len(end_hydro_index_list) == 0:
                                raise Exception('Code Error!')
                            else:
                                end_hydro_index = min(end_hydro_index_list)

                        elif end_hydro_index > max(hydro_pos):
                            factor = 2
                            end_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] <= end_hydro_index]
                            if len(end_hydro_index_list) == 0:
                                raise Exception('Code Error!')
                            else:
                                end_hydro_index = max(end_hydro_index_list)

                            start_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] < end_hydro_index]
                            if len(start_hydro_index_list) == 0:
                                raise Exception('Code Error!')
                            else:
                                start_hydro_index = max(start_hydro_index_list)

                        else:
                            factor = 3
                            start_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] <= start_hydro_index]
                            end_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] >= end_hydro_index]

                            if len(start_hydro_index_list) == 0 or len(end_hydro_index_list) == 0:
                                raise Exception('Code Error!')
                            else:
                                start_hydro_index = max(start_hydro_index_list)
                                end_hydro_index = min(end_hydro_index_list)

                        year_list.append(year)
                        hydro_index_list.append([start_hydro_index, end_hydro_index])

                    hydro_unique_index_list = np.unique(np.array(hydro_index_list), axis=0).tolist()
                    inform_list = []
                    for _ in hydro_unique_index_list:
                        inform_temp = []
                        inform_temp.append(Point([thal.Thalweg_Linestring.coords[start_hydro_index][0], thal.Thalweg_Linestring.coords[start_hydro_index][1]]))
                        inform_temp.append(Point([thal.Thalweg_Linestring.coords[end_hydro_index][0], thal.Thalweg_Linestring.coords[end_hydro_index][1]]))
                        inform_temp.append(dis2points_via_line(nearest_p, inform_temp[0], thal.Thalweg_Linestring))
                        inform_temp.append(dis2points_via_line(nearest_p, inform_temp[1], thal.Thalweg_Linestring))
                        inform_list.append(inform_temp)

                    for year in year_list:
                        ii = year_list.index(year)
                        start_hydro_index = hydro_index_list[ii][0]
                        end_hydro_index = hydro_index_list[ii][1]

                        iii = hydro_unique_index_list.index(hydro_index_list[ii])
                        dis_to_start_station = inform_list[iii][2]
                        dis_to_end_station = inform_list[iii][3]

                        start_cs = cs_list[hydro_pos.index(start_hydro_index)]
                        end_cs = cs_list[hydro_pos.index(end_hydro_index)]

                        wl_start = np.array(thal.hydro_inform_dic[start_cs][thal.hydro_inform_dic[start_cs]['year'] == year]['water_level/m'])
                        wl_end = np.array(thal.hydro_inform_dic[end_cs][thal.hydro_inform_dic[end_cs]['year'] == year]['water_level/m'])

                        if wl_start.shape[0] != wl_end.shape[0]:
                            raise ValueError(f'The water level of {cs_list[end_hydro_index]} and {cs_list[start_hydro_index]} in year {str(year)} is not consistent')
                        else:
                            if factor == 1:
                                wl_pos = wl_start - (wl_end - wl_start) * dis_to_start_station / (dis_to_end_station - dis_to_start_station)
                            elif factor == 2:
                                wl_pos = wl_start + (wl_end - wl_start) * dis_to_start_station / (dis_to_start_station - dis_to_end_station)
                            elif factor == 3:
                                wl_pos = wl_start + (wl_end - wl_start) * dis_to_start_station / (dis_to_start_station + dis_to_end_station)

                        yearly_wl[str(year) + '_wl'].append([year, start_cs, end_cs, dis_to_start_station, dis_to_end_station])

                elif isinstance(thal.smoothed_Thalweg, LineString):

                    nearest_p = nearest_points(Point([coord_x, coord_y]), thal.smoothed_Thalweg)[1]
                    start_vertex_index = determin_start_vertex_of_point(nearest_p, thal.smoothed_Thalweg)

                    factor = None
                    hydro_index_list, year_list = [], []
                    for year in year_range:
                        start_hydro_index = copy.deepcopy(start_vertex_index)
                        end_hydro_index = copy.deepcopy(start_vertex_index + 1)

                        if start_hydro_index < min(hydro_pos):
                            factor = 1
                            start_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] >= start_hydro_index]
                            if len(start_hydro_index_list) == 0:
                                raise Exception('Code Error!')
                            else:
                                start_hydro_index = min(start_hydro_index_list)

                            end_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] > start_hydro_index]
                            if len(end_hydro_index_list) == 0:
                                raise Exception('Code Error!')
                            else:
                                end_hydro_index = min(end_hydro_index_list)

                        elif end_hydro_index > max(hydro_pos):
                            factor = 2
                            end_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] <= end_hydro_index]
                            if len(end_hydro_index_list) == 0:
                                raise Exception('Code Error!')
                            else:
                                end_hydro_index = max(end_hydro_index_list)

                            start_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] < end_hydro_index]
                            if len(start_hydro_index_list) == 0:
                                raise Exception('Code Error!')
                            else:
                                start_hydro_index = max(start_hydro_index_list)

                        else:
                            factor = 3
                            start_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] <= start_hydro_index]
                            end_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] >= end_hydro_index]

                            if len(start_hydro_index_list) == 0 or len(end_hydro_index_list) == 0:
                                raise Exception('Code Error!')
                            else:
                                start_hydro_index = max(start_hydro_index_list)
                                end_hydro_index = min(end_hydro_index_list)

                        year_list.append(year)
                        hydro_index_list.append([start_hydro_index, end_hydro_index])

                    hydro_unique_index_list = np.unique(np.array(hydro_index_list), axis=0).tolist()
                    for _ in hydro_unique_index_list:

                        start_hydro_index = _[0]
                        end_hydro_index = _[1]
                        dis_to_start_station = dis2points_via_line(nearest_p, [thal.smoothed_Thalweg.coords[start_hydro_index][0], thal.smoothed_Thalweg.coords[start_hydro_index][1]], thal.smoothed_Thalweg)
                        dis_to_end_station = dis2points_via_line(nearest_p, [thal.smoothed_Thalweg.coords[end_hydro_index][0], thal.smoothed_Thalweg.coords[end_hydro_index][1]], thal.smoothed_Thalweg)

                        start_cs = cs_list[hydro_pos.index(start_hydro_index)]
                        end_cs = cs_list[hydro_pos.index(end_hydro_index)]

                        year_unique_list = [year_list[__] for __ in range(len(hydro_index_list)) if hydro_index_list[__] == _]
                        if max(year_unique_list) - min(year_unique_list) == len(year_unique_list) - 1:
                            wl_start_series = (thal.hydro_inform_dic[start_cs]['year'] >= min(year_unique_list)) & (thal.hydro_inform_dic[start_cs]['year'] <= max(year_unique_list))
                            wl_end_series = (thal.hydro_inform_dic[end_cs]['year'] >= min(year_unique_list)) & (thal.hydro_inform_dic[end_cs]['year'] <= max(year_unique_list))
                        else:
                            wl_start_series, wl_end_series = None, None
                            for year in year_unique_list:
                                if wl_start_series is None:
                                    wl_start_series = (thal.hydro_inform_dic[start_cs]['year'] == year)
                                else:
                                    wl_start_series = wl_start_series | (thal.hydro_inform_dic[start_cs]['year'] == year)

                                if wl_end_series is None:
                                    wl_end_series = (thal.hydro_inform_dic[end_cs]['year'] == year)
                                else:
                                    wl_end_series = wl_end_series | (thal.hydro_inform_dic[end_cs]['year'] == year)

                        wl_start = thal.hydro_inform_dic[start_cs][wl_start_series]['water_level/m'].reset_index(drop=True)
                        wl_end = thal.hydro_inform_dic[end_cs][wl_end_series]['water_level/m'].reset_index(drop=True)

                        if wl_start.shape[0] != wl_end.shape[0]:
                            raise ValueError(f'The water level of {cs_list[hydro_pos.index(start_hydro_index)]} and {cs_list[hydro_pos.index(end_hydro_index)]} in year {str(year)} is not consistent')
                        else:
                            if factor == 1:
                                wl_pos = wl_start - (wl_end - wl_start) * dis_to_start_station / (dis_to_end_station - dis_to_start_station)
                            elif factor == 2:
                                wl_pos = wl_start + (wl_end - wl_start) * dis_to_start_station / (dis_to_start_station - dis_to_end_station)
                            elif factor == 3:
                                wl_pos = wl_start + (wl_end - wl_start) * dis_to_start_station / (dis_to_start_station + dis_to_end_station)

                        for year_temp in year_unique_list:
                            yearly_wl[str(year) + '_wl'].append([year_temp, start_cs, end_cs, dis_to_start_station, dis_to_end_station])
                else:
                    raise Exception('Code Error')
                pbar.update()

        for year in year_range:
            df[str(year) + '_wl'] = yearly_wl[str(year) + '_wl']
        return df
    except:
        print(traceback.format_exc())


def flood_frequency_based_hypsometry(df: pd.DataFrame, thal, year_range, geotransform: list, cs_list: list, hydro_pos: list, hydro_datacube: bool):

    try:

        # Drop unnes water level
        all_year_list, year_doy, year_domain = [], [], []
        for _ in cs_list:
            unne_series = None
            for year in year_range:
                if unne_series is None:
                    unne_series = (thal.hydro_inform_dic[_]['year'] != year)
                else:
                    unne_series = unne_series & (thal.hydro_inform_dic[_]['year'] != year)

            thal.hydro_inform_dic[_] = thal.hydro_inform_dic[_].drop(thal.hydro_inform_dic[_][unne_series].index).reset_index(drop=True)
            year_domain.append(np.unique(np.array(thal.hydro_inform_dic[_]['year'])).tolist())

        # Generate year list
        for year_ in year_range:
            all_year_list.append(year_)
            year_doy.append(datetime(year=year_ + 1, month=1, day=1).toordinal() - datetime(year=year_, month=1, day=1).toordinal())

        # Define the output data
        ul_x, x_res, ul_y, y_res = geotransform
        df = df.reset_index(drop=True)
        wl, fr, yearly_wl = [], [], []

        with tqdm(total=df.shape[0], desc=f'Process the inundation frequency', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
            # t1_all, t2_all, t3_all, t4_all, t5_all, t6_all = 0, 0, 0, 0, 0, 0
            for len_t in range(df.shape[0]):
                y_temp, x_temp, if_value = int(df['y'][len_t]), int(df['x'][len_t]), df['if'][len_t]
                if hydro_datacube:
                    yearly_wl_temp = [[] for _ in year_range]

                if ~np.isnan(if_value):
                    coord_x, coord_y = ul_x + (x_temp + 0.5) * x_res, ul_y + (y_temp + 0.5) * y_res
                    wl_all = []
                    hydro_pos_temp = copy.deepcopy(hydro_pos)

                    if thal.smoothed_Thalweg is None:
                        nearest_p = nearest_points(Point([coord_x, coord_y]), thal.Thalweg_Linestring)[1]
                        start_vertex_index = determin_start_vertex_of_point(nearest_p, thal.Thalweg_Linestring)

                        # Determine the hydrostation
                        factor = None
                        hydro_index_list, year_list = [], []
                        for year in year_range:
                            start_hydro_index = copy.deepcopy(start_vertex_index)
                            end_hydro_index = copy.deepcopy(start_vertex_index + 1)

                            if start_hydro_index < min(hydro_pos):
                                factor = 1
                                start_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] >= start_hydro_index]
                                if len(start_hydro_index_list) == 0:
                                    raise Exception('Code Error!')
                                else:
                                    start_hydro_index = min(start_hydro_index_list)

                                end_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] > start_hydro_index]
                                if len(end_hydro_index_list) == 0:
                                    raise Exception('Code Error!')
                                else:
                                    end_hydro_index = min(end_hydro_index_list)

                            elif end_hydro_index > max(hydro_pos):
                                factor = 2
                                end_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] <= end_hydro_index]
                                if len(end_hydro_index_list) == 0:
                                    raise Exception('Code Error!')
                                else:
                                    end_hydro_index = max(end_hydro_index_list)

                                start_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] < end_hydro_index]
                                if len(start_hydro_index_list) == 0:
                                    raise Exception('Code Error!')
                                else:
                                    start_hydro_index = max(start_hydro_index_list)

                            else:
                                factor = 3
                                start_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] <= start_hydro_index]
                                end_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] >= end_hydro_index]

                                if len(start_hydro_index_list) == 0 or len(end_hydro_index_list) == 0:
                                    raise Exception('Code Error!')
                                else:
                                    start_hydro_index = max(start_hydro_index_list)
                                    end_hydro_index = min(end_hydro_index_list)

                            year_list.append(year)
                            hydro_index_list.append([start_hydro_index, end_hydro_index])

                        hydro_unique_index_list = np.unique(np.array(hydro_index_list), axis=0).tolist()
                        inform_list = []
                        for _ in hydro_unique_index_list:
                            inform_temp = []
                            # _ 0/1 coord
                            inform_temp.append(Point([thal.Thalweg_Linestring.coords[start_hydro_index][0], thal.Thalweg_Linestring.coords[start_hydro_index][1]]))
                            inform_temp.append(Point([thal.Thalweg_Linestring.coords[end_hydro_index][0], thal.Thalweg_Linestring.coords[end_hydro_index][1]]))
                            # _ 2/3 dis
                            inform_temp.append(dis2points_via_line(nearest_p, inform_temp[0], thal.Thalweg_Linestring))
                            inform_temp.append(dis2points_via_line(nearest_p, inform_temp[1], thal.Thalweg_Linestring))
                            inform_list.append(inform_temp)

                        for year in year_list:
                            ii = year_list.index(year)
                            start_hydro_index = hydro_index_list[ii][0]
                            end_hydro_index = hydro_index_list[ii][1]

                            iii = hydro_unique_index_list.index(hydro_index_list[ii])
                            dis_to_start_station = inform_list[iii][2]
                            dis_to_end_station = inform_list[iii][3]

                            start_cs = cs_list[hydro_pos.index(start_hydro_index)]
                            end_cs = cs_list[hydro_pos.index(end_hydro_index)]

                            wl_start = np.array(thal.hydro_inform_dic[start_cs][thal.hydro_inform_dic[start_cs]['year'] == year]['water_level/m'])
                            wl_end = np.array(thal.hydro_inform_dic[end_cs][thal.hydro_inform_dic[end_cs]['year'] == year]['water_level/m'])

                            if wl_start.shape[0] != wl_end.shape[0]:
                                raise ValueError(f'The water level of {cs_list[end_hydro_index]} and {cs_list[start_hydro_index]} in year {str(year)} is not consistent')
                            else:
                                if factor == 1:
                                    wl_pos = wl_start - (wl_end - wl_start) * dis_to_start_station / (dis_to_end_station - dis_to_start_station)
                                elif factor == 2:
                                    wl_pos = wl_start + (wl_end - wl_start) * dis_to_start_station / (dis_to_start_station - dis_to_end_station)
                                elif factor == 3:
                                    wl_pos = wl_start + (wl_end - wl_start) * dis_to_start_station / (dis_to_start_station + dis_to_end_station)

                            wl_all.extend(wl_pos.flatten().tolist())

                    elif isinstance(thal.smoothed_Thalweg, LineString):

                        nearest_p = nearest_points(Point([coord_x, coord_y]), thal.smoothed_Thalweg)[1]
                        start_vertex_index = determin_start_vertex_of_point(nearest_p, thal.smoothed_Thalweg)

                        # Determine the hydrostation
                        # t1 = time.time()
                        factor = None
                        hydro_index_list, year_list = [], []
                        for year in year_range:
                            start_hydro_index = copy.deepcopy(start_vertex_index)
                            end_hydro_index = copy.deepcopy(start_vertex_index + 1)

                            if start_hydro_index < min(hydro_pos):
                                factor = 1
                                start_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] >= start_hydro_index]
                                if len(start_hydro_index_list) == 0:
                                    raise Exception('Code Error!')
                                else:
                                    start_hydro_index = min(start_hydro_index_list)

                                end_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] > start_hydro_index]
                                if len(end_hydro_index_list) == 0:
                                    raise Exception('Code Error!')
                                else:
                                    end_hydro_index = min(end_hydro_index_list)

                            elif end_hydro_index > max(hydro_pos):
                                factor = 2
                                end_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] <= end_hydro_index]
                                if len(end_hydro_index_list) == 0:
                                    raise Exception('Code Error!')
                                else:
                                    end_hydro_index = max(end_hydro_index_list)

                                start_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] < end_hydro_index]
                                if len(start_hydro_index_list) == 0:
                                    raise Exception('Code Error!')
                                else:
                                    start_hydro_index = max(start_hydro_index_list)

                            else:
                                factor = 3
                                start_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] <= start_hydro_index]
                                end_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] >= end_hydro_index]

                                if len(start_hydro_index_list) == 0 or len(end_hydro_index_list) == 0:
                                    raise Exception('Code Error!')
                                else:
                                    start_hydro_index = max(start_hydro_index_list)
                                    end_hydro_index = min(end_hydro_index_list)

                            year_list.append(year)
                            hydro_index_list.append([start_hydro_index, end_hydro_index])

                        # t1_all += time.time() - t1
                        hydro_unique_index_list = np.unique(np.array(hydro_index_list), axis=0).tolist()
                        for _ in hydro_unique_index_list:

                            # get the hydro station index
                            start_hydro_index = _[0]
                            end_hydro_index = _[1]

                            # Calculate the dis between nearest p with start and end station
                            t2 = time.time()
                            dis_to_start_station = dis2points_via_line(nearest_p, [thal.smoothed_Thalweg.coords[start_hydro_index][0], thal.smoothed_Thalweg.coords[start_hydro_index][1]], thal.smoothed_Thalweg)
                            dis_to_end_station = dis2points_via_line(nearest_p, [thal.smoothed_Thalweg.coords[end_hydro_index][0], thal.smoothed_Thalweg.coords[end_hydro_index][1]], thal.smoothed_Thalweg)
                            # t2_all += time.time() - t2

                            # Retrieve the water series of start and end stations
                            # t3 = time.time()
                            start_cs = cs_list[hydro_pos.index(start_hydro_index)]
                            end_cs = cs_list[hydro_pos.index(end_hydro_index)]

                            year_unique_list = [year_list[__] for __ in range(len(hydro_index_list)) if hydro_index_list[__] == _]
                            if max(year_unique_list) - min(year_unique_list) == len(year_unique_list) - 1:
                                wl_start_series = (thal.hydro_inform_dic[start_cs]['year'] >= min(year_unique_list)) & (thal.hydro_inform_dic[start_cs]['year'] <= max(year_unique_list))
                                wl_end_series = (thal.hydro_inform_dic[end_cs]['year'] >= min(year_unique_list)) & (thal.hydro_inform_dic[end_cs]['year'] <= max(year_unique_list))
                            else:
                                wl_start_series, wl_end_series = None, None
                                for year in year_unique_list:
                                    if wl_start_series is None:
                                        wl_start_series = (thal.hydro_inform_dic[start_cs]['year'] == year)
                                    else:
                                        wl_start_series = wl_start_series | (thal.hydro_inform_dic[start_cs]['year'] == year)

                                    if wl_end_series is None:
                                        wl_end_series = (thal.hydro_inform_dic[end_cs]['year'] == year)
                                    else:
                                        wl_end_series = wl_end_series | (thal.hydro_inform_dic[end_cs]['year'] == year)
                            # t3_all += time.time() - t3

                            # Interpolate the water level
                            # t4 = time.time()
                            wl_start = thal.hydro_inform_dic[start_cs][wl_start_series]['water_level/m'].reset_index(drop=True)
                            wl_end = thal.hydro_inform_dic[end_cs][wl_end_series]['water_level/m'].reset_index(drop=True)

                            if wl_start.shape[0] != wl_end.shape[0]:
                                raise ValueError(f'The water level of {cs_list[hydro_pos.index(start_hydro_index)]} and {cs_list[hydro_pos.index(end_hydro_index)]} in year {str(year)} is not consistent')
                            else:
                                if factor == 1:
                                    wl_pos = wl_start - (wl_end - wl_start) * dis_to_start_station / (dis_to_end_station - dis_to_start_station)
                                elif factor == 2:
                                    wl_pos = wl_start + (wl_end - wl_start) * dis_to_start_station / (dis_to_start_station - dis_to_end_station)
                                elif factor == 3:
                                    wl_pos = wl_start + (wl_end - wl_start) * dis_to_start_station / (dis_to_start_station + dis_to_end_station)
                            # t4_all += time.time() - t4

                            wl_pos = list(wl_pos)
                            wl_all.extend(wl_pos)

                            if hydro_datacube:
                                for year_temp in year_unique_list:
                                    yearly_wl_temp[all_year_list.index(year_temp)] = [year_temp, start_cs, end_cs, dis_to_start_station, dis_to_end_station]

                    else:
                        raise Exception('Code Error')

                    t6 = time.time()
                    if if_value == 1:
                        wl.append(np.nan)
                        fr.append(1)
                    else:
                        wl_all = np.sort(np.array(wl_all))
                        inun_freq = np.linspace(1, 1 / wl_all.shape[0], wl_all.shape[0])
                        wl_factor = False
                        for _ in range(inun_freq.shape[0] - 1):
                            if (inun_freq[_] - if_value) * (inun_freq[_ + 1] - if_value) < 0:
                                wl.append(wl_all[_])
                                fr.append(inun_freq[_])
                                wl_factor = True
                                break
                            elif inun_freq[_] - if_value == 0:
                                wl.append(wl_all[_])
                                fr.append(inun_freq[_])
                                wl_factor = True
                                break
                            elif inun_freq[_ + 1] - if_value == 0:
                                wl.append(wl_all[_ + 1])
                                fr.append(inun_freq[_ + 1])
                                wl_factor = True
                                break
                        # for _ in range(wl_unique.shape[0] - 1):
                        #     if_lower = np.sum(wl_all <= wl_unique[_]) / wl_all.shape[0]
                        #     if_higher = np.sum(wl_all <= wl_unique[_ + 1]) / wl_all.shape[0]
                        #     if if_lower <= if_value and if_higher > if_value:
                        #         wl.append(wl_unique[_])
                        #         fr.append(np.sum(wl_all >= wl_unique[_]) / wl_all.shape[0])
                        #         wl_factor = True
                        #         break
                        if wl_factor is False:
                            wl.append(np.nan)
                            fr.append(np.nan)

                    # t6_all += t6 + time.time()
                    # if np.mod(len_t, 5000) == 0:
                    #     print(f'{str(t1_all)}s, {str(t2_all)}s, {str(t3_all)}s, {str(t4_all)}s, {str(t5_all)}s, {str(t6_all)}s')
                elif np.isnan(if_value):
                    wl.append(np.nan)
                    fr.append(np.nan)

                if hydro_datacube:
                    yearly_wl.append(yearly_wl_temp)

                pbar.update()
        df['wl'] = wl
        df['fr'] = fr
        if hydro_datacube:
            df['yearly_wl'] = yearly_wl
        return df
    except:
        print(traceback.format_exc())


def retrieve_correct_filename(file_name: str):
    file_list = file_name.split('\\')
    sep = '\\'
    if len(file_list) == 1:
        raise Exception('filename cannot be a folder!')
    else:
        for _ in range(1, len(file_list)):
            folder = [__ for __ in file_list[: _]]
            folder_name = sep.join(folder) + sep
            up_folder = [__ for __ in file_list[: _ - 1]]
            up_folder_name = sep.join(up_folder) + sep
            if os.path.exists(folder_name):
                pass
            else:
                dir_all = os.listdir(up_folder_name)
                dir_all = [up_folder_name + __ for __ in dir_all]
                dir_all_ = copy.copy(dir_all)
                try:
                    for __ in dir_all:
                        if os.path.isdir(__):
                            dir_all_.extend([os.path.join(__, file) for file in os.listdir(__)])
                except:
                    pass

                new_path = None
                file_temp = [__ for __ in file_list[_:]]
                for file_ in file_temp:
                    for dir_ in dir_all_:
                        if dir_.endswith(file_) and '.' in file_:
                            return os.path.join(dir_, file_)
                        elif dir_.endswith(file_) and '.' not in file_:
                            file_name = os.path.join(dir_, sep.join(file_temp[file_temp.index(file_) + 1:]))
                            if os.path.isfile(file_name):
                                return file_name
                        else:
                            pass

                return None

def hydrodc_csv2matrix(hydroinform_df, hydroinform_dic, Xsize, Ysize, year):

    # Check if hydro station data is import
    if hydroinform_dic is None:
        raise Exception('Please input the hydro inform first')

    # Get the year list and array size
    hydro_inform = list(hydroinform_df['yearly_wl'])
    hydro_inform_list = []
    hydro_list = []
    hydro_temp = {}
    for _ in hydroinform_dic.keys():
        hydro_temp[_] = list(hydroinform_dic[_][hydroinform_dic[_]['year'] == year]['water_level/m'])
    hydro_list.append(hydro_temp)

    # Define the matrix
    mem = psutil.virtual_memory().available
    if hydroinform_df.shape[0] / (Xsize * Ysize) < 0.2 or Xsize * Ysize * 4 * 365 > psutil.virtual_memory().available:
        if not os.path.exists(f'{outputfolder}{str(year)}\\SMsequence.npz.npy'):
            # Define the sparse matrix
            doy_list = [year * 1000 + _ for _ in range(1, datetime(year=year + 1, month=1, day=1).toordinal() - datetime(year=year, month=1, day=1).toordinal() + 1)]
            sm_list = [sm.lil_matrix((Ysize, Xsize)) for _ in range(len(doy_list))]

            with tqdm(total=len(hydro_inform), desc=f'Generate hydro datacube',
                      bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
                for _ in range(len(hydro_inform)):
                    y, x = int(y_l[_]), int(x_l[_])
                    # print(str(y) + str(x))
                    wl_start_series = np.array(hydro_dic[hydro_inform[_][1]])
                    wl_end_series = np.array(hydro_dic[hydro_inform[_][2]])
                    # print(str(wl_start_series))
                    wl_start_dis = hydro_inform[_][3]
                    wl_end_dis = hydro_inform[_][4]
                    wl_inter = wl_start_series + (wl_end_series - wl_start_series) * wl_start_dis / (
                                wl_start_dis + wl_end_dis)
                    # print(str(wl_inter))
                    # print(str(wl_end_dis))
                    for __ in range(len(wl_inter)):
                        sm_list[__][y, x] = wl_inter[__]
                    pbar.update()

            print(f'Start saving the hydro datacube of year {str(year)}!')
            st = time.time()
            # for _ in range(len(sm_list)):
            #     sm_list[_] = sm_list[_].tocsr()
            ND_temp = NDSparseMatrix(*sm_list, SM_namelist=doy_list)
            ND_temp.save(f'{outputfolder}{str(year)}\\')
            print(f'Finish saving the hydro datacube of year {str(year)} in {str(time.time() - st)}!')

    else:
        doy_list = [year * 1000 + _ for _ in range(1, datetime(year=year + 1, month=1, day=1).toordinal() - datetime(year=year, month=1, day=1).toordinal() + 1)]
        dc = np.zeros((Ysize, Xsize, len(doy_list))) * np.nan
        np.save(f'{outputfolder}{str(year)}\\', dc)


def get_value_through_river_crosssection(num, coord, slope, nodata, itr, tiffile, length, outputfolder, limit=10000):

    gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'NO')
    time1, time2, time3, time4 = 0, 0, 0, 0

    # print(str(num))
    left_value, right_value = [], []
    slope_sta_left_x, slope_sta_left_y = -slope[0][1], slope[0][0]
    slope_end_left_x, slope_end_left_y = -slope[1][1], slope[1][0]
    slope_sta_right_x, slope_sta_right_y = slope[0][1], -slope[0][0]
    slope_end_right_x, slope_end_right_y = slope[1][1], -slope[1][0]
    sta_coord_, end_coord_ = coord[0], coord[1]
    # print(str(sta_coord_), str(end_coord_))
    in_left_floodplain, in_right_floodplain = True, True
    left_pix, right_pix = 0, 0
    ds_temp = gdal.Open(tiffile)
    [ul_x, x_res, xt, ul_y, yt, y_res] = ds_temp.GetGeoTransform()
    xmax, xmin, ymax, ymin = ul_x + x_res * ds_temp.RasterXSize, ul_x, ul_y, ul_y + y_res * ds_temp.RasterYSize

    while in_left_floodplain:

        try:
            st = time.time()
            bounds = [[sta_coord_[0] + slope_sta_left_x * left_pix * itr, sta_coord_[1] + slope_sta_left_y * left_pix * itr],
                      [sta_coord_[0] + slope_sta_left_x * (left_pix + 1) * itr, sta_coord_[1] + slope_sta_left_y * (left_pix + 1) * itr],
                      [end_coord_[0] + slope_end_left_x * (left_pix + 1) * itr, end_coord_[1] + slope_end_left_y * (left_pix + 1) * itr],
                      [end_coord_[0] + slope_end_left_x * left_pix * itr, end_coord_[1] + slope_end_left_y * left_pix * itr]]
            bounds_arr = np.array(bounds)
            minx, miny, maxx, maxy = np.min(bounds_arr[:, 0]), np.min(bounds_arr[:, 1]), np.max(bounds_arr[:, 0]), np.max(bounds_arr[:, 1])
            time1 += time.time() - st

            if minx < xmin or miny < ymin or maxx > xmax or maxy > ymax or left_pix > limit/itr:
                in_left_floodplain = False
            else:

                st = time.time()
                gdal.Warp(f'/vsimem/l{str(num)}_{str(left_pix)}.vrt', tiffile, outputBounds=(minx, miny, maxx, maxy), targetAlignedPixels=False, multithread=True)
                ds = gdal.Open(f'/vsimem/l{str(num)}_{str(left_pix)}.vrt')
                # gdal.Warp(f'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11_nc\\temp\\l{str(num)}_{str(left_pix)}.vrt', tiffile, outputBounds=(minx, miny, maxx, maxy), targetAlignedPixels=False)
                # ds = gdal.Open(f'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11_nc\\temp\\l{str(num)}_{str(left_pix)}.vrt')
                [ul_x, x_res, xt, ul_y, yt, y_res] = ds.GetGeoTransform()
                arr_ = ds.GetRasterBand(1).ReadAsArray()
                time2 += time.time() - st

                st = time.time()
                for y_ in range(ds.RasterYSize):
                    for x_ in range(ds.RasterXSize):
                        coord_ = (ul_x + x_res * x_, ul_y + y_ * y_res)
                        if not is_point_in_polygon(coord_, bounds):
                            arr_[y_, x_] = nodata

                if np.isnan(nodata):
                    if np.isnan(arr_).all():
                        in_left_floodplain = False
                    else:
                        arr_[arr_ == -200] = nodata
                        if np.isnan(arr_).all():
                            left_value.append(np.nan)
                        else:
                            left_value.append(np.nanmean(arr_))
                else:
                    if (arr_ == nodata).all():
                        in_left_floodplain = False
                    else:
                        arr_[arr_ == -200] = nodata
                        if (arr_ == nodata).all():
                            left_value.append(np.nan)
                        else:
                            left_value.append(np.nanmean(arr_))
                ds = None
                gdal.Unlink(f'/vsimem/l{str(num)}_{str(left_pix)}.vrt')
                left_pix += 1
                time3 += time.time() - st
                print(str(left_pix))
        except:
            print((minx, miny, maxx, maxy, left_pix))
            print(traceback.format_exc())
            raise Exception('-1')

    while in_right_floodplain:

        try:
            st = time.time()
            bounds = [[sta_coord_[0] + slope_sta_right_x * right_pix * itr, sta_coord_[1] + slope_sta_right_y * right_pix * itr],
                      [sta_coord_[0] + slope_sta_right_x * (right_pix + 1) * itr, sta_coord_[1] + slope_sta_right_y * (right_pix + 1) * itr],
                      [end_coord_[0] + slope_end_right_x * (right_pix + 1) * itr, end_coord_[1] + slope_end_right_y * (right_pix + 1) * itr],
                      [end_coord_[0] + slope_end_right_x * right_pix * itr, end_coord_[1] + slope_end_right_y * right_pix * itr]]
            bounds_arr = np.array(bounds)
            minx, miny, maxx, maxy = np.min(bounds_arr[:, 0]), np.min(bounds_arr[:, 1]), np.max(bounds_arr[:, 0]), np.max(bounds_arr[:, 1])
            time1 += time.time() - st

            if minx < xmin or miny < ymin or maxx > xmax or maxy > ymax or right_pix > limit/itr:
                in_right_floodplain = False
            else:
                st = time.time()
                gdal.Warp(f'/vsimem/l{str(num)}_{str(right_pix)}.vrt', tiffile, outputBounds=(minx, miny, maxx, maxy), targetAlignedPixels=False, multithread=True)
                ds = gdal.Open(f'/vsimem/l{str(num)}_{str(right_pix)}.vrt')
                # gdal.Warp(f'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11_nc\\temp\\l{str(num)}_{str(right_pix)}.vrt', tiffile, outputBounds=(minx, miny, maxx, maxy), targetAlignedPixels=False)
                # ds = gdal.Open(f'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11_nc\\temp\\l{str(num)}_{str(right_pix)}.vrt')
                [ul_x, x_res, xt, ul_y, yt, y_res] = ds.GetGeoTransform()
                arr_ = ds.GetRasterBand(1).ReadAsArray()
                time2 += time.time() - st

                st = time.time()
                for y_ in range(ds.RasterYSize):
                    for x_ in range(ds.RasterXSize):
                        coord_ = (ul_x + x_res * x_, ul_y + y_ * y_res)
                        if not is_point_in_polygon(coord_, bounds):
                            arr_[y_, x_] = nodata

                if np.isnan(nodata):
                    if np.isnan(arr_).all():
                        in_right_floodplain = False
                    else:
                        arr_[arr_ == -200] = nodata
                        if np.isnan(arr_).all():
                            right_value.append(np.nan)
                        else:
                            right_value.append(np.nanmean(arr_))
                else:
                    if (arr_ == nodata).all():
                        in_right_floodplain = False
                    else:
                        arr_[arr_ == -200] = nodata
                        if (arr_ == nodata).all():
                            right_value.append(np.nan)
                        else:
                            right_value.append(np.nanmean(arr_))
                ds = None
                gdal.Unlink(f'/vsimem/l{str(num)}_{str(right_pix)}.vrt')
                right_pix += 1
                time3 += time.time() - st
                print(str(right_pix))
        except:
            print((minx, miny, maxx, maxy, right_pix))
            print(traceback.format_exc())
            raise Exception('-1')

    # gdal.Unlink(f'/vsimem/resized_{str(int(num % 32))}.vrt')
    st = time.time()
    new_arr = np.zeros([2, len(left_value) + len(right_value)])
    new_arr[0, :] = np.linspace(-len(left_value) + 1, len(right_value),  len(left_value) + len(right_value))
    new_arr[1, 0: len(left_value)] = np.array(left_value)[::-1]
    new_arr[1, len(left_value):] = np.array(right_value)
    np.save(f'{outputfolder}line_{str(num)}.npy', new_arr)
    time4 += time.time() - st
    print(f'{str(num)}/{str(length)} Complete! S1: {str(time1)}s; S2: {str(time2)}s; S3: {str(time3)}s; S4: {str(time4)}s;')


def is_point_in_polygon(p, polygon):
    # p is the point (x, y)
    # polygon is a list of tuples [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] defining the vertices of the polygon
    count = 0
    x_intercept = p[0]
    y_intercept = p[1]
    n = len(polygon)

    for i in range(n):
        v1 = polygon[i]
        v2 = polygon[(i + 1) % n]
        # Check if point aligns horizontally with vertex
        if y_intercept >= min(v1[1], v2[1]) and y_intercept <= max(v1[1], v2[1]):
            # Calculate the x coordinate of the intersecting point
            if v1[1] != v2[1]:  # Avoid division by zero
                x_intersect = (y_intercept - v1[1]) * (v2[0] - v1[0]) / (v2[1] - v1[1]) + v1[0]
                if x_intersect > x_intercept:
                    count += 1

    return count % 2 != 0