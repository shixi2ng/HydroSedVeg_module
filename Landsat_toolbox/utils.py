from osgeo import gdal
import pandas as pd
from osgeo import gdal_array, osr
import sys
import collections
import pandas
import numpy as np
import os
import shutil
import datetime
import copy
from scipy.signal import convolve2d
import time
from itertools import chain
from collections import Counter
import basic_function as bf
import matplotlib.pyplot as plt
import tarfile
import traceback
import libarchive


def union_list(small_list, big_list) -> list:
    union_list_temp = []
    if type(small_list) != list or type(big_list) != list:
        raise TypeError('Please input list for union list')

    for i in small_list:
        if i not in big_list:
            print(f'{i} is not supported!')
        else:
            union_list_temp.append(i)
    return union_list_temp


def seven_para_logistic_function(x, m1, m2, m3, m4, m5, m6, m7):
    return m1 + (m2 - m7 * x) * ((1 / (1 + np.exp((m3 - x) / m4))) - (1 / (1 + np.exp((m5 - x) / m6))))


def two_term_fourier(x, a0, a1, b1, a2, b2, w):
    return a0 + a1 * np.cos(w * x) + b1 * np.sin(w * x) + a2 * np.cos(2 * w * x)+b2 * np.sin(2 * w * x)


def unzip_Landsat_tarfile(tar_path, extract_path, unzipped_para=True):

    try:
        if unzipped_para:
            with libarchive.file_reader(tar_path) as entries:
                for entry in entries:
                    out_path = os.path.join(extract_path, os.path.basename(entry.pathname))
                    if os.path.exists(out_path):
                        continue  # Pass existing files

                    # Write in
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with open(out_path, "wb") as f_out:
                        for block in entry.get_blocks():
                            f_out.write(block)
            print(f'Successfully unzip Landsat file {tar_path}')
        else:
            unzipped_file = tarfile.TarFile(tar_path)
            unzipped_file.close()

        if True in [_ in tar_path for _ in ['LE07', 'LC08', 'LT04', 'LT05', 'LC09']]:
            _ = ['LE07', 'LC08', 'LT04', 'LT05', 'LC09'][[_ in tar_path for _ in ['LE07', 'LC08', 'LT04', 'LT05', 'LC09']].index(True)]
            Sensor_type = tar_path[tar_path.find(_): tar_path.find(_) + 4]
            FileID = tar_path[tar_path.find(_): tar_path.find('.tar')]
            Tile = tar_path[tar_path.find('L2S') + 5: tar_path.find('L2S') + 11]
            Date = tar_path[tar_path.find('L2S') + 12: tar_path.find('L2S') + 20]
            Tier_level = tar_path[tar_path.find('_T') + 1: tar_path.find('_T') + 3]
            File_path = tar_path
        else:
            raise Exception('Please make sure the file is from Landsat 7, 8, 4, 5, 9')
        return True, Sensor_type, FileID, Tile, Date, Tier_level, File_path
    except:
        print(traceback.format_exc())
        return False, None, None, None, None, None, tar_path


def trim_axs(axs, N):
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


class Readable_key_dic(pd.DataFrame):
    def __init__(self, ori_dic):
        super(Readable_key_dic, self).__init__()
        self.ori_dic = ori_dic
        self.ori_pd = pd.DataFrame.from_dict(ori_dic, orient='index')

    def export(self, path):
        self.ori_pd.to_excel(path)


def mostCommon(nd_array, indicator_array, nan_value=0):
    nd_list = nd_array.tolist()
    flatList = chain.from_iterable(nd_list)
    a = Counter(flatList)
    res = np.nan
    for i in range(np.unique(nd_list).shape[0]):
        if a.most_common(i + 1)[i][0] != nan_value and indicator_array[int(np.argwhere(nd_array == a.most_common(i + 1)[i][0])[0, 0]), int(np.argwhere(nd_array == a.most_common(i + 1)[i][0])[0, 1])] != 2:
            res = a.most_common(i + 1)[i][0]
            break

    if not np.isnan(res):
        return res
    else:
        return None


def confusion_matrix_2_raster(raster_predict, raster_real, nan_value=np.nan):
    if raster_predict.shape != raster_real.shape:
        print('Please make sure the two raster share the same grid and extent')
        sys.exit(-1)
    unique_value_list = np.unique(np.concatenate((np.unique(raster_predict), np.unique(raster_real))))
    unique_value_list = np.sort(np.array([i for i in unique_value_list if i != nan_value]))
    confusion_matrix = np.zeros([unique_value_list.shape[0] + 1, unique_value_list.shape[0] + 1]).astype(object)
    confusion_matrix[0, 0] = 'Confusion Matrix'
    confusion_matrix[0, 1:] = unique_value_list.astype(str)
    confusion_matrix[1:, 0] = unique_value_list.T.astype(str)
    for para_real in unique_value_list:
        for para_predict in unique_value_list:
            confusion_matrix[np.argwhere(unique_value_list == para_real)[0] + 1, np.argwhere(unique_value_list == para_predict)[0] + 1] = np.sum(np.logical_and(raster_real == para_real, raster_predict == para_predict))
    return confusion_matrix


def fill_landsat7_gap(ori_tif, mask_tif='Landsat7_default'):
    # Get the boundary
    driver = gdal.GetDriverByName('GTiff')
    if mask_tif == 'Landsat7_default':
        gap_mask_file = ori_tif.split('_02_T')[0] + '_gap_mask.TIF'
        if not os.path.exists(gap_mask_file):
            ori_ds = gdal.Open(ori_tif)
            ori_array = bf.file2raster(ori_tif)

            row_min = 0
            row_max = ori_array.shape[0]
            for row in range(ori_array.shape[0] - 1):
                row_state = ori_array[row, :] == 0
                next_row_state = ori_array[row + 1, :] == 0
                if row_state.all() == 1 and next_row_state.all() == 0:
                    row_min = max(row_min, row + 1)
                elif row_state.all() == 0 and next_row_state.all() == 1:
                    row_max = min(row_max, row)
            row_min_column_pos = int(np.mean(np.argwhere(ori_array[row_min,:] != 0)))
            row_max_column_pos = int(np.mean(np.argwhere(ori_array[row_max,:] != 0)))

            col_min = 0
            col_max = ori_array.shape[1]
            for col in range(ori_array.shape[1] - 1):
                col_state = ori_array[:, col] == 0
                next_col_state = ori_array[:, col + 1] == 0
                if col_state.all() == 1 and next_col_state.all() == 0:
                    col_min = max(col_min, col + 1)
                elif col_state.all() == 0 and next_col_state.all() == 1:
                    col_max = min(col_max, col)
            col_min_row_pos = int(np.mean(np.argwhere(ori_array[:, col_min] != 0)))
            col_max_row_pos = int(np.mean(np.argwhere(ori_array[:, col_max] != 0)))
            ul_slope = (row_min - col_min_row_pos) / (row_min_column_pos - col_min)
            ul_offset = row_min - row_min_column_pos * ul_slope
            ur_slope = (col_max_row_pos - row_min) / (col_max - row_min_column_pos)
            ur_offset = row_min - row_min_column_pos * ur_slope
            ll_slope = (row_max - col_min_row_pos) / (row_max_column_pos - col_min)
            ll_offset = row_max - row_max_column_pos * ll_slope
            lr_slope = (col_max_row_pos - row_max) / (col_max - row_max_column_pos)
            lr_offset = row_max - row_max_column_pos * lr_slope

            array_mask = np.ones_like(ori_array)
            for col in range(array_mask.shape[1]):
                for row in range(array_mask.shape[0]):
                    if row - ul_slope * col > ul_offset:
                        if row - ur_slope * col > ur_offset:
                            if row - ll_slope * col < ll_offset:
                                if row - lr_slope * col < lr_offset:
                                    array_mask[row, col] = 0
            array_mask[ori_array != 0] = 1
            bf.write_raster(ori_ds, array_mask, gap_mask_file, '', raster_datatype=gdal.GDT_Int16, nodatavalue=0)
            ori_ds = None
        ori_ds = gdal.Open(ori_tif)
        dst_ds = driver.CreateCopy(ori_tif.split('.TIF')[0] + "_SLC.TIF", ori_ds, strict=0)
        mask_ds = gdal.Open(gap_mask_file)
        mask_band = mask_ds.GetRasterBand(1)
        dst_band = dst_ds.GetRasterBand(1)
        gdal.FillNodata(targetBand=dst_band, maskBand=mask_band, maxSearchDist=14, smoothingIterations=0)
        ori_ds = None
        mask_ds = None
        dst_ds = None

    #
    #
    # src = rasterio.open(ori_tif)
    # raster_temp = src.read(1)
    # mask_temp = copy.copy(raster_temp)
    # mask_temp[mask_temp != 0] = 1
    # mask_result = np.zeros_like(mask_temp).astype(np.uint8)
    # if not os.path.exists(ori_tif.split('.TIF')[0] + '_MASK.TIF'):
    #     for y in range(mask_temp.shape[0]):
    #         for x in range(mask_temp.shape[1]):
    #             if mask_temp[y, x] == 0:
    #                 if np.sum(mask_temp[max(y - 10, 0): y, x]) == 0 or np.sum(mask_temp[y: min(mask_temp.shape[0] - 1, y + 10), x]) == 0:
    #                     mask_result[y, x] = 1
    #     with rasterio.open(ori_tif.split('.TIF')[0] + '_MASK.TIF', 'w', driver='GTiff',
    #                   height=src.shape[0], width=src.shape[1], count=1,
    #                   dtype=rasterio.uint8, crs=src.crs, transform=src.transform) as out:
    #         out.write(mask_result, 1)
    #
    # ds_temp = gdal.Open(ori_tif)
    # ds_temp2 = gdal.Open(ori_tif.split('.TIF')[0] + '_MASK.TIF')
    # mask_result = ds_temp2.GetRasterBand(1)
    # driver_tiff = gdal.GetDriverByName("GTiff")
    # gap_filled = driver_tiff.CreateCopy(ori_tif.split('.TIF')[0] + '_SLC.TIF', ds_temp, strict=0)
    # raster_temp = gap_filled.GetRasterBand(1)
    # gdal.FillNodata(raster_temp, mask_result, 10, 2)
    # gap_filled = None
    # ds_temp = None
    # ds_temp2 = None
    # gap_filled = cv2.inpaint(raster_temp, mask_result, 3, cv2.INPAINT_TELEA)
    # with rasterio.open(ori_tif.split('.TIF')[0] + '_SLC.TIF', 'w', driver='GTiff',
    #                    height=src.shape[0], width=src.shape[1], count=1,
    #                    dtype=rasterio.uint16, crs=src.crs, transform=src.transform) as out_slc:
    #     out_slc.write(gap_filled, 1)


def generate_error_inf(self, reverse_type=True):
    if self[0, self.shape[0] - 1] == 'Error':
        return self
    elif self[0, self.shape[0] - 1] != 'Error':
        new_matrix = np.zeros([self.shape[0] + 1, self.shape[1] + 1]).astype(object)
        new_matrix[0: self.shape[0], 0: self.shape[1]] = self
        i = 0
        new_matrix[new_matrix.shape[0] - 1, new_matrix.shape[1] - 1] = str(np.sum([self[i, i] for i in range(1, self.shape[0])]) / np.sum(self[1:self.shape[0], 1: self.shape[1]]) * 100)[0:4] + '%'
        while i < new_matrix.shape[0] - 1:
            if i == 0:
                if reverse_type is False:
                    new_matrix[i, new_matrix.shape[1] - 1] = 'Commission Error'
                    new_matrix[new_matrix.shape[0] - 1, i] = 'Omission Error'
                else:
                    new_matrix[i, new_matrix.shape[1] - 1] = 'Omission Error'
                    new_matrix[new_matrix.shape[0] - 1, i] = 'Commission Error'
            else:
                new_matrix[i, new_matrix.shape[1] - 1] = str((np.sum(self[i, 1:]) - self[i, i]) / np.sum(self[i, 1:]) * 100)[0:4] + '%'
                new_matrix[new_matrix.shape[0] - 1, i] = str((np.sum(self[1:, i]) - self[i, i]) / np.sum(self[1:, i]) * 100)[0:4] + '%'
            i += 1
        return new_matrix


def xlsx_save(self, output_path):
    temp_df = pandas.DataFrame(self)
    temp_df.to_excel(output_path, index=False)


def retrieve_srs(ds_temp):
    proj = osr.SpatialReference(wkt=ds_temp.GetProjection())
    srs_temp = proj.GetAttrValue('AUTHORITY', 1)
    srs_temp = 'EPSG:' + str(srs_temp)
    return srs_temp


def excel2water_level_array(excel_file_path, Year_range, shoal_name):
    excel_temp = pandas.read_excel(excel_file_path)
    excel_temp.sort_values(by=['Year', 'Month', 'Day'], ascending=True, inplace=True)
    start_year = min(Year_range)
    end_year = max(Year_range)
    start_num = -1
    end_num = -1
    for i in range(1, excel_temp.shape[0]):
        try:
            if excel_temp['Year'][i] == start_year and excel_temp['Year'][i - 1] != start_year:
                start_num = i
            elif excel_temp['Year'][i] == end_year and i == excel_temp.shape[0] - 1:
                end_num = i
            elif excel_temp['Year'][i] == end_year and excel_temp['Year'][i + 1] != end_year:
                end_num = i
        except:
            pass
    if start_num == -1 or end_num == -1:
        print('Please make sure the water level data from all required years is correctly input ')
        sys.exit(-1)
    else:
        out_temp = np.zeros([end_num - start_num + 1, 2])
        out_temp[:, 0] = excel_temp['Year'][start_num: end_num + 1].astype(int) * 10000 + excel_temp['Month'][start_num: end_num + 1].astype(int) * 100 + excel_temp['Day'][start_num: end_num + 1].astype(int)
        out_temp[:, 1] = excel_temp[shoal_name][start_num: end_num + 1].astype(float)
    return out_temp


def water_pixel_cor(dem_raster_array_f, water_pixel, water_ori=0, method='FourP'):
    if water_pixel.shape[1] != 2:
        print('Please make sure the water pixel coordinate is stored in a 2darray')
        sys.exit(-1)
    if method == 'FourP':
        water_list_temp = np.zeros([1, 2]).astype(np.int16)
        y_max = dem_raster_array_f.shape[0]
        x_max = dem_raster_array_f.shape[1]
        i_temp = water_ori
        while i_temp < water_pixel.shape[0]:
            water_list_temp_1 = np.zeros([1, 2]).astype(np.int16)
            x_centre_temp = water_pixel[i_temp, 1]
            y_centre_temp = water_pixel[i_temp, 0]
            for i_temp1 in range(1, y_max):
                if y_centre_temp - i_temp1 < 0:
                    break
                elif dem_raster_array_f[y_centre_temp - i_temp1, x_centre_temp] == 1:
                    water_list_temp_1 = np.append(water_list_temp_1, np.array([[y_centre_temp - i_temp1, x_centre_temp]]), axis=0)
                elif dem_raster_array_f[y_centre_temp - i_temp1, x_centre_temp] == 0 or np.isnan(dem_raster_array_f[y_centre_temp - i_temp1, x_centre_temp]):
                    break

            for i_temp1 in range(1, y_max):
                if y_centre_temp + i_temp1 > y_max - 1:
                    break
                elif dem_raster_array_f[y_centre_temp + i_temp1, x_centre_temp] == 1:
                    water_list_temp_1 = np.append(water_list_temp_1, np.array([[y_centre_temp + i_temp1, x_centre_temp]]), axis=0)
                elif dem_raster_array_f[y_centre_temp + i_temp1, x_centre_temp] == 0 or np.isnan(dem_raster_array_f[y_centre_temp + i_temp1, x_centre_temp]):
                    break

            for i_temp1 in range(1, x_max):
                if x_centre_temp - i_temp1 < 0:
                    break
                elif dem_raster_array_f[y_centre_temp, x_centre_temp - i_temp1] == 1:
                    water_list_temp_1 = np.append(water_list_temp_1, np.array([[y_centre_temp, x_centre_temp - i_temp1]]), axis=0)
                elif dem_raster_array_f[y_centre_temp, x_centre_temp - i_temp1] == 0 or np.isnan(dem_raster_array_f[y_centre_temp, x_centre_temp - i_temp1]):
                    break

            for i_temp1 in range(1, x_max):
                if x_centre_temp + i_temp1 > x_max - 1:
                    break
                elif dem_raster_array_f[y_centre_temp, x_centre_temp + i_temp1] == 1:
                    water_list_temp_1 = np.append(water_list_temp_1, np.array([[y_centre_temp, x_centre_temp + i_temp1]]), axis=0)
                elif dem_raster_array_f[y_centre_temp, x_centre_temp + i_temp1] == 0 or np.isnan(dem_raster_array_f[y_centre_temp, x_centre_temp + i_temp1]):
                    break

            water_list_temp_1 = np.delete(water_list_temp_1, 0, 0)
            if water_list_temp_1.shape[0] > 0:
                water_list_temp_1 = np.unique(water_list_temp_1, axis=0)
                for i_temp2 in range(water_list_temp_1.shape[0]):
                    dem_raster_array_f[water_list_temp_1[i_temp2, 0], water_list_temp_1[i_temp2, 1]] = False
                water_list_temp = np.append(water_list_temp, water_list_temp_1, axis=0)
            i_temp += 1

        water_list_temp = np.delete(water_list_temp, 0, 0)
        water_ori = water_pixel.shape[0]
        water_pixel = np.append(water_pixel, water_list_temp, axis=0)
        water_length_m = water_pixel.shape[0]
        if water_length_m == water_ori:
            sole_factor = True
        else:
            sole_factor = False
        return water_pixel, water_ori, sole_factor, dem_raster_array_f
    elif method == 'EightP':
        pass


def inundated_area_detection(dem_raster_array_f, water_pixel, water_level_indicator=None, water_ori=None, sole_factor=False, initial_factor=False):
    if water_level_indicator is None or water_pixel is None:
        print('Please input the the water level data or the coordinate of water original pixel！')
        sys.exit(-1)
    elif initial_factor is False:
        dem_raster_array_f[dem_raster_array_f < water_level_indicator] = True
        dem_raster_array_f[dem_raster_array_f >= water_level_indicator] = False
        water_pixel, water_ori, sole_factor, dem_raster_array_f = water_pixel_cor(dem_raster_array_f, water_pixel, water_ori=0)
        dem_raster_array_f[water_pixel[0, 0], water_pixel[0, 1]] = False
        initial_factor = True

    if sole_factor:
        return water_pixel
    else:
        water_pixel, water_ori, sole_factor, dem_raster_array_f = water_pixel_cor(dem_raster_array_f, water_pixel, water_ori=water_ori)
        water_pixel = inundated_area_detection(dem_raster_array_f, water_pixel, water_level_indicator=water_level_indicator, water_ori=water_ori, sole_factor=sole_factor, initial_factor=initial_factor)
        return water_pixel


def inundation_detection_surveyed_daily_water_level(dem_raster_array, water_level_array, veg_raster, year_factor=None):
    if dem_raster_array.shape[0] != veg_raster.shape[0] or dem_raster_array.shape[1] != veg_raster.shape[1]:
        print('please double check the dem and veg file consistency')
        sys.exit(-1)
    if year_factor is None:
        year_factor = 'year'
    elif year_factor != 'year' and year_factor != 'day':
        print('please check the year factor')
        sys.exit(-1)
    dem_raster_array_temp = copy.copy(dem_raster_array)
    i = 0
    while i < dem_raster_array_temp.shape[0] * dem_raster_array_temp.shape[1]:
        pos_temp = np.argwhere(dem_raster_array_temp == np.nanmin(dem_raster_array_temp))
        if veg_raster[pos_temp[0, 0], pos_temp[0, 1]] == 60:
            break
        dem_raster_array_temp[pos_temp[0, 0], pos_temp[0, 1]] = np.max(np.max(dem_raster_array_temp, axis=1), axis=0)
        i += 1

    inundation_detection_sample = copy.copy(dem_raster_array_temp)
    inundation_detection_sample[~np.isnan(inundation_detection_sample)] = 0
    inundation_detection_sample[np.isnan(inundation_detection_sample)] = -2
    inundation_detection_sample.astype(np.int8)
    inundation_height_sample = copy.copy(dem_raster_array_temp)
    inundation_height_sample[~np.isnan(inundation_height_sample)] = 0
    inundation_height_sample.astype(np.float16)
    inundation_date_array = np.zeros([0])
    inundation_detection_cube = np.zeros([inundation_detection_sample.shape[0], inundation_detection_sample.shape[1], 1]).astype(np.int8)
    inundation_height_cube = np.zeros([inundation_detection_sample.shape[0], inundation_detection_sample.shape[1], 1]).astype(np.float16)
    initial_factor = True

    if dem_raster_array_temp[pos_temp[0, 0], pos_temp[0, 1]] > max(water_level_array[:, 1]):
        print('There is no water in the area! Something error')
        sys.exit(-1)
    else:
        for i in range(water_level_array.shape[0]):
            if year_factor == 'year' or water_level_array[i, 0]//10000 == int(year_factor):
                print('Start processing the inundation file from ' + str(water_level_array[i, 0]))
                start_time = time.time()
                inundation_date_array = np.append(inundation_date_array, np.array([water_level_array[i, 0]]), axis=0)
                inundation_detection = copy.copy(inundation_detection_sample)
                inundation_height = copy.copy(inundation_height_sample)
                water_level_temp = water_level_array[i, 1]
                dem_raster_array_temp = copy.copy(dem_raster_array)
                water_pixel = inundated_area_detection(dem_raster_array_temp, np.array([[pos_temp[0, 0], pos_temp[0, 1]]]), water_level_indicator=water_level_temp, sole_factor=False)

                for i_temp in range(water_pixel.shape[0]):
                    inundation_detection[int(water_pixel[i_temp, 0]), int(water_pixel[i_temp, 1])] = 1
                    inundation_height[water_pixel[i_temp, 0], water_pixel[i_temp, 1]] = water_level_temp - dem_raster_array[water_pixel[i_temp, 0], water_pixel[i_temp, 1]]
                inundation_detection_cube = np.append(inundation_detection_cube, inundation_detection[:, :, np.newaxis], axis=2)
                inundation_height_cube = np.append(inundation_height_cube, inundation_height[:, :, np.newaxis], axis=2)
                if initial_factor is True:
                    inundation_detection_cube = np.delete(inundation_detection_cube, 0, 2)
                    inundation_height_cube = np.delete(inundation_height_cube, 0, 2)
                    initial_factor = False
                end_time = time.time()
                print('The file was correctly processed in ' + str(end_time - start_time) + 's! ' + str(water_level_array[i, 0]))
    return inundation_detection_cube, inundation_height_cube, inundation_date_array


def dataset2array(ds_temp, Band_factor=True):
    temp_band = ds_temp.GetRasterBand(1)
    temp_array = gdal_array.BandReadAsArray(temp_band).astype(np.float32)
    temp_array[temp_array == temp_band.GetNoDataValue()] = np.nan
    if Band_factor:
        temp_array = temp_array * 0.0000275 - 0.2
    return temp_array


def generate_dry_wet_ratio(path, inundated_value, nan_inundated_value):
    file_path = bf.file_filter(path, ['.TIF'])
    date_list = []
    doy_list = []
    for i in file_path:
        for length in range(len(i)):
            try:
                date_temp = int(i[length: length + 8])
                date_list.append(date_temp)
                break
            except:
                pass

            try:
                doy_temp = int(i[length: length + 7])
                doy_list.append(doy_temp)
                break
            except:
                pass
    if len(doy_list) == len(date_list) == len(file_path):
        doy_list = bf.date2doy(date_list)
    elif len(date_list) == len(file_path):
        doy_list = bf.date2doy(date_list)
    elif len(doy_list) == len(file_path):
        pass
    else:
        print('Consistency error occurred during data composition!')
        sys.exit(-1)
    date_list = bf.doy2date(doy_list)
    dry_ratio_list = []
    wet_ratio_list = []
    dry_wet_ratio_list = []
    for i in file_path:
        ds_temp = gdal.Open(i)
        array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
        array_temp[array_temp == inundated_value] = 1
        array_temp[array_temp == nan_inundated_value] = 2
        all_pixel = np.count_nonzero(array_temp == 1) + np.count_nonzero(array_temp == 2)
        if all_pixel != 0:
            dry_ratio_list.append(np.count_nonzero(array_temp == 2) / all_pixel)
            wet_ratio_list.append(np.count_nonzero(array_temp == 1) / all_pixel)
        else:
            dry_ratio_list.append(np.nan)
            wet_ratio_list.append(np.nan)

        if np.count_nonzero(array_temp == 1) != 0:
            dry_wet_ratio_list.append(np.count_nonzero(array_temp == 2) / np.count_nonzero(array_temp == 1))
        else:
            dry_wet_ratio_list.append(np.nan)
    dic_temp = {'file path': file_path, 'date': date_list, 'dry_ratio':dry_ratio_list, 'wet_ratio': wet_ratio_list, 'dry_wet_ratio': dry_wet_ratio_list}
    df_temp = pandas.DataFrame(dic_temp)
    df_temp.to_excel(path + 'dry_wet_ratio.xlsx')


def composition(re_doy_index, doy_list, file_list, nan_value, composition_strategy, composition_output_folder, itr, time_coverage, year, inundated_value, nan_inundated_value, dry_wet_ratio_threshold, Landsat_7_influence=True, metadata_file_path=None):
    if Landsat_7_influence and metadata_file_path is None:
        print('Please input the metadata file path!')
        sys.exit(-1)
    if not os.path.exists(composition_output_folder + 'composite_Year_' + str(year) + '_' + time_coverage + '_' + str(itr) + '.TIF'):
        if len(re_doy_index) >= 2:
            file_directory = {}
            for file_num in range(len(re_doy_index)):
                file_directory['temp_ds_' + str(file_num)] = gdal.Open(file_list[re_doy_index[file_num]])
                file_temp_array = file_directory['temp_ds_' + str(file_num)].GetRasterBand(1).ReadAsArray()
                if file_num == 0:
                    file_temp_cube = file_temp_array.reshape(file_temp_array.shape[0], file_temp_array.shape[1], 1)
                else:
                    file_temp_cube = np.concatenate((file_temp_cube, file_temp_array.reshape(file_temp_array.shape[0], file_temp_array.shape[1], 1)), axis=2)
            sequenced_doy_list = []
            for i in range(len(re_doy_index)):
                sequenced_doy_list.append(doy_list[re_doy_index[i]])
            sequenced_doy_array = np.array(sequenced_doy_list)
            if composition_strategy == 'dry_wet_ratio_sequenced':
                file_temp_temp_cube = copy.copy(file_temp_cube)
                file_temp_temp_cube[file_temp_temp_cube == inundated_value] = 1
                file_temp_temp_cube[file_temp_temp_cube == nan_inundated_value] = 2
                dry_wet_ratio_cube = np.count_nonzero(file_temp_temp_cube == 1, axis=(0, 1)) / (np.count_nonzero(file_temp_temp_cube == 2, axis=(0, 1)) + np.count_nonzero(file_temp_temp_cube == 1, axis=(0, 1)))
                i_t = 0
                while i_t < dry_wet_ratio_cube.shape[0]:
                    if dry_wet_ratio_cube[i_t] > dry_wet_ratio_threshold[1] or dry_wet_ratio_cube[i_t] < dry_wet_ratio_threshold[0]:
                        dry_wet_ratio_cube = np.delete(dry_wet_ratio_cube, i_t, axis=0)
                        file_temp_cube = np.delete(file_temp_cube, i_t, axis=2)
                        sequenced_doy_array = np.delete(sequenced_doy_array, i_t, axis=0)
                        i_t -= 1
                    i_t += 1
                file_temp_temp_cube = copy.copy(file_temp_cube)
                sequenced_doy_array_temp = copy.copy(sequenced_doy_array)
                if itr == 'dry':
                    for i in range(dry_wet_ratio_cube.shape[0]):
                        max_pos = np.argmax(dry_wet_ratio_cube)
                        file_temp_temp_cube[:, :, -1 * (i + 1)] = file_temp_cube[:, :, max_pos]
                        sequenced_doy_array_temp[-1 * (i + 1)] = sequenced_doy_array_temp[max_pos]
                        dry_wet_ratio_cube[max_pos] = -1
                elif itr == 'wet':
                    for i in range(dry_wet_ratio_cube.shape[0]):
                        max_pos = np.argmax(dry_wet_ratio_cube)
                        file_temp_temp_cube[:, :, i] = file_temp_cube[:, :, max_pos]
                        sequenced_doy_array_temp[i] = sequenced_doy_array_temp[max_pos]
                        dry_wet_ratio_cube[max_pos] = -1
                file_temp_cube = copy.copy(file_temp_temp_cube)
                sequenced_doy_array = copy.copy(sequenced_doy_array_temp)

            if Landsat_7_influence:
                metadata = pd.read_excel(metadata_file_path, engine='openpyxl')
                metadata_array = np.array([metadata['Date'], metadata['Data_Type']])
                i = 0
                Landsat_7_cube = None
                while i < sequenced_doy_array.shape[0]:
                    pos_temp = np.argwhere(metadata_array[0, :] ==  bf.doy2date(sequenced_doy_array[i]))
                    pos_temp = pos_temp.flatten()[0]
                    if metadata_array[1, pos_temp] == 'LE07_L2SP':
                        if Landsat_7_cube is None:
                            Landsat_7_cube = file_temp_cube[:, :, i].reshape([file_temp_cube.shape[0], file_temp_cube.shape[1], 1])
                        else:
                            Landsat_7_cube = np.concatenate((Landsat_7_cube, file_temp_cube[:, :, i].reshape([file_temp_cube.shape[0], file_temp_cube.shape[1], 1])), axis=2)
                        file_temp_cube = np.delete(file_temp_cube, i, axis=2)
                        sequenced_doy_array = np.delete(sequenced_doy_array, i, axis=0)
                        i -= 1
                    i += 1
                if Landsat_7_cube is not None:
                    file_temp_cube = np.concatenate((file_temp_cube, Landsat_7_cube), axis=2)

            if file_temp_cube != []:
                file_output = np.ones([file_temp_cube.shape[0], file_temp_cube.shape[1]]) * nan_value
                ori_type = file_temp_cube.dtype
                file_temp_cube = file_temp_cube.astype(np.float)
                file_temp_cube[file_temp_cube == nan_value] = np.nan
                if composition_strategy == 'first':
                    for y in range(file_temp_cube.shape[0]):
                        for x in range(file_temp_cube.shape[1]):
                            temp_set = file_temp_cube[y, x, :].flatten()
                            temp = nan_value
                            if composition_strategy == 'first':
                                for set_index in range(temp_set.shape[0]):
                                    if temp_set[set_index] != nan_value:
                                        temp = temp_set[set_index]
                                        break
                            file_output[y, x] = temp
                elif composition_strategy == 'last':
                    for y in range(file_temp_cube.shape[0]):
                        for x in range(file_temp_cube.shape[1]):
                            temp_set = file_temp_cube[y, x, :].flatten()
                            temp = nan_value
                            set_index = temp_set.shape[0] - 1
                            while set_index >= 0:
                                if temp_set[set_index] != nan_value:
                                    temp = temp_set[set_index]
                                    break
                                set_index -= 1
                            file_output[y, x] = temp
                elif composition_strategy == 'mean':
                    file_output = np.nanmean(file_temp_cube, axis=2)
                elif composition_strategy == 'dry_wet_ratio_sequenced':
                    for y in range(file_temp_cube.shape[0]):
                        for x in range(file_temp_cube.shape[1]):
                            temp_set = file_temp_cube[y, x, :].flatten()
                            temp = nan_value
                            for set_index in range(temp_set.shape[0]):
                                if temp_set[set_index] != nan_value:
                                    temp = temp_set[set_index]
                                    break
                            file_output[y, x] = temp
                elif composition_strategy == 'max':
                    file_output = np.nanmax(file_temp_cube, axis=2)
                elif composition_strategy == 'min':
                    file_output = np.nanmin(file_temp_cube, axis=2)
                file_output = file_output.astype(ori_type)
                bf.write_raster(file_directory['temp_ds_0'], file_output, composition_output_folder, 'composite_Year_' + str(year) + '_' + time_coverage + '_' + str(itr) + '.TIF', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)
        elif len(re_doy_index) == 1:
            file_ds = gdal.Open(file_list[re_doy_index[0]])
            file_array = file_ds.GetRasterBand(1).ReadAsArray()
            file_array[file_array == inundated_value] = 1
            file_array[file_array == nan_inundated_value] = 2
            dry_wet_ratio = np.count_nonzero(file_array == 1, axis=(0, 1)) / (np.count_nonzero(file_array == 2, axis=(0, 1)) + np.count_nonzero(file_array == 1, axis=(0, 1)))
            if 0.5 <= dry_wet_ratio <= 0.95:
                shutil.copyfile(file_list[re_doy_index[0]], composition_output_folder + 'composite_Year_' + str(year) + '_' + time_coverage + '_' + str(itr) + '.TIF')


def data_composition(file_path, metadata_path, time_coverage=None, composition_strategy=None, file_format=None, nan_value=-32768, user_defined_monsoon=None, inundated_indicator=None, dry_wet_ratio_threshold=None, Landsat7_influence=False):
    # Determine the time range
    all_time_coverage = ['month', 'year', 'monsoon']
    if time_coverage is None:
        time_coverage = 'month'
    elif time_coverage not in all_time_coverage:
        print('Please choose a supported time coverage')
        sys.exit(-1)

    # Determine the inundated indicator
    if inundated_indicator is None:
        inundated_value = 1
        nan_inundated_value = 0
    elif type(inundated_indicator) != list or len(inundated_indicator) != 2:
        print('Please double check the datatype of the inundated_indicator!')
        sys.exit(-1)
    else:
        inundated_value = inundated_indicator[0]
        nan_inundated_value = inundated_indicator[1]

    # Determine the monsoon
    # doy_list_beg = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    # doy_list_end = [31, 59, 90, 120, 151, 181, 212, 243, 273, ]
    if time_coverage != 'monsoon':
        if user_defined_monsoon is not None:
            print('The time coverage is not monsoon')
    else:
        if user_defined_monsoon is None:
            monsoon_beg = 6
            monsoon_end = 10
        elif type(user_defined_monsoon) != list or len(user_defined_monsoon) != 2 or (type(user_defined_monsoon[0]) != float and type(user_defined_monsoon[0]) != int) or (type(user_defined_monsoon[1]) != float and type(user_defined_monsoon[1]) != int):
            print('Please double check the datatype of the user defined monsoon!')
            sys.exit(-1)
        elif user_defined_monsoon[0] < 1 or user_defined_monsoon[0] > 12 or user_defined_monsoon[1] < 1 or user_defined_monsoon[1] > 12 or user_defined_monsoon[0] > user_defined_monsoon[1]:
            monsoon_beg = 6
            monsoon_end = 10
        else:
            monsoon_beg = int(user_defined_monsoon[0])
            monsoon_end = int(user_defined_monsoon[1])

    # Determine the composition strategy
    all_supported_composition_strategy = ['first', 'last', 'mean', 'dry_wet_ratio_sequenced', 'max']
    if composition_strategy is None:
        composition_strategy = 'first'
    elif composition_strategy not in all_supported_composition_strategy:
        print('Please choose a supported composition strategy!')
        sys.exit(-1)
    elif composition_strategy == 'dry_wet_ratio_sequenced' and time_coverage != 'monsoon':
        composition_strategy = 'first'
        if dry_wet_ratio_threshold is None:
            dry_wet_ratio_threshold = 0.95
    elif composition_strategy != 'dry_wet_ratio_sequenced' and time_coverage == 'monsoon':
        composition_strategy = 'dry_wet_ratio_sequenced'
        if dry_wet_ratio_threshold is None:
            dry_wet_ratio_threshold = 0.95

    # Determine the file format
    if file_format is None:
        file_format = '.TIF'
    # Determine the file list
    file_list = bf.file_filter(file_path, [file_format])
    if len(file_list) == 0:
        print('Please input the correct file format!')
        sys.exit(-1)

    # Create the composition output folder
    bf.create_folder(file_path + 'composition_output\\')
    composition_output_folder = file_path + 'composition_output\\' + time_coverage + '\\'
    bf.create_folder(composition_output_folder)
    date_list = []
    doy_list = []
    for i in file_list:
        for length in range(len(i)):
            try:
                date_temp = int(i[length: length + 8])
                date_list.append(date_temp)
                break
            except:
                pass
            try:
                doy_temp = int(i[length: length + 7])
                doy_list.append(doy_temp)
                break
            except:
                pass
    if len(doy_list) == len(date_list) == len(file_list):
        doy_list = bf.date2doy(date_list)
    elif len(date_list) == len(file_list):
        doy_list = bf.date2doy(date_list)
    elif len(doy_list) != len(file_list):
        print('Consistency error occurred during data composition!')
        sys.exit(-1)

    doy_list = np.sort(np.array(doy_list))
    doy_list = doy_list.tolist()
    # Create composite band
    year_list = np.unique(np.array(doy_list) // 1000)
    for year in year_list:
        if time_coverage == 'month':
            for month in range(1, 13):
                re_doy_index = []
                for doy_index in range(len(doy_list)):
                    if doy_list[doy_index] // 1000 == year and datetime.date.fromordinal(datetime.date(int(year), 1, 1).toordinal() + np.mod(doy_list[doy_index], 1000) - 1).month == month:
                        re_doy_index.append(doy_index)
                composition(re_doy_index, doy_list, file_list, nan_value, composition_strategy, composition_output_folder, month, time_coverage, year, inundated_value, nan_inundated_value, dry_wet_ratio_threshold=dry_wet_ratio_threshold, metadata_file_path=metadata_path, Landsat_7_influence=Landsat7_influence)
        elif time_coverage == 'year':
            re_doy_index = []
            for doy_index in range(len(doy_list)):
                if doy_list[doy_index] // 1000 == year:
                    re_doy_index.append(doy_index)
            composition(re_doy_index, doy_list, file_list, nan_value, composition_strategy, composition_output_folder, year, time_coverage, year, inundated_value, nan_inundated_value, dry_wet_ratio_threshold=dry_wet_ratio_threshold, metadata_file_path=metadata_path, Landsat_7_influence=Landsat7_influence)
        elif time_coverage == 'monsoon':
            for i in ['wet', 'dry']:
                re_doy_index = []
                if i == 'wet':
                    for doy_index in range(len(doy_list)):
                        if doy_list[doy_index] // 1000 == year and monsoon_beg <= datetime.date.fromordinal(datetime.date(int(year), 1, 1).toordinal() + np.mod(doy_list[doy_index], 1000) - 1).month <= monsoon_end:
                            re_doy_index.append(doy_index)
                elif i == 'dry':
                    for doy_index in range(len(doy_list)):
                        if doy_list[doy_index] // 1000 == year and (monsoon_beg > datetime.date.fromordinal(datetime.date(int(year), 1, 1).toordinal() + np.mod(doy_list[doy_index], 1000) - 1).month or monsoon_end < datetime.date.fromordinal(datetime.date(int(year), 1, 1).toordinal() + np.mod(doy_list[doy_index], 1000) - 1).month):
                            re_doy_index.append(doy_index)
                composition(re_doy_index, doy_list, file_list, nan_value, composition_strategy, composition_output_folder, i, time_coverage, year, inundated_value, nan_inundated_value, dry_wet_ratio_threshold=dry_wet_ratio_threshold, metadata_file_path=metadata_path, Landsat_7_influence=Landsat7_influence)
        else:
            print('Unknown error occurred!')
            sys.exit(-1)
            pass


def surrounding_max_half_window(array, cor, water_pixel_v=1):
    for i_tt in range(1, min(array.shape[0], array.shape[1])):
        if cor[0] - i_tt < 0:
            y_min = 0
        else:
            y_min = cor[0] - i_tt

        if cor[1] - i_tt < 0:
            x_min = 0
        else:
            x_min = cor[1] - i_tt

        if cor[0] + i_tt > array.shape[0]:
            y_max = array.shape[0] + 1
        else:
            y_max = cor[0] + i_tt + 1

        if cor[1] + i_tt > array.shape[1]:
            x_max = array.shape[1] + 1
        else:
            x_max = cor[1] + i_tt + 1

        array_temp = array[y_min: y_max, x_min: x_max]
        array_temp[array_temp != water_pixel_v] = 0
        array_temp[array_temp == water_pixel_v] = 1
        if np.sum(array_temp) != (y_max - y_min) * (x_max - x_min):
            break
    return i_tt - 1


def surrounding_pixel_cor(water_pixel_under_invest, water_center_pixel_list, surrounding_nan_water_pixel_list, array, x_max, y_max, detection_method='EightP', water_pixel_value=1):
    if len(water_pixel_under_invest[0]) != 2:
        print('Please input the pixel x-y coordinate as a 2-row list')
        sys.exit(-1)
    else:
        surrounding_water_pixel_list_t = []
        water_center_pixel_list_t = []
        surrounding_nan_water_pixel_list_t = []
        water_center_pixel_list_t_2 = []
        for i in range(len(water_pixel_under_invest)):

            for i_tt in range(1, min(array.shape[0], array.shape[1])):
                if water_pixel_under_invest[i][0] - i_tt < 0:
                    y_min = 0
                else:
                    y_min = water_pixel_under_invest[i][0] - i_tt

                if water_pixel_under_invest[i][1] - i_tt < 0:
                    x_min = 0
                else:
                    x_min = water_pixel_under_invest[i][1] - i_tt

                if water_pixel_under_invest[i][0] + i_tt >= array.shape[0]:
                    y_max = array.shape[0] - 1
                else:
                    y_max = water_pixel_under_invest[i][0] + i_tt + 1

                if water_pixel_under_invest[i][1] + i_tt >= array.shape[1]:
                    x_max = array.shape[1] - 1
                else:
                    x_max = water_pixel_under_invest[i][1] + i_tt + 1

                array_temp = array[y_min: y_max, x_min: x_max]
                if np.sum(array_temp[array_temp == water_pixel_value]) != (y_max - y_min) * (x_max - x_min):
                    break

            surrounding_pixel_list = np.zeros([y_max - y_min, x_max - x_min, 2], dtype=np.int64)
            for i_t_y in range(y_max - y_min):
                surrounding_pixel_list[i_t_y, :, 0] = y_min + i_t_y
            for i_t_x in range(x_max - x_min):
                surrounding_pixel_list[:, i_t_x, 1] = x_min + i_t_x

            if detection_method == 'EightP':
                if i_tt > 1:
                    water_center_list = copy.copy(surrounding_pixel_list[1: -1, 1: -1, :])
                    water_center_list = np.reshape(water_center_list, ((y_max - y_min - 2) * (x_max - x_min - 2), 2))
                    water_center_list = water_center_list.tolist()
                    water_center_pixel_list_t.extend(water_center_list)
                    # for i_tttt in range(len(water_center_list)):
                    #     water_pixel_temp = water_center_list[i_tttt]
                    #     s2_b = time.time()
                    #     if water_pixel_temp not in water_center_pixel_list_t_2:
                    #         s3_b = time.time()
                    #         water_center_pixel_list_t_2.append(water_pixel_temp)
                    #         s3_e = time.time()
                    #         phase3_time += s3_e - s3_b
                surrounding_all_list = surrounding_pixel_list[0, :, :].tolist()
                surrounding_all_list.extend(surrounding_pixel_list[-1, :, :].tolist())
                surrounding_all_list.extend(surrounding_pixel_list[:, 0, :].tolist())
                surrounding_all_list.extend(surrounding_pixel_list[:, -1, :].tolist())
                for i_ttttt in range(len(surrounding_all_list)):
                    s_temp = surrounding_all_list[i_ttttt]
                    if array[s_temp[0], s_temp[1]] == water_pixel_value:
                        surrounding_water_pixel_list_t.append(s_temp)
                    else:
                        surrounding_nan_water_pixel_list_t.append(s_temp)
            elif detection_method == 'FourP':
                pass
        water_center_pixel_list.extend(water_center_pixel_list_t)
        water_center_pixel_list = np.unique(np.array(water_center_pixel_list), axis=0).tolist()
        if len(surrounding_water_pixel_list_t) != 0:
            surrounding_water_pixel_list_t = np.unique(np.array(surrounding_water_pixel_list_t), axis=0).tolist()
        surrounding_water_pixel_list = [i for i in surrounding_water_pixel_list_t if i not in water_center_pixel_list]
        water_center_pixel_list.extend(surrounding_water_pixel_list)
        surrounding_nan_water_pixel_list.extend(surrounding_nan_water_pixel_list_t)
        try:
            surrounding_nan_water_pixel_list = np.unique(np.array(surrounding_nan_water_pixel_list), axis=0).tolist()
        except:
            surrounding_nan_water_pixel_list = []
    return surrounding_water_pixel_list, surrounding_nan_water_pixel_list, water_center_pixel_list


def detect_sole_inundated_area(array, water_pixel_list, around_water_pixel_list, around_nanwater_pixel_list, water_pixel_value=1, conditional_factor=False, detection_method=None, nodata_value=None):
    if nodata_value is None:
        print('Please input the nodata value!')
        sys.exit(-1)

    if detection_method is None:
        print('Please input the correct detection method EightP or FourP!')
        sys.exit(-1)

    if water_pixel_list is None:
        print('Please input the original water pixel')
        sys.exit(-1)
    elif type(water_pixel_list) != list:
        print('Please input the water pixel in a list type!')
        sys.exit(-1)

    if around_water_pixel_list is None or around_nanwater_pixel_list is None:
        around_nanwater_pixel_list = []
        around_water_pixel_list = water_pixel_list

    if conditional_factor is False:
        y_max = array.shape[0]
        x_max = array.shape[1]
        around_water_pixel_list, around_nanwater_pixel_list, water_pixel_list = surrounding_pixel_cor(around_water_pixel_list, water_pixel_list, around_nanwater_pixel_list, array, x_max, y_max, detection_method=detection_method,water_pixel_value=water_pixel_value)
        conditional_factor = len(around_water_pixel_list) == 0
        array_sole_area = detect_sole_inundated_area(array, water_pixel_list, around_water_pixel_list, around_nanwater_pixel_list, conditional_factor=conditional_factor,
                                                     water_pixel_value=water_pixel_value,
                                                     detection_method=detection_method, nodata_value=nodata_value)
        return array_sole_area
    else:
        array_sole_area = copy.copy(array)
        array_sole_area[array_sole_area != nodata_value] = nodata_value
        for i_t in range(len(water_pixel_list)):
            if array[int(water_pixel_list[i_t][0]), int(water_pixel_list[i_t][1])] == nodata_value:
                print('Code error')
                sys.exit(-1)
            else:
                array_sole_area[int(water_pixel_list[i_t][0]), int(water_pixel_list[i_t][1])] = 1
        for i_t in range(len(around_nanwater_pixel_list)):
            if array[int(around_nanwater_pixel_list[i_t][0]), int(around_nanwater_pixel_list[i_t][1])] == water_pixel_value:
                print('Code error')
                sys.exit(-1)
            elif array[int(around_nanwater_pixel_list[i_t][0]), int(around_nanwater_pixel_list[i_t][1])] != nodata_value:
                array_sole_area[int(around_nanwater_pixel_list[i_t][0]), int(around_nanwater_pixel_list[i_t][1])] = 2
        return array_sole_area


def DEM_fix_sole(DEM_array_f, sole_temp_array_f, inundated_temp_array_f, around_pixel_list=None, water_pixel_list=None, indicator_list=None, overall_factor=False, nan_water_value=None, sole_max=None, initial_factor=True):
    y_max = sole_temp_array_f.shape[0]
    x_max = sole_temp_array_f.shape[1]
    if sole_max is None:
        sole_max = mostCommon(sole_temp_array_f, inundated_temp_array_f)
    if nan_water_value is None:
        nan_water_value = 255
    if indicator_list is None:
        print('Please double check the indicator within the DEM_fix_sole Function!')
        sys.exit(-1)
    if type(indicator_list) == int:
        indicator_list = [indicator_list]
    if water_pixel_list is None and around_pixel_list is None:
        water_pixel_list = np.argwhere(sole_temp_array_f == indicator_list[0])
        for i_tt in range(water_pixel_list.shape[0]):
            surround_pixel, water_pixel_non = surrounding_pixel_cor(water_pixel_list[i_tt], x_max, y_max, window_size=0)
            if i_tt == 0:
                around_pixel_list = surround_pixel
            else:
                around_pixel_list = np.append(around_pixel_list, surround_pixel, axis=0)
        around_pixel_list = np.unique(around_pixel_list, axis=0)

        nan_factor = False
        for i_temp in range(around_pixel_list.shape[0]):
            if around_pixel_list[i_temp].tolist() not in water_pixel_list.tolist() and nan_factor is False:
                around_p_temp = np.array([around_pixel_list[i_temp]])
                nan_factor = True
            elif around_pixel_list[i_temp].tolist() not in water_pixel_list.tolist() and nan_factor is True:
                around_p_temp = np.append(around_p_temp, np.array([around_pixel_list[i_temp]]), axis=0)
        around_pixel_list = around_p_temp

    if overall_factor is False:
        near_water_factor = False
        dem_factor = False
        another_indicator_exist = False
        dem_around_min = 1000
        dem_water_max = -1
        for i_temp in range(around_pixel_list.shape[0]):
            if initial_factor is True and inundated_temp_array_f[around_pixel_list[i_temp][0], around_pixel_list[i_temp][1]] == 1:
                near_water_factor = True
            elif initial_factor is False and inundated_temp_array_f[around_pixel_list[i_temp][0], around_pixel_list[i_temp][1]] == 1 and sole_temp_array_f[around_pixel_list[i_temp][0], around_pixel_list[i_temp][1]] == sole_max:
                near_water_factor = True
            elif inundated_temp_array_f[around_pixel_list[i_temp][0], around_pixel_list[i_temp][1]] == nan_water_value:
                dem_around_min = min(dem_around_min, DEM_array_f[around_pixel_list[i_temp][0], around_pixel_list[i_temp][1]])
            elif inundated_temp_array_f[around_pixel_list[i_temp][0], around_pixel_list[i_temp][1]] == 2:
                another_indicator_exist = True
        initial_factor = False

        for i_temp_2 in range(water_pixel_list.shape[0]):
            dem_water_max = max(dem_water_max, DEM_array_f[water_pixel_list[i_temp_2][0], water_pixel_list[i_temp_2][1]])
        if dem_water_max <= dem_around_min + 1:
            dem_factor = True
        # Recursion
        if near_water_factor is True and another_indicator_exist is False:
            overall_factor = True

        # elif dem_factor is False:
        #     i_temp_3 = 0
        #     i_temp_31 = 0
        #     while i_temp_3 < around_pixel_list.shape[0]:
        #         if DEM_array_f[around_pixel_list[i_temp_3][0], around_pixel_list[i_temp_3][1]] <= dem_water_max and inundated_temp_array_f[around_pixel_list[i_temp_3][0], around_pixel_list[i_temp_3][1]] == nan_water_value:
        #             water_pixel_list = np.append(water_pixel_list, np.array([around_pixel_list[i_temp_3]]), axis=0)
        #             around_pixel_list = np.delete(around_pixel_list, i_temp_3, 0)
        #             i_temp_31 += 1
        #             i_temp_3 -= 1
        #         i_temp_3 += 1
        #
        #     i_temp_32 = 0
        #     while i_temp_32 < i_temp_31:
        #         i_temp_32 += 1
        #         surround_pixel, water_pixel_non = surrounding_pixel_cor(water_pixel_list[-i_temp_32], x_max, y_max, window_size=0)
        #         around_pixel_list = np.append(around_pixel_list, surround_pixel, axis=0)
        #     around_pixel_list = np.unique(around_pixel_list, axis=0)
        #
        #     i_temp_6 = 0
        #     while i_temp_6 < around_pixel_list.shape[0]:
        #         if around_pixel_list[i_temp_6].tolist() in water_pixel_list.tolist():
        #             around_pixel_list = np.delete(around_pixel_list, i_temp_6, 0)
        #             i_temp_6 -= 1
        #         i_temp_6 += 1

        elif another_indicator_exist is True:
            for i_temp_4 in range(around_pixel_list.shape[0]):
                if inundated_temp_array_f[around_pixel_list[i_temp_4][0], around_pixel_list[i_temp_4][1]] == 2:
                    if int(sole_temp_array_f[around_pixel_list[i_temp_4][0], around_pixel_list[i_temp_4][1]]) not in indicator_list:
                        indicator_list.append(int(sole_temp_array_f[around_pixel_list[i_temp_4][0], around_pixel_list[i_temp_4][1]]))
                        water_pixel_list_t = np.argwhere(sole_temp_array_f == int(sole_temp_array_f[around_pixel_list[i_temp_4][0], around_pixel_list[i_temp_4][1]]))
                        water_pixel_list = np.append(water_pixel_list, water_pixel_list_t, axis=0)
                        for i_temp_5 in range(water_pixel_list_t.shape[0]):
                            surround_pixel, water_pixel_non = surrounding_pixel_cor(water_pixel_list_t[i_temp_5], x_max, y_max, window_size=0)
                            around_pixel_list = np.append(around_pixel_list, surround_pixel, axis=0)
                        around_pixel_list = np.unique(around_pixel_list, axis=0)
            i_temp_6 = 0
            while i_temp_6 < around_pixel_list.shape[0]:
                if around_pixel_list[i_temp_6].tolist() in water_pixel_list.tolist():
                    around_pixel_list = np.delete(around_pixel_list, i_temp_6, 0)
                    i_temp_6 -= 1
                i_temp_6 += 1

        elif near_water_factor is False:
            for i_temp_7 in range(around_pixel_list.shape[0]):
                if DEM_array_f[around_pixel_list[i_temp_7][0], around_pixel_list[i_temp_7][1]] == dem_around_min:
                    water_pixel_list = np.append(water_pixel_list, np.array([around_pixel_list[i_temp_7]]), axis=0)
                    surround_pixel, water_pixel_non = surrounding_pixel_cor(around_pixel_list[i_temp_7], x_max, y_max, window_size=0)
                    around_pixel_list = np.append(around_pixel_list, surround_pixel, axis=0)
            around_pixel_list = np.unique(around_pixel_list, axis=0)

            i_temp_8 = 0
            while i_temp_8 < around_pixel_list.shape[0]:
                if around_pixel_list[i_temp_8].tolist() in water_pixel_list.tolist():
                    around_pixel_list = np.delete(around_pixel_list, i_temp_8, 0)
                    i_temp_8 -= 1
                i_temp_8 += 1
        indicator_list, water_pixel_list, around_pixel_list = DEM_fix_sole(DEM_array_f, sole_temp_array_f, inundated_temp_array_f, around_pixel_list=around_pixel_list, water_pixel_list=water_pixel_list, indicator_list=indicator_list, overall_factor=overall_factor, nan_water_value=None, initial_factor=initial_factor)
    elif overall_factor is True:
        return indicator_list, water_pixel_list, around_pixel_list
    return indicator_list, water_pixel_list, around_pixel_list


def complement_all_inundated_area(DEM_array_f, sole_temp_array_f, inundated_temp_array_f):
    sole_temp_list = list(range(sole_temp_array_f.max()))
    inundated_temp_array_ff = copy.copy(inundated_temp_array_f)
    for i in range(sole_temp_array_f.max()):
        indi_temp = i + 1
        if indi_temp in sole_temp_list:
            i_list = np.argwhere(sole_temp_array_f == indi_temp)
            inundated_indi_f = inundated_temp_array_f[i_list[0, 0], i_list[0, 1]]
            if inundated_indi_f == 2:
                indicator_list, water_pixel_list, around_pixel_nan = DEM_fix_sole(DEM_array_f, sole_temp_array_f, inundated_temp_array_f, indicator_list=indi_temp)
                for i_temp in indicator_list:
                    sole_temp_list[:] = [ele for ele in sole_temp_list if ele != i_temp]
                for i_temp1 in range(water_pixel_list.shape[0]):
                    if inundated_temp_array_ff[water_pixel_list[i_temp1, 0], water_pixel_list[i_temp1, 1]] == 255:
                        inundated_temp_array_ff[water_pixel_list[i_temp1, 0], water_pixel_list[i_temp1, 1]] = 3
                print('Successfully complement the ' + str(indi_temp) + 'inundated area!')
            elif inundated_indi_f == 255:
                print('Some inconsistency error occurred!')
                sys.exit(-1)
    return inundated_temp_array_ff


def identify_all_inundated_area(inundated_array, inundated_pixel_indicator=None, nanvalue_pixel_indicator=None, surrounding_pixel_identification_factor=False, input_detection_method=None):
    value_list = np.unique(inundated_array)

    if input_detection_method is None:
        input_detection_method = 'EightP'
    elif input_detection_method != 'EightP' and input_detection_method != 'FourP':
        print('Please mention current inundated area detection method only consist EightP and FourP!')
        sys.exit(-1)

    if inundated_pixel_indicator is None:
        inundated_pixel_indicator = 1
    if nanvalue_pixel_indicator is None:
        nanvalue_pixel_indicator = 255
    if inundated_pixel_indicator not in value_list.tolist() or nanvalue_pixel_indicator not in value_list.tolist():
        print('Please double check the inundated indicator set in the inundated area mapping!')

    inundated_ori = copy.copy(inundated_array)
    inundated_sole_water_map = np.zeros([inundated_array.shape[0], inundated_array.shape[1]], dtype=np.int32)
    inundated_identified_f = copy.copy(inundated_array)
    indicator = 1
    for y in range(inundated_array.shape[0]):
        for x in range(inundated_array.shape[1]):
            if inundated_identified_f[y, x] == inundated_pixel_indicator:
                start_time = time.time()
                array_sole_area = detect_sole_inundated_area(inundated_ori, [[y, x]], None, None, water_pixel_value=inundated_pixel_indicator, detection_method=input_detection_method, nodata_value=nanvalue_pixel_indicator)
                inundated_sole_water_map[array_sole_area == 1] = indicator
                if surrounding_pixel_identification_factor:
                    inundated_sole_water_map[np.logical_and(array_sole_area == 2, inundated_sole_water_map != 0)] += indicator * -10000
                    inundated_sole_water_map[np.logical_and(array_sole_area == 2, inundated_sole_water_map == 0)] = indicator * -1
                inundated_identified_f[array_sole_area == 1] = nanvalue_pixel_indicator
                end_time = time.time()
                print(str(indicator) + ' finished in' + str(end_time - start_time) + 's')
                indicator += 1
    inundated_sole_water_map[inundated_array == nanvalue_pixel_indicator] = 0
    return inundated_sole_water_map


def remove_all_file_and_folder(filter_list):
    for file in filter_list:
        if os.path.isdir(str(file)):
            try:
                shutil.rmtree(file)
            except:
                print('folder cannot be removed')
        elif os.path.isfile(str(file)):
            try:
                os.remove(file)
            except:
                print('file cannot be removed')
        else:
            print('Something went wrong during the file removal')


def check_vi_file_consistency(l2a_output_path_f, VI_list_f):
    vi_file = []
    c_word = ['.tif']
    r_word = ['.ovr']
    for vi in VI_list_f:
        if not os.path.exists(l2a_output_path_f + vi):
            print(vi + 'folders are missing')
            sys.exit(-1)
        else:
            redundant_file_list = bf.file_filter(l2a_output_path_f + vi + '\\', r_word)
            remove_all_file_and_folder(redundant_file_list)
            tif_file_list = bf.file_filter(l2a_output_path_f + vi + '\\', c_word)
            vi_temp = []
            for tif_file in tif_file_list:
                vi_temp.append(tif_file[tif_file.find('\\20') + 2:tif_file.find('\\20') + 15])
            vi_file.append(vi_temp)
    for i in range(len(vi_file)):
        if not collections.Counter(vi_file[0]) == collections.Counter(vi_file[i]):
            print('VIs did not share the same file numbers')
            sys.exit(-1)


def eliminating_all_not_required_file(file_path_f, filename_extension=None):
    if filename_extension is None:
        filename_extension = ['txt', 'tif', 'TIF', 'json', 'jpeg', 'xml']
    filter_name = ['.']
    tif_file_list = bf.file_filter(file_path_f, filter_name)
    for file in tif_file_list:
        if file.split('.')[-1] not in filename_extension:
            try:
                os.remove(file)
            except:
                raise Exception(f'file {file} cannot be removed')

        if str(file[-8:]) == '.aux.xml':
            try:
                os.remove(file)
            except:
                raise Exception(f'file {file} cannot be removed')


def remove_same_element_from_lists(list1, list2):
    list2_temp = copy.copy(list2)
    list1_temp = []
    for element in list1:
        if element not in list2_temp:
            list1_temp.append(element)
        elif element in list2_temp:
            list2_temp.remove(element)
    return list1_temp, list2_temp


def print_missed_file(list1, list2, filter_name1, filter_name2):
    if len(list1) != 0:
        for i in list1:
            print('The ' + i + ' of ' + filter_name2 + ' is missing!')
    if len(list2) != 0:
        for i in list2:
            print('The ' + i + ' of ' + filter_name1 + ' is missing!')


def file_consistency_check(filepath_list_f, filename_filter_list, files_in_same_folder=True, specific_beg='L2SP', specific_end='02_T'):
    file_consistency_factor = True
    if files_in_same_folder:
        if len(filepath_list_f) == 1:
            if len(filename_filter_list) > 1:
                file_consistency_temp_dic = {}
                i = 0
                for filename_filter in filename_filter_list:
                    file_path_temp = bf.file_filter(filepath_list_f[0], [filename_filter])
                    file_information_temp = []
                    for file_path in file_path_temp:
                        file_information_temp.append(file_path[file_path.find(specific_beg): file_path.find(specific_end) - 1])
                    file_consistency_temp_dic['filtered_' + filename_filter + '_file_information'] = file_information_temp
                    if i == 0:
                        file_num = len(file_consistency_temp_dic['filtered_' + filename_filter + '_file_information'])
                        filename_ori = filename_filter
                    else:
                        if len(file_consistency_temp_dic['filtered_' + filename_filter + '_file_information']) > file_num:
                            list_temp_1, list_temp_2 = remove_same_element_from_lists(file_consistency_temp_dic['filtered_' + filename_filter + '_file_information'], file_consistency_temp_dic['filtered_' + filename_ori + '_file_information'])
                            print_missed_file(list_temp_1, list_temp_2, filename_filter, filename_ori)
                            file_consistency_factor = False
                            filename_ori = filename_filter
                            file_num = len(file_consistency_temp_dic['filtered_' + filename_filter + '_file_information'])
                        elif len(file_consistency_temp_dic['filtered_' + filename_filter + '_file_information']) < file_num:
                            list_temp_1, list_temp_2 = remove_same_element_from_lists(file_consistency_temp_dic['filtered_' + filename_filter + '_file_information'], file_consistency_temp_dic['filtered_' + filename_ori + '_file_information'])
                            print_missed_file(list_temp_1, list_temp_2, filename_filter, filename_ori)
                            file_consistency_factor = False
                    i += 1
            else:
                print('''Caution! If you want to check files' consistency, please make sure there is more than one filter name in the list''')
        else:
            print('''Caution! If you want to check files' consistency in the same folder, please make sure there is only one file path in the list''')

    elif not files_in_same_folder:
        if len(filepath_list_f) == len(filename_filter_list) and len(filename_filter_list) > 1:
            i = 0
            file_consistency_temp_dic = {}
            while i < len(filepath_list_f):
                file_consistency_temp_dic[filename_filter_list[i]] = bf.file_filter(filepath_list_f[i], filename_filter_list[i])
                if i == 0:
                    file_num = len(file_consistency_temp_dic[filename_filter_list[i]])
                    filename_ori = filename_filter_list[i]
                else:
                    if len(file_consistency_temp_dic[filename_filter_list[i]]) > file_num:
                        print('There are some files missing in the ' + filename_ori + ' folder.')
                        file_num = len(file_consistency_temp_dic[filename_filter_list[i]])
                        filename_ori = filename_filter_list[i]
                        file_consistency_factor = False
                    elif len(file_consistency_temp_dic[filename_filter_list[i]]) < file_num:
                        print('There are some files missing in the ' + filename_filter_list[i] + ' folder.')
                        file_consistency_factor = False
                i += 1
        else:
            print('Please make sure file path and its specific filename are corresponded and respectively set in filepath list and filename filter list ')
    # Outcome
    if not file_consistency_factor:
        print('Consistency issues detected, please check the output information above')
        sys.exit(-1)
    else:
        print('All input files are consistent')


def neighbor_average_convolve2d(array, size=4):
    kernel = np.ones((2 * size + 1, 2 * size + 1))
    kernel[4, 4] = 0
    neighbor_sum = convolve2d(
        array, kernel, mode='same',
        boundary='fill', fillvalue=0)
    return neighbor_sum




def remove_sole_pixel(twod_array, Nan_value=0, half_size_window=2):
    if len(twod_array.shape) != 2:
        print('Please correctly inputting a 2d array!')
        sys.exit(-1)
    else:
        twod_array_temp = copy.copy(twod_array)
        for y in range(twod_array.shape[0]):
            for x in range(twod_array.shape[1]):
                if twod_array[y, x] != Nan_value and np.count_nonzero(twod_array[y - half_size_window: y + half_size_window + 1, x - half_size_window: x + half_size_window + 1] == Nan_value) == (2 * half_size_window + 1) ** 2 - 1:
                    twod_array_temp[y, x] = Nan_value
    return twod_array_temp


def cor_to_pixel(two_corner_coordinate, study_area_example_file_path):
    pixel_limitation_f = {}
    if len(two_corner_coordinate) == 2:
        UL_corner = two_corner_coordinate[0]
        LR_corner = two_corner_coordinate[1]
        if len(UL_corner) == len(LR_corner) == 2:
            upper_limit = UL_corner[1]
            lower_limit = LR_corner[1]
            right_limit = LR_corner[0]
            left_limit = UL_corner[0]
            dataset_temp_list = bf.file_filter(study_area_example_file_path, ['.TIF'])
            temp_dataset = gdal.Open(dataset_temp_list[0])
            # TEMP_warp = gdal.Warp(study_area_example_file_path + '\\temp.tif', temp_dataset, dstSRS='EPSG:4326')
            # temp_band = temp_dataset.GetRasterBand(1)
            # temp_cols = temp_dataset.RasterXSize
            # temp_rows = temp_dataset.RasterYSize
            temp_transform = temp_dataset.GetGeoTransform()
            temp_xOrigin = temp_transform[0]
            temp_yOrigin = temp_transform[3]
            temp_pixelWidth = temp_transform[1]
            temp_pixelHeight = -temp_transform[5]
            pixel_limitation_f['x_max'] = max(int((right_limit - temp_xOrigin) / temp_pixelWidth), int((left_limit - temp_xOrigin) / temp_pixelWidth))
            pixel_limitation_f['y_max'] = max(int((temp_yOrigin - lower_limit) / temp_pixelHeight), int((temp_yOrigin - upper_limit) / temp_pixelHeight))
            pixel_limitation_f['x_min'] = min(int((right_limit - temp_xOrigin) / temp_pixelWidth), int((left_limit - temp_xOrigin) / temp_pixelWidth))
            pixel_limitation_f['y_min'] = min(int((temp_yOrigin - lower_limit) / temp_pixelHeight), int((temp_yOrigin - upper_limit) / temp_pixelHeight))
        else:
            print('Please make sure input all corner pixel with two coordinate in list format')
    else:
        print('Please mention the input coordinate should contain the coordinate of two corner pixel')
    try:
        # TEMP_warp.dispose()
        os.remove(study_area_example_file_path + '\\temp.tif')
    except:
        print('please remove the temp file manually')
    return pixel_limitation_f


def print_single_stacked_Zvalue(output_folder, xy_list, datacube, offset_list, nodata_value, doy_list):

    # Generate the zValue series
    for _ in range(xy_list.shape[0]):
        x, y = xy_list[_, 1], xy_list[_, 0]
        x_rel, y_rel = x - offset_list[1], y - offset_list[0]

        if not os.path.exists(output_folder + f'stacked_{str(x)}_{str(y)}.png'):
            dc_temp = datacube[y_rel, x_rel, :].flatten()
            doy_list_ = np.array(copy.copy(doy_list))
            doy_list_ = np.delete(doy_list_, np.argwhere(dc_temp == nodata_value))
            dc_temp = np.delete(dc_temp, np.argwhere(dc_temp == nodata_value))
            if not os.path.exists(output_folder + f'csv_data\\stacked_{str(x)}_{str(y)}.csv'):
                bf.create_folder(output_folder + f'csv_data\\')
                df_ = pd.DataFrame({'DOY': doy_list_, 'vi': dc_temp})
                df_.to_csv(output_folder + f'csv_data\\stacked_{str(x)}_{str(y)}.csv')
            
            if dc_temp.shape[0] != 0:
                fig1, ax1 = plt.subplots(figsize=(12, 8), constrained_layout=True)
                ax1.scatter(doy_list_, dc_temp, s=8**2, color=(196/256, 80/256, 80/256), edgecolor=(0/256, 0/256, 0/256), linewidth=2, zorder=4)
                plt.savefig(output_folder + f'stacked_{str(x)}_{str(y)}.png')
                fig1 = None
                ax1 = None
                plt.close()
        
        