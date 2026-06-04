# coding=utf-8
from osgeo import gdal
import sys
import pandas as pd
import numpy as np
import os
import zipfile
import shutil
import scipy.sparse as sp
import copy
import time
from basic_function import Path
import basic_function as bf
import concurrent.futures
from itertools import repeat
from zipfile import ZipFile
import traceback
import pywt
import psutil
import sympy
from scipy import sparse as sm
from .utils import retrieve_srs, write_raster, remove_all_file_and_folder
from .built_in_index import built_in_index
from lxml import etree
from NDsm import NDSparseMatrix
import json
import Sentinel2_toolbox

global topts
topts = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'PREDICTOR=2'])

# Set np para
np.seterr(divide='ignore', invalid='ignore')


# Convert the str-type mathematical expression into python-readable type
def convert_index_func(expr: str):
    f = sympy.sympify(expr)
    dep_list = sorted(f.free_symbols, key=str)
    num_f = sympy.lambdify(dep_list, f)
    return dep_list, num_f


class Sentinel2_ds(object):

    def __init__(self, ori_zipfile_folder, work_env=None):
        # Define var
        self.S2_metadata = None
        self._subset_failure_file = []
        self.output_bounds = np.array([])
        self.raw_10m_bounds = np.array([])
        self.ROI = None
        self.ROI_name = None
        self.ori_folder = Path(ori_zipfile_folder).path_name
        self.S2_metadata_size = np.nan
        self.date_list = []
        self.main_coordinate_system = None
        self._index_exprs_dic = {}

        # Define key variables (kwargs)
        self._size_control_factor = False
        self._cloud_removal_para = False
        self._vi_clip_factor = False
        self._sparsify_matrix_factor = False
        self._cloud_clip_seq = None
        self._pansharp_factor = True

        # Define mosaic para
        self._mosaic_infr = {}
        self._mosaic_overwritten_para = False

        # Define 2dc para
        self._dc_infr = {}
        self._dc_overwritten_para = False
        self._inherit_from_logfile = None
        self._remove_nan_layer = False
        self._manually_remove_para = False
        self._manually_remove_datelist = None

        # Remove all the duplicated data
        dup_data = bf.file_filter(self.ori_folder, [f'.{str(_)}.zip' for _ in range(10)], and_or_factor='and')
        for dup in dup_data:
            os.remove(dup)

        # Generate the original zip file list
        self.orifile_list = bf.file_filter(self.ori_folder, ['.zip', 'S2'], and_or_factor='and', subfolder_detection=True)
        self.orifile_list = [i for i in self.orifile_list if 'S2' in i.split('\\')[-1] and '.zip' in i.split('\\')[-1]]
        if not self.orifile_list:
            print('There has no Sentinel zipfiles in the input dir')
            sys.exit(-1)

        # Initialise the work environment
        if work_env is None:
            try:
                self._work_env = Path(os.path.dirname(os.path.dirname(self.ori_folder)) + '\\').path_name
            except:
                print('There has no base dir for the ori_folder and the ori_folder will be treated as the work env')
                self._work_env = bf.Path(self.ori_folder).path_name
        else:
            self._work_env = Path(work_env).path_name

        # Create cache path
        self.cache_folder = self._work_env + 'cache\\'
        self.trash_folder = self._work_env + 'trash\\'
        bf.create_folder(self.cache_folder)
        bf.create_folder(self.trash_folder)
        bf.create_folder(self._work_env + 'Corrupted_S2_file\\')

        # Create output path
        self.output_path = f'{self._work_env}Sentinel2_L2A_Output\\'
        self.shpfile_path = f'{self._work_env}shpfile\\'
        self.log_filepath = f'{self._work_env}logfile\\'
        bf.create_folder(self.output_path)
        bf.create_folder(self.log_filepath)
        bf.create_folder(self.shpfile_path)

        # Constant
        self._band_name_list = ['B01_60m.jp2', 'B02_10m.jp2', 'B03_10m.jp2', 'B04_10m.jp2', 'B05_20m.jp2', 'B06_20m.jp2',
                                'B07_20m.jp2', 'B08_10m.jp2', 'B8A_20m.jp2', 'B09_60m.jp2', 'B11_20m.jp2', 'B12_20m.jp2']
        self._band_output_list = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
        self._all_supported_index_list = ['RGB', 'QI', 'all_band', '4visual', 'NDVI', 'MNDWI', 'EVI', 'EVI2', 'OSAVI', 'GNDVI',
                                          'NDVI_RE', 'NDVI_RE2', 'AWEI', 'AWEInsh', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9',
                                          'B11', 'B12', 'NDVI_20m', 'OSAVI_20m']

    def save_log_file(func):
        def wrapper(self, *args, **kwargs):

            ############################################################################################################
            # Document the log file and para file
            # The log file contained the information for each run and para file documented the args of each func
            ############################################################################################################

            time_start = time.time()
            c_time = time.ctime()
            log_file = open(f"{self.log_filepath}log.txt", "a+")
            if os.path.exists(f"{self.log_filepath}para_file.txt"):
                para_file = open(f"{self.log_filepath}para_file.txt", "r+")
            else:
                para_file = open(f"{self.log_filepath}para_file.txt", "w+")
            error_inf = None

            para_txt_all = para_file.read()
            para_ori_txt = para_txt_all.split('#' * 70 + '\n')
            para_txt = para_txt_all.split('\n')
            contain_func = [txt for txt in para_txt if txt.startswith('Process Func:')]

            try:
                func(self, *args, **kwargs)
            except:
                error_inf = traceback.format_exc()
                print(error_inf)

            # Header for the log file
            log_temp = ['#' * 70 + '\n', f'Process Func: {func.__name__}\n', f'Start time: {c_time}\n',
                        f'End time: {time.ctime()}\n', f'Total processing time: {str(time.time() - time_start)}\n']

            # Create args and kwargs list
            args_f = 0
            args_list = ['*' * 25 + 'Arguments' + '*' * 25 + '\n']
            kwargs_list = []
            for i in args:
                args_list.extend([f"args{str(args_f)}:{str(i)}\n"])
            for k_key in kwargs.keys():
                kwargs_list.extend([f"{str(k_key)}:{str(kwargs[k_key])}\n"])
            para_temp = ['#' * 70 + '\n', f'Process Func: {func.__name__}\n', f'Start time: {c_time}\n',
                         f'End time: {time.ctime()}\n', f'Total processing time: {str(time.time() - time_start)}\n']
            para_temp.extend(args_list)
            para_temp.extend(kwargs_list)
            para_temp.append('#' * 70 + '\n')

            log_temp.extend(args_list)
            log_temp.extend(kwargs_list)
            log_file.writelines(log_temp)
            for func_key, func_processing_name in zip(['metadata', 'subset', 'ds2sdc', 'mosaic'], ['constructing metadata', 'executing subset and clip', '2sdc', 'mosaic2seqtif']):
                if func_key in func.__name__:
                    if error_inf is None:
                        log_file.writelines([f'Status: Finished {func_processing_name}!\n', '#' * 70 + '\n'])
                        metadata_line = [q for q in contain_func if func_key in q]
                        if len(metadata_line) == 0:
                            para_file.writelines(para_temp)
                            para_file.close()
                        elif len(metadata_line) == 1:
                            for para_ori_temp in para_ori_txt:
                                if para_ori_temp != '' and metadata_line[0] not in para_ori_temp:
                                    para_temp.extend(['#' * 70 + '\n', para_ori_temp, '#' * 70 + '\n'])
                                    para_file.close()
                                    para_file = open(f"{self.log_filepath}para_file.txt", "w+")
                                    para_file.writelines(para_temp)
                                    para_file.close()
                        elif len(metadata_line) > 1:
                            print('Code error! ')
                            sys.exit(-1)
                    else:
                        log_file.writelines([f'Status: Error in {func_processing_name}!\n', 'Error information:\n', error_inf + '\n', '#' * 70 + '\n'])
        return wrapper

    def _retrieve_para(self, required_para_name_list: list, protected_var=False, **kwargs):

        if not os.path.exists(f'{self.log_filepath}para_file.txt'):
            print('The para file is not established yet')
            sys.exit(-1)
        else:
            para_file = open(f"{self.log_filepath}para_file.txt", "r+")
            para_raw_txt = para_file.read().split('\n')

        for para in required_para_name_list:
            for q in para_raw_txt:
                para = str(para)
                if q.startswith(para + ':'):
                    if q.split(para + ':')[-1] == 'None':
                        if not protected_var:
                            self.__dict__[para] = None
                        else:
                            self.__dict__['_' + para] = None
                    elif q.split(para + ':')[-1] == 'True':
                        if not protected_var:
                            self.__dict__[para] = True
                        else:
                            self.__dict__['_' + para] = True
                    elif q.split(para + ':')[-1] == 'False':
                        if not protected_var:
                            self.__dict__[para] = False
                        else:
                            self.__dict__['_' + para] = True
                    elif q.split(para + ':')[-1].startswith('['):
                        if not protected_var:
                            self.__dict__[para] = list(q.split(para + ':')[-1][1: -1])
                        else:
                            self.__dict__['_' + para] = list(q.split(para + ':')[-1][1: -1])
                    elif q.split(para + ':')[-1].startswith('('):
                        if not protected_var:
                            self.__dict__[para] = tuple(q.split(para + ':')[-1][1: -1])
                        else:
                            self.__dict__['_' + para] = tuple(q.split(para + ':')[-1][1: -1])
                    else:
                        try:
                            t = float(q.split(para + ':')[-1])
                            if not protected_var:
                                self.__dict__[para] = float(q.split(para + ':')[-1])
                            else:
                                self.__dict__['_' + para] = float(q.split(para + ':')[-1])
                        except:
                            if not protected_var:
                                self.__dict__[para] = q.split(para + ':')[-1]
                            else:
                                self.__dict__['_' + para] = q.split(para + ':')[-1]


    @save_log_file
    def construct_metadata(self):

        #########################################################################
        # Construct the metadata based on the zip file in the ori folders
        # While the corrupted files will be moved into a trash folder
        #########################################################################

        # Start constructing metadata
        print('---------------------------- Start the construction of the Sentinel2 ds metadata ----------------------------')
        start_temp = time.time()

        # process input files
        if os.path.exists(self._work_env + 'Metadata.xlsx'):
            metadata_num = pd.read_excel(self._work_env + 'Metadata.xlsx').shape[0]
        else:
            metadata_num = 0

        if not os.path.exists(self._work_env + 'Metadata.xlsx') or metadata_num != len(self.orifile_list):
            corrupted_ori_file, corrupted_file_date, product_path, product_name, sensor_type, sensing_date, orbit_num, tile_num, width, height = ([] for i in range(10))
            corrupted_factor = 0
            for ori_file in self.orifile_list:
                try:
                    unzip_file = zipfile.ZipFile(ori_file)
                    unzip_file.close()
                    file_name = ori_file.split('\\')[-1]
                    product_path.append(ori_file)
                    sensing_date.append(file_name[file_name.find('_20') + 1: file_name.find('_20') + 9])
                    orbit_num.append(file_name[file_name.find('_R') + 2: file_name.find('_R') + 5])
                    tile_num.append(file_name[file_name.find('_T') + 2: file_name.find('_T') + 7])
                    sensor_type.append(file_name[file_name.find('S2'): file_name.find('S2') + 10])
                    # print(file_information)
                except:
                    if (not os.path.exists(self._work_env + 'Corrupted_S2_file')) and corrupted_factor == 0:
                        os.makedirs(self._work_env + 'Corrupted_S2_file')
                        corrupted_factor = 1
                    print(f'This file is corrupted {ori_file}!')
                    file_name = ori_file.split('\\')[-1]
                    corrupted_ori_file.append(file_name)
                    corrupted_file_date.append(file_name[file_name.find('_20') + 1: file_name.find('_20') + 9])

            # Move the corrupted files
            for corrupted_file_name in corrupted_ori_file:
                shutil.move(self.ori_folder + corrupted_file_name, self._work_env + 'Corrupted_S2_file\\' + corrupted_file_name)

            # Construct corrupted metadata
            Corrupted_metadata = pd.DataFrame({'Corrupted_file_name': corrupted_ori_file, 'File_Date': corrupted_file_date})
            if not os.path.exists(self._work_env + 'Corrupted_metadata.xlsx'):
                Corrupted_metadata.to_excel(self._work_env + 'Corrupted_metadata.xlsx')
            else:
                Corrupted_metadata_old_version = pd.read_excel(self._work_env + 'Corrupted_metadata.xlsx')
                Corrupted_metadata_old_version = pd.concat([Corrupted_metadata_old_version, Corrupted_metadata], axis=0, ignore_index=True)
                Corrupted_metadata_old_version.drop_duplicates()
                Corrupted_metadata_old_version.to_excel(self._work_env + 'Corrupted_metadata.xlsx')

            self.S2_metadata = pd.DataFrame({'Product_Path': product_path, 'Sensing_Date': sensing_date,
                                             'Orbit_Num': orbit_num, 'Tile_Num': tile_num, 'Sensor_Type': sensor_type})

            # Process duplicate file
            duplicate_file_list = []
            i = 0
            while i <= self.S2_metadata.shape[0] - 1:
                file_inform = str(self.S2_metadata['Sensing_Date'][i]) + '_' + self.S2_metadata['Tile_Num'][i]
                q = i + 1
                while q <= self.S2_metadata.shape[0] - 1:
                    file_inform2 = str(self.S2_metadata['Sensing_Date'][q]) + '_' + self.S2_metadata['Tile_Num'][q]
                    if file_inform2 == file_inform:
                        if len(self.S2_metadata['Product_Path'][i]) > len(self.S2_metadata['Product_Path'][q]):
                            duplicate_file_list.append(self.S2_metadata['Product_Path'][i])
                        elif len(self.S2_metadata['Product_Path'][i]) < len(self.S2_metadata['Product_Path'][q]):
                            duplicate_file_list.append(self.S2_metadata['Product_Path'][q])
                        else:
                            if int(os.path.getsize(self.S2_metadata['Product_Path'][i])) > int(os.path.getsize(self.S2_metadata['Product_Path'][q])):
                                duplicate_file_list.append(self.S2_metadata['Product_Path'][q])
                            else:
                                duplicate_file_list.append(self.S2_metadata['Product_Path'][i])
                        break
                    q += 1
                i += 1

            duplicate_file_list = list(dict.fromkeys(duplicate_file_list))
            if duplicate_file_list != []:
                for file in duplicate_file_list:
                    shutil.move(file, self._work_env + 'Corrupted_S2_file\\' + file.split('\\')[-1])
                self.orifile_list = bf.file_filter(self.ori_folder, ['.zip', 'S2'], and_or_factor='and',subfolder_detection=True)
                self.orifile_list = [i for i in self.orifile_list if 'S2' in i.split('\\')[-1] and '.zip' in i.split('\\')[-1]]
                self.construct_metadata()
            else:
                self.S2_metadata.to_excel(self._work_env + 'Metadata.xlsx')
                self.S2_metadata = pd.read_excel(self._work_env + 'Metadata.xlsx')
        else:
            self.S2_metadata = pd.read_excel(self._work_env + 'Metadata.xlsx')

        self.S2_metadata.sort_values(by=['Sensing_Date'], ascending=True)
        self.S2_metadata_size = self.S2_metadata.shape[0]
        self.output_bounds = np.zeros([self.S2_metadata_size, 4]) * np.nan
        self.raw_10m_bounds = np.zeros([self.S2_metadata_size, 4]) * np.nan
        self.date_list = self.S2_metadata['Sensing_Date'].drop_duplicates().to_list()
        print(f'Finish in {str(time.time() - start_temp)} sec!')
        print('----------------------------  End the construction of Metadata  ----------------------------')

    def _qi_remove_cloud(self, processed_filepath, qi_filepath, serial_num, dst_nodata=0, **kwargs):
        # Determine the process parameter
        sensing_date = self.S2_metadata['Sensing_Date'][serial_num]
        tile_num = self.S2_metadata['Tile_Num'][serial_num]
        if kwargs['cloud_removal_strategy'] == 'QI_all_cloud':
            cloud_indicator = [0, 1, 2, 3, 8, 9, 10, 11]
        else:
            print('Cloud removal strategy is not supported!')
            return

        # Input ds
        try:
            qi_ds = gdal.Open(qi_filepath)
            processed_ds = gdal.Open(processed_filepath, gdal.GA_Update)
            qi_array = qi_ds.GetRasterBand(1).ReadAsArray()
            processed_array = processed_ds.GetRasterBand(1).ReadAsArray()
        except:
            print(f'Cannot read the qi array or processed array of {str(sensing_date)} {str(tile_num)}!')
            return

        if qi_array is None or processed_array is None:
            print(f'the qi array or processed array of {str(sensing_date)} {str(tile_num)} dont exist!')
            return

        elif qi_array.shape[0] != processed_array.shape[0] or qi_array.shape[1] != processed_array.shape[1]:
            print(f'Consistency issue between the qi array and processed array of {str(sensing_date)} {str(tile_num)}!')
            return

        for indicator in cloud_indicator:
            qi_array[qi_array == indicator] = 64
        qi_array[qi_array == 255] = 64
        processed_array[qi_array == 64] = dst_nodata

        processed_ds.GetRasterBand(1).WriteArray(processed_array)
        processed_ds.FlushCache()
        processed_ds = None
        qi_ds = None

    def _check_metadata_availability(self):
        if self.S2_metadata is None:
            try:
                self.construct_metadata()
            except:
                print('Please manually construct the S2_metadata before further processing!')
                sys.exit(-1)
        else:
            pass

    def _check_output_band_statue(self, band_name, tiffile_serial_num, *args, **kwargs):

        # Define local var
        sensing_date = self.S2_metadata['Sensing_Date'][tiffile_serial_num]
        tile_num = self.S2_metadata['Tile_Num'][tiffile_serial_num]

        # Factor configuration
        if True in [band_temp not in self._band_output_list for band_temp in band_name]:
            print(f'Band {band_name} is not valid!')
            sys.exit(-1)

        if self._vi_clip_factor:
            output_path = f'{self.output_path}Sentinel2_{self.ROI_name}_index\\all_band\\'
        else:
            output_path = f'{self.output_path}Sentinel2_constructed_index\\all_band\\'
        bf.create_folder(output_path)

        # Detect whether the required band was generated before
        try:
            if False in [os.path.exists(f'{output_path}{str(sensing_date)}_{str(tile_num)}_{band_temp}.TIF') for band_temp in band_name]:
                if 'combine_band_factor' in kwargs.keys():
                    kwargs['combine_band_factor'] = False
                self.subset_tiffiles(band_name, tiffile_serial_num, **kwargs)

            # Return output
            if False in [os.path.exists(f'{output_path}{str(sensing_date)}_{str(tile_num)}_{band_temp}.TIF') for band_temp in band_name]:
                print(f'Something error processing {band_name}!')
                return None
            else:
                return [gdal.Open(f'{output_path}{str(sensing_date)}_{str(tile_num)}_{band_temp}.TIF') for band_temp in band_name]

        except:
            return None

    @save_log_file
    def mp_subset(self, *args, **kwargs):

        if 'chunk_size' in kwargs.keys() and type(kwargs['chunk_size']) is int:
            chunk_size = min(os.cpu_count(), kwargs['chunk_size'])
            del kwargs['chunk_size']
        else:
            chunk_size = os.cpu_count()

        if self.S2_metadata is None:
            print('Please construct the S2_metadata before the subset!')
            sys.exit(-1)

        i = range(self.S2_metadata.shape[0])
        # mp process
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(chunk_size * Sentinel2_toolbox.configuration['multiprocess_ratio'])) as executor:
            results = executor.map(self.subset_tiffiles, repeat(args[0]), i, repeat(False), repeat(kwargs))

        results = list(results)
        for result in results:
            if result is not None and result[0] is not None:
                self._subset_failure_file.append(result)

        if self._subset_failure_file != []:
            self._process_subset_failure_file(args[0])

    @save_log_file
    def sequenced_subset(self, *args, **kwargs):

        if self.S2_metadata is None:
            print('Please construct the S2_metadata before the subset!')
            sys.exit(-1)

        # sequenced process
        for i in range(self.S2_metadata.shape[0]):
            [index, tiffile_serial_num, sensing_date, tile_num] = self.subset_tiffiles(args[0], i, **kwargs)
            if index is not None:
                self._subset_failure_file.append([index, tiffile_serial_num, sensing_date, tile_num])

        if self._subset_failure_file != []:
            self._process_subset_failure_file(args[0])

    def _wavelet_pansharpen(self, processed_image, pan_image_list, wavelet_level: int = 1, wavelet_type: str = 'haar'):

        # DataType check
        if type(pan_image_list) is not list:
            raise Exception('please input the pan image list as a list of array')
        elif type(processed_image) is not np.ndarray:
            raise Exception('please input the processed image as an array')

        # Consistency check
        for array_temp in pan_image_list:
            if array_temp.shape != processed_image.shape:
                raise Exception('Consistency Error')

        # Require the most correlated image
        # time1 = time.time()
        correlation_list = []
        temp = processed_image.reshape(-1)

        if np.nansum(temp) == 0:
            return processed_image
        else:
            correlation_list = []
            for array_temp in pan_image_list:
                array_temp[array_temp == 0] = np.nan
                array_temp = array_temp.reshape(-1)
                msk = (~np.ma.masked_invalid(temp).mask & ~np.ma.masked_invalid(array_temp).mask)
                correlation_list.append(np.corrcoef(temp[msk], array_temp[msk])[0][1])
            # print(f'Correlation analysis consumes {str(time.time() - time1)} s')
            most_correlated_image = pan_image_list[np.argmax(np.array(correlation_list))]
            # time2 = time.time()
            # print(f'Correlation analysis consumes {str(time2-time1)} s')

            # Pan sharpening
            pan_coef = pywt.wavedec2(most_correlated_image, wavelet_type, mode='sym', level=wavelet_level)
            processed_coef = pywt.wavedec2(processed_image, wavelet_type, mode='sym', level=wavelet_level)
            # time3 = time.time()
            # print(f'Wavelet analysis consumes {str(time3 - time2)} s')

            pan_coef[0] = processed_coef[0]
            sharped_image = pywt.waverec2(pan_coef, wavelet_type)
            # print(f'Wavelet reconstruction consumes {str(time.time() - time3)} s')
            return sharped_image

    def _subset_indicator_process(self, **kwargs):
        # Detect whether all the indicator are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in ['ROI', 'ROI_name', 'size_control_factor', 'cloud_removal_strategy', 'combine_band_factor', 'pansharp_factor', 'main_coordinate_system']:
                print(f'{kwarg_indicator} is not supported kwargs! Please double check!')

        # process clip parameter
        if self.ROI is None:
            if 'ROI' in kwargs.keys():
                self._vi_clip_factor = True
                if '.shp' in kwargs['ROI'] and os.path.exists(kwargs['ROI']):
                    self.ROI = kwargs['ROI']
                else:
                    print('Please input valid shp file for clip!')
                    sys.exit(-1)

                if 'ROI_name' in kwargs.keys():
                    self.ROI_name = kwargs['ROI_name']
                else:
                    self.ROI_name = self.ROI.split('\\')[-1].split('.')[0]
            else:
                self._vi_clip_factor = False

        # process size control parameter
        if 'size_control_factor' in kwargs.keys():
            if type(kwargs['size_control_factor']) is bool:
                self._size_control_factor = kwargs['size_control_factor']
            else:
                print('Please mention the size_control_factor should be bool type!')
                self._size_control_factor = False
        else:
            self._size_control_factor = False

        # process cloud removal parameter
        if 'cloud_removal_strategy' in kwargs.keys():
            self._cloud_removal_para = True
        else:
            self._cloud_removal_para = False

        # process main_coordinate_system
        if 'main_coordinate_system' in kwargs.keys():
            self.main_coordinate_system = kwargs['main_coordinate_system']
        else:
            self.main_coordinate_system = 'EPSG:32649'

        # process combine band factor
        # This factor determine whether a TIFFILE with all bands integrated is created
        if 'combine_band_factor' in kwargs.keys():
            if type(kwargs['combine_band_factor']) is bool:
                self._combine_band_factor = kwargs['combine_band_factor']
            else:
                raise Exception(f'combine band factor is not under the bool type!')
        else:
            self._combine_band_factor = False

        # Process pansharp_factor
        # This factor control whether low-resolution(60m/120m) bands are sharpened to a high-resolution(10m)
        if 'pansharp_factor' in kwargs.keys():
            if type(kwargs['pansharp_factor']) is bool:
                self._pansharp_factor = kwargs['pansharp_factor']
            else:
                raise Exception(f'pansharp_factor is not under the bool type!')
        else:
            self._pansharp_factor = False

    def generate_10m_output_bounds(self, tiffile_serial_num, **kwargs):

        # Define local var
        sensing_date = self.S2_metadata['Sensing_Date'][tiffile_serial_num]
        tile_num = self.S2_metadata['Tile_Num'][tiffile_serial_num]
        VI = 'all_band'

        # Define the output path
        if self._vi_clip_factor:
            output_path = f'{self.output_path}Sentinel2_{self.ROI_name}_index\\{VI}\\'
        else:
            output_path = f'{self.output_path}Sentinel2_constructed_index\\{VI}\\'
        bf.create_folder(output_path)

        # Create the output bounds based on the 10-m Band2 images
        if self.output_bounds.shape[0] > tiffile_serial_num:
            if True in np.isnan(self.output_bounds[tiffile_serial_num, :]):
                temp_S2file_path = self.S2_metadata.iat[tiffile_serial_num, 1]
                zfile = ZipFile(temp_S2file_path, 'r')
                b2_band_file_name = f'{str(sensing_date)}_{str(tile_num)}_B2'
                if not os.path.exists(output_path + b2_band_file_name + '.TIF'):
                    b2_file = [zfile_temp for zfile_temp in zfile.namelist() if 'B02_10m.jp2' in zfile_temp]
                    if len(b2_file) != 1:
                        print(f'Data issue for the B2 file of all_cloud data ({str(tiffile_serial_num + 1)} of {str(self.S2_metadata.shape[0])})')
                        return
                    else:
                        try:
                            ds_temp = gdal.Open('/vsizip/%s/%s' % (temp_S2file_path, b2_file[0]))
                            ds_cor = ds_temp.GetSpatialRef().GetAttrValue("AUTHORITY", 0) + ':' + ds_temp.GetSpatialRef().GetAttrValue("AUTHORITY", 1)
                            if self.main_coordinate_system != ds_cor:
                                gdal.Warp('/vsimem/' + b2_band_file_name + '_cor.TIF', ds_temp, dstSRS=self.main_coordinate_system)
                                cor_ds = gdal.Open('/vsimem/' + b2_band_file_name + '_cor.TIF')
                                ulx_temp, xres_temp, xskew_temp, uly_temp, yskew_temp, yres_temp = cor_ds.GetGeoTransform()
                                self.output_bounds[tiffile_serial_num, :] = np.array([ulx_temp, uly_temp + yres_temp * cor_ds.RasterYSize, ulx_temp + xres_temp * cor_ds.RasterXSize, uly_temp])
                                gdal.Unlink('/vsimem/' + b2_band_file_name + '_cor.TIF')
                            else:
                                ulx_temp, xres_temp, xskew_temp, uly_temp, yskew_temp, yres_temp = ds_temp.GetGeoTransform()
                                self.output_bounds[tiffile_serial_num, :] = np.array([ulx_temp, uly_temp + yres_temp * ds_temp.RasterYSize, ulx_temp + xres_temp * ds_temp.RasterXSize, uly_temp])
                            band_output_limit = (int(self.output_bounds[tiffile_serial_num, 0]), int(self.output_bounds[tiffile_serial_num, 1]),
                                                 int(self.output_bounds[tiffile_serial_num, 2]), int(self.output_bounds[tiffile_serial_num, 3]))

                            if self._vi_clip_factor:
                                gdal.Warp('/vsimem/' + b2_band_file_name + '.TIF', ds_temp,
                                          dstSRS=self.main_coordinate_system, xRes=10, yRes=10, cutlineDSName=self.ROI,
                                          outputType=gdal.GDT_UInt16, dstNodata=65535, outputBounds=band_output_limit)
                            else:
                                gdal.Warp('/vsimem/' + b2_band_file_name + '.TIF', ds_temp,
                                          dstSRS=self.main_coordinate_system, xRes=10, yRes=10,
                                          outputType=gdal.GDT_UInt16, dstNodata=65535, outputBounds=band_output_limit)
                            gdal.Translate(output_path + b2_band_file_name + '.TIF', '/vsimem/' + b2_band_file_name + '.TIF', options=topts, noData=65535)
                            gdal.Unlink('/vsimem/' + b2_band_file_name + '.TIF')
                        except:
                            print(f'The B2 of {str(sensing_date)}_{str(tile_num)} is not valid')
                            return
                else:
                    ds4bounds = gdal.Open(output_path + b2_band_file_name + '.TIF')
                    ulx, xres, xskew, uly, yskew, yres = ds4bounds.GetGeoTransform()
                    self.output_bounds[tiffile_serial_num, :] = np.array(
                        [ulx, uly + yres * ds4bounds.RasterYSize, ulx + xres * ds4bounds.RasterXSize, uly])
                    ds4bounds = None
        else:
            print('The output bounds has some logical issue!')
            sys.exit(-1)

    def _process_subset_failure_file(self, index_list, **kwargs):
        if self._subset_failure_file != []:
            subset_failure_file_folder = self._work_env + 'Corrupted_S2_file\\subset_failure_file\\'
            bf.create_folder(subset_failure_file_folder)
            for subset_failure_file in self._subset_failure_file:
                # remove all the related file
                related_output_file = bf.file_filter(self.output_path, [str(subset_failure_file[2]), str(subset_failure_file[3])], and_or_factor='and', subfolder_detection=True)
                for file in related_output_file:
                    file_name = file.split('\\')[-1]
                    shutil.move(file, f'{self.trash_folder}{file_name}')

            shutil.rmtree(self.trash_folder)
            bf.create_folder(self.trash_folder)

            for subset_failure_file in self._subset_failure_file:
                # remove all the zip file
                time = 1
                cannot_remove_file = []
                while True:
                    try:
                        zipfile = bf.file_filter(self.ori_folder, [str(subset_failure_file[2]), str(subset_failure_file[3])], and_or_factor='and')
                        for file in zipfile:
                            file_name = file.split('\\')[-1]
                            shutil.move(file, f'{subset_failure_file_folder}{file_name}')
                        break
                    except:
                        time += 1
                        if time >= 3:
                            cannot_remove_file.extend(zipfile)
                            break
                        pass

            if cannot_remove_file != []:
                raise Exception(f'Please remove the files manually {cannot_remove_file}')

            self.construct_metadata()

    def _process_index_list(self, index_list):

        # Process subset index list
        self._index_exprs_dic = built_in_index()
        for index in index_list:
            if index not in self._all_supported_index_list and type(index) is str and '=' in index:
                try:
                    self._index_exprs_dic.add_index(index)
                except:
                    raise ValueError(f'The expression {index} is wrong')
                index_list[index_list.index(index)] = index.split('=')[0]

            elif index in self._all_supported_index_list:
                pass

            else:
                raise NameError(f'The {index} is not a valid index or the expression is wrong')
        self._index_exprs_dic = self._index_exprs_dic.index_dic

        index_list_temp = copy.copy(index_list)
        for len_t in range(len(index_list)):
            if index_list[len_t] in ['B2', 'B3', 'B4', 'B8', 'all_band', '4visual', 'RGB']:
                index_list_temp.remove(index_list[len_t])
                temp_list = [index_list[len_t]]
                temp_list.extend(index_list_temp)
                index_list_temp = copy.copy(temp_list)
        index_list = copy.copy(index_list_temp)

        return index_list

    def subset_tiffiles(self, processed_index_list: list, tiffile_serial_num: int, overwritten_para: bool = False, *args, **kwargs):

        # subset_tiffiles is the core function in subsetting, resampling, clipping images as well as extracting VI and removing clouds.
        # The argument includes
        # ROI = define the path of a .shp file using for clipping all the sentinel-2 images
        # ROI_name = using to generate the roi-specified output folder, the default value is setting as the name of the ROI shp file
        # cloud_remove_strategy = method using to remove clouds, supported methods include QI_all_cloud

        # Define local args

        time1, time2, time3 = 0, 0, 0

        # Retrieve kwargs from args using the mp
        if args != () and type(args[0]) == dict:
            kwargs = copy.copy(args[0])

        # determine the subset indicator
        self._check_metadata_availability()
        self._subset_indicator_process(**kwargs)
        processed_index_list = self._process_index_list(processed_index_list)

        combine_index_list = []
        if self._combine_band_factor:
            array_cube = None
            for i_t in processed_index_list:
                if i_t in ['RGB', 'all_band', '4visual']:
                    combine_index_list.extend([['B2', 'B3', 'B4'], ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'], ['B2', 'B3', 'B4', 'B5', 'B8', 'B11']][['RGB', 'all_band', '4visual'].index(i_t)])
                else:
                    combine_index_list.append(i_t)
            combine_index_array_list = copy.copy(combine_index_list)

        if processed_index_list != []:

            sensing_date = self.S2_metadata['Sensing_Date'][tiffile_serial_num]
            tile_num = self.S2_metadata['Tile_Num'][tiffile_serial_num]
            temp_S2file_path = self.S2_metadata.iat[tiffile_serial_num, 1]

            # Generate the output boundary
            try:
                self.generate_10m_output_bounds(tiffile_serial_num, **kwargs)
            except:
                return 'B2', tiffile_serial_num, sensing_date, tile_num

            band_output_limit = (int(self.output_bounds[tiffile_serial_num, 0]), int(self.output_bounds[tiffile_serial_num, 1]),
                                 int(self.output_bounds[tiffile_serial_num, 2]), int(self.output_bounds[tiffile_serial_num, 3]))

            for index in processed_index_list:
                start_temp = time.time()
                print(f'Start processing \033[1;31m{index}\033[0m data of \033[3;34m{str(sensing_date)} {str(tile_num)}\033[0m ({str(tiffile_serial_num + 1)} of {str(self.S2_metadata_size)})')

                # Generate output folder
                if self._vi_clip_factor:
                    subset_output_path = f'{self.output_path}Sentinel2_{self.ROI_name}_index\\{index}\\'
                    if index in self._band_output_list or index in ['4visual', 'RGB']:
                        subset_output_path = f'{self.output_path}Sentinel2_{self.ROI_name}_index\\all_band\\'
                else:
                    subset_output_path = f'{self.output_path}Sentinel2_constructed_index\\{index}\\'
                    if index in self._band_output_list or index in ['4visual', 'RGB']:
                        subset_output_path = f'{self.output_path}Sentinel2_constructed_index\\all_band\\'

                # Generate qi output folder
                if self._cloud_clip_seq or not self._vi_clip_factor:
                    qi_path = f'{self.output_path}Sentinel2_constructed_index\\QI\\'
                else:
                    qi_path = f'{self.output_path}Sentinel2_{self.ROI_name}_index\\QI\\'

                # Combine bands to a single tif file
                if self._combine_band_factor:
                    folder_name = ''
                    for combine_index_temp in combine_index_list:
                        folder_name = folder_name + str(combine_index_temp) + '_'
                    if self._cloud_clip_seq or not self._vi_clip_factor:
                        combine_band_folder = f'{self.output_path}Sentinel2_constructed_index\\' + folder_name[0:-1] + '\\'
                    else:
                        combine_band_folder = f'{self.output_path}Sentinel2_{self.ROI_name}_index\\' + folder_name[0:-1] + '\\'
                    bf.create_folder(combine_band_folder)
                    if os.path.exists(f'{combine_band_folder}{str(sensing_date)}_{str(tile_num)}.npz'):
                        self._combine_band_factor, combine_index_list = False, []

                bf.create_folder(subset_output_path)
                bf.create_folder(qi_path)

                # Define the file name for VI
                file_name = f'{str(sensing_date)}_{str(tile_num)}_{index}'

                # Generate QI layer
                if (index == 'QI' and overwritten_para) or (index == 'QI' and not overwritten_para and not os.path.exists(qi_path + file_name + '.TIF')):
                    zfile = ZipFile(temp_S2file_path, 'r')
                    band_all = [zfile_temp for zfile_temp in zfile.namelist() if 'SCL_20m.jp2' in zfile_temp]
                    if len(band_all) != 1:
                        print(f'Something error during processing {index} data ({str(tiffile_serial_num + 1)} of {str(self.S2_metadata_size)})')
                        return [index, tiffile_serial_num, sensing_date, tile_num]
                    else:
                        for band_temp in band_all:
                            try:
                                ds_temp = gdal.Open('/vsizip/%s/%s' % (temp_S2file_path, band_temp))

                                if self._vi_clip_factor:
                                    gdal.Warp('/vsimem/' + file_name + 'temp.TIF', ds_temp, xRes=10, yRes=10, dstSRS=self.main_coordinate_system, outputBounds=band_output_limit, outputType=gdal.GDT_Byte, dstNodata=255)
                                    gdal.Warp('/vsimem/' + file_name + '.TIF', '/vsimem/' + file_name + 'temp.TIF', xRes=10, yRes=10, outputBounds=band_output_limit, cutlineDSName=self.ROI)
                                else:
                                    gdal.Warp('/vsimem/' + file_name + '.TIF', ds_temp, xRes=10, yRes=10, dstSRS=self.main_coordinate_system, outputBounds=band_output_limit, outputType=gdal.GDT_Byte, dstNodata=255)
                                gdal.Translate(qi_path + file_name + '.TIF', '/vsimem/' + file_name + '.TIF', options=topts, noData=255, outputType=gdal.GDT_Byte)

                                if self._combine_band_factor and 'QI' in combine_index_list:
                                    temp_ds = gdal.Open(qi_path + file_name + '.TIF')
                                    temp_array = temp_ds.GetRasterBand(1).ReadAsArray()
                                    temp_array = temp_array.astype(np.float16)
                                    temp_array[temp_array == 255] = np.nan
                                    if array_cube is None:
                                        array_cube = np.zeros([temp_array[0].shape[0], temp_array[0].shape[1], len(combine_index_array_list)], dtype=np.int16)

                                    if array_cube.shape[0] == temp_array.shape[0] and array_cube.shape[1] == temp_array.shape[1]:
                                        array_cube[:, :, combine_index_list.index('QI')] = temp_array
                                    else:
                                        print('consistency issuses')
                                        return
                                if self._vi_clip_factor:
                                    gdal.Unlink('/vsimem/' + file_name + 'temp.TIF')
                                gdal.Unlink('/vsimem/' + file_name + '.TIF')

                            except:
                                print(traceback.format_exc())
                                print(f'The {index} of {str(sensing_date)}_{str(tile_num)} is not valid')
                                return [index, tiffile_serial_num, sensing_date, tile_num]

                elif index == 'QI' and os.path.exists(qi_path + file_name + '.TIF') and 'QI' in combine_index_list:
                    temp_ds = gdal.Open(qi_path + file_name + '.TIF')
                    temp_array = temp_ds.GetRasterBand(1).ReadAsArray()
                    temp_array = temp_array.astype(np.float16)
                    temp_array[temp_array == 255] = np.nan
                    if array_cube is None:
                        array_cube = np.zeros([temp_array[0].shape[0], temp_array[0].shape[1], len(combine_index_array_list)], dtype=np.int16)

                    if array_cube.shape[0] == temp_array.shape[0] and array_cube.shape[1] == temp_array.shape[1]:
                        array_cube[:, :, combine_index_list.index('QI')] = temp_array
                    else:
                        print('consistency issues')
                        return

                # Subset band images
                elif index == 'all_band' or index == '4visual' or index == 'RGB' or index in self._band_output_list:
                    # Check the output band
                    if index == 'all_band':
                        band_name_list, band_output_list = self._band_name_list, self._band_output_list
                    elif index == '4visual':
                        band_name_list, band_output_list = ['B02_10m.jp2', 'B03_10m.jp2', 'B04_10m.jp2', 'B08_10m.jp2', 'B05_20m.jp2', 'B11_20m.jp2'], ['B2', 'B3', 'B4', 'B8', 'B5', 'B11']
                    elif index == 'RGB':
                        band_name_list, band_output_list = ['B02_10m.jp2', 'B03_10m.jp2', 'B04_10m.jp2'], ['B2', 'B3', 'B4']
                    elif index in self._band_output_list:
                        band_name_list, band_output_list = [self._band_name_list[self._band_output_list.index(index)]], [index]
                    else:
                        print('Code error!')
                        sys.exit(-1)

                    if overwritten_para or False in [os.path.exists(subset_output_path + str(sensing_date) + '_' + str(tile_num) + '_' + str(band_temp) + '.TIF') for band_temp in band_output_list] or (self._combine_band_factor and True in [band_index_temp in band_output_list for band_index_temp in combine_index_list]):
                        for band_name, band_output in zip(band_name_list, band_output_list):
                            if band_output != 'B2':
                                if self._pansharp_factor and band_output in ['B1', 'B5', 'B6', 'B7', 'B8A', 'B9', 'B11', 'B12'] and False in [os.path.exists(f'{subset_output_path}{str(sensing_date)}_{str(tile_num)}_{str(band_t)}.TIF') for band_t in ['B2', 'B3', 'B4', 'B8']]:
                                    band4sharpen = []
                                    for band_t in ['B2', 'B3', 'B4', 'B8']:
                                        if not os.path.exists(f'{subset_output_path}{str(sensing_date)}_{str(tile_num)}_{str(band_t)}.TIF'):
                                            band4sharpen.append(band_t)
                                    if band4sharpen == []:
                                        raise Exception('Code Error')
                                    else:
                                        if 'combine_band_factor' in kwargs.keys():
                                            kwargs['combine_band_factor'] = False
                                        self.subset_tiffiles(band4sharpen, tiffile_serial_num, *args, **kwargs)
                                else:
                                    zfile = ZipFile(temp_S2file_path, 'r')
                                    all_band_file_name = f'{str(sensing_date)}_{str(tile_num)}_{str(band_output)}'
                                    if not os.path.exists(subset_output_path + all_band_file_name + '.TIF') or overwritten_para:
                                        band_all = [zfile_temp for zfile_temp in zfile.namelist() if band_name in zfile_temp]
                                        if len(band_all) != 1:
                                            print(f'Something error during processing {band_output} of {index} data ({str(tiffile_serial_num + 1)} of {str(self.S2_metadata_size)})')
                                            return [index, tiffile_serial_num, sensing_date, tile_num]
                                        else:
                                            for band_temp in band_all:
                                                try:
                                                    ds_temp = gdal.Open('/vsizip/%s/%s' % (temp_S2file_path, band_temp))
                                                    t1 = time.time()
                                                    if band_output in ['B1', 'B9', 'B11', 'B10', 'B12']:
                                                        if self._vi_clip_factor:
                                                            gdal.Warp('/vsimem/' + all_band_file_name + '.TIF', ds_temp,
                                                                      xRes=10, yRes=10, dstSRS=self.main_coordinate_system, cutlineDSName=self.ROI,
                                                                      outputBounds=band_output_limit, outputType=gdal.GDT_UInt16,
                                                                      dstNodata=65535, resampleAlg=gdal.GRA_Bilinear)
                                                        else:
                                                            gdal.Warp('/vsimem/' + all_band_file_name + '.TIF', ds_temp,
                                                                      xRes=10, yRes=10, dstSRS=self.main_coordinate_system,
                                                                      outputBounds=band_output_limit, outputType=gdal.GDT_UInt16,
                                                                      dstNodata=65535, resampleAlg=gdal.GRA_Bilinear)
                                                    else:
                                                        if self._vi_clip_factor:
                                                            gdal.Warp('/vsimem/' + all_band_file_name + '.TIF', ds_temp,
                                                                      xRes=10, yRes=10, dstSRS=self.main_coordinate_system, cutlineDSName=self.ROI,
                                                                      outputBounds=band_output_limit, outputType=gdal.GDT_UInt16,
                                                                      dstNodata=65535)
                                                        else:
                                                            gdal.Warp('/vsimem/' + all_band_file_name + '.TIF', ds_temp,
                                                                      xRes=10, yRes=10, dstSRS=self.main_coordinate_system,
                                                                      outputBounds=band_output_limit, outputType=gdal.GDT_UInt16,
                                                                      dstNodata=65535)
                                                    time1 = time.time() - t1

                                                    t2 = time.time()
                                                    if self._cloud_removal_para:
                                                        qi_file_path = f'{qi_path}{str(sensing_date)}_{str(tile_num)}_QI.TIF'
                                                        if not os.path.exists(qi_file_path):
                                                            if 'combine_band_factor' in kwargs.keys():
                                                                kwargs['combine_band_factor'] = False
                                                            self.subset_tiffiles(['QI'], tiffile_serial_num, **kwargs)
                                                        try:
                                                            self._qi_remove_cloud('/vsimem/' + all_band_file_name + '.TIF',
                                                                                  qi_file_path, tiffile_serial_num, dst_nodata=65535,
                                                                                  sparse_matrix_factor=self._sparsify_matrix_factor,
                                                                                  **kwargs)
                                                        except:
                                                            return [None, None, None, None]
                                                    time2 = time.time() - t1

                                                    t3 = time.time()
                                                    if self._pansharp_factor and band_output in ['B11', 'B12']:
                                                        high_resolution_image_list = []
                                                        for high_res_band in ['B2', 'B3', 'B4', 'B8']:
                                                            if os.path.exists(f'{subset_output_path}{str(sensing_date)}_{str(tile_num)}_{high_res_band}.TIF'):
                                                                ds_t = gdal.Open(f'{subset_output_path}{str(sensing_date)}_{str(tile_num)}_{high_res_band}.TIF')
                                                                array_t = ds_t.GetRasterBand(1).ReadAsArray()
                                                                array_t = array_t.astype(np.float16)
                                                                array_t[array_t == 65535] = 0
                                                                high_resolution_image_list.append(array_t)

                                                        if high_resolution_image_list == []:
                                                            print('Something went wrong for the code in pan sharpening')
                                                        else:
                                                            process_ds = gdal.Open('/vsimem/' + all_band_file_name + '.TIF', gdal.GA_Update)
                                                            process_image = process_ds.GetRasterBand(1).ReadAsArray()
                                                            process_image = process_image.astype(np.float16)
                                                            process_image[process_image == 65535] = 0
                                                            output_image = self._wavelet_pansharpen(process_image, high_resolution_image_list)
                                                            # output_image = ATPRK.ATPRK_PANsharpen(process_image, high_resolution_image_list)
                                                            output_image = np.round(output_image)
                                                            output_image = output_image.astype(np.int)
                                                            output_image[output_image == 0] = 65535
                                                            process_ds.GetRasterBand(1).WriteArray(output_image)
                                                            process_ds.FlushCache()
                                                            process_ds = None
                                                    gdal.Translate(subset_output_path + all_band_file_name + '.TIF',
                                                                   '/vsimem/' + all_band_file_name + '.TIF', options=topts,
                                                                   noData=65535)
                                                    time3 = time.time() - t3

                                                    if self._combine_band_factor and band_output in combine_index_list:
                                                        temp_ds = gdal.Open(subset_output_path + all_band_file_name + '.TIF')
                                                        temp_array = temp_ds.GetRasterBand(1).ReadAsArray()
                                                        temp_array = temp_array.astype(np.float16)
                                                        temp_array[temp_array == 65535] = np.nan
                                                        if array_cube is None:
                                                            array_cube = np.zeros([temp_array[0].shape[0], temp_array[0].shape[1], len(combine_index_array_list)], dtype=np.int16)

                                                        if array_cube.shape[0] == temp_array.shape[0] and \
                                                                array_cube.shape[1] == temp_array.shape[1]:
                                                            array_cube[:, :, combine_index_list.index(band_output)] = temp_array
                                                        else:
                                                            print('consistency issuses')
                                                            return

                                                    gdal.Unlink('/vsimem/' + all_band_file_name + '.TIF')
                                                    time2 = time.time() - t2
                                                    print(f'Subset {band_output} of {str(sensing_date)}_{str(tile_num)} consumes {str(time1)[0:5]}s, remove cloud consumes {str(time2)[0:5]}s, pan-sharpening consumes {str(time3)[0:5]}s!')
                                                except:
                                                    print(f'The {band_output} of {str(sensing_date)}_{str(tile_num)} is not valid')
                                                    return [index, tiffile_serial_num, sensing_date, tile_num]

                                    elif os.path.exists(subset_output_path + all_band_file_name + '.TIF') and self._combine_band_factor and band_output in combine_index_list:
                                        temp_ds = gdal.Open(subset_output_path + all_band_file_name + '.TIF')
                                        temp_array = temp_ds.GetRasterBand(1).ReadAsArray()
                                        temp_array = temp_array.astype(np.float16)
                                        temp_array[temp_array == 65535] = np.nan
                                        if array_cube is None:
                                            array_cube = np.zeros([temp_array[0].shape[0], temp_array[0].shape[1], len(combine_index_array_list)], dtype=np.int16)

                                        if array_cube.shape[0] == temp_array.shape[0] and array_cube.shape[1] == temp_array.shape[1]:
                                            array_cube[:, :, combine_index_list.index(band_output)] = temp_array
                                        else:
                                            print('consistency issuses')
                                            return

                            else:
                                if not os.path.exists(f'{subset_output_path}\\{str(sensing_date)}_{str(tile_num)}_B2.TIF'):
                                    print('Code error for B2!')
                                    sys.exit(-1)
                                else:
                                    if False in [os.path.exists(
                                        subset_output_path + str(sensing_date) + '_' + str(tile_num) + '_' + str(
                                            band_temp) + '.TIF') for band_temp in band_output_list]:
                                        if self._cloud_removal_para:
                                            qi_file_path = f'{qi_path}{str(sensing_date)}_{str(tile_num)}_QI.TIF'
                                            if not os.path.exists(qi_file_path):
                                                if 'combine_band_factor' in kwargs.keys():
                                                    kwargs['combine_band_factor'] = False
                                                self.subset_tiffiles(['QI'], tiffile_serial_num, **kwargs)
                                            self._qi_remove_cloud(
                                                f'{subset_output_path}\\{str(sensing_date)}_{str(tile_num)}_B2.TIF',
                                                qi_file_path, tiffile_serial_num, dst_nodata=65535,
                                                sparse_matrix_factor=self._sparsify_matrix_factor, **kwargs)

                                    elif self._combine_band_factor and 'B2' in combine_index_list:
                                        temp_ds = gdal.Open(f'{subset_output_path}\\{str(sensing_date)}_{str(tile_num)}_B2.TIF')
                                        temp_array = temp_ds.GetRasterBand(1).ReadAsArray()
                                        temp_array = temp_array.astype(np.float16)
                                        temp_array[temp_array == 65535] = np.nan
                                        if array_cube is None:
                                            array_cube = np.zeros([temp_array[0].shape[0], temp_array[0].shape[1], len(combine_index_array_list)], dtype=np.int16)

                                        if array_cube.shape[0] == temp_array.shape[0] and array_cube.shape[1] == temp_array.shape[1]:
                                            array_cube[:, :, combine_index_list.index('B2')] = temp_array
                                        else:
                                            print('consistency issuses')
                                            return

                    if index == 'RGB':

                        gamma_coef = 1.5
                        RGB_array = {}
                        if self._vi_clip_factor:
                            rgb_output_path = f'{self.output_path}Sentinel2_{self.ROI_name}_index\\RGB\\'
                        else:
                            rgb_output_path = f'{self.output_path}Sentinel2_constructed_index_index\\RGB\\'
                        bf.create_folder(rgb_output_path)

                        if not os.path.exists(f'{rgb_output_path}{str(sensing_date)}_{str(tile_num)}_RGB.TIF') or overwritten_para:

                            for band_name_temp in ['B2', 'B3', 'B4']:
                                band_file = bf.file_filter(subset_output_path, containing_word_list=[f'{str(sensing_date)}_{str(tile_num)}_{band_name_temp}'])
                                ds_temp = gdal.Open(band_file[0])
                                array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
                                array_temp = ((array_temp / 10000) ** (1/gamma_coef) * 255).astype(np.int8)
                                array_temp[array_temp > 255] = 0
                                RGB_array[band_name_temp] = array_temp

                            dst_ds = gdal.GetDriverByName('GTiff').Create(f'{rgb_output_path}{str(sensing_date)}_{str(tile_num)}_RGB.TIF', xsize=array_temp.shape[1], ysize=array_temp.shape[0], bands=3, eType=gdal.GDT_Byte, options=['COMPRESS=LZW', 'PREDICTOR=2'])
                            dst_ds.SetGeoTransform(ds_temp.GetGeoTransform())  # specify coords
                            dst_ds.SetProjection(ds_temp.GetProjection())  # export coords to file
                            for i, band_name_temp in zip([1,2,3], ['B4', 'B3', 'B2']):
                                dst_ds.GetRasterBand(i).WriteArray(RGB_array[band_name_temp])
                                dst_ds.GetRasterBand(i).SetNoDataValue(0)
                            dst_ds.FlushCache()
                            dst_ds = None

                elif not (index == 'QI' or index == 'all_band' or index == '4visual' or index in self._band_output_list):
                    index_construction_indicator = False
                    if not overwritten_para and not os.path.exists(subset_output_path + file_name + '.TIF'):
                        if index in self._index_exprs_dic.keys():
                            dep_list = self._index_exprs_dic[index][0]
                            dep_list = [str(dep) for dep in dep_list]
                            ds_list = self._check_output_band_statue(dep_list, tiffile_serial_num, **kwargs)
                            if ds_list is not None:
                                try:
                                    if self._sparsify_matrix_factor:
                                        array_list = []
                                        for ds_temp in ds_list:
                                            array_temp = sp.csr_matrix(ds_temp.GetRasterBand(1).ReadAsArray().astype(np.float32))
                                            array_temp[array_temp == 65535] = np.nan
                                            array_list.append(array_temp)
                                    else:
                                        array_list = []
                                        for ds_temp in ds_list:
                                            array_temp = ds_temp.GetRasterBand(1).ReadAsArray().astype(np.float32)
                                            array_temp[array_temp == 65535] = np.nan
                                            array_list.append(array_temp)
                                    output_array = self._index_exprs_dic[index][1](*array_list)
                                    array_list = None
                                except:
                                    index_construction_indicator = True
                            else:
                                index_construction_indicator = True
                        else:
                            raise Exception('Code error concerning the conversion of index expression')

                        if self._combine_band_factor:
                            if array_cube is None:
                                array_cube = np.zeros([output_array.shape[0], output_array.shape[1], len(combine_index_array_list)], dtype=np.int16)

                            if array_cube.shape[0] == output_array.shape[0] and array_cube.shape[1] == output_array.shape[1]:
                                array_cube[:, :, combine_index_list.index(index)] = output_array
                            else:
                                print('consistency issuses')
                                return

                    elif self._combine_band_factor and os.path.exists(subset_output_path + file_name + '.TIF'):
                        time3 = time.time()
                        gdal.Warp('/vsimem/' + file_name + '.TIF', subset_output_path + file_name + '.TIF', outputBounds=band_output_limit, xRes=10, yRes=10)
                        output_ds = gdal.Open('/vsimem/' + file_name + '.TIF')
                        output_array = output_ds.GetRasterBand(1).ReadAsArray()

                        if array_cube is None:
                            array_cube = np.zeros([output_array.shape[0], output_array.shape[1], len(combine_index_array_list)], dtype=np.int16)

                        if array_cube.shape[0] == output_array.shape[0] and array_cube.shape[1] == output_array.shape[1]:
                            array_cube[:, :, combine_index_list.index(index)] = output_array
                        else:
                            print('consistency issuses')
                            return
                        print(f'copy array to list consumes {str(time.time() - time3)}')
                        gdal.Unlink('/vsimem/' + file_name + '.TIF')
                        output_ds = None

                    if index_construction_indicator:
                        return [index, tiffile_serial_num, sensing_date, tile_num]

                    if not os.path.exists(subset_output_path + file_name + '.TIF'):
                        # Output the VI
                        # output_array[np.logical_or(output_array > 1, output_array < -1)] = np.nan
                        if self._size_control_factor is True:
                            output_array[np.isnan(output_array)] = -3.2768
                            output_array = output_array * 10000
                            output_array.astype(int)
                            write_raster(ds_list[0], output_array, '/vsimem/', file_name + '.TIF', raster_datatype=gdal.GDT_Int16)
                            data_type = gdal.GDT_Int16
                        else:
                            write_raster(ds_list[0], output_array, '/vsimem/', file_name + '.TIF', raster_datatype=gdal.GDT_Float32)
                            data_type = gdal.GDT_Float32

                        if self._vi_clip_factor:
                            gdal.Warp('/vsimem/' + file_name + '2.TIF', '/vsimem/' + file_name + '.TIF', xRes=10, yRes=10, cutlineDSName=self.ROI, cropToCutline=True, outputType=data_type, outputBounds=band_output_limit)
                        else:
                            gdal.Warp('/vsimem/' + file_name + '2.TIF', '/vsimem/' + file_name + '.TIF', xRes=10, yRes=10, outputType=data_type, outputBounds=band_output_limit)

                        gdal.Translate(subset_output_path + file_name + '.TIF', '/vsimem/' + file_name + '2.TIF', options=topts)
                        gdal.Unlink('/vsimem/' + file_name + '.TIF')
                        gdal.Unlink('/vsimem/' + file_name + '2.TIF')

                print(f'Finish processing \033[1;31m{index}\033[0m data of \033[3;34m{str(sensing_date)} {str(tile_num)}\033[0m in \033[1;31m{str(time.time() - start_temp)}\033[0m s ({str(tiffile_serial_num + 1)} of {str(self.S2_metadata_size)})')

                # Generate SA map (NEED TO FIX)
                if not os.path.exists(self.output_path + 'ROI_map\\' + self.ROI_name + '_map.npy'):

                    if self._vi_clip_factor:
                        file_list = bf.file_filter(f'{self.output_path}Sentinel2_{self.ROI_name}_index\\QI\\', ['.TIF'], and_or_factor='and')
                    else:
                        file_list = bf.file_filter(f'{self.output_path}Sentinel2_constructed_index\\QI\\', ['.TIF'], and_or_factor='and')
                    bf.create_folder(self.output_path + 'ROI_map\\')
                    ds_temp = gdal.Open(file_list[0])
                    array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
                    array_temp[:, :] = 1
                    write_raster(ds_temp, array_temp, self.cache_folder, 'temp_' + self.ROI_name + '.TIF',
                                 raster_datatype=gdal.GDT_Int16)
                    if retrieve_srs(ds_temp) != self.main_coordinate_system:
                        gdal.Warp('/vsimem/' + 'ROI_map\\' + self.ROI_name + '_map.TIF',
                                  self.cache_folder + 'temp_' + self.ROI_name + '.TIF',
                                  dstSRS=self.main_coordinate_system, cutlineDSName=self.ROI, cropToCutline=True,
                                  xRes=10, yRes=10, dstNodata=-32768)
                    else:
                        gdal.Warp('/vsimem/' + 'ROI_map\\' + self.ROI_name + '_map.TIF',
                                  self.cache_folder + 'temp_' + self.ROI_name + '.TIF', cutlineDSName=self.ROI,
                                  cropToCutline=True, dstNodata=-32768, xRes=10, yRes=10)
                    ds_ROI_array = gdal.Open('/vsimem/' + 'ROI_map\\' + self.ROI_name + '_map.TIF')
                    ds_sa_array = ds_ROI_array.GetRasterBand(1).ReadAsArray()

                    if (ds_sa_array == -32768).all() == False:
                        np.save(self.output_path + 'ROI_map\\' + self.ROI_name + '_map.npy', ds_sa_array)
                        if retrieve_srs(ds_temp) != self.main_coordinate_system:
                            gdal.Warp(self.output_path + 'ROI_map\\' + self.ROI_name + '_map.TIF',
                                      self.cache_folder + 'temp_' + self.ROI_name + '.TIF',
                                      dstSRS=self.main_coordinate_system, cutlineDSName=self.ROI, cropToCutline=True,
                                      xRes=10, yRes=10, dstNodata=-32768)
                        else:
                            gdal.Warp(self.output_path + 'ROI_map\\' + self.ROI_name + '_map.TIF',
                                      self.cache_folder + 'temp_' + self.ROI_name + '.TIF', cutlineDSName=self.ROI,
                                      cropToCutline=True, dstNodata=-32768, xRes=10, yRes=10)
                    gdal.Unlink('/vsimem/' + 'ROI_map\\' + self.ROI_name + '_map.TIF')
                    ds_temp = None
                    ds_ROI_array = None
                    remove_all_file_and_folder(bf.file_filter(self.cache_folder, ['temp', '.TIF'], and_or_factor='and'))

            if self._combine_band_factor:
                time4 = time.time()
                np.savez_compressed(f'{combine_band_folder}{str(sensing_date)}_{str(tile_num)}.npz', array_cube)
                print(f'Write band combination .npz consumes \033[1;31m{str(time.time() - time4)}\033[0m s')
        else:
            print('Caution! the input variable VI_list should be a list and make sure all of them are in Capital Letter')
            sys.exit(-1)
        return [None, None, None, None]

    def check_subset_intergrality(self, indicator, **kwargs):
        if self.ROI_name is None and ('ROI' not in kwargs.keys() and 'ROI_name' not in kwargs.keys()):
            check_path = f'{self.output_path}{str(indicator)}\\'
        elif self.ROI_name is None and ('ROI' in kwargs.keys() or 'ROI_name' in kwargs.keys()):
            self._subset_indicator_process(**kwargs)
            if self.ROI_name is None:
                print()

    def _process_mosaic2seqtif_para(self, **kwargs):
        # Detect whether all the indicators are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in (
                    'inherit_from_logfile', 'ROI', 'ROI_name', 'overwritten_para', 'size_control_factor'):
                raise NameError(f'{kwarg_indicator} is not supported kwargs! Please double check!')

        # process clipped_overwritten_para
        if 'overwritten_para' in kwargs.keys():
            if type(kwargs['overwritten_para']) is bool:
                self._dc_overwritten_para = kwargs['overwritten_para']
            else:
                raise TypeError('Please mention the overwritten_para should be bool type!')
        else:
            self._mosaic_overwritten_para = False

        # process inherit from logfile
        if 'inherit_from_logfile' in kwargs.keys():
            if type(kwargs['inherit_from_logfile']) is bool:
                self._inherit_from_logfile = kwargs['inherit_from_logfile']
            else:
                raise TypeError('Please mention the dc_overwritten_para should be bool type!')
        else:
            self._inherit_from_logfile = False

        # process ROI_NAME
        if 'ROI_name' in kwargs.keys():
            self.ROI_name = kwargs['ROI_name']
        elif self.ROI_name is None and self._inherit_from_logfile:
            self._retrieve_para(['ROI_name'])
        elif self.ROI_name is None:
            raise Exception('Notice the ROI name was missed!')

        # process ROI
        if 'ROI' in kwargs.keys():
            self.ROI = kwargs['ROI']
        elif self.ROI is None and self._inherit_from_logfile:
            self._retrieve_para(['ROI'])
        elif self.ROI is None:
            raise Exception('Notice the ROI was missed!')

        # Retrieve size control factor
        if 'size_control_factor' in kwargs.keys():
            if type(kwargs['size_control_factor']) is bool:
                self._size_control_factor = kwargs['size_control_factor']
            else:
                raise TypeError('Please mention the size_control_factor should be bool type!')
        elif self._inherit_from_logfile:
            self._retrieve_para(['size_control_factor'], protected_var=True)
        else:
            self._size_control_factor = False

    @save_log_file
    def seq_mosaic2seqtif(self, index_list, *args, **kwargs):

        # sequenced process
        for index in index_list:
            self._mosaic2seqtif(index, *args, **kwargs)

    @save_log_file
    def mp_mosaic2seqtif(self, index_list, *args, **kwargs):

        # Define the Chunk size based on the computer
        if 'chunk_size' in kwargs.keys() and type(kwargs['chunk_size']) is int:
            chunk_size = min(os.cpu_count(), kwargs['chunk_size'])
            del kwargs['chunk_size']
        elif 'chunk_size' in kwargs.keys() and kwargs['chunk_size'] == 'auto':
            chunk_size = os.cpu_count()
        else:
            chunk_size = os.cpu_count()

        # MP process
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(chunk_size * Sentinel2_toolbox.configuration['multiprocess_ratio'])) as executor:
            executor.map(self._mosaic2seqtif, index_list, repeat(kwargs))
    
    def _mosaic2seqtif(self, index, *args, **kwargs):
        
        # for the MP
        if args != () and type(args[0]) == dict:
            kwargs = copy.copy(args[0])

        # process para
        self._process_mosaic2seqtif_para(**kwargs)

        # Remove all files which not meet the requirements
        band_list = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11']
        ROI_midname = 'constructed' if self.ROI_name is None else self.ROI_name
        self._mosaic_infr[index + 'input_path'] = self.output_path + f'Sentinel2_{ROI_midname}_index\\all_band\\' if index in band_list else self.output_path + f'Sentinel2_{ROI_midname}_index\\{index}\\'

        # Path check
        if not os.path.exists(self._mosaic_infr[index + 'input_path']):
            raise Exception('Please validate the roi name and vi for datacube output!')

        # Create vrt and seqtif output folder
        self._mosaic_infr[index + 'seq_output_path'] = self.output_path + f'Sentinel2_{ROI_midname}_index\\all_band_seq\\' if index in band_list else self.output_path + f'Sentinel2_{ROI_midname}_index\\{index}_seq\\'
        self._mosaic_infr[index + 'vrt_output_path'] = self.output_path + f'Sentinel2_{ROI_midname}_index\\all_band_vrt\\' if index in band_list else self.output_path + f'Sentinel2_{ROI_midname}_index\\{index}_vrt\\'
        bf.create_folder(self._mosaic_infr[index + 'seq_output_path'])
        bf.create_folder(self._mosaic_infr[index + 'vrt_output_path'])

        # Consistency check
        if len(bf.file_filter(self._mosaic_infr[index + 'input_path'], [f'{index}.TIF'], and_or_factor='and')) != self.S2_metadata_size:
            raise Exception(f'{index} of the {self.ROI_name} is not consistent')

        # Start Mosaic
        print(f'Start mosaic all the tiffiles of \033[0;31m{index}\033[0m.')
        start_time = time.time()

        # Retrieve VAR
        VI_stack_list = bf.file_filter(self._mosaic_infr[index + 'input_path'], [index, '.TIF'], and_or_factor='and')
        VI_stack_list.sort()
        doy_list = [int(filepath_temp.split('\\')[-1][0:8]) for filepath_temp in VI_stack_list]
        doy_list = np.unique(np.array(doy_list))
        doy_list = doy_list.tolist()
        ds_temp = gdal.Open(VI_stack_list[0])
        nodata_value = ds_temp.GetRasterBand(1).GetNoDataValue()
        ds_temp = gdal.Open(self.output_path + 'ROI_map\\' + self.ROI_name + '_map.TIF')
        ulx_temp, xres_temp, xskew_temp, uly_temp, yskew_temp, yres_temp = ds_temp.GetGeoTransform()
        output_bounds = (ulx_temp, uly_temp + yres_temp * ds_temp.RasterYSize, ulx_temp + xres_temp * ds_temp.RasterXSize, uly_temp)

        int_min = np.iinfo(int).min
        i = 0
        while i < len(doy_list):

            t1 = time.time()
            if not os.path.exists(f"{self._mosaic_infr[index + 'seq_output_path']}{str(doy_list[i])}_{index}.TIF") or self._mosaic_overwritten_para:

                tiffile_list_temp = [temp for temp in VI_stack_list if str(doy_list[i]) in temp]
                if len(tiffile_list_temp) == 1:
                    vrt = gdal.BuildVRT(f"{self._mosaic_infr[index + 'vrt_output_path']}{str(doy_list[i])}_{index}.vrt",
                                        tiffile_list_temp, outputBounds=output_bounds, xRes=10, yRes=10,
                                        srcNodata=nodata_value, VRTNodata=nodata_value)
                    vrt = None

                elif len(tiffile_list_temp) > 1:
                    vrt = gdal.BuildVRT(f"{self._mosaic_infr[index + 'vrt_output_path']}{str(doy_list[i])}_{index}.vrt",
                                        tiffile_list_temp, outputBounds=output_bounds, xRes=10, yRes=10,
                                        srcNodata=nodata_value, VRTNodata=nodata_value)
                    vrt = None

                    vrt_tree = etree.parse(f"{self._mosaic_infr[index + 'vrt_output_path']}{str(doy_list[i])}_{index}.vrt")
                    vrt_root = vrt_tree.getroot()
                    vrtband1 = vrt_root.findall(".//VRTRasterBand[@band='1']")[0]

                    vrtband1.set("subClass", "VRTDerivedRasterBand")
                    pixelFunctionType = etree.SubElement(vrtband1, 'PixelFunctionType')
                    pixelFunctionType.text = "find_max"
                    pixelFunctionLanguage = etree.SubElement(vrtband1, 'PixelFunctionLanguage')
                    pixelFunctionLanguage.text = "Python"
                    pixelFunctionCode = etree.SubElement(vrtband1, 'PixelFunctionCode')
                    pixelFunctionCode.text = etree.CDATA("""
                    import numpy as np
                    
                    def find_max(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize, raster_ysize, buf_radius, gt, **kwargs):
                         np.amax(in_ar, axis=0, initial=255, out=out_ar)
            
                    """)

                else:
                    raise Exception('Code error')

                gdal.Translate(f"{self._mosaic_infr[index + 'seq_output_path']}{str(doy_list[i])}_{index}.TIF",
                               f"{self._mosaic_infr[index + 'vrt_output_path']}{str(doy_list[i])}_{index}.vrt",
                               options=topts, noData=nodata_value)

            print(f'Mosaic the \033[0;34m{str(doy_list[i])}\033[0m \033[0;31m{index}\033[0m Tif file in {str(time.time() - t1)[0:5]}s (layer {str(i + 1)} of {str(len(doy_list))})')
            i += 1

        print(f'Finish mosaic all the tiffiles of \033[0;31m{index}\033[0m.')

    def _process_2sdc_para(self, **kwargs):
        # Detect whether all the indicators are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in (
            'inherit_from_logfile', 'ROI', 'ROI_name', 'dc_overwritten_para', 'remove_nan_layer',
            'manually_remove_datelist', 'size_control_factor'):
                raise NameError(f'{kwarg_indicator} is not supported kwargs! Please double check!')

        # process clipped_overwritten_para
        if 'dc_overwritten_para' in kwargs.keys():
            if type(kwargs['dc_overwritten_para']) is bool:
                self._dc_overwritten_para = kwargs['dc_overwritten_para']
            else:
                raise TypeError('Please mention the dc_overwritten_para should be bool type!')
        else:
            self._clipped_overwritten_para = False

        # process inherit from logfile
        if 'inherit_from_logfile' in kwargs.keys():
            if type(kwargs['inherit_from_logfile']) is bool:
                self._inherit_from_logfile = kwargs['inherit_from_logfile']
            else:
                raise TypeError('Please mention the dc_overwritten_para should be bool type!')
        else:
            self._inherit_from_logfile = False

        # process remove_nan_layer
        if 'remove_nan_layer' in kwargs.keys():
            if type(kwargs['remove_nan_layer']) is bool:
                self._remove_nan_layer = kwargs['remove_nan_layer']
            else:
                raise TypeError('Please mention the remove_nan_layer should be bool type!')
        else:
            self._remove_nan_layer = False

        # process remove_nan_layer
        if 'manually_remove_datelist' in kwargs.keys():
            if type(kwargs['manually_remove_datelist']) is list:
                self._manually_remove_datelist = kwargs['manually_remove_datelist']
                self._manually_remove_para = True
            else:
                raise TypeError('Please mention the manually_remove_datelist should be list type!')
        else:
            self._manually_remove_datelist = False
            self._manually_remove_para = False

        # process ROI_NAME
        if 'ROI_name' in kwargs.keys():
            self.ROI_name = kwargs['ROI_name']
        elif self.ROI_name is None and self._inherit_from_logfile:
            self._retrieve_para(['ROI_name'])
        elif self.ROI_name is None:
            raise Exception('Notice the ROI name was missed!')

        # process ROI
        if 'ROI' in kwargs.keys():
            self.ROI = kwargs['ROI']
        elif self.ROI is None and self._inherit_from_logfile:
            self._retrieve_para(['ROI'])
        elif self.ROI is None:
            raise Exception('Notice the ROI was missed!')

        # Retrieve size control factor
        if 'size_control_factor' in kwargs.keys():
            if type(kwargs['size_control_factor']) is bool:
                self._size_control_factor = kwargs['size_control_factor']
            else:
                raise TypeError('Please mention the size_control_factor should be bool type!')
        elif self._inherit_from_logfile:
            self._retrieve_para(['size_control_factor'], protected_var=True)
        else:
            self._size_control_factor = False

    @save_log_file
    def seq_ds2sdc(self, index_list, *args, **kwargs):

        # sequenced process
        for index in index_list:
            self._ds2sdc(index, *args, **kwargs)

    @save_log_file
    def mp_ds2sdc(self, index_list, *args, **kwargs):

        if 'chunk_size' in kwargs.keys() and type(kwargs['chunk_size']) is int:
            chunk_size = min(os.cpu_count(), kwargs['chunk_size'])
            del kwargs['chunk_size']
        # elif 'chunk_size' in kwargs.keys() and kwargs['chunk_size'] == 'auto':
        #     chunk_size = os.cpu_count()
        else:
            chunk_size = os.cpu_count()

        # mp process
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(chunk_size * Sentinel2_toolbox.configuration['multiprocess_ratio'])) as executor:
            executor.map(self._ds2sdc, index_list, repeat(kwargs))

        self._sdc_consistency_check(index_list, **kwargs)

    def _sdc_consistency_check(self, index_list, **kwargs):

        self._process_2sdc_para(**kwargs)

        doy_list = []
        ROI_midname = 'constructed' if self.ROI_name is None else self.ROI_name
        for index_temp in index_list:
            self._dc_infr[index_temp] = self.output_path + f'Sentinel2_{ROI_midname}_datacube\\' + index_temp + '_sequenced_datacube\\'
            if index_temp in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11']:
                self._dc_infr[index_temp + '_input'] = self.output_path + f'Sentinel2_{ROI_midname}_index\\all_band_seq\\'
            else:
                self._dc_infr[index_temp + '_input'] = self.output_path + f'Sentinel2_{ROI_midname}_index\\' + index_temp + '_seq\\'

            doy_temp = np.load(self._dc_infr[index_temp] + 'doy.npy')
            for i in doy_temp:
                if i not in doy_list:
                    doy_list.append(i)

        for index_temp in index_list:
            doy_temp = np.load(self._dc_infr[index_temp] + 'doy.npy')
            doy_fail = []
            for q in doy_list:
                if q not in doy_temp:
                    doy_fail.append(q)

            for doy_fail_temp in doy_fail:
                files = bf.file_filter(self._dc_infr[index_temp + '_input'], containing_word_list=[str(doy_fail_temp), str(index_temp), '.TIF'], and_or_factor='and')
                for file in files:
                    try:
                        os.remove(file)
                    except:
                        raise Exception(f'Please manually delete {file}')

            self._mosaic2seqtif(index_temp)
            ND_matrix = NDSparseMatrix()
            ND_matrix.load(self._dc_infr[index_temp] + index_temp + '_sequenced_datacube\\')
            for doy_fail_temp in doy_fail:
                files = bf.file_filter(self._dc_infr[index_temp + '_input'], containing_word_list=[str(doy_fail_temp), str(index_temp), '.TIF'], and_or_factor='and')
                if len(files) != 1:
                    raise Exception('Code Error in sdc consistency check')
                else:
                    ds_temp = gdal.Open(files[0])
                    array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
                    nodata_value = ds_temp.GetRasterBand(1).GetNoDataValue()

                    if self._size_control_factor and index_temp not in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11']:
                        array_temp = array_temp.astype(np.uint16) + 32768
                    elif np.isnan(nodata_value):
                        array_temp[np.isnan(array_temp)] = 0
                    else:
                        array_temp[array_temp == nodata_value] = 0

                ND_matrix.append(ND_matrix._matrix_type(array_temp), doy_fail_temp)

            if len(doy_list) == ND_matrix.shape[2]:
                ND_matrix.save(self._dc_infr[index_temp] + index_temp + '_sequenced_datacube\\', overwritten_para=False)
                np.save(self._dc_infr[index_temp] + 'doy.npy', doy_list)
            else:
                raise Exception('Code Error in sdc consistency check')

    def _ds2sdc(self, index, *args, **kwargs):

        # for the MP
        if args != () and type(args[0]) == dict:
            kwargs = copy.copy(args[0])

        # process clip parameter
        self._process_2sdc_para(**kwargs)

        # Inputfile path
        band_list = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11']
        ROI_midname = 'constructed' if self.ROI_name is None else self.ROI_name
        self._dc_infr[index + 'input_path'] = self.output_path + f'Sentinel2_{ROI_midname}_index\\all_band_seq\\' if index in band_list else self.output_path + f'Sentinel2_{ROI_midname}_index\\{index}_seq\\'

        # path check
        if not os.path.exists(self._dc_infr[index + 'input_path']):
            raise Exception('Please validate the roi name and vi for datacube output!')

        # Create output folder
        self._dc_infr[index] = self.output_path + f'Sentinel2_{ROI_midname}_datacube\\' + index + '_sequenced_datacube\\'
        bf.create_folder(self._dc_infr[index])

        if len(bf.file_filter(self._dc_infr[index + 'input_path'], [f'{index}.TIF'], and_or_factor='and', exclude_word_list=['xml', 'aux'])) != np.unique(np.array(self.S2_metadata['Sensing_Date'])).shape[0]:
            raise ValueError(f'{index} of the {self.ROI_name} is not consistent')

        print(f'Start output the Sentinel2 dataset of \033[0;31m{index}\033[0m to sequenced datacube.')
        start_time = time.time()

        if self._dc_overwritten_para or not os.path.exists(self._dc_infr[index] + 'doy.npy') or not os.path.exists(self._dc_infr[index] + 'metadata.json'):

            sa_map = np.load(bf.file_filter(self.output_path + 'ROI_map\\', [self.ROI_name, '.npy'], and_or_factor='and')[0], allow_pickle=True)
            if self.ROI_name is None:
                print('Start processing ' + index + ' datacube.')
                metadata_dic = {'ROI_name': None, 'index': index, 'Datatype': 'float', 'ROI': None, 'ROI_array': None, 'ROI_tif': None,
                              'sdc_factor': True, 'coordinate_system': self.main_coordinate_system, 'size_control_factor': self._size_control_factor,
                              'oritif_folder': self._dc_infr[index + 'input_path'], 'dc_group_list': None, 'tiles': None}
            else:
                print('Start processing ' + index + ' datacube of the ' + self.ROI_name + '.')
                metadata_dic = {'ROI_name': self.ROI_name, 'index': index, 'Datatype': 'float', 'ROI': self.ROI, 'ROI_array': self.output_path + 'ROI_map\\' + self.ROI_name + '_map.npy', 'ROI_tif': self.output_path + 'ROI_map\\' + self.ROI_name + '_map.TIF',
                              'sdc_factor': True, 'coordinate_system': self.main_coordinate_system, 'size_control_factor': self._size_control_factor,
                              'oritif_folder': self._dc_infr[index + 'input_path'], 'dc_group_list': None, 'tiles': None}

            # Retrieve Var
            VI_stack_list = bf.file_filter(self._dc_infr[index + 'input_path'], [f'{index}.TIF'], and_or_factor='and', exclude_word_list=['aux', 'xml'])
            VI_stack_list.sort()
            doy_list = [int(filepath_temp.split('\\')[-1][0:8]) for filepath_temp in VI_stack_list]
            doy_list = np.unique(np.array(doy_list))
            doy_list = list(doy_list)
            nodata_value = gdal.Open(VI_stack_list[0])
            nodata_value = nodata_value.GetRasterBand(1).GetNoDataValue()

            # Evaluate the size and sparsity of sdc
            mem = psutil.virtual_memory()
            dc_max_size = int((mem.free) * 0.90)
            cols, rows = sa_map.shape[1], sa_map.shape[0]
            sparsify = np.sum(sa_map == -32768) / (sa_map.shape[0] * sa_map.shape[1])
            _huge_matrix = True if len(doy_list) * cols * rows * 2 > dc_max_size else False
            _sparse_matrix = True if sparsify > 0.9 else False

            # Generate the cube
            if _huge_matrix:
                if _sparse_matrix:

                    i = 0
                    data_cube = NDSparseMatrix()
                    data_valid_array = np.zeros([len(doy_list)], dtype=int)
                    while i < len(doy_list):

                        try:
                            t1 = time.time()
                            if not os.path.exists(f"{self._dc_infr[index + 'input_path']}{str(doy_list[i])}_{index}.TIF"):
                                raise Exception(f'The {str(doy_list[i])}_{index} is not properly generated!')
                            else:
                                array_temp = gdal.Open(f"{self._dc_infr[index + 'input_path']}{str(doy_list[i])}_{index}.TIF")
                                array_temp = array_temp.GetRasterBand(1).ReadAsArray()

                            if self._size_control_factor and index not in band_list:
                                array_temp = array_temp.astype(int) + 32768
                                zoffset = 32768
                            elif np.isnan(nodata_value):
                                array_temp[np.isnan(array_temp)] = 0
                                zoffset = 0
                            else:
                                array_temp[array_temp == nodata_value] = 0
                                zoffset = 0

                            nodata_value = 0
                            sm_temp = sm.csr_matrix(array_temp.astype(np.uint16))
                            data_cube.append(sm_temp, name=doy_list[i])
                            data_valid_array[i] = 1 if sm_temp.data.shape[0] == 0 else 0

                            print(f'Assemble the {str(doy_list[i])} into the sdc using {str(time.time()-t1)[0:5]}s (layer {str(i)} of {str(len(doy_list))})')
                            i += 1
                        except:
                            error_inf = traceback.format_exc()
                            print(error_inf)

                    # remove nan layer
                    if self._manually_remove_para is True and self._manually_remove_datelist is not None:
                        i_temp = 0
                        while i_temp < len(doy_list):
                            if doy_list[i_temp] in self._manually_remove_datelist:
                                data_valid_array[i] = 1
                                self._manually_remove_datelist.remove(doy_list[i_temp])
                            i_temp += 1

                    elif self._manually_remove_para is True and self._manually_remove_datelist is None:
                        raise ValueError('Please correctly input the manual input date list')

                    if self._remove_nan_layer or self._manually_remove_para:
                        i_temp = 0
                        while i_temp < len(doy_list):
                            if data_valid_array[i_temp]:
                                if doy_list[i_temp] in data_cube.SM_namelist:
                                    data_cube.remove_layer(doy_list[i_temp])
                                doy_list.remove(doy_list[i_temp])
                                data_valid_array = np.delete(data_valid_array, i_temp, 0)
                                i_temp -= 1
                            i_temp += 1

                    # Save the sdc
                    np.save(self._dc_infr[index] + f'doy.npy', doy_list)
                    bf.create_folder(f'{self._dc_infr[index]}{str(index)}_sequenced_datacube\\')
                    data_cube.save(f'{self._dc_infr[index]}{str(index)}_sequenced_datacube\\')

                else:
                    pass

            else:
                i = 0
                data_cube_list = []
                data_valid_array = np.zeros([len(doy_list)], dtype=int)
                while i < len(doy_list):

                    t1 = time.time()
                    if not os.path.exists(f"{self._dc_infr[index + 'input_path']}{str(doy_list[i])}_{index}.TIF"):
                        raise Exception(f'The {str(doy_list[i])}_{index} is not properly generated!')
                    else:
                        ds_temp = gdal.Open(f"{self._dc_infr[index + 'input_path']}{str(doy_list[i])}_{index}.TIF")
                        array_temp = ds_temp.GetRasterBand(1).ReadAsArray()

                    data_cube_list.append(array_temp)
                    if self._remove_nan_layer:
                        data_valid_array[i] = 1 if np.array([array_temp == nodata_value]).all() else 0
                    print(f'Assemble the {str(doy_list[i])} into the sdc using {str(time.time() - t1)[0:5]}s (layer {str(i)} of {str(len(doy_list))})')
                    i += 1

                # remove nan layer
                if self._manually_remove_para is True and self._manually_remove_datelist is not None:
                    if self._remove_nan_layer:
                        i_temp = 0
                        while i_temp < len(doy_list):
                            if doy_list[i_temp] in self._manually_remove_datelist:
                                data_valid_array[i] = 1
                                self._manually_remove_datelist.remove(doy_list[i_temp])
                            i_temp += 1
                    else:
                        data_valid_array = np.zeros([len(doy_list)], dtype=int)
                        i_temp = 0
                        while i_temp < len(doy_list):
                            if doy_list[i_temp] in self._manually_remove_datelist:
                                data_valid_array[i] = 1
                                self._manually_remove_datelist.remove(doy_list[i_temp])
                            i_temp += 1

                elif self._manually_remove_para is True and self._manually_remove_datelist is None:
                    raise ValueError('Please correctly input the manual input date list')

                st = time.time()
                if np.sum(data_valid_array) > 0:
                    pos_list = [_ for _ in range(data_valid_array.shape[0]) if data_valid_array[_] == 1]
                    data_cube_list_new = [_ for idx, _ in enumerate(data_cube_list) if idx not in pos_list]
                    doy_list_new = [_ for idx, _ in enumerate(doy_list) if idx not in pos_list]
                    doy_list = doy_list_new
                    data_cube_list = None
                    data_cube = np.stack(data_cube_list_new, axis=2)
                else:
                    data_cube = np.stack(data_cube_list, axis=2)
                print(f'Stack all the layers into the sdc using {str(time.time() - st)[0:5]}s (total layer of {str(len(doy_list))})')

                # Save the sdc
                zoffset = 0
                np.save(self._dc_infr[index] + f'doy.npy', doy_list)
                np.save(f'{self._dc_infr[index]}{str(index)}_sequenced_datacube.npy', data_cube)

            # Save the metadata dic
            metadata_dic['sparse_matrix'], metadata_dic['huge_matrix'] = _sparse_matrix, _huge_matrix
            metadata_dic['Zoffset'], metadata_dic['Nodata_value'] = zoffset, nodata_value
            with open(self._dc_infr[index] + 'metadata.json', 'w') as js_temp:
                json.dump(metadata_dic, js_temp)

        print(f'Finished writing the sdc in \033[1;31m{str(time.time() - start_time)} s\033[0m.')


class Sentinel2_dc(object):
    def __init__(self, dc_filepath, work_env=None):

        # Check the dcfile path
        self.dc_filepath = bf.Path(dc_filepath).path_name

        # Def key var
        self.ROI_name, self.ROI, self.ROI_tif = None, None, None
        self.index, self.Datatype, self.coordinate_system = None, None, None
        self.dc_group_list, self.tiles = None, None
        self.sdc_factor, self.sparse_matrix, self.size_control_factor, self.huge_matrix = False, False, False, False
        self.Zoffset, self.Nodata_value = None, None

        # Check work env
        if work_env is not None:
            self._work_env = Path(work_env).path_name
        else:
            self._work_env = Path(os.path.dirname(os.path.dirname(self.dc_filepath))).path_name
        self.root_path = Path(os.path.dirname(os.path.dirname(self._work_env))).path_name

        # Define the basic var name
        self._fund_factor = ('ROI_name', 'index', 'Datatype', 'ROI', 'ROI_array', 'sdc_factor',
                             'coordinate_system', 'oritif_folder', 'ROI_tif', 'sparse_matrix',
                             'huge_matrix', 'size_control_factor', 'dc_group_list', 'tiles', 'Zoffset', 'Nodata_value')

        # Read the metadata file
        metadata_file = bf.file_filter(self.dc_filepath, ['metadata.json'])
        if len(metadata_file) == 0:
            raise ValueError('There has no valid sdc or the metadata file of the sdc was missing!')
        elif len(metadata_file) > 1:
            raise ValueError('There has more than one metadata file in the dir')
        else:
            try:
                with open(metadata_file[0]) as js_temp:
                    dc_metadata = json.load(js_temp)
                if not isinstance(dc_metadata, dict):
                    raise Exception('Please make sure the metadata file is a dictionary constructed in python!')
                else:
                    for dic_name in self._fund_factor:
                        if dic_name not in dc_metadata.keys():
                            raise Exception(f'The {dic_name} is not in the dc metadata, double check!')
                        else:
                            self.__dict__[dic_name] = dc_metadata[dic_name]
            except:
                print(traceback.format_exc())
                raise Exception('Something went wrong when reading the metadata!')

        start_time = time.time()
        print(f'Start loading the Sentinel2 dc of \033[1;31m{self.index}\033[0m in the \033[1;34m{self.ROI_name}\033[0m')

        # Read doy or date file of the Datacube
        try:
            if self.sdc_factor is True:
                # Read doylist
                doy_file = bf.file_filter(self.dc_filepath, ['doy.npy'])
                if len(doy_file) == 0:
                    raise ValueError('There has no valid doy file or file was missing!')
                elif len(doy_file) > 1:
                    raise ValueError('There has more than one doy file in the dc dir')
                else:
                    sdc_doylist = np.load(doy_file[0], allow_pickle=True)
                    self.sdc_doylist = [int(sdc_doy) for sdc_doy in sdc_doylist]
            else:
                raise TypeError('Please construct a sdc (running the latest version)')
        except:
            raise Exception('Something went wrong when reading the doy list!')

        # Read datacube
        try:
            if self.sparse_matrix and self.huge_matrix:
                if os.path.exists(self.dc_filepath + f'{self.index}_sequenced_datacube\\'):
                    self.dc = NDSparseMatrix().load(self.dc_filepath + f'{self.index}_sequenced_datacube\\')
                else:
                    raise Exception('Please double check the code if the sparse huge matrix is generated properly')
            elif not self.huge_matrix:
                self.dc_filename = bf.file_filter(self.dc_filepath, ['sequenced_datacube.npy'])
                if len(self.dc_filename) == 0:
                    raise ValueError('There has no valid dc or the dc was missing!')
                elif len(self.dc_filename) > 1:
                    raise ValueError('There has more than one date file in the dc dir')
                else:
                    self.dc = np.load(self.dc_filename[0], allow_pickle=True)
            elif self.huge_matrix and not self.sparse_matrix:
                self.dc_filename = bf.file_filter(self.dc_filepath, ['sequenced_datacube', '.npy'], and_or_factor='and')
        except:
            raise Exception('Something went wrong when reading the datacube!')

        # Autotrans sparse matrix
        if self.sparse_matrix and self.dc._matrix_type == sm.coo_matrix:
            self._autotrans_sparse_matrix()

        # Size check
        if len(self.sdc_doylist) != self.dc.shape[2]:
            if self.sparse_matrix:
                self.sdc_doylist = self.dc.SM_namelist
                self.save(self.dc_filepath)
                self.__init__(self.dc_filepath)
            else:
                raise ValueError('The dc and doy is not consistent!')

        # Size calculation and shape definition
        self.dc_XSize, self.dc_YSize, self.dc_ZSize = self.dc.shape[1], self.dc.shape[0], self.dc.shape[2]

        print(f'Finish loading the Sentinel2 dc of \033[1;31m{self.index}\033[0m for the \033[1;34m{self.ROI_name}\033[0m using \033[1;31m{str(time.time() - start_time)}\033[0ms')

    def __sizeof__(self):
        return self.dc.__sizeof__() + self.sdc_doylist.__sizeof__()

    def _autotrans_sparse_matrix(self):

        if not isinstance(self.dc, NDSparseMatrix):
            raise TypeError('The autotrans sparse matrix is specified for the NDsm!')

        for _ in self.dc.SM_namelist:
            if isinstance(self.dc.SM_group[_], sm.coo_matrix):
                self.dc.SM_group[_] = sm.csr_matrix(self.dc.SM_group[_])
        self.dc._update_size_para()
        self.dc._matrix_type = sm.csr_matrix
        self.save(self.dc_filepath)

    def save(self, output_path: str):
        start_time = time.time()
        print(f'Start saving the Sentinel2 dc of \033[1;31m{self.index}\033[0m in the \033[1;34m{self.ROI_name}\033[0m')

        if not os.path.exists(output_path):
            bf.create_folder(output_path)
        output_path = bf.Path(output_path).path_name

        metadata_dic = {'ROI_name': self.ROI_name, 'index': self.index, 'Datatype': self.Datatype, 'ROI': self.ROI, 'ROI_array': self.ROI_array,
                        'ROI_tif': self.ROI_tif, 'sdc_factor': self.sdc_factor, 'coordinate_system': self.coordinate_system,
                        'sparse_matrix': self.sparse_matrix, 'huge_matrix': self.huge_matrix, 'size_control_factor': self.size_control_factor,
                        'oritif_folder': self.oritif_folder, 'dc_group_list': self.dc_group_list, 'tiles': self.tiles,
                        'Zoffset': self.Zoffset, 'Nodata_value': self.Nodata_value}

        doy = self.sdc_doylist
        np.save(f'{output_path}doy.npy', doy)
        with open(f'{output_path}metadata.json', 'w') as js_temp:
            json.dump(metadata_dic, js_temp)

        if self.sparse_matrix:
            self.dc.save(f'{output_path}{str(self.index)}_sequenced_datacube\\')
        else:
            np.save(f'{output_path}{str(self.index)}_sequenced_datacube.npy', self.dc)

        print(f'Finish saving the Sentinel2 dc of \033[1;31m{self.index}\033[0m for the \033[1;34m{self.ROI_name}\033[0m using \033[1;31m{str(time.time() - start_time)}\033[0ms')

