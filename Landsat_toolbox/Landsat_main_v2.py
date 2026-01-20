import concurrent.futures
from itertools import repeat
from .built_in_index import built_in_index
from datetime import date
import glob
from RSDatacube.utils import *
from Landsat_toolbox.utils import *
import Landsat_toolbox
import re
import shutil
from collections import defaultdict


global topts
topts = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'PREDICTOR=2'])
gdal.SetConfigOption('GDAL_VRT_ENABLE_PYTHON', 'YES')
gdal.UseExceptions()

# Set np para
np.seterr(divide='ignore', invalid='ignore')


class Landsat_l2_ds(object):

    def __init__(self, original_file_path, work_env: str = None):
        # Define var
        self.ori_folder = original_file_path
        self.Landsat_metadata = None
        self.orifile_list = []
        self.date_list = []
        self.Landsat_metadata_size = np.nan

        # Define key variables for index construction
        self._size_control_factor = False
        self._cloud_removal_para = False
        self._overwritten_para = False
        self._scan_line_correction = False
        self.vi_output_path_dic = {}
        self.construction_issue_factor = False
        self.construction_failure_files = []
        self._index_exprs_dic = {}
        self._oli_harmonisation = True

        # Define key var for VI clip
        self.clipped_vi_path_dic = {}
        self.main_coordinate_system = None
        self.ROI = None
        self.ROI_name = None

        # Define key var for to datacube
        self._dc_infr = {}
        self._dc_overwritten_para = False
        self._inherit_from_logfile = None
        self._remove_nan_layer = False
        self._manually_remove_para = False
        self._manually_remove_datelist = None
        self._skip_invalid_file = False

        # Remove all the duplicated data
        dup_data = bf.file_filter(self.ori_folder, [f'.{str(_)}.zip' for _ in range(10)], and_or_factor='and')
        for dup in dup_data:
            os.remove(dup)

        # Initialise the work environment
        if work_env is None:
            try:
                self._work_env = os.path.dirname(os.path.dirname(self.ori_folder)) + '\\'
            except:
                print('There has no base dir for the ori_folder and the ori_folder will be treated as the work env')
                self._work_env = self.ori_folder
        else:
            self._work_env = bf.Path(work_env).path_name

        # Create output path
        self.unzipped_folder = self._work_env + 'Landsat_original_tiffile\\'
        self.log_filepath = f'{self._work_env}Log\\'
        bf.create_folder(self.log_filepath)
        bf.create_folder(self.unzipped_folder)

        # Create cache path
        self.cache_folder = self._work_env + 'cache\\'
        self.trash_folder = self._work_env + 'trash\\'
        bf.create_folder(self.cache_folder)
        bf.create_folder(self.trash_folder)

        # Constant
        self._band_output_list = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'QA_PIXEL', 'gap_mask']
        self._all_supported_index_list = ['RGB', 'QI', 'all_band', '4visual', 'NDVI', 'MNDWI', 'EVI', 'EVI2', 'OSAVI',
                                          'GNDVI', 'NDVI_RE', 'NDVI_RE2', 'AWEI', 'AWEInsh', 'SVVI', 'TCGREENESS',
                                          'BLUE', 'GREEN', 'RED', 'NIR', 'SWIR', 'SWIR2', 'TIR']
        self._band_tab = {'LE07_bandnum': ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'),
                          'LE07_bandname': ('BLUE', 'GREEN', 'RED', 'NIR', 'SWIR', 'TIR', 'SWIR2'),
                          'LT04_bandnum': ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'),
                          'LT04_bandname': ('BLUE', 'GREEN', 'RED', 'NIR', 'SWIR', 'TIR', 'SWIR2', 'PAN'),
                          'LT05_bandnum': ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'),
                          'LT05_bandname': ('BLUE', 'GREEN', 'RED', 'NIR', 'SWIR', 'TIR', 'SWIR2', 'PAN'),
                          'LC08_bandnum': ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B10'),
                          'LC08_bandname': ('AER', 'BLUE', 'GREEN', 'RED', 'NIR', 'SWIR', 'SWIR2', 'PAN', 'TIR')}
        self._band_sup = ('AER', 'BLUE', 'GREEN', 'RED', 'NIR', 'SWIR', 'SWIR2', 'PAN', 'TIR')
        self._OLI2ETM_harmonised_factor = {"B2_band_OLS": (0.8850, 0.0183),
                                           "B3_band_OLS": (0.9317, 0.0123),
                                           "B4_band_OLS": (0.9372, 0.0123),
                                           "B5_band_OLS": (0.8339, 0.0448),
                                           "B6_band_OLS": (0.8639, 0.0306),
                                           "B7_band_OLS": (0.9165, 0.0116),
                                           "B1_band_OLS": (1.0000, 0.0000),
                                           "B8_band_OLS": (1.0000, 0.0000),
                                           "B10_band_OLS": (1.000, 0.0000)}

    def save_log_file(func):
        def wrapper(self, *args, **kwargs):

            #########################################################################
            # Document the log file and para file
            # The difference between log file and para file is that the log file contains the information for each run/debug
            # While the para file only comprised of the parameter for the latest run/debug
            #########################################################################

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
            para_temp = ['#' * 70 + '\n', f'Process Func: {func.__name__}\n', f'Start time: {c_time}\n', f'End time: {time.ctime()}\n', f'Total processing time: {str(time.time() - time_start)}\n']
            para_temp.extend(args_list)
            para_temp.extend(kwargs_list)
            para_temp.append('#' * 70 + '\n')

            log_temp.extend(args_list)
            log_temp.extend(kwargs_list)
            log_file.writelines(log_temp)
            for func_key, func_processing_name in zip(['metadata', 'index', 'clip', 'datacube'], ['constructing metadata', 'executing construction', 'executing clip', '2dc']):
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

    @save_log_file
    def construct_metadata(self, unzipped_para=False):
        print('----------------------- Start the construction of the Landsat ds metadata -----------------------')

        # Construct corrupted file folder
        corrupted_file_folder = os.path.join(self._work_env, 'Corrupted_zip_file\\')
        bf.create_folder(corrupted_file_folder)
        bf.create_folder(f'{corrupted_file_folder}all_clear\\')

        # Read the ori Landsat zip file
        self.orifile_list = bf.file_filter(self.ori_folder, ['.tar', 'L2SP'], and_or_factor='and', subfolder_detection=True)
        if len(self.orifile_list) == 0:
            raise ValueError('There has no valid Landsat L2 data in the original folder!')

        # Drop duplicate files
        file_groups = defaultdict(list)

        # 用正则匹配 base_name 和是否含括号后缀
        pattern = re.compile(r"^(.*?)(\(\d+\))?\.tar$", re.IGNORECASE)
        for f in self.orifile_list:
            f_clean = f.replace(" ", "")  # 去除所有空格
            match = pattern.match(f_clean)
            if match:
                base_name = match.group(1).strip()
                file_groups[base_name].append(f)

        moved, renamed, skipped = [], [], []
        for base_name, versions in file_groups.items():
            full_paths = [os.path.join(self.ori_folder, f) for f in versions]
            sizes = [(f, os.path.getsize(f)) for f in full_paths]
            sizes.sort(key=lambda x: x[1], reverse=True)
            target_path = os.path.join(self.ori_folder, base_name + ".tar")

            if len(versions) == 1 and not bool(re.search(r"\(\d+\)", versions[0])):
                continue

            # Case 1: 只有一个带括号版本 → 去括号重命名
            elif len(sizes) == 1 and bool(re.search(r"\(\d+\)", versions[0])):
                src_path = sizes[0][0]
                try:
                    os.rename(src_path, target_path)
                    renamed.append((src_path, target_path))
                except Exception as e:
                    skipped.append((src_path, str(e)))
                continue

            # Case 2: 多个版本，保留最大，重命名为 base_name.tar
            else:
                keep_file = sizes[0][0]  # 最大文件

                # Step 1: 先将其余小文件移到 corrupted
                for f, _ in sizes[1:]:
                    try:
                        shutil.move(f, os.path.join(corrupted_file_folder, os.path.basename(f)))
                        moved.append(f)
                    except Exception as e:
                        skipped.append((f, str(e)))

                # Step 2: 再处理最大文件重命名
                try:
                    if os.path.exists(target_path) and os.path.abspath(target_path) != os.path.abspath(keep_file):
                        os.remove(target_path)
                    if os.path.abspath(keep_file) != os.path.abspath(target_path):
                        os.rename(keep_file, target_path)
                        renamed.append((keep_file, target_path))
                except Exception as e:
                    skipped.append((keep_file, str(e)))

        # Process invalid files
        if len(skipped) > 0:
            print(skipped)
            raise Exception('Above file was unable to move, should manually process！')

        ##################################################################
        # Landsat 9 and Landsat 7 duplicate
        # SINCE the Landsat 9 was not struggled with the scan line corrector issue
        # It has a priority than the Landsat 7
        ##################################################################

        date_tile_combined_list = [filepath_temp.split('\\')[-1].split('.')[0].split('_L2SP_')[-1][0: 15] for filepath_temp in self.orifile_list]
        l79_dup_list = []
        for date_tile_temp in date_tile_combined_list:
            if date_tile_combined_list.count(date_tile_temp) == 2:
                l79_dup_list.append(date_tile_temp)
            elif date_tile_combined_list.count(date_tile_temp) > 2:
                raise Exception(f'There are more than 2 files sharing the same sensing date {str(date_tile_temp)}. Check it manually!')
        l79_dup_list = list(set(l79_dup_list))

        for date_tile_temp in l79_dup_list:
            l79_list = bf.file_filter(self.ori_folder, [str(date_tile_temp)])
            if len(l79_list) == 2 and (('LE07' in l79_list[0] and 'LC09' in l79_list[1]) or ('LE07' in l79_list[1] and 'LC09' in l79_list[0])):
                l7_file = [_ for _ in l79_list if 'LE07' in _][0]
                l7_file_name = l7_file.split('\\')[-1]
                try:
                    os.rename(l7_file, f'{corrupted_file_folder}{l7_file_name}')
                except:
                    raise Exception(f'The Landsat 7 duplicate file {str(l7_file)} is not processed!')
            elif len(l79_list) == 2 and (('LE07' in l79_list[0] and 'LC08' in l79_list[1]) or ('LE07' in l79_list[1] and 'LC08' in l79_list[0])):
                l7_file = [_ for _ in l79_list if 'LE07' in _][0]
                l7_file_name = l7_file.split('\\')[-1]
                try:
                    os.rename(l7_file, f'{corrupted_file_folder}{l7_file_name}')
                except:
                    raise Exception(f'The Landsat 7 duplicate file {str(l7_file)} is not processed!')
            elif len(l79_list) == 2 and (('LC09' in l79_list[0] and 'LC08' in l79_list[1]) or ('LC09' in l79_list[1] and 'LC08' in l79_list[0])):
                l7_file = [_ for _ in l79_list if 'LC08' in _][0]
                l7_file_name = l7_file.split('\\')[-1]
                try:
                    os.rename(l7_file, f'{corrupted_file_folder}{l7_file_name}')
                except:
                    raise Exception(f'The Landsat 7 duplicate file {str(l7_file)} is not processed!')
            elif len(l79_list) == 1 and 'LC09' in l79_list[0]:
                pass
            else:
                raise Exception(f'Something went wrong with the Landsat file under {date_tile_temp}!')

        # Update the ori Landsat zip file
        self.orifile_list = bf.file_filter(self.ori_folder, ['.tar', 'L2SP'], and_or_factor='and', subfolder_detection=True)
        if len(self.orifile_list) == 0:
            raise ValueError('There has no valid Landsat L2 data in the original folder!')

        # Generate metadata
        if (os.path.exists(self._work_env + 'Metadata.xlsx') and len(self.orifile_list) != pd.read_excel(self._work_env + 'Metadata.xlsx').shape[0]) or not os.path.exists(self._work_env + 'Metadata.xlsx'):
            File_path, FileID, Sensor_type, Tile, Date, Tier_level = ([] for _ in range(6))
            result_list = []

            with tqdm(total=len(self.orifile_list), desc=f'Obtain the metadata of Landsat dataset', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
                # Use ProcessPoolExecutor to execute the function in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:

                    # Submit tasks to the executor
                    futures = [executor.submit(unzip_Landsat_tarfile, file_, path_) for file_, path_ in zip(self.orifile_list, repeat(self.unzipped_folder))]

                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()  # Get the result (optional)
                        result_list.append(result) # Print the result (optional)
                        pbar.update(1)

            # Process results after all futures have completed
            for result in result_list:
                success, sensor_type, file_id, tile, date, tier_level, file_path = result
                if success:
                    Sensor_type.append(sensor_type)
                    FileID.append(file_id)
                    Tile.append(tile)
                    Date.append(date)
                    Tier_level.append(tier_level)
                    File_path.append(file_path)
                else:
                    print(f"Failed to process file: {file_path}")
                    shutil.move(file_path, corrupted_file_folder + file_path[file_path.find('L2S') - 5:])

            File_metadata = pandas.DataFrame({'File_Path': File_path, 'FileID': FileID, 'Sensor_Type': Sensor_type, 'Tile_Num': Tile, 'Date': Date, 'Tier_Level': Tier_level})
            File_metadata.to_excel(self._work_env + 'Metadata.xlsx')

        self.Landsat_metadata = pandas.read_excel(self._work_env + 'Metadata.xlsx')
        self.Landsat_metadata.sort_values(by=['Date'], ascending=True)

        # Move all Tier2 file to T2 folder
        Landsat_t2_metadata = self.Landsat_metadata.loc[self.Landsat_metadata['Tier_Level'] == 'T2'].reset_index(drop=True)
        bf.create_folder(corrupted_file_folder + 't2_file\\')
        for __ in range(Landsat_t2_metadata.shape[0]):
            shutil.move(Landsat_t2_metadata['File_Path'][__], corrupted_file_folder + 't2_file\\' + Landsat_t2_metadata['FileID'][__] + '.tar')

        # Resort the metadata
        self.Landsat_metadata = self.Landsat_metadata.loc[self.Landsat_metadata['Tier_Level'] == 'T1']
        self.Landsat_metadata = self.Landsat_metadata.reset_index(drop=True)
        self.Landsat_metadata_size = self.Landsat_metadata.shape[0]
        self.date_list = self.Landsat_metadata['Date'].drop_duplicates().sort_values().tolist()

        # Remove all files which not meet the requirements
        eliminating_all_not_required_file(self.unzipped_folder)
        print('---------------------- Finish the construction of the Landsat ds metadata -----------------------')

        # Remove the unzipped tiffile of corrupted files before
        bf.create_folder(corrupted_file_folder + 'all_clear\\')
        corrupted_file_list = bf.file_filter(corrupted_file_folder, ['.tar', 'L2SP'], and_or_factor='and')
        for corrupted_filename in [_.split('\\')[-1].split('.tar')[0] for _ in corrupted_file_list]:
            folder = [landsat_t for landsat_t in os.listdir(self._work_env) if 'Landsat' in landsat_t]
            for fold_temp in folder:
                if 'original' in fold_temp:
                    unzipped_corrupted_files = bf.file_filter(f'{self._work_env}{fold_temp}\\', [corrupted_filename])
                    for temp in unzipped_corrupted_files:
                        try:
                            os.remove(temp)
                        except:
                            raise Exception(f'Failed to remove the corrupted unzipped tiffile {temp}!')
                else:
                    Tile_temp = (corrupted_filename[corrupted_filename.find('L2S') + 5: corrupted_filename.find('L2S') + 11])
                    Date_temp = (corrupted_filename[corrupted_filename.find('L2S') + 12: corrupted_filename.find('L2S') + 20])
                    unzipped_corrupted_files = bf.file_filter(f'{self._work_env}{fold_temp}\\', [str(Tile_temp), str(Date_temp)], and_or_factor='and', subfolder_detection=True)
                    for temp in unzipped_corrupted_files:
                        try:
                            os.remove(temp)
                        except:
                            raise Exception(f'Failed to remove the corrupted unzipped tiffile {temp}!')

            shutil.move(corrupted_file_folder + corrupted_filename + '.tar', corrupted_file_folder + 'all_clear\\' + corrupted_filename + '.tar')
    
    def _check_metadata_availability(self):
        # Check metadata availability
        if self.Landsat_metadata is None:
            try:
                self.construct_metadata()
            except:
                raise Exception('Please manually generate the Landsat metadata before further processing!')
    
    def _process_index_construction_para(self, **kwargs):
        # Detect whether all the indicators are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in ('ROI', 'ROI_name', 'size_control_factor', 'cloud_removal_para', 'scan_line_correction',
                                       'main_coordinate_system', 'overwritten_factor', 'metadata_range', 'issued_files', 'harmonising_data'):
                raise NameError(f'{kwarg_indicator} is not supported kwargs! Please double check!')

        # process harmonising data parameter
        if 'harmonising_data' in kwargs.keys():
            if isinstance(kwargs['harmonising_data'], bool):
                self._harmonising_data = kwargs['harmonising_data']
            else:
                raise TypeError('Please mention the harmonising_data should be bool type!')
        else:
            self._harmonising_data = False

        # process cloud removal parameter
        if 'cloud_removal_para' in kwargs.keys():
            if isinstance(kwargs['cloud_removal_para'], bool):
                self._cloud_removal_para = kwargs['cloud_removal_para']
            else:
                raise TypeError('Please mention the cloud_removal_para should be bool type!')
        else:
            self._cloud_removal_para = False

        # process construction overwrittern parameter
        if 'overwritten_para' in kwargs.keys():
            if type(kwargs['overwritten_para']) is bool:
                self._overwritten_para = kwargs['overwritten_para']
            else:
                raise TypeError('Please mention the overwritten_para should be bool type!')
        else:
            self._overwritten_para = False

        # process scan line correction
        if 'scan_line_correction' in kwargs.keys():
            if type(kwargs['scan_line_correction']) is bool:
                self._scan_line_correction = kwargs['scan_line_correction']
            else:
                raise TypeError('Please mention the scan_line_correction should be bool type!')
        else:
            self._scan_line_correction = False

        # process size control parameter
        if 'size_control_factor' in kwargs.keys():
            if type(kwargs['size_control_factor']) is bool:
                self._size_control_factor = kwargs['size_control_factor']
            else:
                raise TypeError('Please mention the size_control_factor should be bool type!')
        else:
            self._size_control_factor = False

        # process main_coordinate_system
        if 'main_coordinate_system' in kwargs.keys():
            self.main_coordinate_system = kwargs['main_coordinate_system']
        else:
            self.main_coordinate_system = 'EPSG:32649'

        # process ROI and ROI name
        if 'ROI' in kwargs.keys():
            ROI = kwargs['ROI']
            if self.ROI is None or self.ROI != ROI:
                if '.shp' in ROI and os.path.exists(ROI):
                    self.ROI = ROI
                else:
                    raise ValueError('Please input valid shp file for clip!')

        if self.ROI is not None:
            if 'ROI_name' in kwargs.keys():
                self.ROI_name = kwargs['ROI_name']
            else:
                self.ROI_name = self.ROI.split('\\')[-1].split('.shp')[0]

        # Create shapefile path
        shp_file_path = self._work_env + 'study_area_shapefile\\'
        bf.create_folder(shp_file_path)

        # Move all roi file into the new folder with specific sa name
        if self.ROI is not None:
            if not os.path.exists(shp_file_path + self.ROI_name + '.shp'):
                file_all = bf.file_filter(bf.Path(os.path.dirname(self.ROI)).path_name, [os.path.basename(self.ROI).split('.')[0]], subfolder_detection=True)
                roi_remove_factor = True
                for ori_file in file_all:
                    try:
                        shutil.copyfile(ori_file, shp_file_path + os.path.basename(ori_file))
                    except:
                        roi_remove_factor = False

                if os.path.exists(shp_file_path + self.ROI_name + '.shp') and roi_remove_factor:
                    self.ROI = shp_file_path + self.ROI_name + '.shp'
                else:
                    pass
            else:
                self.ROI = shp_file_path + self.ROI_name + '.shp'

    def _process_index_list(self, index_list):
        index_exprs = built_in_index()
        # Process processed index list
        for index in index_list:

            if not isinstance(index, str):
                raise TypeError(f'The {str(index)} is not under right type')

            if '=' in index:
                index_name = index.split('=')[0].replace(' ', '')
                if index_name in self._all_supported_index_list:
                    pass
                else:
                    try:
                        index_exprs.add_index(index)
                    except:
                        raise ValueError(f'The expression {index} is wrong')
                index_list[index_list.index(index)] = index_name
                self._all_supported_index_list.append(index_name)
                self._index_exprs_dic = index_exprs.index_dic

            elif index in self._all_supported_index_list:
                self._index_exprs_dic = index_exprs.index_dic

            else:
                raise NameError(f'The {index} is not a valid index or the expression is wrong')

        return index_list

    @save_log_file
    def mp_construct_index(self, index_list: list, **kwargs):
        """

        :param index_list: list of indices to generate. All supported indices could be found in the self.__init__ function
        :param kwargs: ROI: The ROI file used to cut the output indices (Should under the type of shpfile);
        ROI_name: The name of ROI user specified or the file name of ROI will be used;
        size_control_factor: Whether converted the float type arr into int16 to save the storage space or not;
        cloud_removal_para: Whether used the QA_pixel layer to remove the cloud-contaminated pixels or not;
        scan_line_correction: Whether fixed the scanline correction error for Landsat 7 ETM+ or not;
        main_coordinate_system: The ESPG style coordinate system for output raster;
        overwritten_factor Whether overwritten the current result;
        'metadata_range';
        issued_files: Do not change the value unless you manually encounter some issued files;
        'harmonising_data: Whether harmonising the Landsat OLI sensor data into Landsat TM range or not'
        """
        # Metadata check
        if self.Landsat_metadata is None:
            raise Exception('Please construct the Landsat_metadata before the subset!')

        # MP process
        i = range(self.Landsat_metadata_size)
        kwargs['metadata_range'] = i
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers= int(os.cpu_count() * Landsat_toolbox.configuration['multiprocess_ratio'])) as executor:
                res = executor.map(self.construct_landsat_index, repeat(index_list), i, repeat(kwargs))
        except OSError:
            print('The OSError during the thread close for py38')

        res = list(set(list(res)))
        res.remove(None)

        # Process issue file
        self._process_issued_files(index_list, res, **kwargs)

    @save_log_file
    def sequenced_construct_index(self, index_list, **kwargs):

        # Metadata check
        if self.Landsat_metadata is None:
            raise Exception('Please construct the Landsat_metadata before the subset!')

        # Sequenced process
        kwargs['metadata_range'] = range(self.Landsat_metadata_size)
        res = []
        for i in range(self.Landsat_metadata_size):
            res.append(self.construct_landsat_index(index_list, i, **kwargs))

        # Process issue file
        self._process_issued_files(index_list, res, **kwargs)

    def _process_issued_files(self, index_, filenum_list, **kwargs):
        if filenum_list != []:
            corrupted_file_folder = os.path.join(self._work_env, 'Corrupted_zip_file\\')
            bf.create_folder(corrupted_file_folder)
            issue_files = open(f"{self.log_filepath}construction_failure_files.txt", "w+")
            issue_files.writelines(['#' * 50 + 'Construction issue files' + '#' * 50])
            for _ in filenum_list:
                try:
                    res_temp = self.construct_landsat_index(index_, _, **kwargs)
                    if res_temp == _:
                        res_temp = True
                    elif res_temp is None:
                        res_temp = False
                    else:
                        raise Exception('Code error concerning process issued files!')
                except:
                    res_temp = True

                if res_temp:
                    issue_files.writelines([q + '\n' for q in self.construction_failure_files])
                    shutil.move(self.Landsat_metadata['File_Path'][_], corrupted_file_folder + self.Landsat_metadata['File_Path'][_].split('\\')[-1])

            print('Please manually rerun the program')
            issue_files.writelines(['#' * 50 + 'Construction issue files' + '#' * 50])
            issue_files.close()

    def _safe_retrieve_band_arr(self, band_name_list, tiffile_serial_num):

        # Define local var
        sensing_date = self.Landsat_metadata['Date'][tiffile_serial_num]
        tile_num = self.Landsat_metadata['Tile_Num'][tiffile_serial_num]
        sensor_type = self.Landsat_metadata['Sensor_Type'][tiffile_serial_num]

        # Factor configuration
        if True in [band_temp not in self._band_output_list for band_temp in band_name_list]:
            print(f'Band {band_name_list} is not valid!')
            sys.exit(-1)

        # Detect whether the required band was generated before
        break_factor, arr_dic, bound_temp = False, {}, None
        while True:

            # Get the ds
            try:
                # Return output
                file_len_list = [len(bf.file_filter(self.unzipped_folder, [f'{str(band_temp)}.TIF', f'{str(tile_num)}_{str(sensing_date)}', str(sensor_type)], and_or_factor='and', exclude_word_list=['.ovr', 'xml'])) for band_temp in band_name_list]
                if True in [_ == 0 for _ in file_len_list]:
                    file_len_t = [_ == 0 for _ in file_len_list]
                    print(f'The {str(band_name_list[file_len_t.index(True)])} of \033[1;31m Date:{str(sensing_date)} Tile:{str(tile_num)}\033[0m was missing!')
                    raise Exception(-1)
                elif True in [_ > 1 for _ in file_len_list]:
                    file_len_t = [_ > 1 for _ in file_len_list]
                    print(f'There are more than one tiffile for \033[1;31m Sensor:{str(sensor_type)} Date:{str(sensing_date)} Tile:{str(tile_num)} Band: {str(band_name_list[file_len_t.index(True)])}\033[0m!')
                    raise Exception(-1)
                else:
                    for band_temp in band_name_list:
                        ds_temp = gdal.Open(bf.file_filter(self.unzipped_folder, [f'{str(band_temp)}.TIF', f'{str(tile_num)}_{str(sensing_date)}', str(sensor_type)], and_or_factor='and', exclude_word_list=['.ovr', 'xml'])[0])

                        if ds_temp is None:
                            print(f'The {str(band_temp)} of \033[1;31m Date:{str(sensing_date)} Tile:{str(tile_num)}\033[0m might be corrupted!')
                            raise Exception(-1)

                        arr_dic[band_temp] = ds_temp.GetRasterBand(1).ReadAsArray().astype(np.float32)
                        nodata_value = ds_temp.GetRasterBand(1).GetNoDataValue()

                        # Reset the nodata value
                        if nodata_value is None:
                            pass
                        elif ~np.isnan(nodata_value):
                            arr_dic[band_temp][arr_dic[band_temp] == nodata_value] = np.nan

                        if band_temp == 'QA_PIXEL':
                            pass
                        elif self._band_tab[f'{sensor_type}_bandname'][self._band_tab[f'{sensor_type}_bandnum'].index(band_temp)] == 'TIR':
                            # Remove invalid pixel
                            arr_dic[band_temp][np.logical_or(arr_dic[band_temp] > 61440, arr_dic[band_temp] < 293)] = np.nan

                            # Refactor the surface temperature
                            arr_dic[band_temp] = arr_dic[band_temp] * 0.00341802 + 149 - 273.15
                        else:
                            # Remove invalid pixel
                            arr_dic[band_temp][np.logical_or(arr_dic[band_temp] > 43636, arr_dic[band_temp]< 7273)] = np.nan

                            # Refactor the surface reflectance
                            arr_dic[band_temp] = arr_dic[band_temp] * 0.0000275 - 0.2

                        # harmonise the Landsat 8
                        if self._harmonising_data and sensor_type in ['LC08'] and band_temp not in ['QA_PIXEL', 'gap_mask']:
                            arr_dic[band_temp] = arr_dic[band_temp] * self._OLI2ETM_harmonised_factor[f'{band_temp}_band_OLS'][0] + self._OLI2ETM_harmonised_factor[f'{band_temp}_band_OLS'][1]

                        if bound_temp is None:
                            ulx_temp, xres_temp, xskew_temp, uly_temp, yskew_temp, yres_temp = ds_temp.GetGeoTransform()
                            bound_temp = (ulx_temp, uly_temp + yres_temp * ds_temp.RasterYSize,  ulx_temp + xres_temp * ds_temp.RasterXSize, uly_temp)

                    return arr_dic, bound_temp, ds_temp

            except Exception:
                if break_factor:
                    print(f"The file {self.Landsat_metadata['File_Path'][tiffile_serial_num]} might be corrupted. Please manually check!")
                    return None, None, None

            except:
                if break_factor:
                    print(traceback.format_exc())
                    raise Exception('Code error during the retrieval of arrays for Landsat')

            break_factor, ds_temp, arr_dic = True, None, {}
            unzipped_file = tarfile.TarFile(self.Landsat_metadata['File_Path'][tiffile_serial_num])
            issued_files = bf.file_filter(self.unzipped_folder, [f'{str(tile_num)}_{str(sensing_date)}', str(sensor_type)], and_or_factor='and')
            for issued_file_temp in issued_files:
                try:
                    os.remove(issued_file_temp)
                except:
                    print(f'Please manually remove {str(issued_file_temp)} and rerun the program!')

            try:
                unzipped_file.extractall(path=self.unzipped_folder)
                unzipped_file.close()
            except:
                print(f"The file {self.Landsat_metadata['File_Path'][tiffile_serial_num]} is corrupted")
                return None, None, None
    
    def construct_landsat_index(self, index_list, i, *args, **kwargs):

        try:
            start_time = time.time()
            # Process VI list
            if index_list == []:
                raise ValueError('None of the input index is supported')
            elif 'FVC' in index_list and 'NDVI' not in index_list:
                index_list.remove('FVC')
                index_list.extend(['NDVI', 'MNDWI', 'FVC'])
            # index_list.append('Watermask')

            # Retrieve kwargs from args using the mp
            if args != () and type(args[0]) == dict:
                kwargs = copy.copy(args[0])

            # Determine the subset indicator
            self._process_index_construction_para(**kwargs)
            self._check_metadata_availability()
            index_list = self._process_index_list(index_list)

            # Retrieve the file inform
            fileid = self.Landsat_metadata.FileID[i]
            filedate = self.Landsat_metadata['Date'][i]
            tile_num = self.Landsat_metadata['Tile_Num'][i]
            sensor_type = self.Landsat_metadata['Sensor_Type'][i]

            # Detect VI existence
            self.vi_output_path_dic = {}
            dep_dic = {}

            # Get all unique bands
            for _ in index_list:
                self.vi_output_path_dic[_] = f'{self._work_env}Landsat_constructed_index\\{str(_)}\\' if self.ROI is None else f'{self._work_env}Landsat_{str(self.ROI_name)}_index\\{str(_)}\\'
                file_name = f'{str(filedate)}_{str(tile_num)}_{str(_)}.TIF'

                if not os.path.exists(f'{self.vi_output_path_dic[_]}{str(filedate)}_{str(tile_num)}_{str(_)}.TIF') or self._overwritten_para:
                    bf.create_folder(self.vi_output_path_dic[_])

                    if _ in self._index_exprs_dic.keys():
                        dep_list = self._index_exprs_dic[_][0]
                        dep_list = [str(dep) for dep in dep_list]
                        if 'LE07' in fileid:
                            dep_list = [self._band_tab['LE07_bandnum'][self._band_tab['LE07_bandname'].index(dep_t)] for dep_t in dep_list]
                        elif 'LT05' in fileid or 'LT04' in fileid:
                            dep_list = [self._band_tab['LT05_bandnum'][self._band_tab['LT05_bandname'].index(dep_t)] for dep_t in dep_list]
                        elif 'LC08' in fileid or 'LC09' in fileid:
                            dep_list = [self._band_tab['LC08_bandnum'][self._band_tab['LC08_bandname'].index(dep_t)] for dep_t in dep_list]
                        else:
                            raise Exception('The Original Tiff files are not belonging to Landsat 5, 7, 8 OR 9')
                        dep_dic[_] = dep_list
                    else:
                        raise Exception(f'Code error: the {str(_)} for {str(filedate)}_{str(tile_num)} is not in the index expression dic')

            # Generate the unique dep list
            unique_dep_list = []
            for _ in dep_dic.keys():
                for __ in dep_dic[_]:
                    if __ not in unique_dep_list:
                        unique_dep_list.append(__)

            # Read all unique tif files and QA files
            if len(unique_dep_list) > 0:
                arr_dic, bound_temp, ds_temp = self._safe_retrieve_band_arr(unique_dep_list, i)
                if arr_dic is None:
                    raise Exception(f'Error during the retrival of Band Array of {fileid}')

                if self._cloud_removal_para:
                    # qi_folder = f'{thalweg_temp._work_env}Landsat_constructed_index\\QI\\' if thalweg_temp.ROI is None else f'{thalweg_temp._work_env}Landsat_{str(thalweg_temp.ROI_name)}_index\\QI\\'
                    # bf.create_folder(qi_folder)
                    QA_dic, bound_temp, ds_temp = self._safe_retrieve_band_arr(['QA_PIXEL'], i)
                    if QA_dic is None:
                        raise Exception(f'Error during the retrieval of QA_PIXEL file for {fileid}')

                    QI_arr = QA_dic['QA_PIXEL']
                    QI_arr = self._process_QA_band(QI_arr, i)

            # Calculate each index
            for _ in index_list:
                file_name = f'{str(filedate)}_{str(tile_num)}_{str(_)}.TIF'
                if not os.path.exists(f'{self.vi_output_path_dic[_]}{str(filedate)}_{str(tile_num)}_{str(_)}.TIF') or self._overwritten_para:

                    # Generate the output arr
                    output_array = self._index_exprs_dic[_][1](*[arr_dic[__] for __ in dep_dic[_]])

                    # Cloud removal procedure
                    if self._cloud_removal_para:
                        try:
                            output_array = QI_arr * output_array
                            # bf.write_raster(ds_list[0], output_array, qi_folder, file_name + '.TIF', raster_datatype=gdal.GDT_Int16)
                        except ValueError:
                            raise ValueError(f'QI and BAND array for {str(tile_num)} {str(filedate)} {str(sensor_type)} is not compatible')

                    # Landsat 7 scan line correction
                    if self._scan_line_correction:
                        fill_landsat7_gap(output_array)
                        pass

                    # Output to tiffile
                    if self._size_control_factor:
                        if _ in self._band_sup:
                            output_array[np.isnan(output_array)] = 0
                            output_array = output_array * 10000
                            output_array.astype(np.uint16)
                            bf.write_raster(ds_temp, output_array, '/vsimem/', file_name + '.TIF', raster_datatype=gdal.GDT_UInt16)
                            data_type = gdal.GDT_UInt16
                            nodata_value = 0
                        else:
                            output_array[np.isnan(output_array)] = -3.2768
                            output_array = output_array * 10000
                            output_array.astype(np.int16)
                            bf.write_raster(ds_temp, output_array, '/vsimem/', file_name + '.TIF', raster_datatype=gdal.GDT_Int16)
                            data_type = gdal.GDT_Int16
                            nodata_value = -32768
                    else:
                        bf.write_raster(ds_temp, output_array, '/vsimem/', file_name + '.TIF', raster_datatype=gdal.GDT_Float32)
                        data_type = gdal.GDT_Float32
                        nodata_value = np.nan

                    if self.ROI is not None:
                        gdal.Warp('/vsimem/' + file_name + '2.TIF', '/vsimem/' + file_name + '.TIF', xRes=30, yRes=30, dstSRS=self.main_coordinate_system, cutlineDSName=self.ROI, cropToCutline=True, outputType=data_type, outputBounds=bound_temp, srcNodata =nodata_value, dstNodata =nodata_value)
                    else:
                        gdal.Warp('/vsimem/' + file_name + '2.TIF', '/vsimem/' + file_name + '.TIF', xRes=30, yRes=30, dstSRS=self.main_coordinate_system, outputType=data_type, outputBounds=bound_temp, srcNodata =nodata_value, dstNodata =nodata_value)

                    gdal.Translate(f'{self.vi_output_path_dic[_]}{str(filedate)}_{str(tile_num)}_{str(_)}.TIF', '/vsimem/' + file_name + '2.TIF', options=topts)
                    gdal.Unlink('/vsimem/' + file_name + '.TIF')
                    gdal.Unlink('/vsimem/' + file_name + '2.TIF')

                    print(f'The \033[1;31m{str(_)}\033[0m of \033[1;33m{str(filedate)} {str(tile_num)}\033[0m were constructed in \033[1;34m{str(time.time() - start_time)}s\033[0m ({str(i + 1)} of {str(self.Landsat_metadata_size)})')
                    start_time = time.time()
                else:
                    print(f'The \033[1;31m{str(_)}\033[0m of \033[1;33m{str(filedate)} {str(tile_num)}\033[0m has been constructed ({str(i + 1)} of {str(self.Landsat_metadata_size)})')
                    start_time = time.time()

                # Generate SA map (NEED TO FIX)
                if self.ROI is not None and kwargs['metadata_range'].index(i) == 0:
                    if not os.path.exists(self._work_env + 'ROI_map\\' + self.ROI_name + '_map.npy'):

                        file_list = bf.file_filter(f'{self.unzipped_folder}\\',  ['.TIF'], and_or_factor='and')
                        bf.create_folder(self._work_env + 'ROI_map\\')
                        ds_temp = gdal.Open(file_list[0])
                        if retrieve_srs(ds_temp) != self.main_coordinate_system:
                            gdal.Warp(self.cache_folder + 'temp_' + self.ROI_name + '.TIF', ds_temp,
                                      dstSRS=self.main_coordinate_system, cutlineDSName=self.ROI, cropToCutline=True,
                                      xRes=30, yRes=30, dstNodata=-32768)
                        else:
                            gdal.Warp(self.cache_folder + 'temp_' + self.ROI_name + '.TIF', ds_temp,
                                      cutlineDSName=self.ROI, cropToCutline=True, dstNodata=-32768, xRes=30, yRes=30)

                        ds_temp = gdal.Open(self.cache_folder + 'temp_' + self.ROI_name + '.TIF')
                        array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
                        array_temp[:, :] = 1
                        bf.write_raster(ds_temp, array_temp, self.cache_folder, 'temp2_' + self.ROI_name + '.TIF', raster_datatype=gdal.GDT_Int16)
                        gdal.Warp('/vsimem/' + 'ROI_map\\' + self.ROI_name + '_map.TIF',
                                  self.cache_folder + 'temp2_' + self.ROI_name + '.TIF',
                                  cutlineDSName=self.ROI, cropToCutline=True,
                                  xRes=30, yRes=30, dstNodata=-32768)
                        gdal.Translate(self._work_env + 'ROI_map\\' + self.ROI_name + '_map.TIF', '/vsimem/' + 'ROI_map\\' + self.ROI_name + '_map.TIF', options=topts)

                        ds_ROI_array = gdal.Open(self._work_env + 'ROI_map\\' + self.ROI_name + '_map.TIF')
                        ds_sa_array = ds_ROI_array.GetRasterBand(1).ReadAsArray()

                        if (ds_sa_array == -32768).all() == False:
                            np.save(self._work_env + 'ROI_map\\' + self.ROI_name + '_map.npy', ds_sa_array)

                        gdal.Unlink('/vsimem/' + 'ROI_map\\' + self.ROI_name + '_map.TIF')
                        ds_temp, ds_ROI_array = None, None
                        remove_all_file_and_folder(bf.file_filter(self.cache_folder, ['temp', '.TIF'], and_or_factor='and'))

                # Generate Watermask map
                water_mask_path = f'{self._work_env}Landsat_constructed_index\\watermask\\' if self.ROI is None else f'{self._work_env}Landsat_{str(self.ROI_name)}_index\\watermask\\'
                bf.create_folder(water_mask_path)
                if not os.path.exists(water_mask_path + str(filedate) + '_' + str(tile_num) + '_watermask.TIF'):

                    QI_filelist = bf.file_filter(self.unzipped_folder,[f'{str(tile_num)}_{str(filedate)}', sensor_type, 'QA_PIXEL'],  and_or_factor='and', exclude_word_list=[f'{str(filedate)}_02'])
                    if len(QI_filelist) != 1:
                        raise ValueError(f'There are more than one QI file for {str(filedate)} {str(tile_num)}!')
                    else:
                        QI_ds = gdal.Open(QI_filelist[0])
                        QI_arr = QI_ds.GetRasterBand(1).ReadAsArray()
                        WATER_temp_array = copy.copy(QI_arr)
                        QI_arr[~np.isnan(QI_arr)] = 1
                        WATER_temp_array[np.logical_and(np.floor_divide(np.mod(WATER_temp_array, 256), 128) != 1, ~np.isnan(np.floor_divide(np.mod(WATER_temp_array, 256), 128)))] = 0
                        WATER_temp_array[np.divide(np.mod(WATER_temp_array, 256), 128) == 1] = 1
                        WATER_temp_array[np.isnan(WATER_temp_array)] = 65535
                        bf.write_raster(QI_ds, WATER_temp_array, water_mask_path, str(filedate) + '_' + str(tile_num) + '_watermask.TIF',  raster_datatype=gdal.GDT_UInt16)
            return None
        except:
            print(traceback.format_exc())
            # return i
            return None

    def _process_QA_band(self, QI_temp_array, tiffile_serial_num):

        # s1_time = time.time()
        if not isinstance(QI_temp_array, np.ndarray):
            raise TypeError('The qi temp array was under a wrong format!')

        if not isinstance(tiffile_serial_num, int):
            raise TypeError('The tiffile serial num was under a wrong format!')
        else:
            sensor_type = self.Landsat_metadata['Sensor_Type'][tiffile_serial_num]
            fileid = self.Landsat_metadata.FileID[tiffile_serial_num]

        QI_temp_array = QI_temp_array.astype(np.float32)
        QI_temp_array[QI_temp_array == 1] = np.nan

        if sensor_type in ['LC08', 'LC09']:
            QI_temp_array[np.floor_divide(QI_temp_array, 256) > 86] = np.nan
            QI_temp_array_temp = copy.copy(QI_temp_array)
            QI_temp_array_temp[~np.isnan(QI_temp_array_temp)] = 0
            QI_temp_array_temp[np.isnan(QI_temp_array_temp)] = 1
            QI_neighbor_average = neighbor_average_convolve2d(QI_temp_array_temp, size=7)
            QI_temp_array[np.logical_and(np.logical_or(QI_temp_array == 22080, QI_temp_array == 22208), QI_neighbor_average > 3)] = np.nan
            QI_temp_array[np.logical_and(np.logical_and(np.mod(QI_temp_array, 128) != 64,
                                                        np.mod(QI_temp_array, 128) != 2),
                                         np.logical_and(np.mod(QI_temp_array, 128) != 0,
                                                        np.mod(QI_temp_array, 128) != 66))] = np.nan

        elif sensor_type == 'LE07':
            QI_temp_array[np.floor_divide(QI_temp_array, 256) > 21] = np.nan
            QI_temp_array_temp = copy.copy(QI_temp_array)
            QI_temp_array_temp[~np.isnan(QI_temp_array_temp)] = 0
            QI_temp_array_temp[np.isnan(QI_temp_array_temp)] = 1
            QI_neighbor_average = neighbor_average_convolve2d(QI_temp_array_temp, size=7)
            QI_temp_array[np.logical_and(np.logical_or(QI_temp_array == 5696, QI_temp_array == 5760), QI_neighbor_average > 3)] = np.nan
            QI_temp_array[np.logical_and(np.logical_and(np.mod(QI_temp_array, 128) != 64,
                                                        np.mod(QI_temp_array, 128) != 2),
                                         np.logical_and(np.mod(QI_temp_array, 128) != 0,
                                                        np.mod(QI_temp_array, 128) != 66))] = np.nan

            if self._scan_line_correction:
                gap_mask_array, t, tt = self._safe_retrieve_band_arr(['gap_mask'], tiffile_serial_num)
                if gap_mask_array is None:
                    raise Exception(f'Error during the retrival of Band Array of {fileid}')
                else:
                    gap_mask_array = gap_mask_array['gap_mask']
                QI_temp_array[gap_mask_array == 0] = 1

        elif sensor_type in ['LT05', 'LT04']:
            QI_temp_array[np.floor_divide(QI_temp_array, 256) > 21] = np.nan
            QI_temp_array_temp = copy.copy(QI_temp_array)
            QI_temp_array_temp[~np.isnan(QI_temp_array_temp)] = 0
            QI_temp_array_temp[np.isnan(QI_temp_array_temp)] = 1
            QI_neighbor_average = neighbor_average_convolve2d(QI_temp_array_temp, size=7)
            QI_temp_array[np.logical_and(np.logical_or(QI_temp_array == 5696, QI_temp_array == 5760),
                                         QI_neighbor_average > 3)] = np.nan
            QI_temp_array[np.logical_and(np.logical_and(np.mod(QI_temp_array, 128) != 64,
                                                        np.mod(QI_temp_array, 128) != 2),
                                         np.logical_and(np.mod(QI_temp_array, 128) != 0,
                                                        np.mod(QI_temp_array, 128) != 66))] = np.nan

        else:
            raise ValueError(f'This {sensor_type} is not supported Landsat data!')

        # print(f's1 time {str(time.time() - start_time)}')
        QI_temp_array[~np.isnan(QI_temp_array)] = 1
        return QI_temp_array

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

    def _process_2dc_para(self, **kwargs):
        # Detect whether all the indicators are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in ('skip_invalid_file', 'inherit_from_logfile', 'ROI', 'ROI_name', 'dc_overwritten_para', 'remove_nan_layer', 'manually_remove_datelist', 'size_control_factor', 'cloud_removal_para'):
                raise NameError(f'{kwarg_indicator} is not supported kwargs! Please double check!')

        # process clipped_overwritten_para
        if 'dc_overwritten_para' in kwargs.keys():
            if type(kwargs['dc_overwritten_para']) is bool:
                self._dc_overwritten_para = kwargs['dc_overwritten_para']
            else:
                raise TypeError('Please mention the dc_overwritten_para should be bool type!')
        else:
            self._dc_overwritten_para = False

        # process inherit from logfile
        if 'inherit_from_logfile' in kwargs.keys():
            if type(kwargs['inherit_from_logfile']) is bool:
                self._inherit_from_logfile = kwargs['inherit_from_logfile']
            else:
                raise TypeError('Please mention the dc_overwritten_para should be bool type!')
        else:
            self._inherit_from_logfile = False

        # process ship invalid file
        if 'skip_invalid_file' in kwargs.keys():
            if type(kwargs['skip_invalid_file']) is bool:
                self._skip_invalid_file = kwargs['skip_invalid_file']
            else:
                raise TypeError('Please mention the skip_invalid_file should be bool type!')
        else:
            self._skip_invalid_file = False

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

        # process ROI
        if 'ROI' in kwargs.keys():
            self.ROI = kwargs['ROI']
        elif self.ROI is None and self._inherit_from_logfile:
            self._retrieve_para(['ROI'])
        elif self.ROI is None:
            raise Exception('Notice the ROI was missed!')

        # process ROI_NAME
        if 'ROI_name' in kwargs.keys():
            self.ROI_name = kwargs['ROI_name']
        elif self.ROI_name is None and self._inherit_from_logfile:
            self._retrieve_para(['ROI_name'])
            if self.ROI_name is None:
                self.ROI_name = self.ROI.split('\\')[-1].split('.shp')[0]
        elif self.ROI_name is None:
            raise Exception('Notice the ROI name was missed!')

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

        # Retrieve cloud removal factor
        if 'cloud_removal_para' in kwargs.keys():
            if type(kwargs['cloud_removal_para']) is bool:
                self._cloud_removal_para = kwargs['cloud_removal_para']
            else:
                raise TypeError('Please mention the cloud_removal_para should be bool type!')
        elif self._inherit_from_logfile:
            self._retrieve_para(['cloud_removal_para'], protected_var=True)
        else:
            self._cloud_removal_para = False

    @save_log_file
    def mp_ds2landsatdc(self, index_list: list, *args, **kwargs):

        # Define the chunk size
        if 'chunk_size' in kwargs.keys() and type(kwargs['chunk_size']) is int:
            chunk_size = min(os.cpu_count(), kwargs['chunk_size'])
            del kwargs['chunk_size']
        # elif 'chunk_size' in kwargs.keys() and kwargs['chunk_size'] == 'auto':
        #     chunk_size = os.cpu_count()
        else:
            chunk_size = int(os.cpu_count() * Landsat_toolbox.configuration['multiprocess_ratio'])

        # MP process
        with concurrent.futures.ProcessPoolExecutor(max_workers=chunk_size) as executor:
            executor.map(self.ds2landsatdc, index_list, repeat(kwargs))

    @save_log_file
    def seq_ds2landsatdc(self, index_list, *args, **kwargs):
        # sequenced process
        for index in index_list:
            self.ds2landsatdc(index, *args, **kwargs)

    def ds2landsatdc(self, _, *args, **kwargs):
        # for the MP
        if args != () and type(args[0]) == dict:
            kwargs = copy.copy(args[0])

        # Process clip parameter
        self._process_2dc_para(**kwargs)

        # Define the input path
        self._dc_infr[_ + 'input_path'] = self._work_env + f'Landsat_constructed_index\\{_}\\' if self.ROI_name is None else self._work_env + f'Landsat_{self.ROI_name}_index\\{_}\\'

        if not os.path.exists(self._dc_infr[_ + 'input_path']):
            raise Exception(f'Please validate the roi name and {str(_)} for ds2dc!')
        elif not self._skip_invalid_file and len(bf.file_filter(self._dc_infr[_ + 'input_path'], [_, '.TIF'], and_or_factor='and')) != self.Landsat_metadata_size:
            raise ValueError(f'{_} of the {self.ROI_name} is not consistent')

        eliminating_all_not_required_file(self._dc_infr[_ + 'input_path'])

        # Define the output path
        self._dc_infr[_] = self._work_env + 'Landsat_constructed_datacube\\' + _ + '_datacube\\' if self.ROI_name is None else self._work_env + 'Landsat_' + self.ROI_name + '_datacube\\' + _ + '_datacube\\'
        bf.create_folder(self._dc_infr[_])

        # Construct the dc
        if self._dc_overwritten_para or not os.path.exists(self._dc_infr[_] + _ + '_datacube.npy') or not os.path.exists(self._dc_infr[_] + 'date.npy') or not os.path.exists(self._dc_infr[_] + 'metadata.json'):

            sa_map = np.load(bf.file_filter(self._work_env + 'ROI_map\\', [self.ROI_name, '.npy'], and_or_factor='and')[0], allow_pickle=True)
            if self.ROI_name is None or self.ROI is None:
                raise ValueError('ROI needs to be specified before the Landsat dc construction')
            else:
                print('Start processing ' + _ + ' datacube of the ' + self.ROI_name + '.')
                self.main_coordinate_system = retrieve_srs(gdal.Open(self._work_env + 'ROI_map\\' + self.ROI_name + '_map.TIF',))
                sa_map = np.load(bf.file_filter(self._work_env + 'ROI_map\\', [self.ROI_name, '.npy'], and_or_factor='and')[0], allow_pickle=True)
                metadata_dic = {'ROI_name': self.ROI_name, 'index': _, 'ROI': self.ROI,
                                'ROI_array': self._work_env + 'ROI_map\\' + self.ROI_name + '_map.npy',
                                'ROI_tif': self._work_env + 'ROI_map\\' + self.ROI_name + '_map.TIF',
                                'sdc_factor': True, 'coordinate_system': self.main_coordinate_system,
                                'size_control_factor': self._size_control_factor, 'oritif_folder': self._dc_infr[_ + 'input_path'],
                                'dc_group_list': None, 'tiles': None}

            # Get the doy list
            start_time = time.time()
            stack_file_list = bf.file_filter(self._dc_infr[_ + 'input_path'], [_, '.TIF'], and_or_factor='and', exclude_word_list=['aux', 'xml'])
            stack_file_list.sort()

            doy_list = [int(filepath_temp.split('\\')[-1][0:8]) for filepath_temp in stack_file_list]
            doy_list = list(set(doy_list))
            doy_list_ = []
            for doy_ in doy_list:
                if doy_ in list(self.Landsat_metadata['Date']):
                    doy_list_.append(doy_)
            doy_list = doy_list_
            doy_list.sort()

            # Evaluate the size and sparsity of sdc
            mem = psutil.virtual_memory()
            dc_max_size = int((mem.free) * 0.90)
            temp_ds = gdal.Open(stack_file_list[0])
            cols, rows = temp_ds.RasterXSize, temp_ds.RasterYSize
            sparsify = np.sum(sa_map == -32768) / (sa_map.shape[0] * sa_map.shape[1])
            _huge_matrix = True if len(doy_list) * cols * rows * 2 > dc_max_size else False
            _sparse_matrix = True if sparsify > 0.9 else False

            # Generate the datacube
            # There are 2 * 2 * 2 types of datacubes, determined by their size, sparsity, and inclusion of band data
            # Define the var for different types
            if _sparse_matrix:
                i, nodata_value, dtype_temp, dtype_out = 0, None, None, None
                data_cube = NDSparseMatrix()
                data_valid_array = np.zeros([len(doy_list)], dtype=int)
            else:
                i, nodata_value, dtype_temp, dtype_out = 0, None, None, None
                data_cube = []
                data_valid_array = np.zeros([len(doy_list)], dtype=int)

            # Generate the bound
            ds_temp = gdal.Open(metadata_dic['ROI_tif'])
            ulx_temp, xres_temp, xskew_temp, uly_temp, yskew_temp, yres_temp = ds_temp.GetGeoTransform()
            bound_temp = (ulx_temp, uly_temp + yres_temp * ds_temp.RasterYSize, ulx_temp + xres_temp * ds_temp.RasterXSize, uly_temp)

            with tqdm(total=len(doy_list), desc=f'Assemble the \033[1;33m{str(_)}\033[0m into the Landsat sdc', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
                while i < len(doy_list):
                    try:
                        # if _ not in self._band_sup:
                        file_list = bf.file_filter(self._dc_infr[_ + 'input_path'], [str(doy_list[i]), f'{_}.TIF'], and_or_factor='and')
                        if len(file_list) == 0:
                            if not self._skip_invalid_file:
                                raise Exception(f'The {str(doy_list[i])}_{_} is not properly generated!')
                            else:
                                array_list = [np.ones(array_list[0].shape[0], array_list[0].shape[1]) * nodata_value]
                        else:
                            try:
                                ds_list = [gdal.Open(file_temp) for file_temp in file_list]
                                array_list = [ds_temp.GetRasterBand(1).ReadAsArray() for ds_temp in ds_list]
                                nodata_list = [ds_temp.GetRasterBand(1).GetNoDataValue() for ds_temp in ds_list]
                                dtype_list = [array_temp.dtype for array_temp in array_list]
                            except RuntimeError:
                                ds_list, array_list, nodata_list, dtype_list = None, None, None, None
                                for file_temp in file_list:
                                    date_temp = file_temp.split('\\')[-1].split('_')[0]
                                    tile = file_temp.split('\\')[-1].split('_')[1]
                                    index_name = file_temp.split('\\')[-1].split('_')[2]

                                try:
                                    os.remove(file_temp)
                                except PermissionError:
                                    pass

                                try:
                                    self.construct_landsat_index([index_name], self.Landsat_metadata[(self.Landsat_metadata['Date'] == int(date_temp)) & (self.Landsat_metadata['Tile_Num'] == int(tile))].index[0], inherit_from_logfile=True)
                                    ds_list = [gdal.Open(file_temp) for file_temp in file_list]
                                    array_list = [ds_temp.GetRasterBand(1).ReadAsArray() for ds_temp in ds_list]
                                    nodata_list = [ds_temp.GetRasterBand(1).GetNoDataValue() for ds_temp in ds_list]
                                    dtype_list = [array_temp.dtype for array_temp in array_list]
                                except RuntimeError:
                                    self._process_issued_files(index_name, [self.Landsat_metadata[(self.Landsat_metadata['Date'] == int(date_temp)) & (self.Landsat_metadata['Tile_Num'] == int(tile))].index[0]])
                                    raise Exception(f'The {index_name} of {str(date_temp)} {str(tile)} is corrupted. Please manually rerun the program')

                            if len(list(set(nodata_list))) != 1:
                                raise ValueError(f'The nodata value is not consistent for {str(_)} in {str(doy_list[i])}')
                            elif nodata_value is None:
                                nodata_value = list(set(nodata_list))[0]
                            elif nodata_value != list(set(nodata_list))[0]:
                                raise ValueError(f'The nodata value is not consistent for {str(_)} in {str(doy_list[i])}')

                            if len(list(set(dtype_list))) != 1:
                                raise ValueError(f'The dtype is not consistent for {str(_)} in {str(doy_list[i])}')
                            elif dtype_temp is None:
                                dtype_temp = list(set(dtype_list))[0]
                            elif dtype_temp != list(set(dtype_list))[0]:
                                raise ValueError(f'The dtype is not consistent for {str(_)} in {str(doy_list[i])}')

                        # Mean array
                        if len(array_list) == 1:
                            output_arr = array_list[0]
                        else:
                            output_arr = np.stack(array_list, axis=2)
                            if np.isnan(nodata_value):
                                output_arr = np.nanmean(output_arr, axis=2)
                            else:
                                # Alternative method but raised RuntimeWarning: Mean of empty slice.
                                # output_arr = np.mean(output_arr, axis=2, where=output_arr != nodata_value)
                                output_arr = output_arr.astype(float)
                                output_arr[output_arr == nodata_value] = np.nan
                                output_arr = np.nanmean(output_arr, axis=2)

                        # Convert the nodata value to 0
                        if _sparse_matrix:
                            if np.isnan(nodata_value):
                                output_arr[np.isnan(output_arr)] = 0
                                dtype_temp = np.float16
                            else:
                                output_arr[np.isnan(output_arr)] = nodata_value
                                output_arr = output_arr - nodata_value
                                max_v = np.iinfo(dtype_temp).max - nodata_value
                                min_v = np.iinfo(dtype_temp).min - nodata_value
                                if min_v < 0:
                                    for type_temp in [np.int8, np.int16, np.int32, np.int64]:
                                        if min_v >= np.iinfo(type_temp).min and max_v <= np.iinfo(type_temp).max:
                                            dtype_out = type_temp
                                            break
                                elif min_v >= 0:
                                    for type_temp in [np.uint8, np.uint16, np.uint32, np.uint64]:
                                        if max_v <= np.iinfo(type_temp).max:
                                            dtype_out = type_temp
                                            break
                                if dtype_out is None:
                                    raise Exception('Code error for generating the datatype of output array!')
                        else:
                            dtype_out = type_temp

#                         else:
#                             dtype_out = np.uint16
#                             index_folder = os.path.join(self._work_env + f'Landsat_constructed_index\\{_}\\') if self.ROI_name is None else os.path.join(self._work_env, f'Landsat_{self.ROI_name}_index\\{_}\\')
#                             QA_folder = os.path.join(self._work_env + f'Landsat_constructed_index\\QA_pixel\\') if self.ROI_name is None else os.path.join(self._work_env, f'Landsat_{self.ROI_name}_index\\QA_pixel\\')
#                             bf.create_folder(index_folder)
#
#                             # Get the file
#                             reqfile_path, sensor_type, metadata_num = [], [], []
#                             for metadata_n in range(self.Landsat_metadata.shape[0]):
#                                 if self.Landsat_metadata['Date'][metadata_n] == doy_list[i]:
#                                     reqfile_path.append(self.Landsat_metadata['File_Path'][metadata_n])
#                                     sensor_type.append(self.Landsat_metadata['Sensor_Type'][metadata_n])
#                                     metadata_num.append(metadata_n)
#                             reqfile_list, QAfile_list = [], []
#
#                             # Obtain the filepath for each band tiffile
#                             for reqfile_num in range(len(reqfile_path)):
#                                 band_name = self._band_tab[f"{sensor_type[reqfile_num]}_bandnum"][
#                                     self._band_tab[f"{sensor_type[reqfile_num]}_bandname"].index(_)]
#                                 file_name = os.path.join(self.unzipped_folder, reqfile_path[reqfile_num].split('.tar')[0].split('\\')[-1] + f'_SR_{str(band_name)}.TIF')
#                                 QA_file_name = os.path.join(self.unzipped_folder, reqfile_path[reqfile_num].split('.tar')[0].split('\\')[-1] + f'_QA_PIXEL.TIF')
#                                 if not os.path.exists(file_name):
#                                     raise Exception(f'The band file {file_name} was not existed')
#                                 else:
#                                     reqfile_list.append(file_name)
#
#                                 if not os.path.exists(QA_file_name):
#                                     raise Exception(f'The QA pixel file {file_name} was not existed')
#                                 else:
#                                     QAfile_list.append(QA_file_name)
#
#                             if not os.path.exists(f"{index_folder}{str(doy_list[i])}_{_}.TIF"):
#                                 # Merge the band file
#                                 if len(reqfile_list) == 1:
#                                     vrt = gdal.BuildVRT(f"{self.cache_folder}{str(doy_list[i])}_{_}.vrt",
#                                                         reqfile_list, xRes=30, yRes=30, outputBounds=bound_temp, srcNodata=0, VRTNodata=0, outputSRS=self.main_coordinate_system)
#                                     vrt = None
#
#                                 elif len(reqfile_list) > 1:
#                                     vrt = gdal.BuildVRT(f"{self.cache_folder}{str(doy_list[i])}_{_}.vrt",
#                                                         reqfile_list, xRes=30, yRes=30, outputBounds=bound_temp, srcNodata=0, VRTNodata=0, outputSRS=self.main_coordinate_system)
#                                     vrt = None
#
#                                     vrt_tree = etree.parse(f"{self.cache_folder}{str(doy_list[i])}_{_}.vrt")
#                                     vrt_root = vrt_tree.getroot()
#                                     vrtband1 = vrt_root.findall(".//VRTRasterBand[@band='1']")[0]
#
#                                     vrtband1.set("subClass", "VRTDerivedRasterBand")
#                                     pixelFunctionType = etree.SubElement(vrtband1, 'PixelFunctionType')
#                                     pixelFunctionType.text = "find_max"
#                                     pixelFunctionLanguage = etree.SubElement(vrtband1, 'PixelFunctionLanguage')
#                                     pixelFunctionLanguage.text = "Python"
#                                     pixelFunctionCode = etree.SubElement(vrtband1, 'PixelFunctionCode')
#                                     pixelFunctionCode.text = etree.CDATA("""
# import numpy as np
#
# def find_max(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize, raster_ysize, buf_radius, gt, **kwargs):
#     out_ar[:] = np.max(in_ar, axis=0)
#
# """)
#                                     vrt_tree.write(f"{self.cache_folder}{str(doy_list[i])}_{_}.vrt")
#                                 else:
#                                     raise Exception('Code error')
#
#                                 gdal.Warp(f"/vsimem/{str(doy_list[i])}_{_}.TIF", f"{self.cache_folder}{str(doy_list[i])}_{_}.vrt", cutlineDSName=self.ROI, cropToCutline=True, xRes=30, yRes=30, dstNodata=0, outputBounds=bound_temp, dstSRS=self.main_coordinate_system)
#                                 gdal.Translate(f"{index_folder}{str(doy_list[i])}_{_}.TIF", f"/vsimem/{str(doy_list[i])}_{_}.TIF", options=topts)
#                                 gdal.Unlink(f"/vsimem/{str(doy_list[i])}_{_}.TIF")
#
#                             output_ds = gdal.Open(f"{index_folder}{str(doy_list[i])}_{_}.TIF")
#                             output_arr = output_ds.GetRasterBand(1).ReadAsArray()
#
#                             if self._cloud_removal_para:
#                                 bf.create_folder(QA_folder)
#                                 if not os.path.exists(os.path.join(QA_folder, f"{str(doy_list[i])}_QA.TIF")):
#                                     # Merge the band file
#                                     if len(QAfile_list) == 1:
#                                         vrt = gdal.BuildVRT(f"{self.cache_folder}{str(doy_list[i])}_QA.vrt",
#                                                             QAfile_list, xRes=30, yRes=30, outputBounds=bound_temp,
#                                                             srcNodata=0, VRTNodata=0, outputSRS=self.main_coordinate_system)
#                                         vrt = None
#
#                                     elif len(QAfile_list) > 1:
#                                         vrt = gdal.BuildVRT(f"{self.cache_folder}{str(doy_list[i])}_QA.vrt",
#                                                             QAfile_list, xRes=30, yRes=30, outputBounds=bound_temp,
#                                                             srcNodata=0, VRTNodata=0, outputSRS=self.main_coordinate_system)
#                                         vrt = None
#
#                                         vrt_tree = etree.parse(f"{self.cache_folder}{str(doy_list[i])}_QA.vrt")
#                                         vrt_root = vrt_tree.getroot()
#                                         vrtband1 = vrt_root.findall(".//VRTRasterBand[@band='1']")[0]
#
#                                         vrtband1.set("subClass", "VRTDerivedRasterBand")
#                                         pixelFunctionType = etree.SubElement(vrtband1, 'PixelFunctionType')
#                                         pixelFunctionType.text = "find_max"
#                                         pixelFunctionLanguage = etree.SubElement(vrtband1, 'PixelFunctionLanguage')
#                                         pixelFunctionLanguage.text = "Python"
#                                         pixelFunctionCode = etree.SubElement(vrtband1, 'PixelFunctionCode')
#                                         pixelFunctionCode.text = etree.CDATA("""
# import numpy as np
#
# def find_max(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize, raster_ysize, buf_radius, gt, **kwargs):
#     out_ar[:] = np.max(in_ar, axis=0)
#
# """)
#                                         vrt_tree.write(f"{self.cache_folder}{str(doy_list[i])}_QA.vrt")
#                                     else:
#                                         raise Exception('Code error')
#                                     # gdal.Translate(f"/vsimem/{str(doy_list[i])}_QA.TIF", f"{self.cache_folder}{str(doy_list[i])}_QA.vrt", options=topts, noData=0, outputSRS=self.main_coordinate_system)
#                                     # gdal.Warp(os.path.join(QA_folder, f"{str(doy_list[i])}_QA.TIF"), f"/vsimem/{str(doy_list[i])}_QA.TIF", cutlineDSName=self.ROI, cropToCutline=True, xRes=30, yRes=30, dstNodata=0)
#                                     gdal.Warp(f"/vsimem/{str(doy_list[i])}_QA.TIF", f"{self.cache_folder}{str(doy_list[i])}_QA.vrt", cutlineDSName=self.ROI, cropToCutline=True, xRes=30, yRes=30, dstNodata=0, dstSRS=self.main_coordinate_system)
#                                     gdal.Translate(os.path.join(QA_folder, f"{str(doy_list[i])}_QA.TIF"), f"/vsimem/{str(doy_list[i])}_QA.TIF", options=topts)
#                                     gdal.Unlink(f"/vsimem/{str(doy_list[i])}_QA.TIF")
#
#                                     qa_ds = gdal.Open(os.path.join(QA_folder, f"{str(doy_list[i])}_QA.TIF"), gdal.GA_Update)
#                                     qa_arr = qa_ds.GetRasterBand(1).ReadAsArray()
#                                     qa_arr = self._process_QA_band(qa_arr, metadata_num[0])
#                                     qa_arr = qa_arr.astype(dtype_out)
#                                     qa_ds.GetRasterBand(1).WriteArray(qa_arr)
#                                     qa_ds.GetRasterBand(1).SetNoDataValue(0)
#                                     qa_ds = None
#                                 else:
#                                     qa_ds = gdal.Open(os.path.join(QA_folder, f"{str(doy_list[i])}_QA.TIF"))
#                                     qa_arr = qa_ds.GetRasterBand(1).ReadAsArray()
#                                 output_arr = output_arr * qa_arr

                        if _sparse_matrix:
                            # Convert the output_arr 2 sparse matrix
                            sm_temp = sm.csr_matrix(output_arr.astype(dtype_out))
                            data_cube.append(sm_temp, name=doy_list[i])
                            data_valid_array[i] = 1 if sm_temp.data.shape[0] == 0 else 0
                        else:
                            data_cube.append(output_arr)
                            data_valid_array[i] = 1 if np.all(output_arr == nodata_value) else 0
                        i += 1

                        pbar.update()

                    except:
                        print(traceback.format_exc())
                        raise Exception(f'Dc construction failed during process {str(doy_list[i])}!')

            if not _sparse_matrix:
                data_cube = np.stack(data_cube, axis=2)

            # remove nan layer
            try:
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
            except:
                print(traceback.format_exc())
                raise Exception(f'Dc construction failed during remove nan layer!')

            # Save Landsat dc
            start_time = time.time()
            try:
                if _sparse_matrix:
                    # Save the sdc
                    np.save(self._dc_infr[_] + f'doy.npy', doy_list)
                    bf.create_folder(f'{self._dc_infr[_]}{str(_)}_sequenced_datacube\\')
                    data_cube.save(f'{self._dc_infr[_]}{str(_)}_sequenced_datacube\\')
                    metadata_dic['Zoffset'], metadata_dic['Nodata_value'] = - nodata_value, 0
                else:
                    metadata_dic['Zoffset'], metadata_dic['Nodata_value'] = None, nodata_value
                    np.save(self._dc_infr[_] + f'doy.npy', doy_list)
                    np.save(f'{self._dc_infr[_]}{str(_)}_sequenced_datacube.npy', data_cube)

                # Save the metadata dic
                metadata_dic['Datatype'] = str(np.iinfo(dtype_out).dtype),
                metadata_dic['sparse_matrix'], metadata_dic['huge_matrix'] = _sparse_matrix, _huge_matrix
                with open(self._dc_infr[_] + 'metadata.json', 'w') as js_temp:
                    json.dump(metadata_dic, js_temp)
            except:
                print(traceback.format_exc())
                raise Exception(f'Dc construction failed during saving the datacube!')

            print(f'Finished writing the \033[1;31m{str(_)}\033[0m sdc in \033[1;34m{str(time.time() - start_time)} s\033[0m.')


class Landsat_dc(object):
    def __init__(self, dc_filepath, work_env=None):

        # Check the dcfile path
        self.dc_filepath = bf.Path(dc_filepath).path_name

        # Def key var
        self.ROI_name, self.ROI, self.ROI_tif, self.ROI_array = None, None, None, None
        self.index, self.Datatype, self.coordinate_system = None, None, None
        self.dc_group_list, self.tiles = None, None
        self.sdc_factor, self.sparse_matrix, self.size_control_factor, self.huge_matrix = False, False, False, False
        self.Nodata_value, self.Zoffset = None, None

        # Def Inundation parameter
        self._DSWE_threshold = None
        self._flood_month_list = None
        self.flood_mapping_method = []

        # Check work env
        if work_env is not None:
            self._work_env = bf.Path(work_env).path_name
        else:
            self._work_env = bf.Path(os.path.dirname(os.path.dirname(self.dc_filepath))).path_name
        self.root_path = bf.Path(os.path.dirname(os.path.dirname(self._work_env))).path_name

        # Define the basic var name
        self._fund_factor = ('ROI_name', 'index', 'Datatype', 'ROI', 'ROI_array', 'sdc_factor',
                             'coordinate_system', 'oritif_folder', 'ROI_tif', 'sparse_matrix',
                             'huge_matrix', 'size_control_factor', 'dc_group_list', 'tiles', 'Zoffset', 'Nodata_value')

        # Read header
        metadata_file = bf.file_filter(self.dc_filepath, ['metadata.json'])
        if len(metadata_file) == 0:
            raise ValueError('There has no valid sdc or the metadata file of the sdc was missing!')
        elif len(metadata_file) > 1:
            raise ValueError('There has more than one metadata file in the dir!')
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
                raise Exception('Something went wrong when reading the metadata!')

        start_time = time.time()
        print(f'Start loading the Landsat dc of \033[1;31m{self.index}\033[0m in the \033[1;34m{self.ROI_name}\033[0m')

        # Read doy or date file of the Datacube
        try:
            if self.sdc_factor is True:
                # Read doylist
                doy_file = bf.file_filter(self.dc_filepath, ['doy.npy'], and_or_factor='and')

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
            if self.sparse_matrix:
                if os.path.exists(self.dc_filepath + f'{self.index}_sequenced_datacube\\'):
                    self.dc = NDSparseMatrix().load(self.dc_filepath + f'{self.index}_sequenced_datacube\\')
                else:
                    raise Exception('Please double check the code if the sparse huge matrix is generated properly')
            elif not self.sparse_matrix and self.huge_matrix:
                self.dc_filename = bf.file_filter(self.dc_filepath, ['sequenced_datacube.npy'])
                if len(self.dc_filename) == 0:
                    raise ValueError('There has no valid dc or the dc was missing!')
                elif len(self.dc_filename) > 1:
                    raise ValueError('There has more than one date file in the dc dir')
                else:
                    self.dc = np.load(self.dc_filename[0], allow_pickle=True)
            elif not self.huge_matrix and not self.sparse_matrix:
                self.dc_filename = bf.file_filter(self.dc_filepath, ['sequenced_datacube', '.npy'], and_or_factor='and')
        except:
            raise Exception('Something went wrong when reading the datacube!')

        # autotrans sparse matrix
        if self.sparse_matrix and self.dc._matrix_type == sm.coo_matrix:
            self._autotrans_sparse_matrix()

        # Backdoor metadata check
        self._backdoor_metadata_check()

        # Size calculation and shape definition
        self.dc_XSize, self.dc_YSize, self.dc_ZSize = self.dc.shape[1], self.dc.shape[0], self.dc.shape[2]

        print(f'Finish loading the Landsat dc of \033[1;31m{self.index}\033[0m for the \033[1;34m{self.ROI_name}\033[0m using \033[1;31m{str(time.time() - start_time)}\033[0ms')

    def __sizeof__(self):
        return self.dc.__sizeof__() + self.sdc_doylist.__sizeof__()

    def _backdoor_metadata_check(self):

        backdoor_issue = False
        # Check if metadata is valid and timeliness
        # Problem 1 dir change between multi-devices ROI ROI arr ROI tif
        if not os.path.exists(self.ROI_tif):
            self.ROI_tif = retrieve_correct_filename(self.ROI_tif)
            backdoor_issue = True

        if not os.path.exists(self.ROI_array):
            self.ROI_array = retrieve_correct_filename(self.ROI_array)
            backdoor_issue = True

        if not os.path.exists(self.ROI):
            self.ROI = retrieve_correct_filename(self.ROI)
            backdoor_issue = True

        if not os.path.exists(self.oritif_folder):
            self.oritif_folder = retrieve_correct_filename(self.oritif_folder)
            backdoor_issue = True

        if self.ROI_tif is None or self.ROI_array is None or self.ROI is None or self.oritif_folder is None:
            raise Exception('Please manually change the roi path in the Landsat dc')

        # Problem 2
        if backdoor_issue:
            self.save(self.dc_filepath)
            self.__init__(self.dc_filepath)

    def _autotrans_sparse_matrix(self):

        if not isinstance(self.dc, NDSparseMatrix):
            raise TypeError('The autotrans sparse matrix is specified for the NDsm!')

        for _ in self.dc.SM_namelist:
            if isinstance(self.dc.SM_group[_], sm.coo_matrix):
                self.dc.SM_group[_] = sm.csr_matrix(self.dc.SM_group[_])

        self.dc._update_size_para()
        self.dc._matrix_type = sm.csr_matrix
        self.save(self.dc_filepath)

    def append(self, append_landsat_dc):
        if not isinstance(append_landsat_dc, Landsat_dc):
            raise TypeError('The Landsat dc can only append with Landsat dc')

        # size consistency
        for _ in ['dc_XSize', 'dc_YSize', 'sparse_matrix', 'index', 'Nodata_value', 'Zoffset', 'size_control_factor']:
            if _ not in self.__dict__.keys() or  _ not in append_landsat_dc.__dict__.keys():
                raise ValueError(f'Missing factor {_} in Landsat dc!')
            elif self.__dict__[_] != append_landsat_dc.__dict__[_]:
                raise ValueError(f'The Landsat dcs didnot share consistency in {_}')

        roi_arr_ori = np.load(self.ROI_array)
        roi_arr_app = np.load(append_landsat_dc.ROI_array)

        if not (roi_arr_ori == roi_arr_app).any():
            raise ValueError('The Landsat dcs didnot share consistency roi')

        # Move all the tif from append folder into the original landsat dc folder
        append_tiffile = bf.file_filter(append_landsat_dc.oritif_folder, ['.TIF', '.tif'], and_or_factor='or')
        for _ in append_tiffile:
            filename = _.split('\\')[-1]
            try:
                shutil.copy(_, f"{self.oritif_folder}{filename}")
            except:
                pass

        # re-arrange the doy and dc file
        if self.sparse_matrix:
            for __ in append_landsat_dc.sdc_doylist:
                if __ not in self.sdc_doylist:
                    self.sdc_doylist.append(__)
                    self.dc.append(append_landsat_dc.dc.SM_group[__], name=__)
        else:
            pass

        self.sdc_doylist.sort()
        self.save(self.dc_filepath)
        self.__init__(self.dc_filepath)

    def save(self, output_path: str):
        start_time = time.time()
        print(f'Start saving the Landsat dc of \033[1;31m{self.index}\033[0m in the \033[1;34m{self.ROI_name}\033[0m')

        if not os.path.exists(output_path):
            bf.create_folder(output_path)
        output_path = bf.Path(output_path).path_name

        # Save the datacube
        if self.sparse_matrix:
            self.dc.save(f'{output_path}{str(self.index)}_sequenced_datacube\\')
        else:
            np.save(f'{output_path}{str(self.index)}_sequenced_datacube.npy', self.dc)

        # Save the doy list
        doy = self.sdc_doylist
        np.save(f'{output_path}doy.npy', doy)

        # Save the metadata
        metadata_dic = {'ROI_name': self.ROI_name, 'index': self.index, 'Datatype': self.Datatype, 'ROI': self.ROI,
                        'ROI_array': self.ROI_array, 'Zoffset': self.Zoffset, 'Nodata_value': self.Nodata_value,
                        'ROI_tif': self.ROI_tif, 'sdc_factor': self.sdc_factor,
                        'coordinate_system': self.coordinate_system,
                        'sparse_matrix': self.sparse_matrix, 'huge_matrix': self.huge_matrix,
                        'size_control_factor': self.size_control_factor,
                        'oritif_folder': self.oritif_folder, 'dc_group_list': self.dc_group_list, 'tiles': self.tiles}

        with open(f'{output_path}metadata.json', 'w') as js_temp:
            json.dump(metadata_dic, js_temp)

        print(f'Finish saving the Landsat dc of \033[1;31m{self.index}\033[0m for the \033[1;34m{self.ROI_name}\033[0m using \033[1;31m{str(time.time() - start_time)}\033[0ms')

    def print_stacked_Zvalue(self, output_foldername: str = None):

        print(f'Start print stacked Zvalue of the Landsat dc \033[1;31m{self.index}\033[0m in the \033[1;34m{self.ROI_name}\033[0m')

        # Print stacked zvalue
        if output_foldername is not None and isinstance(output_foldername, str):
            output_folder = bf.Path(self.dc_filepath).path_name + output_foldername + '\\'
        else:
            output_folder = bf.Path(self.dc_filepath).path_name + 'stacked_Zvalue\\'
        bf.create_folder(output_folder)

        # Get the ROI
        roi_temp = np.load(self.ROI_array)
        roi_xy = np.argwhere(roi_temp != -32768)
        roi_xy = pd.DataFrame(roi_xy, columns=['y', 'x'])
        roi_xy = roi_xy.sort_values(['x', 'y'], ascending=True).reset_index(drop=True)

        roi_xy_list, offset_list, dc_list = [], [], []
        cpu_amount = os.cpu_count()
        roi_xy_itr = int(np.floor(roi_xy.shape[0]/cpu_amount))

        for _ in range(cpu_amount):
            if _ != cpu_amount - 1:
                roi_xy_list.append(np.array(roi_xy[_ * roi_xy_itr: (_ + 1) * roi_xy_itr]))
            else:
                roi_xy_list.append(np.array(roi_xy[_ * roi_xy_itr:]))
            offset_list.append([np.nanmin(roi_xy_list[-1][:, 0]), np.nanmin(roi_xy_list[-1][:, 1])])
            if self.sparse_matrix:
                dc_list.append(self.dc.extract_matrix(([np.nanmin(roi_xy_list[-1][:, 0]), np.nanmax(roi_xy_list[-1][:, 0]) + 1], [np.nanmin(roi_xy_list[-1][:, 1]), np.nanmax(roi_xy_list[-1][:, 1]) + 1], ['all'])))
            else:
                dc_list.append(self.dc[np.nanmin(roi_xy_list[-1][:, 0]): np.nanmax(roi_xy_list[-1][:, 0]) + 1, np.nanmin(roi_xy_list[-1][:, 1]): np.nanmax(roi_xy_list[-1][:, 1]) + 1, :])

        doy_list = bf.date2doy(self.sdc_doylist)
        doy_list = [np.mod(__, 1000) for __ in doy_list]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(print_single_stacked_Zvalue, repeat(output_folder), roi_xy_list, dc_list, offset_list, repeat(self.Nodata_value), repeat(doy_list))


class Landsat_dcs(object):
    def __init__(self, *args, work_env=None, auto_harmonised=True):

        # Generate the datacubes list
        self.Landsat_dcs = []
        for args_temp in args:
            if type(args_temp) != Landsat_dc:
                raise TypeError('The Landsat datacubes was a bunch of Landsat datacube!')
            else:
                self.Landsat_dcs.append(args_temp)

        # Validation and consistency check
        if len(self.Landsat_dcs) == 0:
            raise ValueError('Please input at least one valid Landsat datacube')

        if type(auto_harmonised) != bool:
            raise TypeError('Please input the auto harmonised factor as bool type!')
        else:
            harmonised_factor = False

        self.index_list, ROI_list, ROI_name_list, Datatype_list, ds_list, study_area_list, sdc_factor_list, doy_list = [], [], [], [], [], [], [], []
        x_size, y_size, z_size = 0, 0, 0
        for dc_temp in self.Landsat_dcs:
            if x_size == 0 and y_size == 0 and z_size == 0:
                x_size, y_size, z_size = dc_temp.dc_XSize, dc_temp.dc_YSize, dc_temp.dc_ZSize
            elif x_size != dc_temp.dc_XSize or y_size != dc_temp.dc_YSize:
                raise Exception('Please make sure all the datacube share the same size!')
            elif z_size != dc_temp.dc_ZSize:
                if auto_harmonised:
                    harmonised_factor = True
                else:
                    raise Exception('The datacubes is not consistent in the date dimension! Turn auto harmonised fator as True if wanna avoid this problem!')

            self.index_list.append(dc_temp.VI)
            ROI_name_list.append(dc_temp.ROI_name)
            sdc_factor_list.append(dc_temp.sdc_factor)
            ROI_list.append(dc_temp.ROI)
            ds_list.append(dc_temp.ds_file)
            study_area_list.append(dc_temp.sa_map)
            Datatype_list.append(dc_temp.Datatype)

        if x_size != 0 and y_size != 0 and z_size != 0:
            self.dcs_XSize, self.dcs_YSize, self.dcs_ZSize = x_size, y_size, z_size
        else:
            raise Exception('Please make sure all the datacubes was not void')

        # Check the consistency of the roi list
        if len(ROI_list) == 0 or False in [len(ROI_list) == len(self.index_list), len(self.index_list) == len(sdc_factor_list), len(ROI_name_list) == len(sdc_factor_list), len(ROI_name_list) == len(ds_list), len(ds_list) == len(study_area_list), len(study_area_list) == len(Datatype_list)]:
            raise Exception('The ROI list or the index list for the datacubes were not properly generated!')
        elif False in [roi_temp == ROI_list[0] for roi_temp in ROI_list]:
            raise Exception('Please make sure all datacubes were in the same roi!')
        elif False in [sdc_temp == sdc_factor_list[0] for sdc_temp in sdc_factor_list]:
            raise Exception('Please make sure all dcs were consistent!')
        elif False in [roi_name_temp == ROI_name_list[0] for roi_name_temp in ROI_name_list]:
            raise Exception('Please make sure all dcs were consistent!')
        elif False in [(sa_temp == study_area_list[0]).all() for sa_temp in study_area_list]:
            print('Please make sure all dcs were consistent!')
        elif False in [dt_temp == Datatype_list[0] for dt_temp in Datatype_list]:
            raise Exception('Please make sure all dcs were consistent!')

        # Define the field
        self.ROI = ROI_list[0]
        self.ROI_name = ROI_name_list[0]
        self.sdc_factor = sdc_factor_list[0]
        self.Datatype = Datatype_list[0]
        self.sa_map = study_area_list[0]
        self.ds_file = ds_list[0]

        # Read the doy or date list
        if self.sdc_factor is False:
            raise Exception('Please sequenced the datacubes before further process!')
        else:
            doy_list = [temp.sdc_doylist for temp in self.Landsat_dcs]
            if False in [temp.shape[0] == doy_list[0].shape[0] for temp in doy_list] or False in [(temp == doy_list[0]).all() for temp in doy_list]:
                if auto_harmonised:
                    harmonised_factor = True
                else:
                    raise Exception('The datacubes is not consistent in the date dimension! Turn auto harmonised fator as True if wanna avoid this problem!')
            else:
                self.doy_list = self.Landsat_dcs[0].sdc_doylist

        # Harmonised the dcs
        if harmonised_factor:
            self._auto_harmonised_dcs()

        #  Define the output_path
        if work_env is None:
            self.work_env = bf.Path(os.path.dirname(os.path.dirname(self.Landsat_dcs[0].Denv_dc_filepath))).path_name
        else:
            self.work_env = work_env
        
        # Define var for the flood mapping
        self.inun_det_method_dic = {}
        self._variance_num = 2
        self._inundation_overwritten_factor = False
        self._DEM_path = None
        self._DT_std_fig_construction = False
        self._construct_inundated_dc = True
        self._flood_mapping_accuracy_evaluation_factor = False
        self.inundation_para_folder = self.work_env + '\\Landsat_Inundation_Condition\\Inundation_para\\'
        self._sample_rs_link_list = None
        self._sample_data_path = None

        bf.create_folder(self.inundation_para_folder)

        # Define var for the phenological analysis
        self._curve_fitting_algorithm = None
        self._flood_removal_method = None
        self._curve_fitting_dic = {}

        # Define var for NIPY reconstruction
        self._add_NIPY_dc = True
        self._NIPY_overwritten_factor = False

        # Define var for phenology metrics generation
        self._phenology_index_all = ['annual_ave_VI', 'flood_ave_VI', 'unflood_ave_VI', 'max_VI', 'max_VI_doy',
                                     'bloom_season_ave_VI', 'well_bloom_season_ave_VI']
        self._curve_fitting_dic = {}
        self._all_quantify_str = None

        # Define var for flood_free_phenology_metrics
        self._flood_free_pm = ['annual_max_VI', 'average_VI_between_max_and_flood']

    def _auto_harmonised_dcs(self):

        doy_all = np.array([])
        for dc_temp in self.Landsat_dcs:
            doy_all = np.concatenate([doy_all, dc_temp.sdc_doylist], axis=0)
        doy_all = np.sort(np.unique(doy_all))

        i = 0
        while i < len(self.Landsat_dcs):
            m_factor = False
            for doy in doy_all:
                if doy not in self.Landsat_dcs[i].sdc_doylist:
                    m_factor = True
                    self.Landsat_dcs[i].dc = np.insert(self.Landsat_dcs[i].dc, np.argwhere(doy_all == doy)[0], np.nan * np.zeros([self.Landsat_dcs[i].dc_XSize, self.Landsat_dcs[i].dc_YSize, 1]), axis=2)

            if m_factor:
                self.Landsat_dcs[i].sdc_doylist = copy.copy(doy_all)
                self.Landsat_dcs[i].dc_ZSize = self.Landsat_dcs[i].dc.shape[2]

            i += 1

        z_size, doy_list = 0, []
        for dc_temp in self.Landsat_dcs:
            if z_size == 0:
                z_size = dc_temp.dc_ZSize
            elif z_size != dc_temp.dc_ZSize:
                raise Exception('Auto harmonised failure!')
            doy_list.append(dc_temp.sdc_doylist)

        if False in [temp.shape[0] == doy_list[0].shape[0] for temp in doy_list] or False in [(temp == doy_list[0]).all() for temp in doy_list]:
            raise Exception('Auto harmonised failure!')

        self.dcs_ZSize = z_size
        self.doy_list = doy_list[0]

    def append(self, dc_temp: Landsat_dc) -> None:
        if type(dc_temp) is not Landsat_dc:
            raise TypeError('The appended data should be a Landsat_dc!')

        for indicator in ['ROI', 'ROI_name', 'sdc_factor']:
            if dc_temp.__dict__[indicator] != self.__dict__[indicator]:
                raise ValueError('The appended datacube is not consistent with the original datacubes')

        if self.dcs_XSize != dc_temp.dc_XSize or self.dcs_YSize != dc_temp.dc_YSize or self.dcs_ZSize != dc_temp.dc_ZSize:
            raise ValueError('The appended datacube has different size compared to the original datacubes')

        if (self.doy_list != dc_temp.sdc_doylist).any():
            raise ValueError('The appended datacube has doy list compared to the original datacubes')

        self.index_list.append(dc_temp.VI)
        self.Landsat_dcs.append(dc_temp)

    def extend(self, dcs_temp) -> None:
        if type(dcs_temp) is not Landsat_dcs:
            raise TypeError('The appended data should be a Landsat_dcs!')

        for indicator in ['ROI', 'ROI_name', 'sdc_factor', 'dcs_XSize', 'dcs_YSize', 'dcs_ZSize', 'doy_list']:
            if dcs_temp.__dict__[indicator] != self.__dict__[indicator]:
                raise ValueError('The appended datacube is not consistent with the original datacubes')

        self.index_list.extend(dcs_temp.index_list)
        self.Landsat_dcs.extend(dcs_temp)

    def inundation_detection(self, flood_mapping_method, **kwargs):

        # Check the inundation datacube type, roi name and index type
        inundation_output_path = f'{self.work_env}Landsat_Inundation_Condition\\'
        self.inun_det_method_dic['flood_mapping_approach_list'] = flood_mapping_method
        self.inun_det_method_dic['inundation_output_path'] = inundation_output_path

        # Process the detection method
        self._process_inundation_para(**kwargs)
        
        # Determine the flood mapping method
        rs_dem_factor, DT_factor, DSWE_factor, AWEI_factor = False, False, False, False
        if type(flood_mapping_method) is list:
            for method in flood_mapping_method:
                if method not in self._flood_mapping_method:
                    raise ValueError(f'The flood mapping method {str(method)} was not supported! Only support DSWE, DT rs_dem and AWEI!')
                elif method == 'DSWE':
                    DSWE_factor = True
                elif method == 'DT':
                    DT_factor = True
                elif method == 'AWEI':
                    AWEI_factor = True
                elif method == 'rs_dem':
                    rs_dem_factor = True
        else:
            raise TypeError('Please input the flood mapping method as a list')

        # Main process
        # Flood mapping Method 1 SATE DEM
        if rs_dem_factor:
            if self._DEM_path is None:
                raise Exception('Please input a valid dem_path ')

            if 'MNDWI' in self.index_list:
                doy_temp = self.Landsat_dcs[self.index_list.index('MNDWI')].sdc_doy_list
                mndwi_dc_temp = self.Landsat_dcs[self.index_list.index('MNDWI')].VI
                year_range = range(int(np.true_divide(doy_temp[0], 1000)), int(np.true_divide(doy_temp[-1], 1000) + 1))
                if len(year_range) == 1:
                    raise Exception('Caution! The time span should be larger than two years in order to retrieve interannual inundation information!')

                # Create Inundation Map
                self.inun_det_method_dic['year_range'] = year_range
                self.inun_det_method_dic['rs_dem_inundation_folder'] = f"{self.inun_det_method_dic['inundation_output_path']}Landsat_Inundation_Condition\\{self.ROI_name}_rs_dem_inundated\\"
                bf.create_folder(self.inun_det_method_dic['rs_dem_inundation_folder'])
                for year in year_range:
                    if self._inundation_overwritten_factor or not os.path.exists(self.inun_det_method_dic['rs_dem_inundation_folder'] + str(year) + '_inundation_map.TIF'):
                        inundation_map_regular_month_temp = np.zeros((self.dcs_XSize, self.dcs_YSize), dtype=np.uint8)
                        inundation_map_inundated_month_temp = np.zeros((self.dcs_XSize, self.dcs_YSize), dtype=np.uint8)
                        for doy in doy_temp:
                            if str(year) in str(doy):
                                if str((date.fromordinal(date(year, 1, 1).toordinal() + np.mod(doy, 1000) - 1)).month) not in self._flood_month_list:
                                    inundation_map_regular_month_temp[(mndwi_dc_temp[:, :, np.argwhere(doy_temp == doy)]).reshape(mndwi_dc_temp.shape[0], -1) > self._MNDWI_threshold] = 1
                                elif str((date.fromordinal(date(year, 1, 1).toordinal() + np.mod(doy, 1000) - 1)).month) in self._flood_month_list:
                                    inundation_map_inundated_month_temp[(mndwi_dc_temp[:, :, np.argwhere(doy_temp == doy)]).reshape(mndwi_dc_temp.shape[0], -1) > self._MNDWI_threshold] = 2
                        inundation_map_inundated_month_temp[inundation_map_regular_month_temp == 1] = 1
                        inundation_map_inundated_month_temp[inundation_map_inundated_month_temp == 0] = 255
                        remove_sole_pixel(inundation_map_inundated_month_temp, Nan_value=255, half_size_window=2)
                        MNDWI_temp_ds = gdal.Open((bf.file_filter(self.work_env + 'Landsat_clipped_MNDWI\\', ['MNDWI']))[0])
                        bf.write_raster(MNDWI_temp_ds, inundation_map_inundated_month_temp, self.inun_det_method_dic['rs_dem_inundation_folder'], str(year) + '_inundation_map.TIF')
                        self.inun_det_method_dic[str(year) + '_inundation_map'] = inundation_map_inundated_month_temp
                np.save(self.inun_det_method_dic['rs_dem_inundation_folder'] + self.ROI_name + '_rs_dem_inundated_dic.npy', self.inun_det_method_dic)

                # This section will generate sole inundated area and reconstruct with individual satellite DEM
                print('The DEM fix inundated area procedure could consumes bunch of time! Caution!')
                rs_dem_inundated_dic = np.load(self.inun_det_method_dic['rs_dem_inundation_folder'] + self.ROI_name + '_rs_dem_inundated_dic.npy', allow_pickle=True).item()
                for year in rs_dem_inundated_dic['year_range']:
                    if not os.path.exists(rs_dem_inundated_dic['rs_dem_inundation_folder'] + str(year) + '_sole_water.TIF'):
                        try:
                            ds_temp = gdal.Open(rs_dem_inundated_dic['rs_dem_inundation_folder'] + str(year) + '_inundation_map.TIF')
                        except:
                            raise Exception('Please double check whether the yearly inundation map was properly generated!')

                        temp_array = ds_temp.GetRasterBand(1).ReadAsArray.astype(np.uint8)
                        sole_water = identify_all_inundated_area(temp_array, nan_water_pixel_indicator=None)
                        bf.write_raster(ds_temp, sole_water, rs_dem_inundated_dic['rs_dem_inundation_folder'], str(year) + '_sole_water.TIF')

                DEM_ds = gdal.Open(self._DEM_path + 'dem_' + self.ROI_name + '.tif')
                DEM_band = DEM_ds.GetRasterBand(1)
                DEM_array = gdal_array.BandReadAsArray(DEM_band).astype(np.uint32)

                for year in rs_dem_inundated_dic['year_range']:
                    if not os.path.exists(rs_dem_inundated_dic['inundation_folder'] + str(year) + '_sole_water.TIF'):
                        print('Please double check the sole water map!')
                    elif not os.path.exists(rs_dem_inundated_dic['inundation_folder'] + str(year) + '_sole_water_fixed.TIF'):
                        try:
                            sole_ds_temp = gdal.Open(rs_dem_inundated_dic['inundation_folder'] + str(year) + '_sole_water.TIF')
                            inundated_ds_temp = gdal.Open(rs_dem_inundated_dic['inundation_folder'] + str(year) + '_inundation_map.TIF')
                        except:
                            print('Sole water Map can not be opened!')
                            sys.exit(-1)
                        sole_temp_band = sole_ds_temp.GetRasterBand(1)
                        inundated_temp_band = inundated_ds_temp.GetRasterBand(1)
                        sole_temp_array = gdal_array.BandReadAsArray(sole_temp_band).astype(np.uint32)
                        inundated_temp_array = gdal_array.BandReadAsArray(inundated_temp_band).astype(np.uint8)
                        inundated_array_ttt = complement_all_inundated_area(DEM_array, sole_temp_array, inundated_temp_array)
                        bf.write_raster(DEM_ds, inundated_array_ttt, rs_dem_inundated_dic['inundation_folder'], str(year) + '_sole_water_fixed.TIF')

        elif not rs_dem_factor:

            if DSWE_factor:
                print(f'The inundation area of {self.ROI_name} was mapping using DSWE algorithm!')
                start_time = time.time()
                # Implement the Dynamic Water Surface Extent inundation detection method
                if 'NIR' not in self.index_list or 'MIR2' not in self.index_list or 'MNDWI' not in self.index_list:
                    raise Exception('Please make sure the NIR and MIR2 is properly input when implementing the inundation detection through DSWE method')
                else:
                    self.inun_det_method_dic['DSWE_' + self.ROI_name] = inundation_output_path + self.ROI_name + '_DSWE\\'
                    bf.create_folder(self.inun_det_method_dic['DSWE_' + self.ROI_name])

                inundated_dc = np.array([])
                if not os.path.exists(self.inun_det_method_dic['DSWE_' + self.ROI_name] + 'doy.npy') or not os.path.exists(self.inun_det_method_dic['DSWE_' + self.ROI_name] + 'inundated_dc.npy'):

                    # Input the SDC VI array
                    MNDWI_sdc = self.Landsat_dcs[self.index_list.index('MNDWI')].dc
                    NIR_sdc = self.Landsat_dcs[self.index_list.index('NIR')].dc
                    MIR2_sdc = self.Landsat_dcs[self.index_list.index('MIR2')].dc

                    date_temp = 0
                    while date_temp < len(self.doy_list):
                        if np.all(np.isnan(MNDWI_sdc[:, :, date_temp])) is True:
                            self.doy_list = np.delete(self.doy_list, date_temp, axis=0)
                            MNDWI_sdc = np.delete(MNDWI_sdc, date_temp, axis=2)
                            NIR_sdc = np.delete(NIR_sdc, date_temp, axis=2)
                            MIR2_sdc = np.delete(MIR2_sdc, date_temp, axis=2)
                            date_temp -= 1
                        date_temp += 1

                    bf.create_folder(self.inun_det_method_dic['DSWE_' + self.ROI_name] + 'individual_tif\\')
                    doy_index = 0
                    for doy in self.doy_list:
                        if not os.path.exists(self.inun_det_method_dic['DSWE_' + self.ROI_name] + 'individual_tif\\DSWE_' + str(doy) + '.TIF') or self._inundation_overwritten_factor:
                            year_t = doy // 1000
                            date_t = np.mod(doy, 1000)
                            day_t = datetime.date.fromordinal(datetime.date(year_t, 1, 1).toordinal() + date_t - 1)
                            day_str = str(day_t.year * 10000 + day_t.month * 100 + day_t.day)

                            inundated_array = np.zeros([MNDWI_sdc.shape[0], MNDWI_sdc.shape[1]]).astype(np.int16)
                            for y_temp in range(MNDWI_sdc.shape[0]):
                                for x_temp in range(MNDWI_sdc.shape[1]):
                                    if MNDWI_sdc[y_temp, x_temp, doy_index] == -32768 or np.isnan(NIR_sdc[y_temp, x_temp, doy_index]) or np.isnan(MIR2_sdc[y_temp, x_temp, doy_index]):
                                        inundated_array[y_temp, x_temp] = -2
                                    elif MNDWI_sdc[y_temp, x_temp, doy_index] > self._DSWE_threshold[0]:
                                        inundated_array[y_temp, x_temp] = 1
                                    elif MNDWI_sdc[y_temp, x_temp, doy_index] > self._DSWE_threshold[1] and NIR_sdc[y_temp, x_temp, doy_index] < self._DSWE_threshold[2] and MIR2_sdc[y_temp, x_temp, doy_index] < self._DSWE_threshold[3]:
                                        inundated_array[y_temp, x_temp] = 1
                                    else:
                                        inundated_array[y_temp, x_temp] = 0

                            # inundated_array = reassign_sole_pixel(inundated_array, Nan_value=-32768, half_size_window=2)
                            inundated_array[self.sa_map == -32768] = -2
                            bf.write_raster(gdal.Open(self.ds_file), inundated_array, self.inun_det_method_dic['DSWE_' + self.ROI_name], 'individual_tif\\DSWE_' + str(doy) + '.TIF', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)
                        else:
                            inundated_ds = gdal.Open(self.inun_det_method_dic['DSWE_' + self.ROI_name] + 'individual_tif\\DSWE_' + str(doy) + '.TIF')
                            inundated_array = inundated_ds.GetRasterBand(1).ReadAsArray()

                        if inundated_dc.size == 0:
                            inundated_dc = np.zeros([inundated_array.shape[0], inundated_array.shape[1], 1])
                            inundated_dc[:, :, 0] = inundated_array
                        else:
                            inundated_dc = np.concatenate((inundated_dc, inundated_array.reshape((inundated_array.shape[0], inundated_array.shape[1], 1))), axis=2)
                        doy_index += 1

                    self.inun_det_method_dic['DSWE_doy_file'] = self.inun_det_method_dic['DSWE_' + self.ROI_name] + 'doy.npy'
                    self.inun_det_method_dic['DSWE_dc_file'] = self.inun_det_method_dic['DSWE_' + self.ROI_name] + 'inundated_dc.npy'
                    np.save(self.inun_det_method_dic['DSWE_doy_file'], self.doy_list)
                    np.save(self.inun_det_method_dic['DSWE_dc_file'], inundated_dc)

                # Create annual inundation map
                self.inun_det_method_dic['DSWE_annual_' + self.ROI_name] = self.work_env + 'Landsat_Inundation_Condition\\' + self.ROI_name + '_DSWE\\annual\\'
                bf.create_folder(self.inun_det_method_dic['DSWE_annual_' + self.ROI_name])
                inundated_dc = np.load(self.inun_det_method_dic['DSWE_' + self.ROI_name] + 'inundated_dc.npy')
                doy_array = np.load(self.inun_det_method_dic['DSWE_' + self.ROI_name] + 'doy.npy')
                year_array = np.unique(doy_array // 1000)
                temp_ds = gdal.Open(bf.file_filter(self.inun_det_method_dic['DSWE_' + self.ROI_name] + 'individual_tif\\', ['.TIF'])[0])
                for year in year_array:
                    annual_inundated_map = np.zeros([inundated_dc.shape[0], inundated_dc.shape[1]])
                    annual_inundated_map[self.sa_map == -32768] = -32768
                    if not os.path.exists(self.inun_det_method_dic['DSWE_annual_' + self.ROI_name] + 'DSWE_' + str(year) + '.TIF') or self._inundation_overwritten_factor:
                        for doy_index in range(doy_array.shape[0]):
                            if doy_array[doy_index] // 1000 == year and np.mod(doy_array[doy_index], 1000) >= 182:
                                annual_inundated_map[inundated_dc[:, :, doy_index] > 0] = 1
                        bf.write_raster(temp_ds, annual_inundated_map, self.inun_det_method_dic['DSWE_annual_' + self.ROI_name], 'DSWE_' + str(year) + '.TIF', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)
                print(f'Flood mapping using DSWE algorithm within {self.ROI_name} consumes {str(time.time()-start_time)}s!')

            if AWEI_factor:

                print(f'The inundation area of {self.ROI_name} was mapping using AWEI algorithm!')
                start_time = time.time()

                # Implement the AWEI inundation detection method
                if 'AWEI' not in self.index_list:
                    raise Exception('Please make sure the AWEI is properly input when implementing the inundation detection through AWEI method')
                else:
                    self.inun_det_method_dic['AWEI_' + self.ROI_name] = inundation_output_path + self.ROI_name + '_AWEI\\'
                    bf.create_folder(self.inun_det_method_dic['AWEI_' + self.ROI_name])
                    bf.create_folder(self.inun_det_method_dic['AWEI_' + self.ROI_name] + 'individual_tif\\')
                    AWEI_sdc = self.Landsat_dcs[self.index_list.index('AWEI')].dc

                # Generate the inundation condition
                for doy in range(AWEI_sdc.shape[2]):
                    if not os.path.exists(self.inun_det_method_dic['AWEI_' + self.ROI_name] + 'individual_tif\\AWEI_' + str(self.doy_list[doy]) + '.TIF'):
                        AWEI_temp = AWEI_sdc[:, :, doy]
                        AWEI_temp[AWEI_temp >= 0] = 1
                        AWEI_temp[AWEI_temp < 0] = 0
                        AWEI_temp[np.isnan(AWEI_temp)] = -2
                        AWEI_sdc[:, :, doy] = AWEI_temp
                        bf.write_raster(gdal.Open(self.ds_file), AWEI_temp, self.inun_det_method_dic['AWEI_' + self.ROI_name] + 'individual_tif\\', 'AWEI_' + str(self.doy_list[doy]) + '.TIF', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)

                self.inun_det_method_dic['AWEI_doy_file'] = self.inun_det_method_dic['AWEI_' + self.ROI_name] + 'doy.npy'
                self.inun_det_method_dic['AWEI_dc_file'] = self.inun_det_method_dic['AWEI_' + self.ROI_name] + 'inundated_dc.npy'
                np.save(self.inun_det_method_dic['AWEI_doy_file'], self.doy_list)
                np.save(self.inun_det_method_dic['AWEI_dc_file'], AWEI_sdc)

                # Create annual inundation map for AWEI
                self.inun_det_method_dic['AWEI_annual_' + self.ROI_name] = self.work_env + 'Landsat_Inundation_Condition\\' + self.ROI_name + '_AWEI\\annual\\'
                bf.create_folder(self.inun_det_method_dic['AWEI_annual_' + self.ROI_name])
                inundated_dc = np.load(self.inun_det_method_dic['AWEI_dc_file'])
                doy_array = np.load(self.inun_det_method_dic['AWEI_doy_file'])
                year_array = np.unique(doy_array // 1000)
                temp_ds = gdal.Open(bf.file_filter(self.inun_det_method_dic['AWEI_' + self.ROI_name] + 'individual_tif\\', ['.TIF'])[0])
                for year in year_array:
                    annual_inundated_map = np.zeros([inundated_dc.shape[0], inundated_dc.shape[1]])
                    annual_inundated_map[self.sa_map == -32768] = -32768
                    if not os.path.exists(self.inun_det_method_dic['AWEI_annual_' + self.ROI_name] + 'AWEI_' + str(year) + '.TIF') or self._inundation_overwritten_factor:
                        for doy_index in range(doy_array.shape[0]):
                            if doy_array[doy_index] // 1000 == year and np.mod(doy_array[doy_index], 1000) >= 182:
                                annual_inundated_map[inundated_dc[:, :, doy_index] > 0] = 1
                        bf.write_raster(temp_ds, annual_inundated_map, self.inun_det_method_dic['AWEI_annual_' + self.ROI_name], 'AWEI_' + str(year) + '.TIF', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)

                print(f'Flood mapping using AWEI algorithm within {self.ROI_name} consumes {str(time.time() - start_time)}s!')

            if DT_factor:

                print(f'The inundation area of {self.ROI_name} was mapping using DT algorithm!')
                start_time = time.time()

                # Flood mapping by DT method (DYNAMIC MNDWI THRESHOLD using time-series MNDWI!)
                if 'MNDWI' not in self.index_list:
                    raise Exception('Please make sure the MNDWI is properly input when implementing the inundation detection through DT method')
                else:
                    self.inun_det_method_dic['DT_' + self.ROI_name] = inundation_output_path + self.ROI_name + '_DT\\'
                    bf.create_folder(self.inun_det_method_dic['DT_' + self.ROI_name])
                    bf.create_folder(self.inun_det_method_dic['DT_' + self.ROI_name] + 'individual_tif\\')
                    MNDWI_sdc = self.Landsat_dcs[self.index_list.index('MNDWI')].dc

                doy_array = copy.copy(self.doy_list)
                date_temp = 0
                while date_temp < doy_array.shape[0]:
                    if np.all(np.isnan(MNDWI_sdc[:, :, date_temp])) is True:
                        doy_array = np.delete(doy_array, date_temp, axis=0)
                        MNDWI_sdc = np.delete(MNDWI_sdc, date_temp, axis=2)
                        date_temp -= 1
                    date_temp += 1

                self.inun_det_method_dic['DT_threshold_map_' + self.ROI_name] = inundation_output_path + f'\\{self.ROI_name}_DT\\DT_threshold\\'
                bf.create_folder(self.inun_det_method_dic['DT_threshold_map_' + self.ROI_name])
                if not os.path.exists(self.inun_det_method_dic['DT_threshold_map_' + self.ROI_name] + 'threshold_map.TIF') or not os.path.exists(self.inun_det_method_dic['DT_threshold_map_' + self.ROI_name] + 'bh_threshold_map.TIF'):
                    doy_array_temp = copy.copy(doy_array)
                    MNDWI_sdc_temp = copy.copy(MNDWI_sdc)
                    threshold_array = np.ones([MNDWI_sdc_temp.shape[0], MNDWI_sdc_temp.shape[1]]) * -2
                    bh_threshold_array = np.ones([MNDWI_sdc_temp.shape[0], MNDWI_sdc_temp.shape[1]]) * -2

                    # Process MNDWI sdc
                    if self._DT_bimodal_histogram_factor:

                        # DT method with empirical threshold
                        for y_temp in range(MNDWI_sdc.shape[0]):
                            for x_temp in range(MNDWI_sdc.shape[1]):
                                doy_array_pixel = np.concatenate(np.mod(doy_array_temp, 1000), axis=None)
                                mndwi_temp = np.concatenate(MNDWI_sdc_temp[y_temp, x_temp, :], axis=None)
                                doy_array_pixel = np.delete(doy_array_pixel, np.argwhere(np.isnan(mndwi_temp) == 1))
                                bh_threshold = bimodal_histogram_threshold(mndwi_temp, init_threshold=0.1)
                                if np.isnan(bh_threshold):
                                    bh_threshold = -2
                                mndwi_temp = np.delete(mndwi_temp, np.argwhere(np.isnan(mndwi_temp) == 1))
                                mndwi_temp = np.delete(mndwi_temp, np.argwhere(np.logical_and(doy_array_pixel >= 182, doy_array_pixel <= 300)))
                                mndwi_temp = np.delete(mndwi_temp, np.argwhere(mndwi_temp < -0.7))
                                all_dry_sum = mndwi_temp.shape[0]
                                mndwi_temp = np.delete(mndwi_temp, np.argwhere(mndwi_temp > bh_threshold))
                                if mndwi_temp.shape[0] < 0.5 * all_dry_sum:
                                    threshold_array[y_temp, x_temp] = -1
                                    bh_threshold_array[y_temp, x_temp] = -1
                                    threshold_array[y_temp, x_temp] = np.nan
                                    bh_threshold_array[y_temp, x_temp] = np.nan
                                else:
                                    mndwi_temp_std = np.nanstd(mndwi_temp)
                                    mndwi_ave = np.mean(mndwi_temp)
                                    threshold_array[y_temp, x_temp] = mndwi_ave + self._variance_num * mndwi_temp_std
                                    bh_threshold_array[y_temp, x_temp] = bh_threshold
                    else:

                        # DT method with empirical threshold
                        for y_temp in range(MNDWI_sdc.shape[0]):
                            for x_temp in range(MNDWI_sdc.shape[1]):
                                doy_array_pixel = np.concatenate(np.mod(doy_array_temp, 1000), axis=None)
                                mndwi_temp = np.concatenate(MNDWI_sdc_temp[y_temp, x_temp, :], axis=None)
                                doy_array_pixel = np.delete(doy_array_pixel,
                                                            np.argwhere(np.isnan(mndwi_temp) == 1))
                                mndwi_temp = np.delete(mndwi_temp, np.argwhere(np.isnan(mndwi_temp) == 1))
                                mndwi_temp = np.delete(mndwi_temp, np.argwhere(
                                    np.logical_and(doy_array_pixel >= 182, doy_array_pixel <= 300)))
                                mndwi_temp = np.delete(mndwi_temp, np.argwhere(mndwi_temp < -0.7))
                                all_dry_sum = mndwi_temp.shape[0]
                                mndwi_temp = np.delete(mndwi_temp, np.argwhere(mndwi_temp > 0.123))
                                if mndwi_temp.shape[0] < 0.50 * all_dry_sum:
                                    threshold_array[y_temp, x_temp] = -1
                                elif mndwi_temp.shape[0] < 5:
                                    threshold_array[y_temp, x_temp] = np.nan
                                else:
                                    mndwi_temp_std = np.nanstd(mndwi_temp)
                                    mndwi_ave = np.mean(mndwi_temp)
                                    threshold_array[y_temp, x_temp] = mndwi_ave + self._variance_num * mndwi_temp_std
                        # threshold_array[threshold_array < -0.50] = np.nan
                        threshold_array[threshold_array > 0.123] = 0.123

                    bf.write_raster(gdal.Open(self.ds_file), threshold_array, self.inun_det_method_dic['DT_threshold_map_' + self.ROI_name], 'threshold_map.TIF', raster_datatype=gdal.GDT_Float32, nodatavalue=np.nan)
                    bf.write_raster(gdal.Open(self.ds_file), bh_threshold_array,
                                 self.inun_det_method_dic['DT_threshold_map_' + self.ROI_name], 'bh_threshold_map.TIF',
                                 raster_datatype=gdal.GDT_Float32, nodatavalue=np.nan)
                else:
                    bh_threshold_ds = gdal.Open(self.inun_det_method_dic['DT_threshold_map_' + self.ROI_name] + 'threshold_map.TIF')
                    threshold_ds = gdal.Open(self.inun_det_method_dic['DT_threshold_map_' + self.ROI_name] + 'bh_threshold_map.TIF')
                    threshold_array = threshold_ds.GetRasterBand(1).ReadAsArray()
                    bh_threshold_array = bh_threshold_ds.GetRasterBand(1).ReadAsArray()

                doy_array_temp = copy.copy(doy_array)
                MNDWI_sdc_temp = copy.copy(MNDWI_sdc)
                self.inun_det_method_dic['DT_' + self.ROI_name] = inundation_output_path + self.ROI_name + '_DT\\'
                bf.create_folder(self.inun_det_method_dic['DT_' + self.ROI_name])
                bf.create_folder(self.inun_det_method_dic['DT_' + self.ROI_name] + 'individual_tif\\')
                inundated_dc = np.array([])
                DT_threshold_ds = gdal.Open(self.inun_det_method_dic['DT_threshold_map_' + self.ROI_name] + 'threshold_map.TIF')
                DT_threshold = DT_threshold_ds.GetRasterBand(1).ReadAsArray().astype(np.float16)
                DT_threshold[np.isnan(DT_threshold)] = 0

                # Construct the MNDWI distribution figure
                if self._DT_std_fig_construction:
                    doy_array_temp = copy.copy(doy_array)
                    MNDWI_sdc_temp = copy.copy(MNDWI_sdc)
                    self.inun_det_method_dic['DT_distribution_fig_' + self.ROI_name] = inundation_output_path + f'\\{self.ROI_name}_DT\\MNDWI_dis_thr_fig\\'
                    bf.create_folder(self.inun_det_method_dic['DT_distribution_fig_' + self.ROI_name])

                    # Generate the MNDWI distribution at the pixel level
                    for y_temp in range(MNDWI_sdc.shape[0]):
                        for x_temp in range(MNDWI_sdc.shape[1]):
                            if not os.path.exists(self.inun_det_method_dic['DT_distribution_fig_' + self.ROI_name] + 'MNDWI_distribution_X' + str(x_temp) + '_Y' + str(y_temp) + '.png'):
                                doy_array_pixel = np.concatenate(np.mod(doy_array_temp, 1000), axis=None)
                                mndwi_temp = np.concatenate(MNDWI_sdc_temp[y_temp, x_temp, :], axis=None)
                                mndwi_temp2 = copy.copy(mndwi_temp)
                                mndwi_temp = np.delete(mndwi_temp, np.argwhere(np.logical_and(doy_array_pixel >= 182, doy_array_pixel <= 285)))
                                doy_array_pixel = np.delete(doy_array_pixel, np.argwhere(np.logical_and(doy_array_pixel >= 182, doy_array_pixel <= 285)))
                                mndwi_temp = np.delete(mndwi_temp, np.argwhere(np.isnan(mndwi_temp) == 1))
                                mndwi_temp = np.delete(mndwi_temp, np.argwhere(mndwi_temp > 0))
                                if mndwi_temp.shape[0] != 0:
                                    yy = np.arange(0, 100, 1)
                                    xx = np.ones([100])
                                    mndwi_temp_std = np.std(mndwi_temp)
                                    mndwi_ave = np.mean(mndwi_temp)
                                    plt.xlim(xmax=1, xmin=-1)
                                    plt.ylim(ymax=50, ymin=0)
                                    plt.hist(mndwi_temp2, bins=50, color='#FFA500')
                                    plt.hist(mndwi_temp, bins=20, color='#00FFA5')
                                    plt.plot(xx * mndwi_ave, yy, color='#FFFF00')
                                    plt.plot(xx * bh_threshold_array[y_temp, x_temp], yy, color='#CD0000', linewidth='3')
                                    plt.plot(xx * threshold_array[y_temp, x_temp], yy, color='#0000CD', linewidth='1.5')
                                    plt.plot(xx * (mndwi_ave - mndwi_temp_std), yy, color='#00CD00')
                                    plt.plot(xx * (mndwi_ave + mndwi_temp_std), yy, color='#00CD00')
                                    plt.plot(xx * (mndwi_ave - self._variance_num * mndwi_temp_std), yy, color='#00CD00')
                                    plt.plot(xx * (mndwi_ave + self._variance_num * mndwi_temp_std), yy, color='#00CD00')
                                    plt.savefig(self.inun_det_method_dic['DT_distribution_fig_' + self.ROI_name] + 'MNDWI_distribution_X' + str(x_temp) + '_Y' + str(y_temp) + '.png', dpi=100)
                                    plt.close()

                self.inun_det_method_dic['inundated_doy_file'] = self.inun_det_method_dic['DT_' + self.ROI_name] + 'doy.npy'
                self.inun_det_method_dic['inundated_dc_file'] = self.inun_det_method_dic['DT_' + self.ROI_name] + 'inundated_dc.npy'
                if not os.path.exists(self.inun_det_method_dic['DT_' + self.ROI_name] + 'doy.npy') or not os.path.exists(self.inun_det_method_dic['DT_' + self.ROI_name] + 'inundated_dc.npy'):
                    for date_temp in range(doy_array_temp.shape[0]):
                        if not os.path.exists(self.inun_det_method_dic['DT_' + self.ROI_name] + 'individual_tif\\DT_' + str(doy_array_temp[date_temp]) + '.TIF') or self._inundation_overwritten_factor:
                            MNDWI_array_temp = MNDWI_sdc_temp[:, :, date_temp].reshape(MNDWI_sdc_temp.shape[0], MNDWI_sdc_temp.shape[1])
                            pos_temp = np.argwhere(MNDWI_array_temp > 0)
                            inundation_map = MNDWI_array_temp - DT_threshold
                            inundation_map[inundation_map > 0] = 1
                            inundation_map[inundation_map < 0] = 0
                            inundation_map[np.isnan(inundation_map)] = -2
                            for i in pos_temp:
                                inundation_map[i[0], i[1]] = 1
                            inundation_map = reassign_sole_pixel(inundation_map, Nan_value=-2, half_size_window=2)
                            inundation_map[np.isnan(self.sa_map)] = -32768
                            bf.write_raster(gdal.Open(self.ds_file), inundation_map, self.inun_det_method_dic['DT_' + self.ROI_name] + 'individual_tif\\', 'DT_' + str(doy_array_temp[date_temp]) + '.TIF', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)
                        else:
                            inundated_ds = gdal.Open(self.inun_det_method_dic['DT_' + self.ROI_name] + 'individual_tif\\DT_' + str(doy_array_temp[date_temp]) + '.TIF')
                            inundation_map = inundated_ds.GetRasterBand(1).ReadAsArray()

                        if inundated_dc.size == 0:
                            inundated_dc = np.zeros([inundation_map.shape[0], inundation_map.shape[1], 1])
                            inundated_dc[:, :, 0] = inundation_map
                        else:
                            inundated_dc = np.concatenate((inundated_dc, inundation_map.reshape((inundation_map.shape[0], inundation_map.shape[1], 1))), axis=2)
                    np.save(self.inun_det_method_dic['inundated_doy_file'], doy_array_temp)
                    np.save(self.inun_det_method_dic['inundated_dc_file'], inundated_dc)

                # Create annual inundation map
                self.inun_det_method_dic['DT_annual_' + self.ROI_name] = inundation_output_path + self.ROI_name + '_DT\\annual\\'
                bf.create_folder(self.inun_det_method_dic['DT_annual_' + self.ROI_name])
                inundated_dc = np.load(self.inun_det_method_dic['inundated_dc_file'])
                doy_array = np.load(self.inun_det_method_dic['inundated_doy_file'])
                year_array = np.unique(doy_array // 1000)
                temp_ds = gdal.Open(bf.file_filter(self.inun_det_method_dic['DT_' + self.ROI_name] + 'individual_tif\\', ['.TIF'])[0])
                for year in year_array:
                    annual_inundated_map = np.zeros([inundated_dc.shape[0], inundated_dc.shape[1]])
                    annual_inundated_map[self.sa_map == -32768] = -32768
                    if not os.path.exists(self.inun_det_method_dic['DT_annual_' + self.ROI_name] + 'DT_' + str(year) + '.TIF') or self._inundation_overwritten_factor:
                        for doy_index in range(doy_array.shape[0]):
                            if doy_array[doy_index] // 1000 == year and 182 <= np.mod(doy_array[doy_index], 1000) <= 285:
                                annual_inundated_map[inundated_dc[:, :, doy_index] > 0] = 1
                        bf.write_raster(temp_ds, annual_inundated_map, self.inun_det_method_dic['DT_annual_' + self.ROI_name], 'DT_' + str(year) + '.TIF', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)

                print(f'Flood mapping using DT algorithm within {self.ROI_name} consumes {str(time.time() - start_time)}s!')

            if self._flood_mapping_accuracy_evaluation_factor is True:

                # Factor check
                if self._sample_rs_link_list is None or self._sample_data_path is None:
                    raise ValueError('Please input the sample data path and the accuracy evaluation list!')

                # Initial factor generation
                confusion_dic = {}
                sample_all = glob.glob(self._sample_data_path + self.ROI_name + '\\output\\*.TIF')
                sample_datelist = np.unique(np.array([i[i.find('\\output\\') + 8: i.find('\\output\\') + 16] for i in sample_all]).astype(np.int))
                global_initial_factor = True
                local_initial_factor = True
                AWEI_initial_factor = True

                # Evaluate accuracy
                for sample_date in sample_datelist:
                    pos = np.argwhere(self._sample_rs_link_list == sample_date)
                    if pos.shape[0] == 0:
                        print('Please make sure all the sample are in the metadata file!')
                    else:
                        sample_ds = gdal.Open(self._sample_data_path + self.ROI_name + '\\output\\' +  str(sample_date) + '.TIF')
                        sample_all_temp_raster = sample_ds.GetRasterBand(1).ReadAsArray().astype(np.int16)
                        landsat_doy = self._sample_rs_link_list[pos[0][0], 1] // 10000 * 1000 + datetime.date(self._sample_rs_link_list[pos[0][0], 1] // 10000, np.mod(self._sample_rs_link_list[pos[0][0], 1], 10000) // 100, np.mod(self._sample_rs_link_list[pos[0][0], 1], 100)).toordinal() - datetime.date(self._sample_rs_link_list[pos[0][0], 1] // 10000, 1, 1).toordinal() + 1
                        sample_all_temp_raster[sample_all_temp_raster == 3] = -2
                        sample_all_temp_raster_1 = copy.copy(sample_all_temp_raster).astype(np.float16)
                        sample_all_temp_raster_1[sample_all_temp_raster_1 == -2] = np.nan

                        if DT_factor:

                            landsat_local_temp_ds = gdal.Open(self.inun_det_method_dic['DT_' + self.ROI_name] + 'individual_tif\\DT_' + str(landsat_doy) + '.TIF')
                            landsat_local_temp_raster = landsat_local_temp_ds.GetRasterBand(1).ReadAsArray()
                            confusion_matrix_temp = confusion_matrix_2_raster(landsat_local_temp_raster, sample_all_temp_raster, nan_value=-2)
                            confusion_dic[self.ROI_name + '_DT_' + str(sample_date)] = confusion_matrix_temp
                            landsat_local_temp_raster = landsat_local_temp_raster.astype(np.float16)
                            landsat_local_temp_raster[landsat_local_temp_raster == -2] = np.nan
                            local_error_distribution = landsat_local_temp_raster - sample_all_temp_raster_1
                            local_error_distribution[np.isnan(local_error_distribution)] = 0
                            local_error_distribution[local_error_distribution != 0] = 1
                            if local_initial_factor is True:
                                confusion_matrix_local_sum_temp = confusion_matrix_temp
                                local_initial_factor = False
                                local_error_distribution_sum = local_error_distribution
                            elif local_initial_factor is False:
                                local_error_distribution_sum = local_error_distribution_sum + local_error_distribution
                                confusion_matrix_local_sum_temp[1:, 1:] = confusion_matrix_local_sum_temp[1:, 1: ] + confusion_matrix_temp[1:, 1:]
                            # confusion_pandas = pandas.crosstab(pandas.Series(sample_all_temp_raster, name='Actual'), pandas.Series(landsat_local_temp_raster, name='Predict'))

                        if DSWE_factor:

                            landsat_global_temp_ds = gdal.Open(self.inun_det_method_dic['DSWE_' + self.ROI_name] + 'individual_tif\\DSWE_' + str(landsat_doy) + '.TIF')
                            landsat_global_temp_raster = landsat_global_temp_ds.GetRasterBand(1).ReadAsArray()
                            confusion_matrix_temp = confusion_matrix_2_raster(landsat_global_temp_raster, sample_all_temp_raster, nan_value=-2)
                            confusion_dic[self.ROI_name + '_DSWE_' + str(sample_date)] = confusion_matrix_temp
                            landsat_global_temp_raster = landsat_global_temp_raster.astype(np.float16)
                            landsat_global_temp_raster[landsat_global_temp_raster == -2] = np.nan
                            global_error_distribution = landsat_global_temp_raster - sample_all_temp_raster_1
                            global_error_distribution[np.isnan(global_error_distribution)] = 0
                            global_error_distribution[global_error_distribution != 0] = 1
                            if global_initial_factor is True:
                                confusion_matrix_global_sum_temp = confusion_matrix_temp
                                global_initial_factor = False
                                global_error_distribution_sum = global_error_distribution
                            elif global_initial_factor is False:
                                global_error_distribution_sum = global_error_distribution_sum + global_error_distribution
                                confusion_matrix_global_sum_temp[1:, 1:] = confusion_matrix_global_sum_temp[1:, 1:] + confusion_matrix_temp[1:, 1:]

                        if AWEI_factor:

                            landsat_AWEI_temp_ds = gdal.Open(self.inun_det_method_dic['AWEI_' + self.ROI_name] + 'individual_tif\\AWEI_' + str(landsat_doy) + '.TIF')
                            landsat_AWEI_temp_raster = landsat_AWEI_temp_ds.GetRasterBand(1).ReadAsArray()
                            confusion_matrix_temp = confusion_matrix_2_raster(landsat_AWEI_temp_raster, sample_all_temp_raster, nan_value=-2)
                            confusion_dic[self.ROI_name + '_AWEI_' + str(sample_date)] = confusion_matrix_temp
                            landsat_AWEI_temp_raster = landsat_AWEI_temp_raster.astype(np.float16)
                            landsat_AWEI_temp_raster[landsat_AWEI_temp_raster == -2] = np.nan
                            AWEI_error_distribution = landsat_AWEI_temp_raster - sample_all_temp_raster_1
                            AWEI_error_distribution[np.isnan(AWEI_error_distribution)] = 0
                            AWEI_error_distribution[AWEI_error_distribution != 0] = 1
                            if AWEI_initial_factor is True:
                                confusion_matrix_AWEI_sum_temp = confusion_matrix_temp
                                AWEI_initial_factor = False
                                AWEI_error_distribution_sum = AWEI_error_distribution
                            elif AWEI_initial_factor is False:
                                AWEI_error_distribution_sum = AWEI_error_distribution_sum + AWEI_error_distribution
                                confusion_matrix_AWEI_sum_temp[1:, 1:] = confusion_matrix_AWEI_sum_temp[1:, 1:] + confusion_matrix_temp[1:, 1:]

                confusion_matrix_global_sum_temp = generate_error_inf(confusion_matrix_global_sum_temp)
                confusion_matrix_AWEI_sum_temp = generate_error_inf(confusion_matrix_AWEI_sum_temp)
                confusion_matrix_local_sum_temp = generate_error_inf(confusion_matrix_local_sum_temp)
                confusion_dic['AWEI_acc'] = float(confusion_matrix_AWEI_sum_temp[
                                                        confusion_matrix_AWEI_sum_temp.shape[0] - 1,
                                                        confusion_matrix_AWEI_sum_temp.shape[1] - 1][0:-1])
                confusion_dic['DSWE_acc'] = float(confusion_matrix_global_sum_temp[confusion_matrix_global_sum_temp.shape[0] - 1, confusion_matrix_global_sum_temp.shape[1] - 1][0:-1])
                confusion_dic['DT_acc'] = float(confusion_matrix_local_sum_temp[confusion_matrix_local_sum_temp.shape[0] - 1, confusion_matrix_local_sum_temp.shape[1] - 1][0:-1])
                xlsx_save(confusion_matrix_global_sum_temp, self.work_env + 'Landsat_Inundation_Condition\\DSWE_' + self.ROI_name + '.xlsx')
                xlsx_save(confusion_matrix_local_sum_temp, self.work_env + 'Landsat_Inundation_Condition\\DT_' + self.ROI_name + '.xlsx')
                xlsx_save(confusion_matrix_AWEI_sum_temp, self.work_env + 'Landsat_Inundation_Condition\\AWEI_' + self.ROI_name + '.xlsx')
                bf.write_raster(sample_ds, AWEI_error_distribution_sum, self.work_env + 'Landsat_Inundation_Condition\\', str(self.ROI_name) + '_Error_dis_AWEI.tif', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)
                bf.write_raster(sample_ds, global_error_distribution_sum, self.work_env + 'Landsat_Inundation_Condition\\',
                             str(self.ROI_name) + '_Error_dis_global.tif', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)
                bf.write_raster(sample_ds, local_error_distribution_sum, self.work_env + 'Landsat_Inundation_Condition\\',
                             str(self.ROI_name) + '_Error_dis_local.tif', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)
                np.save(self.work_env + 'Landsat_Inundation_Condition\\Key_dic\\' + self.ROI_name + '_inundation_acc_dic.npy', confusion_dic)

        bf.create_folder(self.work_env + 'Landsat_Inundation_Condition\\Key_dic\\')
        np.save(self.work_env + 'Landsat_Inundation_Condition\\Key_dic\\' + self.ROI_name + '_inundation_dic.npy', self.inun_det_method_dic)

        #     if landsat_detected_inundation_area is True:
        #         try:
        #             confusion_dic = np.load(thalweg_temp.work_env + 'Landsat_key_dic\\' + thalweg_temp.ROI_name + '_inundation_acc_dic.npy', allow_pickle=True).item()
        #         except:
        #             print('Please evaluate the accracy of different methods before detect the inundation area!')
        #             sys.exit(-1)
        #
        #         if confusion_dic['global_acc'] > confusion_dic['local_acc']:
        #             gl_factor = 'global'
        #             inundation_dic = np.load(thalweg_temp.work_env + 'Landsat_key_dic\\' + thalweg_temp.ROI_name + '_global_inundation_dic.npy', allow_pickle=True).item()
        #         elif confusion_dic['global_acc'] <= confusion_dic['local_acc']:
        #             gl_factor = 'local'
        #             inundation_dic = np.load(thalweg_temp.work_env + 'Landsat_key_dic\\' + thalweg_temp.ROI_name + '_local_inundation_dic.npy', allow_pickle=True).item()
        #         else:
        #             print('Systematic error!')
        #             sys.exit(-1)
        #
        #         if os.path.exists(thalweg_temp.work_env + 'Landsat_key_dic\\' + thalweg_temp.ROI_name + '_final_inundation_dic.npy'):
        #             inundation_dic = np.load(thalweg_temp.work_env + 'Landsat_key_dic\\' + thalweg_temp.ROI_name + '_final_inundation_dic.npy', allow_pickle=True).item()
        #         else:
        #             inundation_dic = {}
        #
        #         inundation_dic['final_' + thalweg_temp.ROI_name] = thalweg_temp.work_env + 'Landsat_Inundation_Condition\\' + thalweg_temp.ROI_name + '_final\\'
        #         bf.create_folder(inundation_dic['final_' + thalweg_temp.ROI_name])
        #         if not os.path.exists(inundation_dic['final_' + thalweg_temp.ROI_name] + 'inundated_dc.npy') or not os.path.exists(inundation_dic['final_' + thalweg_temp.ROI_name] + 'doy.npy'):
        #             landsat_inundation_file_list = bf.file_filter(inundation_dic[gl_factor + '_' + thalweg_temp.ROI_name] + 'individual_tif\\', ['.TIF'])
        #             date_array = np.zeros([0]).astype(np.uint32)
        #             inundation_ds = gdal.Open(landsat_inundation_file_list[0])
        #             inundation_raster = inundation_ds.GetRasterBand(1).ReadAsArray()
        #             inundated_area_cube = np.zeros([inundation_raster.shape[0], inundation_raster.shape[1], 0])
        #             for inundation_file in landsat_inundation_file_list:
        #                 inundation_ds = gdal.Open(inundation_file)
        #                 inundation_raster = inundation_ds.GetRasterBand(1).ReadAsArray()
        #                 date_ff = doy2date(np.array([int(inundation_file.split(gl_factor + '_')[1][0:7])]))
        #                 if np.sum(inundation_raster == -2) >= (0.9 * inundation_raster.shape[0] * inundation_raster.shape[1]):
        #                     print('This is a cloud impact image (' + str(date_ff[0]) + ')')
        #                 else:
        #                     if not os.path.exists(inundation_dic['final_' + thalweg_temp.ROI_name] + 'individual_tif\\' + str(date_ff[0]) + '.TIF'):
        #                         inundated_area_mapping = identify_all_inundated_area(inundation_raster, inundated_pixel_indicator=1, nanvalue_pixel_indicator=-2, surrounding_pixel_identification_factor=True, input_detection_method='EightP')
        #                         inundated_area_mapping[thalweg_temp.sa_map == -32768] = -32768
        #                         bf.write_raster(inundation_ds, inundated_area_mapping, inundation_dic['final_' + thalweg_temp.ROI_name] + 'individual_tif\\', str(date_ff[0]) + '.TIF')
        #                     else:
        #                         inundated_area_mapping_ds = gdal.Open(inundation_dic['final_' + thalweg_temp.ROI_name] + 'individual_tif\\' + str(date_ff[0]) + '.TIF')
        #                         inundated_area_mapping = inundated_area_mapping_ds.GetRasterBand(1).ReadAsArray()
        #                     date_array = np.concatenate((date_array, date_ff), axis=0)
        #                     inundated_area_cube = np.concatenate((inundated_area_cube, inundated_area_mapping.reshape([inundated_area_mapping.shape[0], inundated_area_mapping.shape[1], 1])), axis=2)
        #             date_array = date2doy(date_array)
        #             inundation_dic['inundated_doy_file'] = inundation_dic['final_' + thalweg_temp.ROI_name] + 'doy.npy'
        #             inundation_dic['inundated_dc_file'] = inundation_dic['final_' + thalweg_temp.ROI_name] + 'inundated_dc.npy'
        #             np.save(inundation_dic['inundated_dc_file'], inundated_area_cube)
        #             np.save(inundation_dic['inundated_doy_file'], date_array)
        #
        #         # Create the annual inundation map
        #         inundation_dic['final_annual_' + thalweg_temp.ROI_name] = thalweg_temp.work_env + 'Landsat_Inundation_Condition\\' + thalweg_temp.ROI_name + '_final\\annual\\'
        #         bf.create_folder(inundation_dic['final_annual_' + thalweg_temp.ROI_name])
        #         inundated_dc = np.load(inundation_dic['inundated_dc_file'])
        #         doy_array = np.load(inundation_dic['inundated_doy_file'])
        #         year_array = np.unique(doy_array // 1000)
        #         temp_ds = gdal.Open(bf.file_filter(inundation_dic['final_' + thalweg_temp.ROI_name] + 'individual_tif\\', ['.TIF'])[0])
        #         for year in year_array:
        #             annual_inundated_map = np.zeros([inundated_dc.shape[0], inundated_dc.shape[1]])
        #             if not os.path.exists(inundation_dic['final_annual_' + thalweg_temp.ROI_name] + 'final_' + str(year) + '.TIF') or thalweg_temp._inundation_overwritten_factor:
        #                 for doy_index in range(doy_array.shape[0]):
        #                     if doy_array[doy_index] // 1000 == year and 182 <= np.mod(doy_array[doy_index], 1000) <= 285:
        #                         annual_inundated_map[inundated_dc[:, :, doy_index] > 0] = 1
        #                 annual_inundated_map[sa_map == -32768] = -32768
        #                 bf.write_raster(temp_ds, annual_inundated_map, inundation_dic['final_annual_' + thalweg_temp.ROI_name], 'final_' + str(year) + '.TIF', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)
        #         np.save(thalweg_temp.work_env + 'Landsat_key_dic\\' + thalweg_temp.ROI_name + '_final_inundation_dic.npy', inundation_dic)
        #         inundation_approach_dic['approach_list'].append('final')
        #
        #         inundated_area_cube = np.load(inundation_dic['inundated_dc_file'])
        #         date_array = np.load(inundation_dic['inundated_doy_file'])
        #         DEM_ds = gdal.Open(DEM_path + 'dem_' + thalweg_temp.ROI_name + '.tif')
        #         DEM_array = DEM_ds.GetRasterBand(1).ReadAsArray()
        #         if dem_surveyed_date is None:
        #             dem_surveyed_year = int(date_array[0]) // 10000
        #         elif int(dem_surveyed_date) // 10000 > 1900:
        #             dem_surveyed_year = int(dem_surveyed_date) // 10000
        #         else:
        #             print('The dem surveyed date should be input in the format fo yyyymmdd as a 8 digit integer')
        #             sys.exit(-1)
        #
        #         valid_pixel_num = np.sum(~np.isnan(DEM_array))
        #         # The code below execute the dem fix
        #         inundation_dic = np.load(thalweg_temp.work_env + 'Landsat_key_dic\\' + thalweg_temp.ROI_name + '_final_inundation_dic.npy', allow_pickle=True).item()
        #         inundation_dic['DEM_fix_' + thalweg_temp.ROI_name] = thalweg_temp.work_env + 'Landsat_Inundation_Condition\\' + thalweg_temp.ROI_name + '_final\\' + thalweg_temp.ROI_name + '_dem_fixed\\'
        #         bf.create_folder(inundation_dic['DEM_fix_' + thalweg_temp.ROI_name])
        #         if not os.path.exists(inundation_dic['DEM_fix_' + thalweg_temp.ROI_name] + 'fixed_dem_min_' + thalweg_temp.ROI_name + '.tif') or not os.path.exists(inundation_dic['DEM_fix_' + thalweg_temp.ROI_name] + 'fixed_dem_max_' + thalweg_temp.ROI_name + '.tif') or not os.path.exists(inundation_dic['DEM_fix_' + thalweg_temp.ROI_name] + 'inundated_threshold_' + thalweg_temp.ROI_name + '.tif') or not os.path.exists(inundation_dic['DEM_fix_' + thalweg_temp.ROI_name] + 'variation_dem_max_' + thalweg_temp.ROI_name + '.tif') or not os.path.exists(inundation_dic['DEM_fix_' + thalweg_temp.ROI_name] + 'variation_dem_min_' + thalweg_temp.ROI_name + '.tif') or not os.path.exists(inundation_dic['DEM_fix_' + thalweg_temp.ROI_name] + 'dem_fix_num_' + thalweg_temp.ROI_name + '.tif'):
        #             water_level_data = excel2water_level_array(water_level_data_path, Year_range, CrossSection)
        #             year_range = range(int(np.min(water_level_data[:, 0] // 10000)), int(np.max(water_level_data[:, 0] // 10000) + 1))
        #             min_dem_pos = np.argwhere(DEM_array == np.nanmin(DEM_array))
        #             # The first layer displays the maximum variation and second for the minimum and the third represents the
        #             inundated_threshold_new = np.zeros([DEM_array.shape[0], DEM_array.shape[1]])
        #             dem_variation = np.zeros([DEM_array.shape[0], DEM_array.shape[1], 3])
        #             dem_new_max = copy.copy(DEM_array)
        #             dem_new_min = copy.copy(DEM_array)
        #
        #             for i in range(date_array.shape[0]):
        #                 if date_array[i] // 10000 > 2004:
        #                     inundated_temp = inundated_area_cube[:, :, i]
        #                     temp_tif_file = bf.file_filter(inundation_dic['local_' + thalweg_temp.ROI_name], [str(date2doy(date_array[i])) + '.TIF'])
        #                     temp_ds = gdal.Open(temp_tif_file[0])
        #                     temp_raster = temp_ds.GetRasterBand(1).ReadAsArray()
        #                     temp_raster[temp_raster != -2] = 1
        #                     current_pixel_num = np.sum(temp_raster[temp_raster != -2])
        #                     if date_array[i] // 10000 in year_range and current_pixel_num > 1.09 * valid_pixel_num:
        #                         date_pos = np.argwhere(water_level_data == date_array[i])
        #                         if date_pos.shape[0] == 0:
        #                             print('The date is not found!')
        #                             sys.exit(-1)
        #                         else:
        #                             water_level_temp = water_level_data[date_pos[0, 0], 1]
        #                         inundated_array_temp = inundated_area_cube[:, :, i]
        #                         surrounding_mask = np.zeros([inundated_array_temp.shape[0], inundated_array_temp.shape[1]]).astype(np.int16)
        #                         inundated_mask = np.zeros([inundated_array_temp.shape[0], inundated_array_temp.shape[1]]).astype(np.int16)
        #                         surrounding_mask[np.logical_or(inundated_array_temp == -1 * inundated_array_temp[min_dem_pos[0, 0], min_dem_pos[0, 1]], np.mod(inundated_array_temp, 10000) == -1 * inundated_array_temp[min_dem_pos[0, 0], min_dem_pos[0, 1]], inundated_array_temp // 10000 == -1 * inundated_array_temp[min_dem_pos[0, 0], min_dem_pos[0, 1]])] = 1
        #                         inundated_mask[inundated_array_temp == inundated_array_temp[min_dem_pos[0, 0], min_dem_pos[0, 1]]] = 1
        #                         pos_inundated_temp = np.argwhere(inundated_mask == 1)
        #                         pos_temp = np.argwhere(surrounding_mask == 1)
        #                         for i_temp in range(pos_temp.shape[0]):
        #                             if DEM_array[pos_temp[i_temp, 0], pos_temp[i_temp, 1]] < water_level_temp:
        #                                 dem_variation[pos_temp[i_temp, 0], pos_temp[i_temp, 1], 2] += 1
        #                                 if dem_variation[pos_temp[i_temp, 0], pos_temp[i_temp, 1], 1] == 0 or water_level_temp < dem_variation[pos_temp[i_temp, 0], pos_temp[i_temp, 1], 1]:
        #                                     dem_variation[pos_temp[i_temp, 0], pos_temp[i_temp, 1], 1] = water_level_temp
        #                                 if water_level_temp > dem_variation[pos_temp[i_temp, 0], pos_temp[i_temp, 1], 0]:
        #                                     dem_variation[pos_temp[i_temp, 0], pos_temp[i_temp, 1], 0] = water_level_temp
        #                         for i_temp_2 in range(pos_inundated_temp.shape[0]):
        #                             if inundated_threshold_new[pos_inundated_temp[i_temp_2, 0], pos_inundated_temp[i_temp_2, 1]] == 0:
        #                                 inundated_threshold_new[pos_inundated_temp[i_temp_2, 0], pos_inundated_temp[i_temp_2, 1]] = water_level_temp
        #                             elif water_level_temp < inundated_threshold_new[pos_inundated_temp[i_temp_2, 0], pos_inundated_temp[i_temp_2, 1]]:
        #                                 inundated_threshold_new[pos_inundated_temp[i_temp_2, 0], pos_inundated_temp[i_temp_2, 1]] = water_level_temp
        #                                 dem_variation[pos_inundated_temp[i_temp, 0], pos_inundated_temp[i_temp, 1], 2] += 1
        #
        #             dem_max_temp = dem_variation[:, :, 0]
        #             dem_min_temp = dem_variation[:, :, 1]
        #             dem_new_max[dem_max_temp != 0] = 0
        #             dem_new_max = dem_new_max + dem_max_temp
        #             dem_new_min[dem_min_temp != 0] = 0
        #             dem_new_min = dem_new_min + dem_min_temp
        #             bf.write_raster(DEM_ds, dem_new_min, inundation_dic['DEM_fix_' + thalweg_temp.ROI_name], 'fixed_dem_min_' + thalweg_temp.ROI_name + '.tif')
        #             bf.write_raster(DEM_ds, dem_new_max, inundation_dic['DEM_fix_' + thalweg_temp.ROI_name], 'fixed_dem_max_' + thalweg_temp.ROI_name + '.tif')
        #             bf.write_raster(DEM_ds, inundated_threshold_new, inundation_dic['DEM_fix_' + thalweg_temp.ROI_name], 'inundated_threshold_' + thalweg_temp.ROI_name + '.tif')
        #             bf.write_raster(DEM_ds, dem_variation[:, :, 0], inundation_dic['DEM_fix_' + thalweg_temp.ROI_name], 'variation_dem_max_' + thalweg_temp.ROI_name + '.tif')
        #             bf.write_raster(DEM_ds, dem_variation[:, :, 1], inundation_dic['DEM_fix_' + thalweg_temp.ROI_name], 'variation_dem_min_' + thalweg_temp.ROI_name + '.tif')
        #             bf.write_raster(DEM_ds, dem_variation[:, :, 2], inundation_dic['DEM_fix_' + thalweg_temp.ROI_name], 'dem_fix_num_' + thalweg_temp.ROI_name + '.tif')
        #
        #     if surveyed_inundation_detection_factor:
        #         if Year_range is None or CrossSection is None or VEG_path is None or water_level_data_path is None:
        #             print('Please input the required year range, the cross section name or the Veg distribution.')
        #             sys.exit(-1)
        #         DEM_ds = gdal.Open(DEM_path + 'dem_' + thalweg_temp.ROI_name + '.tif')
        #         DEM_array = DEM_ds.GetRasterBand(1).ReadAsArray()
        #         VEG_ds = gdal.Open(VEG_path + 'veg_' + thalweg_temp.ROI_name + '.tif')
        #         VEG_array = VEG_ds.GetRasterBand(1).ReadAsArray()
        #         water_level_data = excel2water_level_array(water_level_data_path, Year_range, CrossSection)
        #         if os.path.exists(thalweg_temp.work_env + 'Landsat_key_dic\\' + thalweg_temp.ROI_name + '_survey_inundation_dic.npy'):
        #             survey_inundation_dic = np.load(thalweg_temp.work_env + 'Landsat_key_dic\\' + thalweg_temp.ROI_name + '_survey_inundation_dic.npy', allow_pickle=True).item()
        #         else:
        #             survey_inundation_dic = {}
        #         survey_inundation_dic['year_range'] = Year_range,
        #         survey_inundation_dic['date_list'] = water_level_data[:, 0],
        #         survey_inundation_dic['CrossSection'] = CrossSection
        #         survey_inundation_dic['study_area'] = thalweg_temp.ROI_name
        #         survey_inundation_dic['surveyed_' + thalweg_temp.ROI_name] = str(thalweg_temp.work_env) + 'Landsat_Inundation_Condition\\' + str(thalweg_temp.ROI_name) + '_survey\\'
        #         bf.create_folder(survey_inundation_dic['surveyed_' + thalweg_temp.ROI_name])
        #         inundated_doy = np.array([])
        #         inundated_dc = np.array([])
        #         if not os.path.exists(survey_inundation_dic['surveyed_' + thalweg_temp.ROI_name] + 'inundated_dc.npy') or not os.path.exists(survey_inundation_dic['surveyed_' + thalweg_temp.ROI_name] + 'doy.npy'):
        #             for year in range(np.amin(water_level_data[:, 0].astype(np.int32) // 10000, axis=0), np.amax(water_level_data[:, 0].astype(np.int32) // 10000, axis=0) + 1):
        #                 if not os.path.exists(survey_inundation_dic['surveyed_' + thalweg_temp.ROI_name] + 'annual_tif\\' + str(year) + '\\inundation_detection_cube.npy') or not os.path.exists(survey_inundation_dic['surveyed_' + thalweg_temp.ROI_name] + 'annual_tif\\' + str(year) + '\\inundation_height_cube.npy') or not os.path.exists(survey_inundation_dic['surveyed_' + thalweg_temp.ROI_name] + 'annual_tif\\' + str(year) + '\\inundation_date.npy') or not os.path.exists(survey_inundation_dic['surveyed_' + thalweg_temp.ROI_name] + 'annual_tif\\' + str(year) + '\\yearly_inundation_condition.TIF') or thalweg_temp._inundation_overwritten_factor:
        #                     inundation_detection_cube, inundation_height_cube, inundation_date_array = inundation_detection_surveyed_daily_water_level(DEM_array, water_level_data, VEG_array, year_factor=year)
        #                     bf.create_folder(survey_inundation_dic['surveyed_' + thalweg_temp.ROI_name] + 'annual_tif\\' + str(year) + '\\')
        #                     np.save(survey_inundation_dic['surveyed_' + thalweg_temp.ROI_name] + 'annual_tif\\' + str(year) + '\\inundation_height_cube.npy', inundation_height_cube)
        #                     np.save(survey_inundation_dic['surveyed_' + thalweg_temp.ROI_name] + 'annual_tif\\' + str(year) + '\\inundation_date.npy', inundation_date_array)
        #                     np.save(survey_inundation_dic['surveyed_' + thalweg_temp.ROI_name] + 'annual_tif\\' + str(year) + '\\inundation_detection_cube.npy', inundation_detection_cube)
        #                     yearly_inundation_condition = np.sum(inundation_detection_cube, axis=2)
        #                     yearly_inundation_condition[sa_map == -32768] = -32768
        #                     bf.write_raster(DEM_ds, yearly_inundation_condition, survey_inundation_dic['surveyed_' + thalweg_temp.ROI_name] + 'annual_tif\\' + str(year) + '\\', 'yearly_inundation_condition.TIF', raster_datatype=gdal.GDT_UInt16)
        #                 else:
        #                     inundation_date_array = np.load(survey_inundation_dic['surveyed_' + thalweg_temp.ROI_name] + 'annual_tif\\' + str(year) + '\\inundation_date.npy')
        #                     inundation_date_array = np.delete(inundation_date_array, np.argwhere(inundation_date_array == 0))
        #                     inundation_detection_cube = np.load(survey_inundation_dic['surveyed_' + thalweg_temp.ROI_name] + 'annual_tif\\' + str(year) + '\\inundation_detection_cube.npy')
        #                     inundation_date_array = date2doy(inundation_date_array.astype(np.int32))
        #
        #                 if inundated_doy.size == 0 or inundated_dc.size == 0:
        #                     inundated_dc = np.zeros([inundation_detection_cube.shape[0], inundation_detection_cube.shape[1], inundation_detection_cube.shape[2]])
        #                     inundated_dc[:, :, :] = inundation_detection_cube
        #                     inundated_doy = inundation_date_array
        #                 else:
        #                     inundated_dc = np.concatenate((inundated_dc, inundation_detection_cube), axis=2)
        #                     inundated_doy = np.append(inundated_doy, inundation_date_array)
        #             survey_inundation_dic['inundated_doy_file'] = survey_inundation_dic['surveyed_' + thalweg_temp.ROI_name] + 'doy.npy'
        #             survey_inundation_dic['inundated_dc_file'] = survey_inundation_dic['surveyed_' + thalweg_temp.ROI_name] + 'inundated_dc.npy'
        #             np.save(survey_inundation_dic['inundated_dc_file'], inundated_dc)
        #             np.save(survey_inundation_dic['inundated_doy_file'], inundated_doy)
        #
        #         survey_inundation_dic['surveyed_annual_' + thalweg_temp.ROI_name] = thalweg_temp.work_env + 'Landsat_Inundation_Condition\\' + thalweg_temp.ROI_name + '_survey\\annual\\'
        #         bf.create_folder(survey_inundation_dic['surveyed_annual_' + thalweg_temp.ROI_name])
        #         doy_array = np.load(survey_inundation_dic['inundated_doy_file'])
        #         year_array = np.unique(doy_array // 1000)
        #         for year in year_array:
        #             temp_ds = gdal.Open(bf.file_filter(survey_inundation_dic['surveyed_' + thalweg_temp.ROI_name] + 'annual_tif\\' + str(year) + '\\', ['.TIF'])[0])
        #             temp_array = temp_ds.GetRasterBand(1).ReadAsArray()
        #             annual_inundated_map = np.zeros([temp_array.shape[0], temp_array.shape[1]])
        #             if not os.path.exists(survey_inundation_dic['surveyed_annual_' + thalweg_temp.ROI_name] + 'survey_' + str(year) + '.TIF') or thalweg_temp._inundation_overwritten_factor:
        #                 annual_inundated_map[temp_array > 0] = 1
        #                 annual_inundated_map[sa_map == -32768] = -32768
        #                 bf.write_raster(temp_ds, annual_inundated_map, survey_inundation_dic['surveyed_annual_' + thalweg_temp.ROI_name], 'survey_' + str(year) + '.TIF', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)
        #         np.save(thalweg_temp.work_env + 'Landsat_key_dic\\' + thalweg_temp.ROI_name + '_survey_inundation_dic.npy', survey_inundation_dic)
        #         inundation_approach_dic['approach_list'].append('survey')
        # inundation_list_temp = np.unique(np.array(inundation_approach_dic['approach_list']))
        # inundation_approach_dic['approach_list'] = inundation_list_temp.tolist()
        # np.save(thalweg_temp.work_env + 'Landsat_key_dic\\' + str(thalweg_temp.ROI_name) + '_inundation_approach_list.npy', inundation_approach_dic)

        # Construct and append the inundated dc
        if self._construct_inundated_dc:
            for method_temp in flood_mapping_method:
                inundated_file_folder = self.inun_det_method_dic[method_temp + '_' + self.ROI_name] + 'individual_tif\\'
                bf.create_folder(self.inun_det_method_dic[method_temp + '_' + self.ROI_name] + 'datacube\\')
                self.files2sdc(inundated_file_folder, method_temp, self.inun_det_method_dic[method_temp + '_' + self.ROI_name] + 'datacube\\')
                self.append(Landsat_dc(self.inun_det_method_dic[method_temp + '_' + self.ROI_name] + 'datacube\\', sdc_factor=self.sdc_factor))

    def calculate_inundation_duration(self, file_folder, file_priority, output_folder, water_level_data,
                                   nan_value=-2, inundated_value=1, generate_inundation_status_factor=True,
                                   generate_max_water_level_factor=False, example_date=None):

        output_folder = output_folder + str(self.ROI_name) + '\\'
        bf.create_folder(output_folder)

        if type(file_folder) == str:
            file_folder = list(file_folder)
        elif type(file_folder) == list:
            pass
        else:
            print('Please input the file folder as a correct data type!')
            sys.exit(-1)

        if generate_inundation_status_factor and (example_date is None):
            print('Please input the required para')
            sys.exit(-1)
        else:
            example_date = str(bf.doy2date(example_date))

        if type(file_priority) != list:
            print('Please input the priority as a list!')
            sys.exit(-1)
        elif len(file_priority) != len(file_folder):
            print('Please make sure the file folder has a same shape as priority')
            sys.exit(-1)

        if type(water_level_data) != np.array:
            try:
                water_level_data = np.array(water_level_data)
            except:
                print('Please input the water level data in a right format!')
                pass

        if water_level_data.shape[1] != 2:
            print('Please input the water level data as a 2d array with shape under n*2!')
            sys.exit(-1)
        water_level_data[:, 0] = water_level_data[:, 0].astype(np.int)
        date_range = [np.min(water_level_data[:, 0]), np.max(water_level_data[:, 0])]
        if not os.path.exists(output_folder + '\\recession.xlsx'):
            recession_list = []
            year_list = np.unique(water_level_data[:, 0] // 10000).astype(np.int)
            for year in year_list:
                recession_turn = 0
                for data_size in range(1, water_level_data.shape[0]):
                    if year * 10000 + 6 * 100 <= water_level_data[data_size, 0] < year * 10000 + 1100:
                        if recession_turn == 0:
                            if (water_level_data[data_size, 1] > water_level_data[data_size - 1, 1] and
                                water_level_data[data_size, 1] > water_level_data[data_size + 1, 1]) or (
                                    water_level_data[data_size, 1] < water_level_data[data_size - 1, 1] and
                                    water_level_data[data_size, 1] < water_level_data[data_size + 1, 1]) or (
                                    water_level_data[data_size - 1, 1] == water_level_data[data_size, 1] ==
                                    water_level_data[data_size + 1, 1]):
                                recession_list.append([water_level_data[data_size, 0], 0])
                            elif (water_level_data[data_size - 1, 1] < water_level_data[data_size, 1] <=
                                  water_level_data[data_size + 1, 1]) or (
                                    water_level_data[data_size - 1, 1] <= water_level_data[data_size, 1] <
                                    water_level_data[data_size + 1, 1]):
                                recession_list.append([water_level_data[data_size, 0], 1])
                                recession_turn = 1
                            elif (water_level_data[data_size - 1, 1] > water_level_data[data_size, 1] >=
                                  water_level_data[data_size + 1, 1]) or (
                                    water_level_data[data_size - 1, 1] >= water_level_data[data_size, 1] >
                                    water_level_data[data_size + 1, 1]):
                                recession_list.append([water_level_data[data_size, 0], -1])
                                recession_turn = -1
                            else:
                                print('error occrrued recession!')
                                sys.exit(-1)

                        elif recession_turn != 0:
                            if (water_level_data[data_size, 1] > water_level_data[data_size - 1, 1] and
                                water_level_data[data_size, 1] > water_level_data[data_size + 1, 1]) or (
                                    water_level_data[data_size, 1] < water_level_data[data_size - 1, 1] and
                                    water_level_data[data_size, 1] < water_level_data[data_size + 1, 1]) or (
                                    water_level_data[data_size - 1, 1] == water_level_data[data_size, 1] ==
                                    water_level_data[data_size + 1, 1]):
                                recession_list.append([water_level_data[data_size, 0], 0])
                            elif (water_level_data[data_size - 1, 1] < water_level_data[data_size, 1] <=
                                  water_level_data[data_size + 1, 1]) or (
                                    water_level_data[data_size - 1, 1] <= water_level_data[data_size, 1] <
                                    water_level_data[data_size + 1, 1]):
                                if recession_turn > 0:
                                    recession_list.append([water_level_data[data_size, 0], recession_turn])
                                else:
                                    recession_turn = -recession_turn + 1
                                    recession_list.append([water_level_data[data_size, 0], recession_turn])
                            elif (water_level_data[data_size - 1, 1] > water_level_data[data_size, 1] >=
                                  water_level_data[data_size + 1, 1]) or (
                                    water_level_data[data_size - 1, 1] >= water_level_data[data_size, 1] >
                                    water_level_data[data_size + 1, 1]):
                                if recession_turn < 0:
                                    recession_list.append([water_level_data[data_size, 0], recession_turn])
                                else:
                                    recession_turn = -recession_turn
                                    recession_list.append([water_level_data[data_size, 0], recession_turn])
                            else:
                                print('error occrrued recession!')
                                sys.exit(-1)
            recession_list = np.array(recession_list)
            recession_list = pd.DataFrame(recession_list)
            recession_list.to_excel(output_folder + '\\recession.xlsx')
        else:
            recession_list = pandas.read_excel(output_folder + '\\recession.xlsx')
            recession_list = np.array(recession_list).astype(np.object)[:, 1:]
            recession_list[:, 0] = recession_list[:, 0].astype(np.int)

        # water_level_data[:, 0] = doy2date(water_level_data[:, 0])
        tif_file = []
        for folder in file_folder:
            if os.path.exists(folder):
                tif_file_temp = bf.file_filter(folder, ['.tif', '.TIF'], and_or_factor='or',
                                            exclude_word_list=['.xml', '.aux', '.cpg', '.dbf', '.lock'])
                tif_file.extend(tif_file_temp)
            else:
                pass
        if len(tif_file) == 0:
            print('None file meet the requirement')
            sys.exit(-1)

        if generate_max_water_level_factor:
            date_list = []
            priority_list = []
            water_level_list = []
            for file_path in tif_file:
                ds_temp = gdal.Open(file_path)
                raster_temp = ds_temp.GetRasterBand(1).ReadAsArray()
                true_false_array = raster_temp == nan_value
                if not true_false_array.all():
                    for length in range(len(file_path)):
                        try:
                            date_temp = int(file_path[length: length + 8])
                            break
                        except:
                            pass

                        try:
                            date_temp = int(file_path[length: length + 7])
                            date_temp = bf.doy2date(date_temp)
                            break
                        except:
                            pass
                    if date_range[0] < date_temp < date_range[1]:
                        date_list.append(date_temp)
                        date_num = np.argwhere(water_level_data == date_temp)[0, 0]
                        water_level_list.append(water_level_data[date_num, 1])
                        for folder_num in range(len(file_folder)):
                            if file_folder[folder_num] in file_path:
                                priority_list.append(file_priority[folder_num])

            if len(date_list) == len(priority_list) and len(priority_list) == len(water_level_list) and len(
                    date_list) != 0:
                information_array = np.ones([3, len(date_list)])
                information_array[0, :] = np.array(date_list)
                information_array[1, :] = np.array(priority_list)
                information_array[2, :] = np.array(water_level_list)
                year_array = np.unique(np.fix(information_array[0, :] // 10000))
                unique_level_array = np.unique(information_array[1, :])
                annual_max_water_level = np.ones([year_array.shape[0], unique_level_array.shape[0]])
                annual_max = []
                for year in range(year_array.shape[0]):
                    water_level_max = 0
                    for date_temp in range(water_level_data.shape[0]):
                        if water_level_data[date_temp, 0] // 10000 == year_array[year]:
                            water_level_max = max(water_level_max, water_level_data[date_temp, 1])
                    annual_max.append(water_level_max)
                    for unique_level in range(unique_level_array.shape[0]):
                        max_temp = 0
                        for i in range(information_array.shape[1]):
                            if np.fix(information_array[0, i] // 10000) == year_array[year] and information_array[1, i] == unique_level_array[unique_level]:
                                max_temp = max(max_temp, information_array[2, i])
                        annual_max_water_level[year, unique_level] = max_temp
                annual_max = np.array(annual_max)
                annual_max_water_level_dic = np.zeros([year_array.shape[0] + 1, unique_level_array.shape[0] + 2])
                annual_max_water_level_dic[1:, 2:] = annual_max_water_level
                annual_max_water_level_dic[1:, 0] = year_array.transpose()
                annual_max_water_level_dic[1:, 1] = annual_max.transpose()
                annual_max_water_level_dic[0, 2:] = unique_level_array
                dic_temp = pd.DataFrame(annual_max_water_level_dic)
                dic_temp.to_excel(output_folder + self.ROI_name + '.xlsx')

        if generate_inundation_status_factor:
            combined_file_path = output_folder + 'Original_Inundation_File\\'
            bf.create_folder(combined_file_path)

            for file_path in tif_file:
                ds_temp = gdal.Open(file_path)
                raster_temp = ds_temp.GetRasterBand(1).ReadAsArray()
                if self.sa_map.shape[0] != raster_temp.shape[0] or self.sa_map.shape[1] != raster_temp.shape[1]:
                    print('Consistency error sa map')
                    sys.exit(-1)
                true_false_array = raster_temp == nan_value
                if not true_false_array.all():
                    for length in range(len(file_path)):
                        try:
                            date_temp = int(file_path[length: length + 8])
                            break
                        except:
                            pass

                        try:
                            date_temp = int(file_path[length: length + 7])
                            date_temp = bf.doy2date(date_temp)
                            break
                        except:
                            pass
                    if date_range[0] < date_temp < date_range[1]:
                        if 6 <= np.mod(date_temp, 10000) // 100 <= 10:
                            if not os.path.exists(combined_file_path + 'inundation_' + str(date_temp) + '.tif'):
                                for folder_num in range(len(file_folder)):
                                    if file_folder[folder_num] in file_path:
                                        priority_temp = file_priority[folder_num]
                                if priority_temp == 0:
                                    shutil.copyfile(file_path,
                                                    combined_file_path + 'inundation_' + str(date_temp) + '.tif')
                                else:
                                    ds_temp_temp = gdal.Open(file_path)
                                    raster_temp_temp = ds_temp_temp.GetRasterBand(1).ReadAsArray()
                                    raster_temp_temp[raster_temp_temp != inundated_value] = nan_value
                                    bf.write_raster(ds_temp_temp, raster_temp_temp, combined_file_path,
                                                 'inundation_' + str(date_temp) + '.tif',
                                                 raster_datatype=gdal.GDT_Int16)

            inundation_file = bf.file_filter(combined_file_path, containing_word_list=['.tif'])
            sole_file_path = output_folder + 'Sole_Inundation_File\\'
            bf.create_folder(sole_file_path)

            date_dc = []
            inundation_dc = []
            sole_area_dc = []
            river_sample = []
            recession_dc = []
            num_temp = 0

            if not os.path.exists(output_folder + 'example.tif'):
                for file in inundation_file:
                    if example_date in file:
                        example_ds = gdal.Open(file)
                        example_raster = example_ds.GetRasterBand(1).ReadAsArray()
                        example_sole = identify_all_inundated_area(example_raster,
                                                                   inundated_pixel_indicator=inundated_value,
                                                                   nanvalue_pixel_indicator=nan_value)
                        unique_sole = np.unique(example_sole.flatten())
                        unique_sole = np.delete(unique_sole, np.argwhere(unique_sole == 0))
                        amount_sole = [np.sum(example_sole == value) for value in unique_sole]
                        example_sole[example_sole != unique_sole[np.argmax(amount_sole)]] = -10
                        example_sole[example_sole == unique_sole[np.argmax(amount_sole)]] = 1
                        example_sole = example_sole.astype(np.float16)
                        example_sole[example_sole == -10] = np.nan
                        river_sample = example_sole
                        bf.write_raster(example_ds, river_sample, output_folder, 'example.tif',
                                     raster_datatype=gdal.GDT_Float32)
                        break
            else:
                example_ds = gdal.Open(output_folder + 'example.tif')
                river_sample = example_ds.GetRasterBand(1).ReadAsArray()

            if river_sample == []:
                print('Error in river sample creation.')
                sys.exit(-1)

            if not os.path.exists(output_folder + 'date_dc.npy') or not os.path.exists(
                    output_folder + 'date_dc.npy') or not os.path.exists(output_folder + 'date_dc.npy'):
                for file in inundation_file:
                    ds_temp3 = gdal.Open(file)
                    raster_temp3 = ds_temp3.GetRasterBand(1).ReadAsArray()
                    for length in range(len(file)):
                        try:
                            date_temp = int(file[length: length + 8])
                            break
                        except:
                            pass
                    date_dc.append(date_temp)

                    if inundation_dc == []:
                        inundation_dc = np.zeros([raster_temp3.shape[0], raster_temp3.shape[1], len(inundation_file)])
                        inundation_dc[:, :, 0] = raster_temp3
                    else:
                        inundation_dc[:, :, num_temp] = raster_temp3

                    if not os.path.exists(sole_file_path + str(date_temp) + '_individual_area.tif'):
                        sole_floodplain_temp = identify_all_inundated_area(raster_temp3,
                                                                           inundated_pixel_indicator=inundated_value,
                                                                           nanvalue_pixel_indicator=nan_value)
                        sole_result = np.zeros_like(sole_floodplain_temp)
                        unique_value = np.unique(sole_floodplain_temp.flatten())
                        unique_value = np.delete(unique_value, np.argwhere(unique_value == 0))
                        for u_value in unique_value:
                            if np.logical_and(sole_floodplain_temp == u_value, river_sample == 1).any():
                                sole_result[sole_floodplain_temp == u_value] = 1
                        sole_result[river_sample == 1] = 0
                        bf.write_raster(ds_temp3, sole_result, sole_file_path, str(date_temp) + '_individual_area.tif',
                                     raster_datatype=gdal.GDT_Int32)
                    else:
                        sole_floodplain_ds_temp = gdal.Open(sole_file_path + str(date_temp) + '_individual_area.tif')
                        sole_floodplain_temp = sole_floodplain_ds_temp.GetRasterBand(1).ReadAsArray()

                    if sole_area_dc == []:
                        sole_area_dc = np.zeros([raster_temp3.shape[0], raster_temp3.shape[1], len(inundation_file)])
                        sole_area_dc[:, :, 0] = sole_floodplain_temp
                    else:
                        sole_area_dc[:, :, num_temp] = sole_floodplain_temp
                    num_temp += 1
                date_dc = np.array(date_dc).transpose()

                if date_dc.shape[0] == sole_area_dc.shape[2] == inundation_dc.shape[2]:
                    np.save(output_folder + 'date_dc.npy', date_dc)
                    np.save(output_folder + 'sole_area_dc.npy', sole_area_dc)
                    np.save(output_folder + 'inundation_dc.npy', inundation_dc)
                else:
                    print('Consistency error during outout!')
                    sys.exit(-1)
            else:
                inundation_dc = np.load(output_folder + 'inundation_dc.npy')
                sole_area_dc = np.load(output_folder + 'sole_area_dc.npy')
                date_dc = np.load(output_folder + 'date_dc.npy')

            date_list = []
            for file_path in inundation_file:
                for length in range(len(file_path)):
                    try:
                        date_temp = int(file_path[length: length + 8])
                        break
                    except:
                        pass
                    date_list.append(date_temp)

            year_list = np.unique(np.fix(date_dc // 10000).astype(np.int))
            annual_inundation_folder = output_folder + 'annual_inundation_status\\'
            annual_inundation_epoch_folder = output_folder + 'annual_inundation_epoch\\'
            annual_inundation_beg_folder = output_folder + 'annual_inundation_beg\\'
            annual_inundation_end_folder = output_folder + 'annual_inundation_end\\'
            bf.create_folder(annual_inundation_folder)
            bf.create_folder(annual_inundation_epoch_folder)
            bf.create_folder(annual_inundation_beg_folder)
            bf.create_folder(annual_inundation_end_folder)

            for year in year_list:
                inundation_temp = []
                sole_area_temp = []
                date_temp = []
                recession_temp = []
                water_level_temp = []
                annual_inundation_epoch = np.zeros_like(self.sa_map).astype(np.float16)
                annual_inundation_status = np.zeros_like(self.sa_map).astype(np.float16)
                annual_inundation_beg = np.zeros_like(self.sa_map).astype(np.float16)
                annual_inundation_end = np.zeros_like(self.sa_map).astype(np.float16)
                annual_inundation_status[np.isnan(self.sa_map)] = np.nan
                annual_inundation_epoch[np.isnan(self.sa_map)] = np.nan
                annual_inundation_beg[np.isnan(self.sa_map)] = np.nan
                annual_inundation_end[np.isnan(self.sa_map)] = np.nan
                water_level_epoch = []
                len1 = 0
                while len1 < date_dc.shape[0]:
                    if date_dc[len1] // 10000 == year:
                        date_temp.append(date_dc[len1])
                        recession_temp.append(recession_list[np.argwhere(recession_list == date_dc[len1])[0, 0], 1])
                        water_level_temp.append(
                            water_level_data[np.argwhere(water_level_data == date_dc[len1])[0, 0], 1])
                        if inundation_temp == [] or sole_area_temp == []:
                            inundation_temp = inundation_dc[:, :, len1].reshape(
                                [inundation_dc.shape[0], inundation_dc.shape[1], 1])
                            sole_area_temp = sole_area_dc[:, :, len1].reshape(
                                [sole_area_dc.shape[0], sole_area_dc.shape[1], 1])
                        else:
                            inundation_temp = np.concatenate((inundation_temp, inundation_dc[:, :, len1].reshape(
                                [inundation_dc.shape[0], inundation_dc.shape[1], 1])), axis=2)
                            sole_area_temp = np.concatenate((sole_area_temp, sole_area_dc[:, :, len1].reshape(
                                [sole_area_dc.shape[0], sole_area_dc.shape[1], 1])), axis=2)
                    len1 += 1

                len1 = 0
                while len1 < water_level_data.shape[0]:
                    if water_level_data[len1, 0] // 10000 == year:
                        if 6 <= np.mod(water_level_data[len1, 0], 10000) // 100 <= 10:
                            recession_temp2 = recession_list[np.argwhere(water_level_data[len1, 0])[0][0], 1]
                            water_level_epoch.append(
                                [water_level_data[len1, 0], water_level_data[len1, 1], recession_temp2])
                    len1 += 1
                water_level_epoch = np.array(water_level_epoch)

                for y_temp in range(annual_inundation_status.shape[0]):
                    for x_temp in range(annual_inundation_status.shape[1]):
                        if self.sa_map[y_temp, x_temp] != -32768:
                            inundation_series_temp = inundation_temp[y_temp, x_temp, :].reshape(
                                [inundation_temp.shape[2]])
                            sole_series_temp = sole_area_temp[y_temp, x_temp, :].reshape([sole_area_temp.shape[2]])
                            water_level_min = np.nan
                            recession_level_min = np.nan

                            inundation_status = False
                            len2 = 0
                            while len2 < len(date_temp):
                                if inundation_series_temp[len2] == 1:
                                    if sole_series_temp[len2] != 1:
                                        if recession_temp[len2] < 0:
                                            sole_local = sole_area_temp[
                                                         max(0, y_temp - 8): min(annual_inundation_status.shape[0],
                                                                                 y_temp + 8),
                                                         max(0, x_temp - 8): min(annual_inundation_status.shape[1],
                                                                                 x_temp + 8), len2]
                                            if np.sum(sole_local == 1) == 0:
                                                inundation_series_temp[len2] == 3
                                            else:
                                                inundation_series_temp[len2] == 2
                                        else:
                                            inundation_local = inundation_temp[max(0, y_temp - 8): min(
                                                annual_inundation_status.shape[0], y_temp + 8), max(0, x_temp - 8): min(
                                                annual_inundation_status.shape[1], x_temp + 8), len2]
                                            sole_local = sole_area_temp[
                                                         max(0, y_temp - 5): min(annual_inundation_status.shape[0],
                                                                                 y_temp + 5),
                                                         max(0, x_temp - 5): min(annual_inundation_status.shape[1],
                                                                                 x_temp + 5), len2]

                                            if np.sum(inundation_local == -2) == 0:
                                                if np.sum(inundation_local == 1) == 1:
                                                    inundation_series_temp[len2] == 6
                                                else:
                                                    if np.sum(sole_local == 1) != 0:
                                                        inundation_series_temp[len2] == 4
                                                    else:
                                                        inundation_series_temp[len2] == 5
                                            else:
                                                if np.sum(inundation_local == 1) == 1:
                                                    inundation_series_temp[len2] == 6
                                                else:
                                                    if np.sum(sole_local == 1) != 0:
                                                        np.sum(sole_local == 1) != 0
                                                        inundation_series_temp[len2] == 4
                                                    else:
                                                        inundation_series_temp[len2] == 5
                                len2 += 1
                            if np.sum(inundation_series_temp == 1) + np.sum(inundation_series_temp == 2) + np.sum(
                                    inundation_series_temp == 3) == 0:
                                annual_inundation_status[y_temp, x_temp] = 0
                                annual_inundation_epoch[y_temp, x_temp] = 0
                                annual_inundation_beg[y_temp, x_temp] = 0
                                annual_inundation_end[y_temp, x_temp] = 0
                            elif np.sum(inundation_series_temp >= 2) / np.sum(inundation_series_temp >= 1) > 0.8:
                                annual_inundation_status[y_temp, x_temp] = 0
                                annual_inundation_epoch[y_temp, x_temp] = 0
                                annual_inundation_beg[y_temp, x_temp] = 0
                                annual_inundation_end[y_temp, x_temp] = 0
                            elif np.sum(inundation_series_temp == 1) != 0:
                                len2 = 0
                                while len2 < len(date_temp):
                                    if inundation_series_temp[len2] == 1:
                                        if np.isnan(water_level_min):
                                            water_level_min = water_level_temp[len2]
                                        else:
                                            water_level_min = min(water_level_temp[len2], water_level_min)
                                    elif inundation_series_temp[len2] == 2:
                                        if np.isnan(recession_level_min):
                                            recession_level_min = water_level_temp[len2]
                                        else:
                                            recession_level_min = min(recession_level_min, water_level_temp[len2])
                                    len2 += 1

                                len3 = 0
                                while len3 < len(date_temp):
                                    if inundation_series_temp[len3] == 3 and (
                                            recession_level_min > water_level_temp[len3] or np.isnan(
                                            recession_level_min)):
                                        len4 = 0
                                        while len4 < len(date_temp):
                                            if inundation_series_temp[len4] == 1 and abs(recession_temp[len4]) == abs(
                                                    recession_temp[len3]):
                                                if np.isnan(recession_level_min):
                                                    recession_level_min = water_level_temp[len3]
                                                else:
                                                    recession_level_min = min(recession_level_min,
                                                                              water_level_temp[len3])
                                                    inundation_series_temp[len3] = 2
                                                break
                                            len4 += 1
                                    len3 += 1

                                len5 = 0
                                while len5 < len(date_temp):
                                    if (inundation_series_temp[len5] == 4 or inundation_series_temp[len5] == 5) and \
                                            water_level_temp[len5] < water_level_min:
                                        len6 = 0
                                        while len6 < len(date_temp):
                                            if inundation_series_temp[len6] == 2 and abs(recession_temp[len6]) == abs(
                                                    recession_temp[len5]):
                                                water_level_min = min(water_level_min, water_level_temp[len5])
                                                break
                                            len6 += 1
                                    len5 += 1

                                annual_inundation_status[y_temp, x_temp] = 1
                                if np.isnan(water_level_min):
                                    print('WATER_LEVEL_1_ERROR!')
                                    sys.exit(-1)
                                elif np.isnan(recession_level_min):
                                    annual_inundation_epoch[y_temp, x_temp] = np.sum(
                                        water_level_epoch[:, 1] >= water_level_min)
                                    date_min = 200000000
                                    date_max = 0
                                    for len0 in range(water_level_epoch.shape[0]):
                                        if water_level_epoch[len0, 1] >= water_level_min:
                                            date_min = min(date_min, water_level_epoch[len0, 0])
                                            date_max = max(date_max, water_level_epoch[len0, 0])
                                    annual_inundation_beg[y_temp, x_temp] = date_min
                                    annual_inundation_end[y_temp, x_temp] = date_max
                                else:
                                    len0 = 0
                                    annual_inundation_epoch = 0
                                    inundation_recession = []
                                    date_min = 200000000
                                    date_max = 0
                                    while len0 < water_level_epoch.shape[0]:
                                        if water_level_epoch[len0, 2] >= 0:
                                            if water_level_epoch[len0, 1] >= water_level_min:
                                                annual_inundation_epoch += 1
                                                inundation_recession.append(water_level_epoch[len0, 2])
                                                date_min = min(date_min, water_level_epoch[len0, 0])
                                                date_max = max(date_max, water_level_epoch[len0, 0])
                                        elif water_level_epoch[len0, 2] < 0 and abs(
                                                water_level_epoch[len0, 2]) in inundation_recession:
                                            if water_level_epoch[len0, 1] >= recession_level_min:
                                                annual_inundation_epoch += 1
                                                date_min = min(date_min, water_level_epoch[len0, 0])
                                                date_max = max(date_max, water_level_epoch[len0, 0])
                                        len0 += 1
                                    annual_inundation_beg[y_temp, x_temp] = date_min
                                    annual_inundation_end[y_temp, x_temp] = date_max

                            elif np.sum(inundation_series_temp == 2) != 0 or np.sum(inundation_series_temp == 3) != 0:
                                len2 = 0
                                while len2 < len(date_temp):
                                    if inundation_series_temp[len2] == 2 or inundation_series_temp[len2] == 3:
                                        if np.isnan(recession_level_min):
                                            recession_level_min = water_level_temp[len2]
                                        else:
                                            recession_level_min = min(recession_level_min, water_level_temp[len2])
                                    len2 += 1

                                len5 = 0
                                while len5 < len(date_temp):
                                    if inundation_series_temp[len5] == 4 or inundation_series_temp[len5] == 5:
                                        len6 = 0
                                        while len6 < len(date_temp):
                                            if inundation_series_temp[len6] == 2 and abs(recession_temp[len6]) == abs(
                                                    recession_temp[len5]):
                                                water_level_min = min(water_level_min, water_level_temp[len5])
                                                break
                                            len6 += 1
                                    len5 += 1

                                if np.isnan(water_level_min):
                                    water_level_min = water_level_epoch[np.argwhere(water_level_epoch == date_temp[
                                        max(water_level_temp.index(recession_level_min) - 3, 0)])[0][0], 1]
                                    if water_level_min < recession_level_min:
                                        water_level_min = recession_level_min

                                annual_inundation_status[y_temp, x_temp] = 1
                                if np.isnan(water_level_min):
                                    print('WATER_LEVEL_1_ERROR!')
                                    sys.exit(-1)
                                elif np.isnan(recession_level_min):
                                    annual_inundation_epoch[y_temp, x_temp] = np.sum(
                                        water_level_epoch[:, 1] >= water_level_min)
                                    date_min = 200000000
                                    date_max = 0
                                    for len0 in range(water_level_epoch.shape[0]):
                                        if water_level_epoch[len0, 1] >= water_level_min:
                                            date_min = min(date_min, water_level_epoch[len0, 0])
                                            date_max = max(date_max, water_level_epoch[len0, 0])
                                    annual_inundation_beg[y_temp, x_temp] = date_min
                                    annual_inundation_end[y_temp, x_temp] = date_max
                                else:
                                    len0 = 0
                                    annual_inundation_epoch = 0
                                    inundation_recession = []
                                    date_min = 200000000
                                    date_max = 0
                                    while len0 < water_level_epoch.shape[0]:
                                        if water_level_epoch[len0, 2] >= 0:
                                            if water_level_epoch[len0, 1] >= water_level_min:
                                                annual_inundation_epoch += 1
                                                inundation_recession.append(water_level_epoch[len0, 2])
                                                date_min = min(date_min, water_level_epoch[len0, 0])
                                                date_max = max(date_max, water_level_epoch[len0, 0])
                                        elif water_level_epoch[len0, 2] < 0 and abs(
                                                water_level_epoch[len0, 2]) in inundation_recession:
                                            if water_level_epoch[len0, 1] >= recession_level_min:
                                                annual_inundation_epoch += 1
                                                date_min = min(date_min, water_level_epoch[len0, 0])
                                                date_max = max(date_max, water_level_epoch[len0, 0])
                                        len0 += 1
                                    annual_inundation_beg[y_temp, x_temp] = date_min
                                    annual_inundation_end[y_temp, x_temp] = date_max

                            elif np.sum(inundation_series_temp == 4) != 0 or np.sum(inundation_series_temp == 5) != 0:
                                len2 = 0
                                while len2 < len(date_temp):
                                    if inundation_series_temp[len2] == 4 or inundation_series_temp[len2] == 5:
                                        if np.isnan(water_level_min):
                                            water_level_min = water_level_temp[len2]
                                        else:
                                            water_level_min = min(water_level_min, water_level_temp[len2])
                                    len2 += 1

                                annual_inundation_status[y_temp, x_temp] = 1
                                if np.isnan(water_level_min):
                                    print('WATER_LEVEL_1_ERROR!')
                                    sys.exit(-1)
                                elif np.isnan(recession_level_min):
                                    annual_inundation_epoch[y_temp, x_temp] = np.sum(
                                        water_level_epoch[:, 1] >= water_level_min)
                                    date_min = 200000000
                                    date_max = 0
                                    for len0 in range(water_level_epoch.shape[0]):
                                        if water_level_epoch[len0, 1] >= water_level_min:
                                            date_min = min(date_min, water_level_epoch[len0, 0])
                                            date_max = max(date_max, water_level_epoch[len0, 0])
                                    annual_inundation_beg[y_temp, x_temp] = date_min
                                    annual_inundation_end[y_temp, x_temp] = date_max
                bf.write_raster(ds_temp, annual_inundation_status, annual_inundation_folder,
                             'annual_' + str(year) + '.tif', raster_datatype=gdal.GDT_Int32)
                bf.write_raster(ds_temp, annual_inundation_epoch, annual_inundation_epoch_folder,
                             'epoch_' + str(year) + '.tif', raster_datatype=gdal.GDT_Int32)
                bf.write_raster(ds_temp, annual_inundation_beg, annual_inundation_beg_folder, 'beg_' + str(year) + '.tif',
                             raster_datatype=gdal.GDT_Int32)
                bf.write_raster(ds_temp, annual_inundation_end, annual_inundation_end_folder, 'end_' + str(year) + '.tif',
                             raster_datatype=gdal.GDT_Int32)

    def area_statitics(self, index, expression, **kwargs):

        # Input the selected dc
        if type(index) == str:
            if index not in self.index_list:
                raise ValueError('The index is not input!')
            else:
                index = [index]
        elif type(index) == list:
            for index_temp in index:
                if index_temp not in self.index_list:
                    raise ValueError('The index is not input!')
        else:
            raise TypeError('Please input the index as a str or list!')

        # Check the threshold
        thr = None
        if type(expression) == str:
            for symbol_temp in ['gte', 'lte', 'gt', 'lt', 'neq', 'eq']:
                if expression.startswith(symbol_temp):
                    try:
                        symbol = symbol_temp
                        thr = float(expression.split(symbol_temp)[-1])
                    except:
                        raise Exception('Please input a valid num')
            if thr is None:
                raise Exception('Please make sure the expression starts with gte lte gt lt eq neq')
        else:
            raise TypeError('Please input the expression as a str!')

        # Define the output path
        output_path = self.work_env + 'Area_statistics\\'
        bf.create_folder(output_path)

        for index_temp in index:
            area_list = []
            output_path_temp = self.work_env + f'Area_statistics\\{index_temp}\\'
            bf.create_folder(output_path_temp)
            index_dc = self.Landsat_dcs[self.index_list.index(index_temp)].dc
            for doy_index in range(self.dcs_ZSize):
                index_doy_temp = index_dc[:,:,doy_index]
                if symbol == 'gte':
                    area = np.sum(index_doy_temp[index_doy_temp >= thr])
                elif symbol == 'lte':
                    area = np.sum(index_doy_temp[index_doy_temp <= thr])
                elif symbol == 'lt':
                    area = np.sum(index_doy_temp[index_doy_temp < thr])
                elif symbol == 'gt':
                    area = np.sum(index_doy_temp[index_doy_temp > thr])
                elif symbol == 'eq':
                    area = np.sum(index_doy_temp[index_doy_temp == thr])
                elif symbol == 'neq':
                    area = np.sum(index_doy_temp[index_doy_temp != thr])
                else:
                    raise Exception('Code error!')
                area_list.append(area * 900)

            area_df = pd.DataFrame({'Doy': bf.doy2date(self.doy_list), f'Area of {index_temp} {symbol} {str(thr)}': area_list})
            area_df.to_excel(output_path_temp + f'area_{index_temp}{expression}.xlsx')

    def _process_file2sdc_para(self, **kwargs):
        pass

    def files2sdc(self, file_folder: str, index: str, output_folder: str, **kwargs) -> None:

        # Process file2sdc para
        self._process_file2sdc_para(**kwargs)

        # Generate sdc
        if not os.path.exists(file_folder):
            raise FileExistsError(f'The {file_folder} folder was missing!')
        else:
            if not os.path.exists(output_folder + 'header.npy') or not os.path.exists(output_folder + 'date.npy') or not os.path.exists(output_folder + str(index) + '_datacube.npy'):
                files = bf.file_filter(file_folder, ['.TIF', str(index)], and_or_factor='and')
                if len(files) != self.dcs_ZSize:
                    raise Exception('Consistent error')
                else:
                    header_dic = {'ROI_name': self.ROI_name, 'VI': index, 'Datatype': 'Int', 'ROI': self.ROI,
                                  'Study_area': self.sa_map, 'sdc_factor': self.sdc_factor, 'ds_file': self.ds_file}
                    data_cube_temp = np.zeros((self.dcs_XSize, self.dcs_YSize, self.dcs_ZSize), dtype=np.float16)

                    i = 0
                    for doy in self.doy_list:
                        file_list = bf.file_filter(file_folder, ['.TIF', str(index), str(doy)], and_or_factor='and')
                        if len(file_list) == 1:
                            temp_ds = gdal.Open(file_list[0])
                            temp_raster = temp_ds.GetRasterBand(1).ReadAsArray()
                            data_cube_temp[:, :, i] = temp_raster
                            i += 1
                        elif len(file_list) >= 1:
                            for q in file_list:
                                temp_ds = gdal.Open(q)
                                temp_raster = temp_ds.GetRasterBand(1).ReadAsArray()
                                data_cube_temp[:, :, i] = temp_raster
                                i += 1
                        else:
                            raise Exception(f'The {str(doy)} file for {index} was missing!')

                    # Write the datacube
                    print('Start writing the ' + index + ' datacube.')
                    start_time = time.time()
                    np.save(output_folder + 'header.npy', header_dic)
                    np.save(output_folder + 'doy.npy', self.doy_list.astype(np.uint32).tolist())
                    np.save(output_folder + str(index) + '_datacube.npy', data_cube_temp.astype(np.float16))
                    end_time = time.time()
                    print('Finished constructing ' + str(index) + ' datacube in ' + str(end_time - start_time) + ' s.')
            else:
                print(f'The sdc of the {index} has already constructed!')

    def quantify_fitness_index_function(self, vi, func, inundated_method=None, output_path=None):

        # Determine vi
        if type(vi) == list:
            for vi_temp in vi:
                if vi_temp not in self.index_list:
                    raise ValueError(f'{vi_temp} is not input!')
        elif type(vi) == str:
            if vi not in self.index_list:
                raise ValueError(f'{vi} is not input!')
            vi = [vi]

        #
        if output_path is None:
            output_path = self.work_env + 'Quantify_fitness\\'
            bf.create_folder(output_path)

        fun_dc = {}
        all_supported_curve_fitting_method = ['seven_para_logistic', 'two_term_fourier']
        if func is None or func == 'seven_para_logistic':
            fun_dc['CFM'] = 'SPL'
            fun_dc['para_num'] = 7
            fun_dc['initial_para_ori'] = [0.10, 0.8802, 108.2, 7.596, 311.4, 7.473, 0.00225]
            fun_dc['initial_para_boundary'] = (
            [0, 0.3, 0, 3, 180, 3, 0.00001], [0.5, 1, 180, 17, 330, 17, 0.01])
            fun_dc['para_ori'] = [0.10, 0.6, 108.2, 7.596, 311.4, 7.473, 0.00015]
            fun_dc['para_boundary'] = (
            [0.10, 0.4, 80, 6.2, 285, 4.5, 0.00015], [0.22, 0.7, 130, 30, 330, 30, 0.00200])
            func = seven_para_logistic_function
        elif func == 'two_term_fourier':
            fun_dc['CFM'] = 'TTF'
            fun_dc['para_num'] = 6
            fun_dc['para_ori'] = [0, 0, 0, 0, 0, 0.017]
            fun_dc['para_boundary'] = (
            [0, -0.5, -0.5, -0.05, -0.05, 0.015], [1, 0.5, 0.5, 0.05, 0.05, 0.019])
            func = two_term_fourier
        elif func not in all_supported_curve_fitting_method:
            ValueError(f'The curve fitting method {self._curve_fitting_algorithm} is not supported!')

        output_dic = {}
        for vi_temp in vi:
            vi_dc = copy.copy(self.Landsat_dcs[self.index_list.index(vi_temp)].dc)
            vi_doy = copy.copy(self.doy_list)

            if vi_temp == 'OSAVI':
                vi_dc = 1.16 * vi_dc

            # inundated method
            if inundated_method is not None:
                if inundated_method in self.index_list:
                    inundated_dc = copy.copy(self.Landsat_dcs[self.index_list.index(inundated_method)].dc)
                    inundated_doy = copy.copy(self.doy_list)
                    vi_dc, vi_doy = self.dc_flood_removal(vi_dc, vi_doy, inundated_dc, inundated_doy)
                elif inundated_method not in self.index_list:
                    raise Exception('The flood detection method is not supported！')

            if vi_temp == 'OSAVI':
                vi_dc = 1.16 * vi_dc
                q = 0
                while q < vi_doy.shape[0]:
                    if 120 < np.mod(vi_doy[q], 1000) < 300:
                        temp = vi_dc[:,:,q]
                        temp[temp < 0.3] = np.nan
                        vi_dc[:,:,q] = temp
                    q += 1

            if vi_temp == 'NDVI':
                q = 0
                while q < vi_doy.shape[0]:
                    if 120 < np.mod(vi_doy[q], 1000) < 300:
                        temp = vi_dc[:,:,q]
                        temp[temp < 0.3] = np.nan
                        vi_dc[:,:,q] = temp
                    q += 1

            if vi_temp == 'EVI':
                q = 0
                while q < vi_doy.shape[0]:
                    if 120 < np.mod(vi_doy[q], 1000) < 300:
                        temp = vi_dc[:,:,q]
                        temp[temp < 0.2] = np.nan
                        vi_dc[:,:,q] = temp
                    q += 1

            vi_valid = np.sum(self.sa_map == 1)
            vi_dc[vi_dc < 0.1] = np.nan
            sum_dc = np.sum(np.sum(~np.isnan(vi_dc), axis=0), axis=0)
            p25_dc = np.sum(np.sum(vi_dc < 0.2, axis=0), axis=0) / vi_valid
            vi_dc = np.nanmean(np.nanmean(vi_dc, axis=0), axis=0)
            vi_doy = np.mod(vi_doy, 1000)

            q = 0
            while q < vi_dc.shape[0]:
                if np.isnan(vi_dc[q]) or vi_dc[q] > 1:
                    vi_dc = np.delete(vi_dc, q)
                    vi_doy = np.delete(vi_doy, q)
                    sum_dc = np.delete(sum_dc, q)
                    p25_dc = np.delete(p25_dc, q)
                    q -= 1
                q += 1

            paras, extra = curve_fit(seven_para_logistic_function, vi_doy, vi_dc, maxfev=500000,
                                     p0=fun_dc['para_ori'], bounds=fun_dc['para_boundary'])
            predicted_y_data = func(vi_doy, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6])
            r_square = (1 - np.sum((predicted_y_data - vi_dc) ** 2) / np.sum((vi_dc - np.mean(vi_dc)) ** 2))

            fig5 = plt.figure(figsize=(15, 8.7), tight_layout=True)
            ax1 = fig5.add_subplot()
            ax1.plot(np.linspace(0, 365, 366),
                     seven_para_logistic_function(np.linspace(0, 365, 366), paras[0], paras[1], paras[2], paras[3],
                                                  paras[4], paras[5], paras[6]), linewidth=5,
                     color=(0 / 256, 0 / 256, 0 / 256))
            ax1.scatter(vi_doy, vi_dc, s=15 ** 2, color=(196 / 256, 120 / 256, 120 / 256), marker='.')
            # plt.show()

            num = 0
            for i in range(vi_dc.shape[0]):
                if predicted_y_data[i] * 0.75 <= vi_dc[i] <= predicted_y_data[i] * 1.25:
                    num += 1
            per = num / vi_dc.shape[0]

            output_dic[f'{vi_temp}_sta'] = [str(r_square), str(per), str(np.std((vi_dc/predicted_y_data)-1)), str(np.min((vi_dc/predicted_y_data)-1)), str(np.max((vi_dc/predicted_y_data)-1)), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]]

            plt.savefig(output_path + vi_temp + '.png', dpi=300)

        name = ''
        for name_temp in vi:
            name = name + '_' + name_temp
        df = pd.DataFrame(output_dic)
        df.to_excel(output_path + f'{self.ROI_name}{name}_sta.xlsx')

    def dc_flood_removal(self, index_dc, vi_doy, inundated_dc, inundated_doy):
        # Consistency check
        if index_dc.shape[0] != inundated_dc.shape[0] or index_dc.shape[1] != inundated_dc.shape[1]:
            print('Consistency error!')
            sys.exit(-1)

        if len(vi_doy) != index_dc.shape[2] or inundated_doy.shape[0] == 0:
            print('Consistency error!')
            sys.exit(-1)

        # Elimination
        vi_doy_index = 0
        while vi_doy_index < len(vi_doy):
            inundated_doy_list = np.argwhere(inundated_doy == vi_doy[vi_doy_index])
            if inundated_doy_list.shape != 0:
                inundated_doy_index = inundated_doy_list[0]
                index_dc_temp = index_dc[:, :, vi_doy_index].reshape([index_dc.shape[0], index_dc.shape[1]])
                inundated_dc_temp = inundated_dc[:, :, inundated_doy_index].reshape([index_dc.shape[0], index_dc.shape[1]])
                index_dc_temp[inundated_dc_temp == 1] = np.nan
                index_dc[:, :, vi_doy_index] = index_dc_temp
            vi_doy_index += 1
        return index_dc, vi_doy

    def flood_free_phenology_metrics_extraction(self, VI_list, flood_indi, output_indicator, **kwargs):

        # Check the VI method
        if type(VI_list) is str and VI_list in self.index_list:
            VI_list = [VI_list]
        elif type(VI_list) is list and False not in [VI_temp in self.index_list for VI_temp in VI_list]:
            pass
        else:
            raise TypeError(f'The input VI {VI_list} was not in supported type (list or str) or some input VI is not in the Landsat_dcs!')

        # Check the flood method
        if type(flood_indi) is str and flood_indi in self.index_list and flood_indi in self._flood_mapping_method:
            pass
        else:
            raise TypeError(f'The input flood {flood_indi} was not in supported type (list or str) or some input flooded is not in the Landsat_dcs!')

        # Determine output indicator
        if type(output_indicator) is str and output_indicator in self._flood_free_pm:
            output_indicator = [output_indicator]
        elif type(output_indicator) is list and False not in [indi_temp in self._flood_free_pm for indi_temp in output_indicator]:
            pass
        else:
            raise TypeError(f'The output_indicator {output_indicator} was not in supported type (list or str) or some output_indicator is not in the Landsat_dcs!')

        # Main procedure
        for index in VI_list:
            # generate output path
            output_path = self.work_env + index + '_flood_free_phenology_metrics\\'
            bf.create_folder(output_path)
            for ind in output_indicator:
                if ind == 'annual_max_VI':
                    # Define the output path
                    output_path_temp = output_path + ind + '\\'
                    annual_output_path = output_path + ind + '\\annual\\'
                    annual_v_output_path = output_path + ind + '\\annual_variation\\'
                    annual_obs_path = output_path + ind + '\\annual_obs\\'
                    bf.create_folder(output_path_temp)
                    bf.create_folder(annual_output_path)
                    bf.create_folder(annual_v_output_path)
                    bf.create_folder(annual_obs_path)

                    flood_dc = copy.copy(self.Landsat_dcs[self.index_list.index(flood_indi)].dc)
                    index_dc = copy.copy(self.Landsat_dcs[self.index_list.index(index)].dc)
                    doy_dc = copy.copy(self.doy_list)
                    year_range = np.sort(np.unique(np.floor(doy_dc/1000)).astype(int))

                    index_dc[flood_dc == 1] = np.nan
                    # initiate the cube
                    output_metrics = np.zeros([self.sa_map.shape[0], self.sa_map.shape[1], year_range.shape[0]]).astype(np.float16)
                    output_obs = np.zeros([self.sa_map.shape[0], self.sa_map.shape[1], year_range.shape[0]]).astype(np.int16)
                    for y in range(self.sa_map.shape[0]):
                        for x in range(self.sa_map.shape[1]):
                            if self.sa_map[y, x] == -32768:
                                output_metrics[y, x, :] = np.nan
                                output_obs[y, x, :] = -32768

                    for year in year_range:
                        for y_t in range(index_dc.shape[0]):
                            for x_t in range(index_dc.shape[1]):
                                obs = doy_dc[np.argwhere(np.floor(doy_dc/1000) == year)].reshape([-1])
                                index_temp = index_dc[y_t, x_t, np.argwhere(np.floor(doy_dc/1000) == year)].reshape([-1])
                                if False in np.isnan(index_temp):
                                    if 90 < np.mod(obs[np.argwhere(index_temp == np.nanmax(index_temp))],1000)[0] < 210:
                                        output_metrics[y_t, x_t, np.argwhere(year_range == year)] = np.nanmax(index_temp)
                                        output_obs[y_t, x_t, np.argwhere(year_range == year)] = obs[np.argwhere(index_temp == np.nanmax(index_temp))[0]]
                                    else:
                                        index_90_210 = np.delete(index_temp, np.argwhere(np.logical_and(90 < np.mod(obs, 1000), np.mod(obs, 1000) < 210)))
                                        obs_90_210 = np.delete(obs, np.argwhere(np.logical_and(90 < np.mod(obs, 1000), np.mod(obs, 1000) < 210)))
                                        if np.nanmax(index_90_210) >= 0.5:
                                            output_metrics[y_t, x_t, np.argwhere(year_range == year)] = np.nanmax(
                                                index_90_210)
                                            output_obs[y_t, x_t, np.argwhere(year_range == year)] = obs_90_210[
                                                np.argwhere(index_90_210 == np.nanmax(index_90_210))[0]]
                                        else:
                                            output_metrics[y_t, x_t, np.argwhere(year_range == year)] = np.nan
                                            output_obs[y_t, x_t, np.argwhere(year_range == year)] = -32768
                                else:
                                    output_metrics[y_t, x_t, np.argwhere(year_range == year)] = np.nan
                                    output_obs[y_t, x_t, np.argwhere(year_range == year)] = -32768
                        bf.write_raster(gdal.Open(self.ds_file), output_obs[:, :, np.argwhere(year_range == year)].reshape([output_obs.shape[0], output_obs.shape[1]]), annual_obs_path, str(year) + '_obs_date.TIF', raster_datatype=gdal.GDT_Int16)
                        bf.write_raster(gdal.Open(self.ds_file), output_metrics[:, :, np.argwhere(year_range == year)].reshape([output_metrics.shape[0], output_metrics.shape[1]]), annual_output_path, str(year) + '_annual_maximum_VI.TIF', raster_datatype=gdal.GDT_Float32)
                        if year + 1 in year_range:
                            bf.write_raster(gdal.Open(self.ds_file),
                                         output_metrics[:, :, np.argwhere(year_range == year + 1)].reshape(
                                             [output_metrics.shape[0], output_metrics.shape[1]]) - output_metrics[:, :, np.argwhere(year_range == year)].reshape(
                                             [output_metrics.shape[0], output_metrics.shape[1]]), annual_v_output_path,
                                         str(year) + '-' + str(year + 1) + '_variation.TIF', raster_datatype=gdal.GDT_Int16)

                    np.save(output_path + 'annual_maximum_VI.npy', output_metrics)
                    np.save(output_path + 'annual_maximum_VI_obs_date.npy', output_obs)

                elif ind == 'average_VI_between_max_and_flood':
                    # Define the output path
                    output_path_temp = output_path + ind + '\\'
                    annual_v_output_path = output_path + ind + '\\annual_variation\\'
                    annual_obs_path = output_path + ind + '\\annual_variation_duration\\'
                    bf.create_folder(output_path_temp)
                    bf.create_folder(annual_v_output_path)
                    bf.create_folder(annual_obs_path)

                    index_dc = copy.copy(self.Landsat_dcs[self.index_list.index(index)].dc)
                    inundated_dc = copy.copy(self.Landsat_dcs[self.index_list.index(flood_indi)].dc)
                    doy_dc = copy.copy(self.doy_list)
                    year_range = np.sort(np.unique(np.floor(doy_dc/1000)).astype(int))

                    # initiate the cube
                    output_metrics = np.zeros([self.sa_map.shape[0], self.sa_map.shape[1], year_range.shape[0] - 1]).astype(np.float16)
                    output_obs = np.zeros([self.sa_map.shape[0], self.sa_map.shape[1], year_range.shape[0] - 1]).astype(np.int16)
                    for y in range(self.sa_map.shape[0]):
                        for x in range(self.sa_map.shape[1]):
                            if self.sa_map[y, x] == -32768:
                                output_metrics[y, x, :] = np.nan
                                output_obs[y, x, :] = -32768

                    SPDL_para_dic = {}

                    for year in year_range[0:-1]:
                        for y_t in range(index_dc.shape[0]):
                            for x_t in range(index_dc.shape[1]):
                                current_year_obs = doy_dc[np.argwhere(np.floor(doy_dc/1000) == year)].reshape([-1])
                                current_year_index_temp = index_dc[y_t, x_t, np.argwhere(np.floor(doy_dc/1000) == year)].reshape([-1])
                                current_year_inundated_temp = inundated_dc[y_t, x_t, np.argwhere(np.floor(doy_dc/1000) == year)].reshape([-1])

                                next_year_obs = doy_dc[np.argwhere(np.floor(doy_dc / 1000) == year + 1)].reshape([-1])
                                next_year_index_temp = index_dc[
                                    y_t, x_t, np.argwhere(np.floor(doy_dc / 1000) == year + 1)].reshape([-1])
                                next_year_inundated_temp = inundated_dc[
                                    y_t, x_t, np.argwhere(np.floor(doy_dc / 1000) == year + 1)].reshape([-1])

                                if False in np.isnan(current_year_index_temp) and False in np.isnan(next_year_index_temp):
                                    current_year_max_obs = current_year_obs[np.nanargmax(current_year_index_temp)]
                                    current_year_inundated_status = 1 in current_year_inundated_temp[np.argwhere(current_year_obs >= current_year_max_obs)]

                                    next_year_max_obs = next_year_obs[np.nanargmax(next_year_index_temp)]
                                    next_year_inundated_status = 1 in next_year_inundated_temp[np.argwhere(next_year_obs >=next_year_max_obs)]

                                    if 90 < np.mod(current_year_max_obs, 1000) < 200 and 90 < np.mod(next_year_max_obs, 1000) < 200:
                                        if current_year_inundated_status or next_year_inundated_status:

                                            if current_year_inundated_status:
                                                for obs_temp in range(np.argwhere(current_year_obs == current_year_max_obs)[0][0], current_year_obs.shape[0]):
                                                    if current_year_inundated_temp[obs_temp] == 1:
                                                        current_year_date = current_year_obs[obs_temp] - current_year_max_obs
                                            else:
                                                current_year_date = 365

                                            if next_year_inundated_status:
                                                for obs_temp in range(np.argwhere(next_year_obs == next_year_max_obs)[0][0], next_year_obs.shape[0]):
                                                    if next_year_inundated_temp[obs_temp] == 1:
                                                        next_year_date = next_year_obs[obs_temp] - next_year_max_obs
                                            else:
                                                next_year_date = 365

                                            if current_year_date >= next_year_date:
                                                current_year_avim = np.nanmean(current_year_index_temp[np.argwhere(np.logical_and(current_year_obs >= current_year_max_obs, current_year_obs <= current_year_obs + next_year_date))])
                                                next_year_avim = np.nanmean(next_year_index_temp[np.argwhere(np.logical_and(next_year_obs >= next_year_max_obs, next_year_obs <= next_year_obs + next_year_date))])
                                            elif current_year_date < next_year_date:
                                                current_year_avim = np.nanmean(current_year_index_temp[np.argwhere(np.logical_and(current_year_obs >= current_year_max_obs, current_year_obs <= current_year_obs + current_year_date))])
                                                next_year_avim = np.nanmean(next_year_index_temp[np.argwhere(np.logical_and(next_year_obs >= next_year_max_obs, next_year_obs <= next_year_obs + current_year_date))])

                                            output_metrics[y_t, x_t, np.argwhere(year_range == year)] = next_year_avim - current_year_avim
                                            output_obs[y_t, x_t, np.argwhere(year_range == year)] = min(current_year_date, next_year_date)
                                        else:
                                            if f'{str(x_t)}_{str(y_t)}_para_ori' not in SPDL_para_dic.keys():
                                                vi_all = index_dc[y_t, x_t, :].reshape([-1])
                                                inundated_all = inundated_dc[y_t, x_t, :].reshape([-1])
                                                vi_all[inundated_all == 1] = np.nan
                                                doy_all = copy.copy(doy_dc)
                                                doy_all = np.mod(doy_all, 1000)
                                                doy_all = np.delete(doy_all, np.argwhere(np.isnan(vi_all)))
                                                vi_all = np.delete(vi_all, np.argwhere(np.isnan(vi_all)))
                                                paras, extras = curve_fit(seven_para_logistic_function, doy_all, vi_all,
                                                                          maxfev=500000,
                                                                          p0=[0.10, 0.8802, 108.2, 7.596, 311.4, 7.473, 0.00225],
                                                                          bounds=([0.08, 0.7, 90, 6.2, 285, 4.5, 0.0015], [0.20, 1.0, 130, 11.5, 330, 8.8, 0.0028]))
                                                SPDL_para_dic[str(x_t) + '_' + str(y_t) + '_para_ori'] = paras
                                                vi_dormancy = []
                                                doy_dormancy = []
                                                vi_max = []
                                                doy_max = []
                                                doy_index_max = np.argmax(seven_para_logistic_function(np.linspace(0, 366, 365), paras[0],
                                                                                  paras[1], paras[2], paras[3], paras[4],
                                                                                  paras[5], paras[6]))
                                                # Generate the parameter boundary
                                                senescence_t = paras[4] - 4 * paras[5]
                                                for doy_index in range(doy_all.shape[0]):
                                                    if 0 < doy_all[doy_index] < paras[2] or paras[4] < doy_all[
                                                        doy_index] < 366:
                                                        vi_dormancy.append(vi_all[doy_index])
                                                        doy_dormancy.append(doy_all[doy_index])
                                                    if doy_index_max - 5 < doy_all[doy_index] < doy_index_max + 5:
                                                        vi_max.append(vi_all[doy_index])
                                                        doy_max.append(doy_all[doy_index])

                                                if vi_max == []:
                                                    vi_max = [np.max(vi_all)]
                                                    doy_max = [doy_all[np.argmax(vi_all)]]

                                                itr = 5
                                                while itr < 10:
                                                    doy_senescence = []
                                                    vi_senescence = []
                                                    for doy_index in range(doy_all.shape[0]):
                                                        if senescence_t - itr < doy_all[doy_index] < senescence_t + itr:
                                                            vi_senescence.append(vi_all[doy_index])
                                                            doy_senescence.append(doy_all[doy_index])
                                                    if doy_senescence != [] and vi_senescence != []:
                                                        break
                                                    else:
                                                        itr += 1

                                                # [0, 0.3, 0, 0, 180, 0, 0], [0.5, 1, 180, 20, 330, 10, 0.01]
                                                # define the para1
                                                if vi_dormancy != []:
                                                    vi_dormancy_sort = np.sort(vi_dormancy)
                                                    vi_max_sort = np.sort(vi_max)
                                                    paras1_max = vi_dormancy_sort[
                                                        int(np.fix(vi_dormancy_sort.shape[0] * 0.95))]
                                                    paras1_min = vi_dormancy_sort[
                                                        int(np.fix(vi_dormancy_sort.shape[0] * 0.05))]
                                                    paras1_max = min(paras1_max, 0.5)
                                                    paras1_min = max(paras1_min, 0)
                                                else:
                                                    paras1_max = 0.5
                                                    paras1_min = 0

                                                # define the para2
                                                paras2_max = vi_max[-1] - paras1_min
                                                paras2_min = vi_max[0] - paras1_max
                                                if paras2_min < 0.2:
                                                    paras2_min = 0.2
                                                if paras2_max > 0.7 or paras2_max < 0.2:
                                                    paras2_max = 0.7

                                                # define the para3
                                                paras3_max = 0
                                                for doy_index in range(len(doy_all)):
                                                    if paras1_min < vi_all[doy_index] < paras1_max and doy_all[
                                                        doy_index] < 180:
                                                        paras3_max = max(float(paras3_max), doy_all[doy_index])

                                                paras3_min = 180
                                                for doy_index in range(len(doy_all)):
                                                    if vi_all[doy_index] > paras1_max:
                                                        paras3_min = min(paras3_min, doy_all[doy_index])

                                                if paras3_min > paras[2] or paras3_min < paras[2] - 15:
                                                    paras3_min = paras[2] - 15

                                                if paras3_max < paras[2] or paras3_max > paras[2] + 15:
                                                    paras3_max = paras[2] + 15

                                                # define the para5
                                                paras5_max = 0
                                                for doy_index in range(len(doy_all)):
                                                    if vi_all[doy_index] > paras1_max:
                                                        paras5_max = max(paras5_max, doy_all[doy_index])
                                                paras5_min = 365
                                                for doy_index in range(len(doy_all)):
                                                    if paras1_min < vi_all[doy_index] < paras1_max and doy_all[
                                                        doy_index] > 180:
                                                        paras5_min = min(paras5_min, doy_all[doy_index])
                                                if paras5_min > paras[4] or paras5_min < paras[4] - 15:
                                                    paras5_min = paras[4] - 15

                                                if paras5_max < paras[4] or paras5_max > paras[4] + 15:
                                                    paras5_max = paras[4] + 15

                                                # define the para 4
                                                if len(doy_max) != 1:
                                                    paras4_max = (np.nanmax(doy_max) - paras3_min) / 4
                                                    paras4_min = (np.nanmin(doy_max) - paras3_max) / 4
                                                else:
                                                    paras4_max = (np.nanmax(doy_max) + 5 - paras3_min) / 4
                                                    paras4_min = (np.nanmin(doy_max) - 5 - paras3_max) / 4
                                                paras4_min = max(3, paras4_min)
                                                paras4_max = min(17, paras4_max)
                                                if paras4_min > 17:
                                                    paras4_min = 3
                                                if paras4_max < 3:
                                                    paras4_max = 17
                                                paras6_max = paras4_max
                                                paras6_min = paras4_min
                                                if doy_senescence == [] or vi_senescence == []:
                                                    paras7_max = 0.01
                                                    paras7_min = 0.00001
                                                else:
                                                    paras7_max = (np.nanmax(vi_max) - np.nanmin(vi_senescence)) / (
                                                                doy_senescence[np.argmin(vi_senescence)] - doy_max[
                                                            np.argmax(vi_max)])
                                                    paras7_min = (np.nanmin(vi_max) - np.nanmax(vi_senescence)) / (
                                                                doy_senescence[np.argmax(vi_senescence)] - doy_max[
                                                            np.argmin(vi_max)])
                                                if np.isnan(paras7_min):
                                                    paras7_min = 0.00001
                                                if np.isnan(paras7_max):
                                                    paras7_max = 0.01
                                                paras7_max = min(paras7_max, 0.01)
                                                paras7_min = max(paras7_min, 0.00001)
                                                if paras7_max < 0.00001:
                                                    paras7_max = 0.01
                                                if paras7_min > 0.01:
                                                    paras7_min = 0.00001
                                                if paras1_min > paras[0]:
                                                    paras1_min = paras[0] - 0.01
                                                if paras1_max < paras[0]:
                                                    paras1_max = paras[0] + 0.01
                                                if paras2_min > paras[1]:
                                                    paras2_min = paras[1] - 0.01
                                                if paras2_max < paras[1]:
                                                    paras2_max = paras[1] + 0.01
                                                if paras3_min > paras[2]:
                                                    paras3_min = paras[2] - 1
                                                if paras3_max < paras[2]:
                                                    paras3_max = paras[2] + 1
                                                if paras4_min > paras[3]:
                                                    paras4_min = paras[3] - 0.1
                                                if paras4_max < paras[3]:
                                                    paras4_max = paras[3] + 0.1
                                                if paras5_min > paras[4]:
                                                    paras5_min = paras[4] - 1
                                                if paras5_max < paras[4]:
                                                    paras5_max = paras[4] + 1
                                                if paras6_min > paras[5]:
                                                    paras6_min = paras[5] - 0.5
                                                if paras6_max < paras[5]:
                                                    paras6_max = paras[5] + 0.5
                                                if paras7_min > paras[6]:
                                                    paras7_min = paras[6] - 0.00001
                                                if paras7_max < paras[6]:
                                                    paras7_max = paras[6] + 0.00001
                                                SPDL_para_dic[f'{str(x_t)}_{str(y_t)}_para_boundary'] = (
                                                [paras1_min, paras2_min, paras3_min, paras4_min, paras5_min, paras6_min,
                                                 paras7_min],
                                                [paras1_max, paras2_max, paras3_max, paras4_max, paras5_max, paras6_max,
                                                 paras7_max])
                                            current_year_index_temp[current_year_inundated_temp == 1] = np.nan
                                            next_year_index_temp[next_year_inundated_temp == 1] = np.nan
                                            if np.sum(~np.isnan(current_year_index_temp)) >= 7 and np.sum(~np.isnan(next_year_index_temp)) >= 7:
                                                current_year_obs_date = np.mod(current_year_obs, 1000)
                                                next_year_obs_date = np.mod(next_year_obs, 1000)
                                                current_year_obs_date = np.delete(current_year_obs_date, np.argwhere(np.isnan(current_year_index_temp)))
                                                next_year_obs_date = np.delete(next_year_obs_date, np.argwhere(np.isnan(next_year_index_temp)))
                                                current_year_index_temp = np.delete(current_year_index_temp, np.argwhere(np.isnan(current_year_index_temp)))
                                                next_year_index_temp = np.delete(next_year_index_temp, np.argwhere(np.isnan(next_year_index_temp)))
                                                paras1, extras1 = curve_fit(seven_para_logistic_function, current_year_obs_date, current_year_index_temp,
                                                                          maxfev=500000,
                                                                          p0=SPDL_para_dic[f'{str(x_t)}_{str(y_t)}_para_ori'],
                                                                          bounds= SPDL_para_dic[f'{str(x_t)}_{str(y_t)}_para_boundary'])
                                                paras2, extras2 = curve_fit(seven_para_logistic_function, next_year_obs_date, next_year_index_temp,
                                                                          maxfev=500000,
                                                                          p0=SPDL_para_dic[f'{str(x_t)}_{str(y_t)}_para_ori'],
                                                                          bounds= SPDL_para_dic[f'{str(x_t)}_{str(y_t)}_para_boundary'])
                                                current_year_avim = (paras1[0] + paras1[1] + seven_para_logistic_function(paras1[4] - 3 * paras1[5], paras1[0],paras1[1],paras1[2],paras1[3], paras1[4],paras1[5],paras1[6])) /2
                                                next_year_avim = (paras2[0] + paras2[1] + seven_para_logistic_function(paras2[4] - 3 * paras2[5], paras2[0], paras2[1], paras2[2], paras2[3], paras2[4], paras2[5], paras2[6])) / 2
                                                output_metrics[y_t, x_t, np.argwhere(year_range == year)] = next_year_avim - current_year_avim
                                                output_obs[y_t, x_t, np.argwhere(year_range == year)] = 365
                                            else:
                                                output_metrics[y_t, x_t, np.argwhere(year_range == year)] = np.nan
                                                output_obs[y_t, x_t, np.argwhere(year_range == year)] = -32768
                                    else:
                                        output_metrics[y_t, x_t, np.argwhere(year_range == year)] = np.nan
                                        output_obs[y_t, x_t, np.argwhere(year_range == year)] = -32768
                                else:
                                    output_metrics[y_t, x_t, np.argwhere(year_range == year)] = np.nan
                                    output_obs[y_t, x_t, np.argwhere(year_range == year)] = -32768

                        bf.write_raster(gdal.Open(self.ds_file), output_obs[:, :, np.argwhere(year_range == year)].reshape([output_obs.shape[0], output_obs.shape[1]]), annual_obs_path, str(year) + '_' + str(year+1) + '_obs_duration.TIF', raster_datatype=gdal.GDT_Int16)
                        bf.write_raster(gdal.Open(self.ds_file), output_metrics[:, :, np.argwhere(year_range == year)].reshape([output_metrics.shape[0], output_metrics.shape[1]]), annual_v_output_path, str(year) + '_' + str(year+1) + '_AVIM_variation.TIF', raster_datatype=gdal.GDT_Float32)

                    np.save(output_path + 'AVIM_variation.npy', output_metrics)
                    np.save(output_path + 'AVIM_duration.npy', output_obs)

    def _process_curve_fitting_para(self, **kwargs):

        # Curve fitting method
        all_supported_curve_fitting_method = ['seven_para_logistic', 'two_term_fourier']
        
        if 'curve_fitting_algorithm' in kwargs.keys():
            self._curve_fitting_algorithm = kwargs['curve_fitting_algorithm']

        if self._curve_fitting_algorithm is None or self._curve_fitting_algorithm == 'seven_para_logistic':
            self._curve_fitting_dic['CFM'] = 'SPL'
            self._curve_fitting_dic['para_num'] = 7
            self._curve_fitting_dic['initial_para_ori'] = [0.10, 0.8802, 108.2, 7.596, 311.4, 7.473, 0.00225]
            self._curve_fitting_dic['initial_para_boundary'] = (
            [0, 0.3, 0, 3, 180, 3, 0.00001], [0.5, 1, 180, 17, 330, 17, 0.01])
            self._curve_fitting_dic['para_ori'] = [0.10, 0.8802, 108.2, 7.596, 311.4, 7.473, 0.00225]
            self._curve_fitting_dic['para_boundary'] = (
            [0.08, 0.7, 90, 6.2, 285, 4.5, 0.0015], [0.20, 1.0, 130, 11.5, 330, 8.8, 0.0028])
            self._curve_fitting_algorithm = seven_para_logistic_function
        elif self._curve_fitting_algorithm == 'two_term_fourier':
            self._curve_fitting_dic['CFM'] = 'TTF'
            self._curve_fitting_dic['para_num'] = 6
            self._curve_fitting_dic['para_ori'] = [0, 0, 0, 0, 0, 0.017]
            self._curve_fitting_dic['para_boundary'] = (
            [0, -0.5, -0.5, -0.05, -0.05, 0.015], [1, 0.5, 0.5, 0.05, 0.05, 0.019])
            self._curve_fitting_algorithm = two_term_fourier
        elif self._curve_fitting_algorithm not in all_supported_curve_fitting_method:
            ValueError(f'The curve fitting method {self._curve_fitting_algorithm} is not supported!')

        # Determine inundation removal method
        if 'flood_removal_method' in kwargs.keys():
            self._flood_removal_method = kwargs['flood_removal_method']

            if self._flood_removal_method not in self._flood_mapping_method:
                raise ValueError(f'The flood removal method {self._flood_removal_method} is not supported!')
        else:
            self._flood_removal_method = None

    def curve_fitting(self, index, **kwargs):
        # check vi
        if index not in self.index_list:
            raise ValueError('Please make sure the vi datacube is constructed!')

        # Process paras
        self._process_curve_fitting_para(**kwargs)

        # Define the vi dc
        index_dc = copy.copy(self.Landsat_dcs[self.index_list.index(index)].dc)
        doy_dc = copy.copy(self.doy_list)

        # Eliminate the inundated value
        if self._flood_removal_method is not None:
            inundated_dc = copy.copy(self.Landsat_dcs[self.index_list.index(self._flood_removal_method)])
            index_dc, doy_dc = self.dc_flood_removal(index_dc, self.doy_list, inundated_dc, self.doy_list)

        # Create output path
        curfit_output_path = self.work_env + index + '_curfit_datacube\\'
        output_path = curfit_output_path + str(self._curve_fitting_dic['CFM']) + '\\'
        bf.create_folder(curfit_output_path)
        bf.create_folder(output_path)
        self._curve_fitting_dic[str(self.ROI) + '_' + str(index) + '_' + str(self._curve_fitting_dic['CFM']) + '_path'] = output_path

        # Generate the initial parameter
        if not os.path.exists(output_path + 'para_boundary.npy'):
            doy_all_s = np.mod(doy_dc, 1000)
            for y_t in range(index_dc.shape[0]):
                for x_t in range(index_dc.shape[1]):
                    if self.sa_map[y_t, x_t] != -32768:
                        vi_all = index_dc[y_t, x_t, :].flatten()
                        doy_all = copy.copy(doy_all_s)
                        vi_index = 0
                        while vi_index < vi_all.shape[0]:
                            if np.isnan(vi_all[vi_index]):
                                vi_all = np.delete(vi_all, vi_index)
                                doy_all = np.delete(doy_all, vi_index)
                                vi_index -= 1
                            vi_index += 1
                        if doy_all.shape[0] >= 7:
                            paras, extras = curve_fit(self._curve_fitting_algorithm, doy_all, vi_all, maxfev=500000, p0=self._curve_fitting_dic['initial_para_ori'], bounds=self._curve_fitting_dic['initial_para_boundary'])
                            self._curve_fitting_dic[str(x_t) + '_' + str(y_t) + '_para_ori'] = paras
                            vi_dormancy = []
                            doy_dormancy = []
                            vi_max = []
                            doy_max = []
                            doy_index_max = np.argmax(self._curve_fitting_algorithm(np.linspace(0,366,365), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]))
                            # Generate the parameter boundary
                            senescence_t = paras[4] - 4 * paras[5]
                            for doy_index in range(doy_all.shape[0]):
                                if 0 < doy_all[doy_index] < paras[2] or paras[4] < doy_all[doy_index] < 366:
                                    vi_dormancy.append(vi_all[doy_index])
                                    doy_dormancy.append(doy_all[doy_index])
                                if doy_index_max - 5 < doy_all[doy_index] < doy_index_max + 5:
                                    vi_max.append(vi_all[doy_index])
                                    doy_max.append(doy_all[doy_index])

                            if vi_max == []:
                                vi_max = [np.max(vi_all)]
                                doy_max = [doy_all[np.argmax(vi_all)]]

                            itr = 5
                            while itr < 10:
                                doy_senescence = []
                                vi_senescence = []
                                for doy_index in range(doy_all.shape[0]):
                                    if senescence_t - itr < doy_all[doy_index] < senescence_t + itr:
                                        vi_senescence.append(vi_all[doy_index])
                                        doy_senescence.append(doy_all[doy_index])
                                if doy_senescence != [] and vi_senescence != []:
                                    break
                                else:
                                    itr += 1

                            # [0, 0.3, 0, 0, 180, 0, 0], [0.5, 1, 180, 20, 330, 10, 0.01]
                            # define the para1
                            if vi_dormancy != []:
                                vi_dormancy_sort = np.sort(vi_dormancy)
                                vi_max_sort = np.sort(vi_max)
                                paras1_max = vi_dormancy_sort[int(np.fix(vi_dormancy_sort.shape[0] * 0.95))]
                                paras1_min = vi_dormancy_sort[int(np.fix(vi_dormancy_sort.shape[0] * 0.05))]
                                paras1_max = min(paras1_max, 0.5)
                                paras1_min = max(paras1_min, 0)
                            else:
                                paras1_max = 0.5
                                paras1_min = 0

                            # define the para2
                            paras2_max = vi_max[-1] - paras1_min
                            paras2_min = vi_max[0] - paras1_max
                            if paras2_min < 0.2:
                                paras2_min = 0.2
                            if paras2_max > 0.7 or paras2_max < 0.2:
                                paras2_max = 0.7

                            # define the para3
                            paras3_max = 0
                            for doy_index in range(len(doy_all)):
                                if paras1_min < vi_all[doy_index] < paras1_max and doy_all[doy_index] < 180:
                                    paras3_max = max(float(paras3_max), doy_all[doy_index])

                            paras3_min = 180
                            for doy_index in range(len(doy_all)):
                                if vi_all[doy_index] > paras1_max:
                                    paras3_min = min(paras3_min, doy_all[doy_index])

                            if paras3_min > paras[2] or paras3_min < paras[2] - 15:
                                paras3_min = paras[2] - 15

                            if paras3_max < paras[2] or paras3_max > paras[2] + 15:
                                paras3_max = paras[2] + 15

                            # define the para5
                            paras5_max = 0
                            for doy_index in range(len(doy_all)):
                                if vi_all[doy_index] > paras1_max:
                                    paras5_max = max(paras5_max, doy_all[doy_index])
                            paras5_min = 365
                            for doy_index in range(len(doy_all)):
                                if paras1_min < vi_all[doy_index] < paras1_max and doy_all[doy_index] > 180:
                                    paras5_min = min(paras5_min, doy_all[doy_index])
                            if paras5_min > paras[4] or paras5_min < paras[4] - 15:
                                paras5_min = paras[4] - 15

                            if paras5_max < paras[4] or paras5_max > paras[4] + 15:
                                paras5_max = paras[4] + 15

                            # define the para 4
                            if len(doy_max) != 1:
                                paras4_max = (np.nanmax(doy_max) - paras3_min) / 4
                                paras4_min = (np.nanmin(doy_max) - paras3_max) / 4
                            else:
                                paras4_max = (np.nanmax(doy_max) + 5 - paras3_min) / 4
                                paras4_min = (np.nanmin(doy_max) - 5 - paras3_max) / 4
                            paras4_min = max(3, paras4_min)
                            paras4_max = min(17, paras4_max)
                            if paras4_min > 17:
                                paras4_min = 3
                            if paras4_max < 3:
                                paras4_max = 17
                            paras6_max = paras4_max
                            paras6_min = paras4_min
                            if doy_senescence == [] or vi_senescence == []:
                                paras7_max = 0.01
                                paras7_min = 0.00001
                            else:
                                paras7_max = (np.nanmax(vi_max) - np.nanmin(vi_senescence)) / (doy_senescence[np.argmin(vi_senescence)] - doy_max[np.argmax(vi_max)])
                                paras7_min = (np.nanmin(vi_max) - np.nanmax(vi_senescence)) / (doy_senescence[np.argmax(vi_senescence)] - doy_max[np.argmin(vi_max)])
                            if np.isnan(paras7_min):
                                paras7_min = 0.00001
                            if np.isnan(paras7_max):
                                paras7_max = 0.01
                            paras7_max = min(paras7_max, 0.01)
                            paras7_min = max(paras7_min, 0.00001)
                            if paras7_max < 0.00001:
                                paras7_max = 0.01
                            if paras7_min > 0.01:
                                paras7_min = 0.00001
                            if paras1_min > paras[0]:
                                paras1_min = paras[0] - 0.01
                            if paras1_max < paras[0]:
                                paras1_max = paras[0] + 0.01
                            if paras2_min > paras[1]:
                                paras2_min = paras[1] - 0.01
                            if paras2_max < paras[1]:
                                paras2_max = paras[1] + 0.01
                            if paras3_min > paras[2]:
                                paras3_min = paras[2] - 1
                            if paras3_max < paras[2]:
                                paras3_max = paras[2] + 1
                            if paras4_min > paras[3]:
                                paras4_min = paras[3] - 0.1
                            if paras4_max < paras[3]:
                                paras4_max = paras[3] + 0.1
                            if paras5_min > paras[4]:
                                paras5_min = paras[4] - 1
                            if paras5_max < paras[4]:
                                paras5_max = paras[4] + 1
                            if paras6_min > paras[5]:
                                paras6_min = paras[5] - 0.5
                            if paras6_max < paras[5]:
                                paras6_max = paras[5] + 0.5
                            if paras7_min > paras[6]:
                                paras7_min = paras[6] - 0.00001
                            if paras7_max < paras[6]:
                                paras7_max = paras[6] + 0.00001
                            self._curve_fitting_dic['para_boundary_' + str(y_t) + '_' + str(x_t)] = ([paras1_min, paras2_min, paras3_min, paras4_min, paras5_min, paras6_min, paras7_min], [paras1_max, paras2_max,  paras3_max,  paras4_max,  paras5_max,  paras6_max,  paras7_max])
                            self._curve_fitting_dic['para_ori_' + str(y_t) + '_' + str(x_t)] = [paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]]
            np.save(output_path + 'para_boundary.npy', self._curve_fitting_dic)
        else:
            self._curve_fitting_dic = np.load(output_path + 'para_boundary.npy', allow_pickle=True).item()

        # Generate the year list
        if not os.path.exists(output_path + 'annual_cf_para.npy') or not os.path.exists(output_path + 'year.npy'):
            year_list = np.sort(np.unique(doy_dc // 1000))
            annual_cf_para_dic = {}
            for year in year_list:
                year = int(year)
                annual_para_dc = np.zeros([self.sa_map.shape[0], self.sa_map.shape[1], self._curve_fitting_dic['para_num'] + 1])
                annual_vi = index_dc[:, :, np.min(np.argwhere(doy_dc // 1000 == year)): np.max(np.argwhere(doy_dc // 1000 == year)) + 1]
                annual_doy = doy_dc[np.min(np.argwhere(doy_dc // 1000 == year)): np.max(np.argwhere(doy_dc // 1000 == year)) + 1]
                annual_doy = np.mod(annual_doy, 1000)

                for y_temp in range(annual_vi.shape[0]):
                    for x_temp in range(annual_vi.shape[1]):
                        if self.sa_map[y_temp, x_temp] != -32768:
                            vi_temp = annual_vi[y_temp, x_temp, :]
                            nan_index = np.argwhere(np.isnan(vi_temp))
                            vi_temp = np.delete(vi_temp, nan_index)
                            doy_temp = np.delete(annual_doy, nan_index)
                            if np.sum(~np.isnan(vi_temp)) >= self._curve_fitting_dic['para_num']:
                                try:
                                    paras, extras = curve_fit(self._curve_fitting_algorithm, doy_temp, vi_temp, maxfev=50000, p0=self._curve_fitting_dic['para_ori_' + str(y_temp) + '_' + str(x_temp)], bounds=self._curve_fitting_dic['para_boundary_' + str(y_temp) + '_' + str(x_temp)])
                                    predicted_y_data = self._curve_fitting_algorithm(doy_temp, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6])
                                    R_square = (1 - np.sum((predicted_y_data - vi_temp) ** 2) / np.sum((vi_temp - np.mean(vi_temp)) ** 2))
                                    annual_para_dc[y_temp, x_temp, :] = np.append(paras, R_square)
                                except:
                                    pass
                            else:
                                annual_para_dc[y_temp, x_temp, :] = np.nan
                        else:
                            annual_para_dc[y_temp, x_temp, :] = np.nan
                annual_cf_para_dic[str(year) + '_cf_para'] = annual_para_dc
            np.save(output_path + 'annual_cf_para.npy', annual_cf_para_dic)
            np.save(output_path + 'year.npy', year_list)
        bf.create_folder(curfit_output_path + 'Key_dic\\' )
        np.save(curfit_output_path + 'Key_dic\\' + self.ROI_name + '_curve_fitting_dic.npy', self._curve_fitting_dic)

    def _process_phenology_metrics_para(self, **kwargs):

        self._curve_fitting_algorithm = None
        if 'curve_fitting_algorithm' in kwargs.keys():
            self._curve_fitting_algorithm = kwargs['curve_fitting_algorithm']

        # Curve fitting method
        all_supported_curve_fitting_method = ['seven_para_logistic', 'two_term_fourier']
        self._curve_fitting_dic = {}
        if self._curve_fitting_algorithm is None or self._curve_fitting_algorithm == 'seven_para_logistic':
            self._curve_fitting_dic['CFM'] = 'SPL'
            self._curve_fitting_dic['para_num'] = 7
            self._curve_fitting_algorithm = seven_para_logistic_function
        elif self._curve_fitting_algorithm == 'two_term_fourier':
            self._curve_fitting_dic['CFM'] = 'TTF'
            self._curve_fitting_dic['para_num'] = 6
            self._curve_fitting_algorithm = two_term_fourier
        elif self._curve_fitting_algorithm not in all_supported_curve_fitting_method:
            print('Please double check the curve fitting method')
            sys.exit(-1)

    def phenology_metrics_generation(self, VI_list, phenology_index, **kwargs):

        # Check the VI method
        if type(VI_list) is str and VI_list in self.index_list:
            VI_list = [VI_list]
        elif type(VI_list) is list and False not in [VI_temp in self.index_list for VI_temp in VI_list]:
            pass
        else:
            raise TypeError(
                f'The input VI {VI_list} was not in supported type (list or str) or some input VI is not in the Landsat_dcs!')

        # Detect the para
        self._process_phenology_metrics_para(**kwargs)

        # Determine the phenology metrics extraction method
        if phenology_index is None:
            phenology_index = ['annual_ave_VI']
        elif type(phenology_index) == str:
            if phenology_index in self._phenology_index_all:
                phenology_index = [phenology_index]
            elif phenology_index not in self._phenology_index_all:
                raise NameError(f'{phenology_index} is not supported!')
        elif type(phenology_index) == list:
            for phenology_index_temp in phenology_index:
                if phenology_index_temp not in self._phenology_index_all:
                    phenology_index.remove(phenology_index_temp)
            if len(phenology_index) == 0:
                print('Please choose the correct phenology index!')
                sys.exit(-1)
        else:
            print('Please choose the correct phenology index!')
            sys.exit(-1)

        for VI in VI_list:
            # input the cf dic
            input_annual_file = self.work_env + VI + '_curfit_datacube\\' + self._curve_fitting_dic['CFM'] + '\\annual_cf_para.npy'
            input_year_file = self.work_env + VI + '_curfit_datacube\\' + self._curve_fitting_dic['CFM'] + '\\year.npy'
            if not os.path.exists(input_annual_file) or not os.path.exists(input_year_file):
                raise Exception('Please generate the cf para before the generation of phenology metrics')
            else:
                cf_para_dc = np.load(input_annual_file, allow_pickle=True).item()
                year_list = np.load(input_year_file)

            phenology_metrics_inform_dic = {}
            root_output_folder = self.work_env + VI + '_phenology_metrics\\' + str(self._curve_fitting_dic['CFM']) + '\\'
            bf.create_folder(root_output_folder)
            for phenology_index_temp in phenology_index:
                phenology_metrics_inform_dic[phenology_index_temp + '_' + VI + '_' + str(self._curve_fitting_dic['CFM']) + '_path'] = root_output_folder + phenology_index_temp + '\\'
                phenology_metrics_inform_dic[phenology_index_temp + '_' + VI + '_' + str(self._curve_fitting_dic['CFM']) + '_year'] = year_list
                bf.create_folder(phenology_metrics_inform_dic[phenology_index_temp + '_' + VI + '_' + str(self._curve_fitting_dic['CFM']) + '_path'])

            # Main procedure
            doy_temp = np.linspace(1, 365, 365)
            for year in year_list:
                year = int(year)
                annual_para = cf_para_dc[str(year) + '_cf_para']
                if not os.path.exists(root_output_folder + 'annual\\' + str(year) + '_phe_metrics.npy'):
                    annual_phe = np.zeros([annual_para.shape[0], annual_para.shape[1], 365])

                    for y_temp in range(annual_para.shape[0]):
                        for x_temp in range(annual_para.shape[1]):
                            if self.sa_map[y_temp, x_temp] == -32768:
                                annual_phe[y_temp, x_temp, :] = np.nan
                            else:
                                if self._curve_fitting_dic['para_num'] == 7:
                                    annual_phe[y_temp, x_temp, :] = self._curve_fitting_algorithm(doy_temp, annual_para[y_temp, x_temp, 0], annual_para[y_temp, x_temp, 1], annual_para[y_temp, x_temp, 2], annual_para[y_temp, x_temp, 3], annual_para[y_temp, x_temp, 4], annual_para[y_temp, x_temp, 5], annual_para[y_temp, x_temp, 6]).reshape([1, 1, 365])
                                elif self._curve_fitting_dic['para_num'] == 6:
                                    annual_phe[y_temp, x_temp, :] = self._curve_fitting_algorithm(doy_temp, annual_para[y_temp, x_temp, 0], annual_para[y_temp, x_temp, 1], annual_para[y_temp, x_temp, 2], annual_para[y_temp, x_temp, 3], annual_para[y_temp, x_temp, 4], annual_para[y_temp, x_temp, 5]).reshape([1, 1, 365])
                    bf.create_folder(root_output_folder + 'annual\\')
                    np.save(root_output_folder + 'annual\\' + str(year) + '_phe_metrics.npy', annual_phe)
                else:
                    annual_phe = np.load(root_output_folder + 'annual\\' + str(year) + '_phe_metrics.npy')

                # Generate the phenology metrics
                for phenology_index_temp in phenology_index:
                    phe_metrics = np.zeros([self.sa_map.shape[0], self.sa_map.shape[1]])
                    phe_metrics[self.sa_map == -32768] = np.nan

                    if not os.path.exists(phenology_metrics_inform_dic[phenology_index_temp + '_' + VI + '_' + str(self._curve_fitting_dic['CFM']) + '_path'] + str(year) + '_phe_metrics.TIF'):
                        if phenology_index_temp == 'annual_ave_VI':
                            phe_metrics = np.mean(annual_phe, axis=2)
                        elif phenology_index_temp == 'flood_ave_VI':
                            phe_metrics = np.mean(annual_phe[:, :, 182: 302], axis=2)
                        elif phenology_index_temp == 'unflood_ave_VI':
                            phe_metrics = np.mean(np.concatenate((annual_phe[:, :, 0:181], annual_phe[:, :, 302:364]), axis=2), axis=2)
                        elif phenology_index_temp == 'max_VI':
                            phe_metrics = np.max(annual_phe, axis=2)
                        elif phenology_index_temp == 'max_VI_doy':
                            phe_metrics = np.argmax(annual_phe, axis=2) + 1
                        elif phenology_index_temp == 'bloom_season_ave_VI':
                            phe_temp = copy.copy(annual_phe)
                            phe_temp[phe_temp < 0.3] = np.nan
                            phe_metrics = np.nanmean(phe_temp, axis=2)
                        elif phenology_index_temp == 'well_bloom_season_ave_VI':
                            phe_temp = copy.copy(annual_phe)
                            max_index = np.argmax(annual_phe, axis=2)
                            for y_temp_temp in range(phe_temp.shape[0]):
                                for x_temp_temp in range(phe_temp.shape[1]):
                                    phe_temp[y_temp_temp, x_temp_temp, 0: max_index[y_temp_temp, x_temp_temp]] = np.nan
                            phe_temp[phe_temp < 0.3] = np.nan
                            phe_metrics = np.nanmean(phe_temp, axis=2)
                        phe_metrics = phe_metrics.astype(np.float16)
                        phe_metrics[self.sa_map == -32768] = np.nan
                        bf.write_raster(gdal.Open(self.ds_file), phe_metrics, phenology_metrics_inform_dic[phenology_index_temp + '_' + VI + '_' + str(self._curve_fitting_dic['CFM']) + '_path'], str(year) + '_phe_metrics.TIF', raster_datatype=gdal.GDT_Float32)
            np.save(self.work_env + VI + '_phenology_metrics\\' + str(self._curve_fitting_dic['CFM']) + '_phenology_metrics.npy', phenology_metrics_inform_dic)

    def _process_quantify_para(self, **kwargs):

        # Detect whether all the indicators are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in ('curve_fitting_algorithm', 'quantify_strategy'):
                raise NameError(f'{kwarg_indicator} is not supported kwargs! Please double check!')

        # Determine quantify strategy
        self._all_quantify_str = ['percentile', 'abs_value']

        if 'quantify_str' in kwargs.keys():
            self._quantify_str = kwargs['quantify_str']
        elif 'quantify_str' not in kwargs.keys():
            self._quantify_str = ['percentile', 'abs_value']

        if type(self._quantify_str) == str:
            if self._quantify_str in self._all_quantify_str:
                self._quantify_str = [self._quantify_str]
            else:
                raise ValueError('The input quantify strategy is not available!')
        elif type(self._quantify_str) == list:
            for strat_temp in self._quantify_str:
                if strat_temp not in self._all_quantify_str:
                    self._quantify_str.remove(strat_temp)
            if len(self._quantify_str) == 0:
                raise ValueError('The input quantify strategy is not available!')
        else:
            raise TypeError('The input quantify strategy is not in supported datatype!')

        # Process the para of curve fitting method
        self._curve_fitting_algorithm = None
        if 'curve_fitting_algorithm' in kwargs.keys():
            self._curve_fitting_algorithm = kwargs['curve_fitting_algorithm']

        all_supported_curve_fitting_method = ['seven_para_logistic', 'two_term_fourier']
        self._curve_fitting_dic = {}
        if self._curve_fitting_algorithm is None or self._curve_fitting_algorithm == 'seven_para_logistic':
            self._curve_fitting_dic['CFM'] = 'SPL'
            self._curve_fitting_dic['para_num'] = 7
            self._curve_fitting_algorithm = seven_para_logistic_function
        elif self._curve_fitting_algorithm == 'two_term_fourier':
            self._curve_fitting_dic['CFM'] = 'TTF'
            self._curve_fitting_dic['para_num'] = 6
            self._curve_fitting_algorithm = two_term_fourier
        elif self._curve_fitting_algorithm not in all_supported_curve_fitting_method:
            raise ValueError('Please double check the input curve fitting method')

    def quantify_vegetation_variation(self, vi, phenology_index, **kwargs):

        # Determine the phenology metrics extraction method
        if type(phenology_index) == str:
            if phenology_index in self._phenology_index_all:
                phenology_index = [phenology_index]
            elif phenology_index not in self._phenology_index_all:
                print('Please choose the correct phenology index!')
                sys.exit(-1)
        elif type(phenology_index) == list:
            for phenology_index_temp in phenology_index:
                if phenology_index_temp not in self._phenology_index_all:
                    phenology_index.remove(phenology_index_temp)
            if len(phenology_index) == 0:
                print('Please choose the correct phenology index!')
                sys.exit(-1)
        else:
            print('Please choose the correct phenology index!')
            sys.exit(-1)

        # process vi
        if type(vi) == str:
            vi = [vi]
        elif type(vi) == list:
            vi = vi
        else:
            raise TypeError('Please input the vi under a supported Type!')

        # Process para
        self._process_quantify_para(**kwargs)

        phenology_metrics_inform_dic = {}
        for vi_temp in vi:
            # Create output folder
            root_output_folder = self.work_env + vi_temp + '_phenology_metrics\\' + str(self._curve_fitting_dic['CFM']) + '_veg_variation\\'
            bf.create_folder(root_output_folder)

            for phenology_index_temp in phenology_index:
                for quantify_st in self._quantify_str:
                    phenology_metrics_inform_dic[phenology_index_temp + '_' + vi_temp + '_' + str(self._curve_fitting_dic['CFM']) + '_' + quantify_st + '_veg_variation_path'] = root_output_folder + phenology_index_temp + '_' + quantify_st + '\\'
                    bf.create_folder(phenology_metrics_inform_dic[phenology_index_temp + '_' + vi_temp + '_' + str(self._curve_fitting_dic['CFM']) + '_' + quantify_st + '_veg_variation_path'])

            # Main process
            for phenology_index_temp in phenology_index:
                file_path = self.work_env + vi_temp + '_phenology_metrics\\' + str(self._curve_fitting_dic['CFM']) + '\\' + phenology_index_temp + '\\'
                year_list = np.load(self.work_env + vi_temp + '_curfit_datacube\\' + str(self._curve_fitting_dic['CFM']) + '\\year.npy')
                year_list = np.sort(year_list).tolist()
                for year in year_list[1:]:
                    last_year_ds = gdal.Open(bf.file_filter(file_path, [str(int(year - 1)), '.TIF'], and_or_factor='and')[0])
                    current_year_ds = gdal.Open(bf.file_filter(file_path, [str(int(year)), '.TIF'], and_or_factor='and')[0])
                    last_year_array = last_year_ds.GetRasterBand(1).ReadAsArray()
                    current_year_array = current_year_ds.GetRasterBand(1).ReadAsArray()
                    for quantify_st in self._quantify_str:
                        if quantify_st == 'percentile':
                            veg_variation_array = (current_year_array - last_year_array) / last_year_array
                        elif quantify_st == 'abs_value':
                            veg_variation_array = current_year_array - last_year_array
                        else:
                            raise Exception('Error phenology metrics')
                        bf.write_raster(last_year_ds, veg_variation_array, phenology_metrics_inform_dic[phenology_index_temp + '_' + vi_temp + '_' + str(self._curve_fitting_dic['CFM']) + '_' + quantify_st + '_veg_variation_path'], str(int(year - 1)) + '_' + str(int(year)) + '_veg_variation.TIF')

    def _process_NIPY_para(self, **kwargs: dict) -> None:
        # Detect whether all the indicators are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in ('NIPY_overwritten_factor', 'add_NIPY_dc'):
                raise NameError(f'{kwarg_indicator} is not supported kwargs! Please double check!')

        if 'add_NIPY_dc' in kwargs.keys():
            if type(kwargs['add_NIPY_dc']) != bool:
                raise TypeError('Please input the add_NIPY_dc as a bool type!')
            else:
                self._add_NIPY_dc = (kwargs['add_NIPY_dc'])
        else:
            self._add_NIPY_dc = True
        
        if 'NIPY_overwritten_factor' in kwargs.keys():
            if type(kwargs['NIPY_overwritten_factor']) != bool:
                raise TypeError('Please input the NIPY_overwritten_factor as a bool type!')
            else:
                self._NIPY_overwritten_factor = (kwargs['NIPY_overwritten_factor'])
        else:
            self._NIPY_overwritten_factor = False

    def NIPY_VI_reconstruction(self, VI, flood_mapping_method, **kwargs):

        # Check the VI method
        if type(VI) is str and VI in self.index_list:
            VI = [VI]
        elif type(VI) is list and False not in [VI_temp in self.index_list for VI_temp in VI]:
            pass
        else:
            raise TypeError(f'The input VI {VI} was not in supported type (list or str) or some input VI is not in the Landsat_dcs!')

        # Check the flood mapping method
        if flood_mapping_method not in self._flood_mapping_method:
            raise TypeError(f'The flood mapping method {flood_mapping_method} is not supported!')
        elif flood_mapping_method not in self.index_list:
            raise TypeError(f'Please construct and add the {flood_mapping_method} dc before using it')

        # Process the para
        self._process_NIPY_para(**kwargs)
        NIPY_para = {}

        for vi_temp in VI:

            # Define vi dc
            vi_doy = copy.copy(self.doy_list)
            vi_sdc = copy.copy(self.Landsat_dcs[self.index_list.index(vi_temp)].dc)

            # Define inundated dic
            inundated_sdc = copy.copy(self.Landsat_dcs[self.index_list.index(flood_mapping_method)].dc)
            inundated_doy = copy.copy(self.doy_list)

            # Process the phenology
            NIPY_para[f'NIPY_{vi_temp}_{self.ROI_name}_dcpath'] = self.work_env + str(vi_temp) + '_NIPY_' + str(flood_mapping_method) + '_sequenced_datacube\\'
            NIPY_para_path = NIPY_para[f'NIPY_{vi_temp}_{self.ROI_name}_dcpath'] + '\\NIPY_para\\'
            bf.create_folder(NIPY_para[f'NIPY_{vi_temp}_{self.ROI_name}_dcpath'])
            bf.create_folder(NIPY_para_path)

            # Preprocess the VI sdc
            for inundated_doy_index in range(inundated_doy.shape[0]):
                vi_doy_index = np.argwhere(vi_doy == inundated_doy[inundated_doy_index])
                if vi_doy_index.size == 1:
                    vi_array_temp = vi_sdc[:, :, vi_doy_index[0]].reshape([vi_sdc.shape[0], vi_sdc.shape[1]])
                    inundated_array_temp = inundated_sdc[:, :, inundated_doy_index].reshape(
                        [inundated_sdc.shape[0], inundated_sdc.shape[1]])
                    vi_array_temp[inundated_array_temp > 0] = np.nan
                    vi_array_temp[vi_array_temp <= 0] = np.nan
                    vi_sdc[:, :, vi_doy_index[0]] = vi_array_temp.reshape([vi_sdc.shape[0], vi_sdc.shape[1], 1])
                else:
                    print('Inundated dc has doy can not be found in vi dc')
                    sys.exit(-1)

            # Input the annual_inundated image
            annual_inundated_path = self.work_env + 'Landsat_Inundation_Condition\\' + self.ROI_name + '_' + flood_mapping_method + '\\annual\\'

            # Main process
            if not os.path.exists(NIPY_para[f'NIPY_{vi_temp}_{self.ROI_name}_dcpath'] + str(vi_temp) + '_NIPY_sequenced_datacube.npy') or not os.path.exists(NIPY_para[f'NIPY_{vi_temp}_{self.ROI_name}_dcpath'] + 'doy.npy') or not os.path.exists(NIPY_para[f'NIPY_{vi_temp}_{self.ROI_name}_dcpath'] + 'header.npy'):

                print(f'Start the reconstruction of {vi_temp} within the {self.ROI_name}')
                start_time = time.time()

                year_list = [int(i[i.find('.TIF') - 4: i.find('.TIF')]) for i in bf.file_filter(annual_inundated_path, ['.TIF'])]
                NIPY_header = {'ROI_name': self.ROI_name, 'VI': f'{vi_temp}_NIPY', 'Datatype': self.Datatype, 'ROI': self.ROI, 'Study_area': self.sa_map, 'ds_file': self.ds_file, 'sdc_factor': self.sdc_factor}
                NIPY_vi_dc = []
                NIPY_doy = []
                for i in range(1, len(year_list)):
                    current_year_inundated_temp_ds = gdal.Open(bf.file_filter(annual_inundated_path, ['.TIF', str(year_list[i])], and_or_factor='and')[0])
                    current_year_inundated_temp_array = current_year_inundated_temp_ds.GetRasterBand(1).ReadAsArray()
                    last_year_inundated_temp_ds = gdal.Open(bf.file_filter(annual_inundated_path, ['.TIF', str(year_list[i - 1])], and_or_factor='and')[0])
                    last_year_inundated_temp_array = last_year_inundated_temp_ds.GetRasterBand(1).ReadAsArray()
                    NIPY_temp = np.zeros([current_year_inundated_temp_array.shape[0], current_year_inundated_temp_array.shape[1], 2])
                    annual_NIPY_dc = np.zeros([vi_sdc.shape[0], vi_sdc.shape[1], 366]) * np.nan
                    annual_NIPY_doy = np.linspace(1, 366, 366) + year_list[i] * 1000
                    doy_init = np.min(np.argwhere(inundated_doy//1000 == year_list[i]))
                    doy_init_f = np.min(np.argwhere(inundated_doy//1000 == year_list[i - 1]))
                    for y_temp in range(vi_sdc.shape[0]):
                        for x_temp in range(vi_sdc.shape[1]):

                            # Obtain the doy beg and end for current year
                            doy_end_current = np.nan
                            doy_beg_current = np.nan
                            if self.sa_map[y_temp, x_temp] == -32768:
                                NIPY_temp[y_temp, x_temp, 0] = doy_beg_current
                                NIPY_temp[y_temp, x_temp, 1] = doy_end_current
                            else:
                                if current_year_inundated_temp_array[y_temp, x_temp] > 0:
                                    # Determine the doy_end_current
                                    # time_s = time.time()
                                    doy_end_factor = False
                                    doy_index = doy_init
                                    while doy_index < inundated_doy.shape[0]:
                                        if int(inundated_doy[doy_index] // 1000) == year_list[i] and inundated_sdc[y_temp, x_temp, doy_index] == 1 and 285 >= np.mod(inundated_doy[doy_index], 1000) >= 152:
                                            doy_end_current = inundated_doy[doy_index]
                                            doy_end_factor = True
                                            break
                                        elif int(inundated_doy[doy_index] // 1000) > year_list[i]:
                                            doy_end_current = year_list[i] * 1000 + 366
                                            doy_beg_current = year_list[i] * 1000
                                            break
                                        doy_index += 1

                                    # check the doy index
                                    if doy_index == 0:
                                        print('Unknown error during phenology processing doy_end_current generation!')
                                        sys.exit(-1)
                                    # p1_time = p1_time + time.time() - time_s

                                    # Determine the doy_beg_current
                                    # time_s = time.time()
                                    if doy_end_factor:
                                        if last_year_inundated_temp_array[y_temp, x_temp] > 0:
                                            while doy_index <= inundated_doy.shape[0]:
                                                if int(inundated_doy[doy_index - 1] // 1000) == year_list[i - 1] and inundated_sdc[y_temp, x_temp, doy_index - 1] == 1:
                                                    break
                                                doy_index -= 1
                                            if doy_index == inundated_doy.shape[0]:
                                                print('Unknown error during phenology processing doy_beg_current generation!')
                                                sys.exit(-1)
                                            else:
                                                doy_beg_current = inundated_doy[doy_index]
                                            # Make sure doy beg temp < doy end temp - 1000
                                            if doy_beg_current < doy_end_current - 1000 or np.isnan(doy_beg_current):
                                                doy_beg_current = doy_end_current - 1000
                                        elif last_year_inundated_temp_array[y_temp, x_temp] == 0:
                                            doy_beg_current = doy_end_current - 1000
                                    # p2_time = p2_time + time.time() - time_s
                                elif current_year_inundated_temp_array[y_temp, x_temp] == 0:
                                    doy_end_current = year_list[i] * 1000 + 366
                                    doy_beg_current = year_list[i] * 1000
                                # time_s = time.time()

                                # Construct NIPY_vi_dc
                                doy_f = doy_init_f
                                while doy_f <= inundated_doy.shape[0] - 1:
                                    if doy_end_current > inundated_doy[doy_f] > doy_beg_current:
                                        doy_index_f = np.argwhere(vi_doy == inundated_doy[doy_f])
                                        doy_temp = int(np.mod(inundated_doy[doy_f], 1000))
                                        if not np.isnan(vi_sdc[y_temp, x_temp, doy_index_f[0][0]]):
                                            annual_NIPY_dc[y_temp, x_temp, doy_temp - 1] = vi_sdc[y_temp, x_temp, doy_index_f[0][0]]
                                    elif inundated_doy[doy_f] > doy_end_current:
                                        break
                                    doy_f = doy_f + 1

                                NIPY_temp[y_temp, x_temp, 0] = doy_beg_current
                                NIPY_temp[y_temp, x_temp, 1] = doy_end_current

                    doy_index_t = 0
                    while doy_index_t < annual_NIPY_doy.shape[0]:
                        if np.isnan(annual_NIPY_dc[:, :, doy_index_t]).all():
                            annual_NIPY_dc = np.delete(annual_NIPY_dc, doy_index_t, axis=2)
                            annual_NIPY_doy = np.delete(annual_NIPY_doy, doy_index_t, axis=0)
                            doy_index_t -= 1
                        doy_index_t += 1

                    if NIPY_vi_dc == []:
                        NIPY_vi_dc = copy.copy(annual_NIPY_dc)
                    else:
                        NIPY_vi_dc = np.append(NIPY_vi_dc, annual_NIPY_dc, axis=2)

                    if NIPY_doy == []:
                        NIPY_doy = copy.copy(annual_NIPY_doy)
                    else:
                        NIPY_doy = np.append(NIPY_doy, annual_NIPY_doy, axis=0)

                    # Consistency check
                    if NIPY_vi_dc.shape[2] != NIPY_doy.shape[0]:
                        raise Exception('Consistency error for the NIPY doy and NIPY VI DC')

                    bf.write_raster(gdal.Open(self.ds_file), NIPY_temp[:,:,0], NIPY_para_path, f'{str(year_list[i])}_NIPY_beg.TIF', raster_datatype=gdal.GDT_Float32)
                    bf.write_raster(gdal.Open(self.ds_file), NIPY_temp[:,:,1], NIPY_para_path, f'{str(year_list[i])}_NIPY_end.TIF', raster_datatype=gdal.GDT_Float32)
                # Save dic and phenology dc
                if NIPY_vi_dc != [] and NIPY_doy != []:
                    np.save(NIPY_para[f'NIPY_{vi_temp}_{self.ROI_name}_dcpath'] + str(vi_temp) + '_NIPY_sequenced_datacube.npy', NIPY_vi_dc)
                    np.save(NIPY_para[f'NIPY_{vi_temp}_{self.ROI_name}_dcpath'] + 'doy.npy', NIPY_doy)
                    np.save(NIPY_para[f'NIPY_{vi_temp}_{self.ROI_name}_dcpath'] + 'header.npy', NIPY_header)

        # Add the NIPY
        if self._add_NIPY_dc:
            for vi_temp in VI:
                self.append(Landsat_dc(NIPY_para[f'NIPY_{vi_temp}_{self.ROI_name}_dcpath'], sdc_factor=self.sdc_factor))

    def _process_analyse_valid_data(self, **kwargs):
        # Detect whether all the indicators are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in ('selected_index'):
                raise NameError(f'{kwarg_indicator} is not supported kwargs! Please double check!')

        if 'selected_index' in kwargs.keys():
            if type(kwargs['selected_index']) != str:
                raise TypeError('Please input the selected_index as a string type!')
            elif kwargs['selected_index'] in self.index_list:
                self._valid_data_analyse_index = kwargs['selected_index']
            else:
                self._valid_data_analyse_index = self.index_list[0]
        else:
            self._valid_data_analyse_index = self.index_list[0]

        if 'NIPY_overwritten_factor' in kwargs.keys():
            if type(kwargs['NIPY_overwritten_factor']) != bool:
                raise TypeError('Please input the NIPY_overwritten_factor as a bool type!')
            else:
                self._NIPY_overwritten_factor = (kwargs['NIPY_overwritten_factor'])
        else:
            self._NIPY_overwritten_factor = False
        pass

    def analyse_valid_data_distribution(self, valid_threshold=0.5, *args, **kwargs):

        # These algorithm is designed to obtain the temporal distribution of cloud-free data
        # Hence the cloud should be removed before the analyse

        # Process the parameter
        self._process_analyse_valid_data(**kwargs)

        # Define local var
        temp_dc = self.Landsat_dcs[self.index_list.index(self._valid_data_analyse_index)].dc
        doy_list = self.doy_list
        sa_map = self.sa_map
        sa_area = np.sum(self.sa_map != -32768)

        # Retrieve the nandata_value in dc
        for q in range(temp_dc.shape[0]):
            for j in range(temp_dc.shape[1]):
                if sa_map[q, j] == -32768:
                    nandata_value = temp_dc[q, j, 0]
                    break

        valid_doy = []
        i = 0
        while i < len(doy_list):
            temp_array = temp_dc[:, :, i]
            if np.isnan(nandata_value):
                valid_portion = np.sum(~np.isnan(temp_array)) / sa_area
            else:
                valid_portion = np.sum(temp_array != nandata_value) / sa_area

            if valid_portion > valid_threshold:
                valid_doy.append(doy_list[i])
            i += 1

        # Output the valid_data_distribution
        date_list = bf.doy2date(valid_doy)
        year_list = [int(str(q)[0:4]) for q in date_list]
        month_list = [int(str(q)[4:6]) for q in date_list]
        day_list = [int(str(q)[6:8]) for q in date_list]

        pd_temp = pd.DataFrame({'DOY':valid_doy, 'Date': date_list, 'Year': year_list, 'Month': month_list, 'Day': day_list})
        pd_temp.to_csv(self.work_env + 'valid_data_distribution' + f'_{str(int(valid_threshold * 100))}per_' + '.csv')

    def phenology_analyse(self, **kwargs):
        pass
#     def landsat_vi2phenology_process(root_path_f, inundation_detection_factor=True, phenology_comparison_factor=True, thalweg_temp._inundation_overwritten_factor=False, inundated_pixel_phe_curve_factor=True, mndwi_threshold=0, VI_list_f=None, thalweg_temp._flood_month_list=None, pixel_limitation_f=None, curve_fitting_algorithm=None, dem_fix_inundated_factor=True, DEM_path=None, water_level_data_path=None, study_area=None, Year_range=None, CrossSection=None, VEG_path=None, file_metadata_f=None, unzipped_file_path_f=None, ROI_mask_f=None, local_std_fig_construction=False, global_local_factor=None, thalweg_temp._variance_num=2, inundation_mapping_accuracy_evaluation_factor=True, sample_rs_link_list=None, sample_data_path=None, dem_surveyed_date=None, initial_dem_fix_year_interval=1, phenology_overview_factor=False, landsat_detected_inundation_area=True, phenology_individual_factor=True, surveyed_inundation_detection_factor=False):
#         global phase0_time, phase1_time, phase2_time, phase3_time, phase4_time
#         # so, this is the Curve fitting Version 1, Generally it is used to implement two basic functions:
#         # (1) Find the inundated pixel by introducing MNDWI with an appropriate threshold and remove it.
#         # (2) Using the remaining data to fitting the vegetation growth curve
#         # (3) Obtaining vegetation phenology information
#
#         #Input all required data in figure plot
#         all_supported_curve_fitting_method = ['seven_para_logistic', 'two_term_fourier']
#         VI_sdc = {}
#         VI_curve_fitting = {}
#         if curve_fitting_algorithm is None or curve_fitting_algorithm == 'seven_para_logistic':
#             VI_curve_fitting['CFM'] = 'SPL'
#             VI_curve_fitting['para_num'] = 7
#             VI_curve_fitting['para_ori'] = [0.10, 0.8802, 108.2, 7.596, 311.4, 7.473, 0.00225]
#             VI_curve_fitting['para_boundary'] = ([0.08, 0.7, 100, 6.2, 301.6, 4.5, 0.0015], [0.12, 1.0, 115, 11.5, 321.5, 8.8, 0.0028])
#             curve_fitting_algorithm = seven_para_logistic_function
#         elif curve_fitting_algorithm == 'two_term_fourier':
#             curve_fitting_algorithm = two_term_fourier
#             VI_curve_fitting['CFM'] = 'TTF'
#             VI_curve_fitting['para_num'] = 6
#             VI_curve_fitting['para_ori'] = [0, 0, 0, 0, 0, 0.017]
#             VI_curve_fitting['para_boundary'] = ([0, -0.5, -0.5, -0.05, -0.05, 0.015], [1, 0.5, 0.5, 0.05, 0.05, 0.019])
#         elif curve_fitting_algorithm not in all_supported_curve_fitting_method:
#             print('Please double check the curve fitting method')
#             sys.exit(-1)
#
#         if phenology_overview_factor or phenology_individual_factor or phenology_comparison_factor:
#             phenology_fig_dic = {'phenology_veg_map': root_path_f + 'Landsat_phenology_curve\\'}
#             doy_factor = False
#             sdc_vi_f = np.load(root_path_f + 'Landsat_key_dic\\' + study_area + '_sdc_vi.npy', allow_pickle=True).item()
#             survey_inundation_dic = np.load(root_path_f + 'Landsat_key_dic\\' + study_area + '_survey_inundation_dic.npy', allow_pickle=True).item()
#             try:
#                 VI_list_f.remove('MNDWI')
#             except:
#                 pass
#             # Input Landsat inundated datacube
#             try:
#                 landsat_inundation_dic = np.load(root_path_f + 'Landsat_key_dic\\' + study_area + '_final_inundation_dic.npy', allow_pickle=True).item()
#                 landsat_inundated_dc = np.load(landsat_inundation_dic['final_' + study_area] + 'inundated_area_dc.npy')
#                 landsat_inundated_date = np.load(landsat_inundation_dic['final_' + study_area] + 'inundated_date_dc.npy')
#                 landsat_inundated_doy = date2doy(landsat_inundated_date)
#             except:
#                 print('Caution! Please detect the inundated area via Landsat!')
#                 sys.exit(-1)
#             # Input VI datacube
#             for vi in VI_list_f:
#                 try:
#                     phenology_fig_dic[vi + '_sdc'] = np.load(sdc_vi_f[vi + '_path'] + vi + '_sequenced_datacube.npy')
#                     if not doy_factor:
#                         phenology_fig_dic['doy'] = np.load(sdc_vi_f[vi + '_path'] + 'doy.npy').astype(int)
#                         phenology_fig_dic['doy_only'] = np.mod(phenology_fig_dic['doy'], 1000)
#                         phenology_fig_dic['year_only'] = phenology_fig_dic['doy'] // 1000
#                         doy_factor = True
#                     for doy in range(phenology_fig_dic['doy'].shape[0]):
#                         doy_inundated = np.argwhere(landsat_inundated_doy == phenology_fig_dic['doy'][doy])
#                         if doy_inundated.shape[0] == 0:
#                             pass
#                         elif doy_inundated.shape[0] > 1:
#                             print('The doy of landsat inundation cube is wrong!')
#                             sys.exit(-1)
#                         else:
#                             phenology_temp = phenology_fig_dic[vi + '_sdc'][:, :, doy]
#                             landsat_inundated_temp = landsat_inundated_dc[:, :, doy_inundated[0, 0]]
#                             phenology_temp[landsat_inundated_temp == 1] = np.nan
#                             phenology_temp[phenology_temp > 0.99] = np.nan
#                             phenology_temp[phenology_temp <= 0] = np.nan
#                             phenology_fig_dic[vi + '_sdc'][:, :, doy] = phenology_temp
#                 except:
#                     print('Please make sure all previous programme has been processed or double check the RAM!')
#                     sys.exit(-1)
#             # Input surveyed result
#             survey_inundation_dic = np.load(root_path_f + 'Landsat_key_dic\\' + study_area + '_survey_inundation_dic.npy', allow_pickle=True).item()
#             yearly_inundation_condition_tif_temp = bf.file_filter(survey_inundation_dic['surveyed_' + study_area], ['.TIF'], subfolder_detection=True)
#             initial_factor = True
#             for yearly_inundated_map in yearly_inundation_condition_tif_temp:
#                 yearly_inundated_map_ds = gdal.Open(yearly_inundated_map[0])
#                 yearly_inundated_map_raster = yearly_inundated_map_ds.GetRasterBand(1).ReadAsArray()
#                 if initial_factor:
#                     yearly_inundated_all = copy.copy(yearly_inundated_map_raster)
#                     initial_factor = False
#                 else:
#                     yearly_inundated_all += yearly_inundated_map_raster
#             date_num_threshold = 100 * len(yearly_inundation_condition_tif_temp)
#             yearly_inundated_all[yearly_inundated_all == 0] = 0
#             yearly_inundated_all[yearly_inundated_all >= date_num_threshold] = 0
#             yearly_inundated_all[yearly_inundated_all > 0] = 1
#             phenology_fig_dic['yearly_inundated_all'] = yearly_inundated_all
#             if not os.path.exists(phenology_fig_dic['phenology_veg_map'] + study_area + 'veg_map.TIF'):
#                 bf.write_raster(yearly_inundated_map_ds, yearly_inundated_all, phenology_fig_dic['phenology_veg_map'], study_area + '_veg_map.TIF')
#             # Input basic para
#             colors = {'colors_NDVI': '#00CD00', 'colors_NDVI_2': '#00EE00',
#                       'colors_NDVI_RE': '#CDBE70', 'colors_NDVI_RE2': '#CDC673',
#                       'colors_GNDVI': '#7D26CD', 'colors_MNDWI': '#FFFF00', 'colors_EVI': '#FFFF00',
#                       'colors_EVI2': '#FFD700', 'colors_OSAVI': '#FF3030', 'colors_last': '#FF0000',
#                       'colors_next': '#0000FF'}
#             markers = {'markers_NDVI': 'o', 'markers_MNDWI': '^', 'markers_EVI': '^',
#                        'markers_EVI2': 'v', 'markers_OSAVI': 'p', 'markers_NDVI_2': 'D',
#                        'markers_NDVI_RE': 'x', 'markers_NDVI_RE2': 'X', 'markers_GNDVI': 'd',
#                        'markers_last': 'o', 'markers_next': 'x'}
#             # Initial setup
#             pg.setConfigOption('background', 'w')
#             line_pen = pg.mkPen((0, 0, 255), width=5)
#             x_tick = [list(zip((15, 44, 75, 105, 136, 166, 197, 228, 258, 289, 320, 351), ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')))]
#         # Create the overview curve of phenology
#         if phenology_overview_factor is True:
#             phenology_fig_dic['overview_curve_path'] = root_path_f + 'Landsat_phenology_curve\\' + study_area + '_overview\\'
#             bf.create_folder(phenology_fig_dic['overview_curve_path'])
#             for vi in VI_list_f:
#                 file_dir = bf.file_filter(phenology_fig_dic['overview_curve_path'], ['.png'])
#                 y_max_temp = phenology_fig_dic[vi + '_sdc'].shape[0]
#                 x_max_temp = phenology_fig_dic[vi + '_sdc'].shape[1]
#                 for y in range(y_max_temp):
#                     for x in range(x_max_temp):
#                         if not phenology_fig_dic['overview_curve_path'] + 'overview_' + vi + '_' + str(x) + '_' + str(y) + '.png' in file_dir:
#                             if phenology_fig_dic['yearly_inundated_all'][y, x] == 1:
#                                 VI_list_temp = phenology_fig_dic[vi + '_sdc'][y, x, :]
#                                 plt.ioff()
#                                 plt.rcParams["font.family"] = "Times New Roman"
#                                 plt.figure(figsize=(6, 3.5))
#                                 ax = plt.axes((0.05, 0.05, 0.95, 0.95))
#                                 plt.title('Multiyear NDVI with dates')
#                                 plt.xlabel('DOY')
#                                 plt.ylabel(str(vi))
#                                 plt.xlim(xmax=365, xmin=0)
#                                 plt.ylim(ymax=1, ymin=0)
#                                 ax.tick_params(axis='x', which='major', labelsize=15)
#                                 plt.xticks([15, 44, 75, 105, 136, 166, 197, 228, 258, 289, 320, 351], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
#                                 area = np.pi * 2 ** 2
#                                 plt.scatter(phenology_fig_dic['doy_only'], VI_list_temp, s=area, c=colors['colors_last'], alpha=1, label=vi + '_last', marker=markers['markers_last'])
#                                 plt.savefig(phenology_fig_dic['overview_curve_path'] + 'overview_' + vi + '_' + str(x) + '_' + str(y) + '.png', dpi=300)
#                                 plt.close()
#
#         if phenology_individual_factor is True:
#             phenology_fig_dic['individual_curve_path'] = root_path_f + 'Landsat_phenology_curve\\' + study_area + '_annual\\'
#             bf.create_folder(phenology_fig_dic['individual_curve_path'])
#             x_temp = np.linspace(0, 365, 10000)
#             for vi in VI_list_f:
#                 surveyed_year_list = [int(i) for i in os.listdir(survey_inundation_dic['surveyed_' + study_area])]
#                 initial_t = True
#                 year_range = range(max(np.min(phenology_fig_dic['year_only']), min(surveyed_year_list)), min(np.max(surveyed_year_list), max(phenology_fig_dic['year_only'])) + 1)
#                 sdc_temp = copy.copy(phenology_fig_dic[vi + '_sdc'])
#                 doy_temp = copy.copy(phenology_fig_dic['doy_only'])
#                 year_temp = copy.copy(phenology_fig_dic['year_only'])
#                 columns = int(np.ceil(np.sqrt(len(year_range))))
#                 rows = int(len(year_range) // columns + 1 * (np.mod(len(year_range), columns) != 0))
#                 for y in range(sdc_temp.shape[0]):
#                     for x in range(sdc_temp.shape[1]):
#                         if phenology_fig_dic['yearly_inundated_all'][y, x] == 1 and not os.path.exists(phenology_fig_dic['individual_curve_path'] + 'annual_' + str(vi) + '_' + str(x) + '_' + str(y) + '.png'):
#                             phase0_s = time.time()
#                             phenology_index_temp = sdc_temp[y, x, :]
#                             nan_pos = np.argwhere(np.isnan(phenology_index_temp))
#                             doy_temp_temp = np.delete(doy_temp, nan_pos)
#                             year_temp_temp = np.delete(year_temp, nan_pos)
#                             phenology_index_temp = np.delete(phenology_index_temp, nan_pos)
#                             if len(year_range) < 3:
#                                 plt.ioff()
#                                 plt.rcParams["font.family"] = "Times New Roman"
#                                 plt.rcParams["font.size"] = "20"
#                                 plt.rcParams["figure.figsize"] = [10, 10]
#                                 ax_temp = plt.figure(figsize=(columns * 6, rows * 3.6), constrained_layout=True).subplots(rows, columns)
#                                 ax_temp = trim_axs(ax_temp, len(year_range))
#                                 for ax, year in zip(ax_temp, year_range):
#                                     if np.argwhere(year_temp_temp == year).shape[0] == 0:
#                                         pass
#                                     else:
#                                         annual_doy_temp = doy_temp_temp[np.min(np.argwhere(year_temp_temp == year)): np.max(np.argwhere(year_temp_temp == year)) + 1]
#                                         annual_phenology_index_temp = phenology_index_temp[np.min(np.argwhere(year_temp_temp == year)): np.max(np.argwhere(year_temp_temp == year)) + 1]
#                                         lineplot_factor = True
#                                         if annual_phenology_index_temp.shape[0] < 7:
#                                             lineplot_factor = False
#                                         else:
#                                             paras, extras = curve_fit(curve_fitting_algorithm, annual_doy_temp, annual_phenology_index_temp, maxfev=5000, p0=VI_curve_fitting['para_ori'], bounds=VI_curve_fitting['para_boundary'])
#                                             predicted_phenology_index = seven_para_logistic_function(annual_doy_temp, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6])
#                                             R_square = (1 - np.sum((predicted_phenology_index - annual_phenology_index_temp) ** 2) / np.sum((annual_phenology_index_temp - np.mean(annual_phenology_index_temp)) ** 2)) * 100
#                                             msg_r_square = (r'$R^2 = ' + str(R_square)[0:5] + '%$')
#                                             # msg_equation = (str(paras[0])[0:4] + '+(' + str(paras[1])[0:4] + '-' + str(paras[6])[0:4] + '* x) * ((1 / (1 + e^((' + str(paras[2])[0:4] + '- x) / ' + str(paras[3])[0:4] + '))) - (1 / (1 + e^((' + str(paras[4])[0:4] + '- x) / ' + str(paras[5])[0:4] + ')))))')
#                                         ax.set_title('annual phenology of year ' + str(year))
#                                         ax.set_xlim(xmax=365, xmin=0)
#                                         ax.set_ylim(ymax=0.9, ymin=0)
#                                         ax.set_xlabel('DOY')
#                                         ax.set_ylabel(str(vi))
#                                         ax.tick_params(axis='x', which='major', labelsize=14)
#                                         ax.tick_params(axis='y', which='major', labelsize=14)
#                                         ax.set_xticks([15, 44, 75, 105, 136, 166, 197, 228, 258, 289, 320, 351])
#                                         ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
#                                         area = np.pi * 4 ** 2
#
#                                         ax.scatter(annual_doy_temp, annual_phenology_index_temp, s=area, c=colors['colors_last'], alpha=1, marker=markers['markers_last'])
#                                         if lineplot_factor:
#                                             # ax.text(5, 0.8, msg_equation, size=14)
#                                             ax.text(270, 0.8, msg_r_square, fontsize=14)
#                                             if VI_curve_fitting['CFM'] == 'SPL':
#                                                 ax.plot(x_temp, seven_para_logistic_function(x_temp, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]), linewidth='3.5', color=colors['colors_next'])
#                                             elif VI_curve_fitting['CFM'] == 'TTF':
#                                                 ax.plot(x_temp, two_term_fourier(x_temp, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5]), linewidth='3.5', color=colors['colors_next'])
#                                 plt.savefig(phenology_fig_dic['individual_curve_path'] + 'annual_' + str(vi) + '_' + str(x) + '_' + str(y) + '.png', dpi=150)
#                                 plt.close()
#                             else:
#                                 # pg.setConfigOptions(antialias=True)
#                                 if initial_t:
#                                     phe_dic = {}
#                                     win = pg.GraphicsLayoutWidget(show=False, title="annual phenology")
#                                     win.setRange(newRect=pg.Qt.QtCore.QRectF(140, 100, 500 * columns-200, 300 * rows-200), disableAutoPixel=False)
#                                     win.resize(500 * columns, 300 * rows)
#                                     year_t = 0
#                                     for r_temp in range(rows):
#                                         for c_temp in range(columns):
#                                             if year_t < len(year_range):
#                                                 year = year_range[year_t]
#                                                 phe_dic['plot_temp_' + str(year)] = win.addPlot(row=r_temp, col=c_temp, title='annual phenology of Year ' + str(year))
#                                                 phe_dic['plot_temp_' + str(year)].setLabel('left', vi)
#                                                 phe_dic['plot_temp_' + str(year)].setLabel('bottom', 'DOY')
#                                                 x_axis = phe_dic['plot_temp_' + str(year)].getAxis('bottom')
#                                                 x_axis.setTicks(x_tick)
#                                                 phe_dic['curve_temp_' + str(year)] = pg.PlotCurveItem(pen=line_pen, name="Phenology_index")
#                                                 phe_dic['plot_temp_' + str(year)].addItem(phe_dic['curve_temp_' + str(year)])
#                                                 phe_dic['plot_temp_' + str(year)].setRange(xRange=(0, 365), yRange=(0, 0.95))
#                                                 phe_dic['scatterplot_temp_' + str(year)] = pg.ScatterPlotItem(size=0.01, pxMode=False)
#                                                 phe_dic['scatterplot_temp_' + str(year)].setPen(pg.mkPen('r', width=10))
#                                                 phe_dic['scatterplot_temp_' + str(year)].setBrush(pg.mkBrush(255, 0, 0))
#                                                 phe_dic['plot_temp_' + str(year)].addItem(phe_dic['scatterplot_temp_' + str(year)])
#                                                 phe_dic['text_temp_' + str(year)] = pg.TextItem()
#                                                 phe_dic['text_temp_' + str(year)].setPos(260, 0.92)
#                                                 phe_dic['plot_temp_' + str(year)].addItem(phe_dic['text_temp_' + str(year)])
#                                             year_t += 1
#                                     initial_t = False
#
#                                 year_t = 0
#                                 for r_temp in range(rows):
#                                     for c_temp in range(columns):
#                                         if year_t < len(year_range):
#                                             year = year_range[year_t]
#                                             if np.argwhere(year_temp_temp == year).shape[0] == 0:
#                                                 phe_dic['curve_temp_' + str(year)].setData(np.array([-2, -1]), np.array([-2, -1]))
#                                                 phe_dic['text_temp_' + str(year)].setText('')
#                                                 phe_dic['scatterplot_temp_' + str(year)].setData(np.array([-2, -1]), np.array([-2, -1]))
#                                             else:
#                                                 phase1_s = time.time()
#                                                 p_min = np.min(np.argwhere(year_temp_temp == year))
#                                                 p_max = np.max(np.argwhere(year_temp_temp == year)) + 1
#                                                 annual_doy_temp = doy_temp_temp[p_min: p_max]
#                                                 annual_phenology_index_temp = phenology_index_temp[p_min: p_max]
#                                                 # plot_temp.enableAutoRange()
#                                                 phase1_time += time.time() - phase1_s
#                                                 phase2_s = time.time()
#                                                 scatter_array = np.stack((annual_doy_temp, annual_phenology_index_temp), axis=1)
#                                                 phe_dic['scatterplot_temp_' + str(year)].setData(scatter_array[:, 0], scatter_array[:, 1])
#                                                 phase2_time += time.time() - phase2_s
#                                                 phase3_s = time.time()
#                                                 if annual_phenology_index_temp.shape[0] >= 7:
#                                                     paras, extras = curve_fit(curve_fitting_algorithm, annual_doy_temp, annual_phenology_index_temp, maxfev=5000, p0=VI_curve_fitting['para_ori'], bounds=VI_curve_fitting['para_boundary'])
#                                                     predicted_phenology_index = seven_para_logistic_function(annual_doy_temp, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6])
#                                                     R_square = (1 - np.sum((predicted_phenology_index - annual_phenology_index_temp) ** 2) / np.sum((annual_phenology_index_temp - np.mean(annual_phenology_index_temp)) ** 2)) * 100
#                                                     msg_r_square = (r'R^2 = ' + str(R_square)[0:5] + '%')
#                                                     phe_dic['curve_temp_' + str(year)].setData(x_temp, seven_para_logistic_function(x_temp, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]))
#                                                     phe_dic['text_temp_' + str(year)].setText(msg_r_square)
#                                                 else:
#                                                     phe_dic['curve_temp_' + str(year)].setData(np.array([-2, -1]), np.array([-2, -1]))
#                                                     phe_dic['text_temp_' + str(year)].setText('')
#                                                 phase3_time += time.time() - phase3_s
#                                         year_t += 1
#                                 # win.show()
#                                 phase4_s = time.time()
#                                 exporter = pg.exporters.ImageExporter(win.scene())
#                                 exporter.export(phenology_fig_dic['individual_curve_path'] + 'annual_' + str(vi) + '_' + str(x) + '_' + str(y) + '.png')
#                                 phase0_time = time.time() - phase0_s
#                                 print('Successfully export the file ' + '(annual_' + str(vi) + '_' + str(x) + '_' + str(y) + ') consuming ' + str(phase0_time) + ' seconds.')
#                                 # win.close()
#                                 phase4_time += time.time() - phase4_s
#
#         if phenology_comparison_factor is True:
#             doy_factor = False
#             try:
#                 VI_list_f.remove('MNDWI')
#             except:
#                 pass
#             inundated_curve_path = root_path_f + 'Landsat_phenology_curve\\'
#             bf.create_folder(inundated_curve_path)
#             for vi in VI_list_f:
#                 try:
#                     VI_sdc[vi + '_sdc'] = np.load(sdc_vi_f[vi + '_path'] + vi + '_sequenced_datacube.npy')
#                     if not doy_factor:
#                         VI_sdc['doy'] = np.load(sdc_vi_f[vi + '_path'] + 'doy.npy').astype(int)
#                         doy_factor = True
#                 except:
#                     print('Please make sure all previous programme has been processed or double check the RAM!')
#                     sys.exit(-1)
#                 if pixel_limitation_f is None:
#                     pixel_l_factor = False
#                 else:
#                     pixel_l_factor = True
#                 vi_inundated_curve_path = inundated_curve_path + vi + '\\'
#                 bf.create_folder(vi_inundated_curve_path)
#                 # Generate the phenology curve of the inundated pixel diagram
#                 if inundated_pixel_phe_curve_factor and not os.path.exists(root_path_f + 'Landsat_key_dic\\inundation_dic.npy'):
#                     print('Mention! Inundation map should be generated before the curve construction.')
#                     sys.exit(-1)
#                 else:
#                     inundated_dic = np.load(root_path_f + 'Landsat_key_dic\\inundation_dic.npy', allow_pickle=True).item()
#                     i = 1
#                     while i < len(inundated_dic['year_range']) - 1:
#                         yearly_vi_inundated_curve_path = vi_inundated_curve_path + str(inundated_dic['year_range'][i]) + '_' + VI_curve_fitting['CFM'] + '\\'
#                         bf.create_folder(yearly_vi_inundated_curve_path)
#                         inundated_year_doy_beg = np.argwhere(VI_sdc['doy'] > inundated_dic['year_range'][i] * 1000)[0]
#                         inundated_year_doy_end = np.argwhere(VI_sdc['doy'] < inundated_dic['year_range'][i + 1] * 1000)[-1]
#                         last_year_doy_beg = np.argwhere(VI_sdc['doy'] > inundated_dic['year_range'][i - 1] * 1000)[0]
#                         last_year_doy_end = inundated_year_doy_beg - 1
#                         next_year_doy_beg = inundated_year_doy_end + 1
#                         next_year_doy_end = np.argwhere(VI_sdc['doy'] < inundated_dic['year_range'][i + 2] * 1000)[-1]
#                         last_year_doy_beg = int(last_year_doy_beg[0])
#                         last_year_doy_end = int(last_year_doy_end[0])
#                         next_year_doy_beg = int(next_year_doy_beg[0])
#                         next_year_doy_end = int(next_year_doy_end[0])
#                         last_year = inundated_dic[str(inundated_dic['year_range'][i - 1]) + '_inundation_map']
#                         inundated_year = inundated_dic[str(inundated_dic['year_range'][i]) + '_inundation_map']
#                         next_year = inundated_dic[str(inundated_dic['year_range'][i + 1]) + '_inundation_map']
#                         inundated_detection_map = np.zeros([last_year.shape[0], last_year.shape[1]], dtype=np.uint8)
#                         inundated_detection_map[np.logical_and(np.logical_and(inundated_year == 2, last_year == 255), next_year == 255)] = 1
#                         inundated_detection_map[np.logical_and(np.logical_and(inundated_year == 2, last_year != 255), next_year != 255)] = 4
#                         inundated_detection_map[np.logical_and(np.logical_and(inundated_year == 2, last_year != 255), next_year == 255)] = 2
#                         inundated_detection_map[np.logical_and(np.logical_and(inundated_year == 2, last_year == 255), next_year != 255)] = 3
#
#                         for y in range(inundated_detection_map.shape[0]):
#                             for x in range(inundated_detection_map.shape[1]):
#                                 if inundated_detection_map[y, x] != 0:
#                                     if (pixel_l_factor and (y in range(pixel_limitation_f['y_min'], pixel_limitation_f['y_max'] + 1) and x in range(pixel_limitation_f['x_min'], pixel_limitation_f['x_max'] + 1))) or not pixel_l_factor:
#                                         last_year_VIs_temp = np.zeros([last_year_doy_end - last_year_doy_beg + 1, 3])
#                                         next_year_VIs_temp = np.zeros([next_year_doy_end - next_year_doy_beg + 1, 3])
#                                         last_year_VI_curve = np.zeros([last_year_doy_end - last_year_doy_beg + 1, 2])
#                                         next_year_VI_curve = np.zeros([next_year_doy_end - next_year_doy_beg + 1, 2])
#
#                                         last_year_VIs_temp[:, 0] = np.mod(VI_sdc['doy'][last_year_doy_beg: last_year_doy_end + 1], 1000)
#                                         last_year_VIs_temp[:, 1] = copy.copy(VI_sdc['MNDWI_sdc'][y, x, last_year_doy_beg: last_year_doy_end + 1])
#                                         last_year_VIs_temp[:, 2] = copy.copy(VI_sdc[vi + '_sdc'][y, x, last_year_doy_beg: last_year_doy_end + 1])
#                                         next_year_VIs_temp[:, 0] = np.mod(VI_sdc['doy'][next_year_doy_beg: next_year_doy_end + 1], 1000)
#                                         next_year_VIs_temp[:, 1] = copy.copy(VI_sdc['MNDWI_sdc'][y, x, next_year_doy_beg: next_year_doy_end + 1])
#                                         next_year_VIs_temp[:, 2] = copy.copy(VI_sdc[vi + '_sdc'][y, x, next_year_doy_beg: next_year_doy_end + 1])
#                                         next_year_VI_curve[:, 0] = np.mod(VI_sdc['doy'][next_year_doy_beg: next_year_doy_end + 1], 1000)
#                                         vi_curve_temp = copy.copy(VI_sdc[vi + '_sdc'][y, x, next_year_doy_beg: next_year_doy_end + 1])
#                                         mndwi_curve_temp = copy.copy(VI_sdc['MNDWI_sdc'][y, x, next_year_doy_beg: next_year_doy_end + 1])
#                                         vi_curve_temp[mndwi_curve_temp > 0] = np.nan
#                                         next_year_VI_curve[:, 1] = vi_curve_temp
#                                         last_year_VI_curve[:, 0] = np.mod(VI_sdc['doy'][last_year_doy_beg: last_year_doy_end + 1], 1000)
#                                         vi_curve_temp = copy.copy(VI_sdc[vi + '_sdc'][y, x, last_year_doy_beg: last_year_doy_end + 1])
#                                         mndwi_curve_temp = copy.copy(VI_sdc['MNDWI_sdc'][y, x, last_year_doy_beg: last_year_doy_end + 1])
#                                         vi_curve_temp[mndwi_curve_temp > 0] = np.nan
#                                         last_year_VI_curve[:, 1] = vi_curve_temp
#
#                                         last_year_VI_curve = last_year_VI_curve[~np.isnan(last_year_VI_curve).any(axis=1), :]
#                                         next_year_VI_curve = next_year_VI_curve[~np.isnan(next_year_VI_curve).any(axis=1), :]
#                                         next_year_VIs_temp = next_year_VIs_temp[~np.isnan(next_year_VIs_temp).any(axis=1), :]
#                                         last_year_VIs_temp = last_year_VIs_temp[~np.isnan(last_year_VIs_temp).any(axis=1), :]
#
#                                         paras_temp = np.zeros([2, VI_curve_fitting['para_num']])
#                                         cf_last_factor = False
#                                         cf_next_factor = False
#                                         try:
#                                             if last_year_VI_curve.shape[0] > VI_curve_fitting['para_num']:
#                                                 paras, extras = curve_fit(curve_fitting_algorithm, last_year_VI_curve[:, 0], last_year_VI_curve[:, 1], maxfev=5000, p0=VI_curve_fitting['para_ori'], bounds=VI_curve_fitting['para_boundary'])
#                                                 paras_temp[0, :] = paras
#                                                 cf_last_factor = True
#                                             else:
#                                                 paras_temp[0, :] = np.nan
#
#                                             if next_year_VI_curve.shape[0] > VI_curve_fitting['para_num']:
#                                                 paras, extras = curve_fit(curve_fitting_algorithm, next_year_VI_curve[:, 0], next_year_VI_curve[:, 1], maxfev=5000, p0=VI_curve_fitting['para_ori'], bounds=VI_curve_fitting['para_boundary'])
#                                                 paras_temp[1, :] = paras
#                                                 cf_next_factor = True
#                                             else:
#                                                 paras_temp[1, :] = np.nan
#                                         except:
#                                             np.save(yearly_vi_inundated_curve_path + str(inundated_dic['year_range'][i]) + '_' + vi + VI_curve_fitting['CFM'], VI_curve_fitting)
#
#                                         VI_curve_fitting[str(inundated_dic['year_range'][i]) + '_' + vi + '_' + str(x) + '_' + str(y) + '_T' + str(inundated_detection_map[y, x])] = paras_temp
#
#                                         x_temp = np.linspace(0, 365, 10000)
#                                         # 'QI', 'NDVI', 'NDWI', 'EVI', 'EVI2', 'OSAVI', 'GNDVI', 'NDVI_RE', 'NDVI_2', 'NDVI_RE2'
#                                         colors = {'colors_NDVI': '#00CD00', 'colors_NDVI_2': '#00EE00',
#                                                   'colors_NDVI_RE': '#CDBE70', 'colors_NDVI_RE2': '#CDC673',
#                                                   'colors_GNDVI': '#7D26CD', 'colors_MNDWI': '#FFFF00', 'colors_EVI': '#FFFF00',
#                                                   'colors_EVI2': '#FFD700', 'colors_OSAVI': '#FF3030', 'colors_last': '#FF0000', 'colors_next': '#0000FF'}
#                                         markers = {'markers_NDVI': 'o', 'markers_MNDWI': '^', 'markers_EVI': '^',
#                                                    'markers_EVI2': 'v', 'markers_OSAVI': 'p', 'markers_NDVI_2': 'D',
#                                                    'markers_NDVI_RE': 'x', 'markers_NDVI_RE2': 'X', 'markers_GNDVI': 'd', 'markers_last': 'o', 'markers_next': 'x'}
#                                         plt.rcParams["font.family"] = "Times New Roman"
#                                         plt.figure(figsize=(10, 6))
#                                         ax = plt.axes((0.1, 0.1, 0.9, 0.8))
#
#                                         plt.xlabel('DOY')
#                                         plt.ylabel(str(vi))
#                                         plt.xlim(xmax=365, xmin=0)
#                                         plt.ylim(ymax=1, ymin=-1)
#                                         ax.tick_params(axis='x', which='major', labelsize=15)
#                                         plt.xticks(
#                                             [15, 44, 75, 105, 136, 166, 197, 228, 258, 289, 320, 351],
#                                             ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
#                                         area = np.pi * 3 ** 2
#
#                                         # plt.scatter(last_year_VIs_temp[:, 0], last_year_VIs_temp[:, 1], s=area, c=colors['colors_last'], alpha=1, label='MNDWI_last', marker=markers['markers_MNDWI'])
#                                         # plt.scatter(next_year_VIs_temp[:, 0], next_year_VIs_temp[:, 1], s=area, c=colors['colors_next'], alpha=1, label='MNDWI_next', marker=markers['markers_MNDWI'])
#                                         plt.scatter(last_year_VI_curve[:, 0], last_year_VI_curve[:, 1], s=area, c=colors['colors_last'], alpha=1, label=vi + '_last', marker=markers['markers_last'])
#                                         plt.scatter(next_year_VI_curve[:, 0], next_year_VI_curve[:, 1], s=area, c=colors['colors_next'], alpha=1, label=vi + '_next', marker=markers['markers_next'])
#
#                                         # plt.show()
#
#                                         if VI_curve_fitting['CFM'] == 'SPL':
#                                             if cf_next_factor:
#                                                 plt.plot(x_temp, seven_para_logistic_function(x_temp, paras_temp[1, 0], paras_temp[1, 1], paras_temp[1, 2], paras_temp[1, 3], paras_temp[1, 4], paras_temp[1, 5], paras_temp[1, 6]),
#                                                          linewidth='1.5', color=colors['colors_next'])
#                                             if cf_last_factor:
#                                                 plt.plot(x_temp, seven_para_logistic_function(x_temp, paras_temp[0, 0], paras_temp[0, 1], paras_temp[0, 2], paras_temp[0, 3], paras_temp[0, 4], paras_temp[0, 5], paras_temp[0, 6]),
#                                                          linewidth='1.5', color=colors['colors_last'])
#                                         elif VI_curve_fitting['CFM'] == 'TTF':
#                                             if cf_next_factor:
#                                                 plt.plot(x_temp, two_term_fourier(x_temp, paras_temp[1, 0], paras_temp[1, 1], paras_temp[1, 2], paras_temp[1, 3], paras_temp[1, 4], paras_temp[1, 5]),
#                                                          linewidth='1.5', color=colors['colors_next'])
#                                             if cf_last_factor:
#                                                 plt.plot(x_temp, two_term_fourier(x_temp, paras_temp[0, 0], paras_temp[0, 1], paras_temp[0, 2], paras_temp[0, 3], paras_temp[0, 4], paras_temp[0, 5]),
#                                                          linewidth='1.5', color=colors['colors_last'])
#                                         plt.savefig(yearly_vi_inundated_curve_path + 'Plot_' + str(inundated_dic['year_range'][i]) + '_' + vi + '_' + str(x) + '_' + str(y) + '_T' + str(inundated_detection_map[y, x]) + '.png', dpi=300)
#                                         plt.close()
#                                         print('Finish plotting Figure ' + str(x) + '_' + str(y) + '_' + vi + 'from year' + str(inundated_dic['year_range'][i]))
#                         np.save(yearly_vi_inundated_curve_path + str(inundated_dic['year_range'][i]) + '_' + vi + VI_curve_fitting['CFM'], VI_curve_fitting)
#                         i += 1
# #
#
# def normalize_and_gamma_correction(data_array, p_gamma=1.52):
#     if type(data_array) != np.ndarray or data_array.shape[2] != 3:
#         print('Please input a correct image araay with three layers (R G B)!')
#         sys.exit(-1)
#     else:
#         data_array = data_array.astype(np.float16)
#         r_max = np.sort(np.unique(data_array[:, :, 0]))[-2]
#         r_min = np.sort(np.unique(data_array[:, :, 0]))[0]
#         g_max = np.sort(np.unique(data_array[:, :, 1]))[-2]
#         g_min = np.sort(np.unique(data_array[:, :, 1]))[0]
#         b_max = np.sort(np.unique(data_array[:, :, 2]))[-2]
#         b_min = np.sort(np.unique(data_array[:, :, 2]))[0]
#         data_array[:, :, 0] = 65536 * (data_array[:, :, 0] - r_min) / (r_max - r_min)
#         data_array[:, :, 2] = 65536 * (data_array[:, :, 2] - b_min) / (b_max - b_min)
#         data_array[data_array >= 65536] = 65536
#         data_array = (65536 * ((data_array / 65536) ** (1 / p_gamma))).astype(np.uint16)
#         data_array[data_array >= 65536] = 65536
#     return data_array
#
#
# def phenology_monitor(demo_path, phenology_indicator):
#     pass

