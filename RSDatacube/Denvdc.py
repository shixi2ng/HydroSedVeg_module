from importlib.metadata import metadata
from scipy import sparse as sm
from .utils import *
from NDsm import NDSparseMatrix
import datetime


class Denv_dc(object):

    ####################################################################################################
    # Denv dc represents "Daily Environment Datacube"
    # It normally contains data like daily temperature and daily radiation, etc.
    # And stack into a 3-D datacube type.
    # Currently, it was integrated into the NCEI and MODIS FPAR toolbox as an output datatype.
    ####################################################################################################

    def __init__(self, Denv_dc_filepath, work_env=None, autofill=True):

        # Check the phemetric path
        self.Denv_dc_filepath = Path(Denv_dc_filepath).path_name

        # Init key var
        self.ROI_name, self.ROI, self.ROI_tif, self.ROI_array = None, None, None, None
        self.index, self.Datatype, self.coordinate_system = None, None, None
        self.dc_group_list, self.tiles, self.oritif_folder = None, None, None
        self.sdc_factor, self.sparse_matrix, self.size_control_factor, self.huge_matrix = False, False, False, False
        self.Denv_factor, self.timescale, self.timerange = False, None, None
        self.compete_doy_list = []
        self.Nodata_value = None

        # Check work env
        if work_env is not None:
            self._work_env = Path(work_env).path_name
        else:
            self._work_env = Path(os.path.dirname(os.path.dirname(self.Denv_dc_filepath))).path_name
        self.root_path = Path(os.path.dirname(os.path.dirname(self._work_env))).path_name

        # Define the basic var name
        self._fund_factor = ('ROI_name', 'index', 'Datatype', 'ROI', 'ROI_array', 'sdc_factor',
                             'coordinate_system', 'oritif_folder', 'ROI_tif', 'sparse_matrix', 'Nodata_value',
                             'huge_matrix', 'size_control_factor', 'dc_group_list', 'tiles', 'timescale', 'timerange', 'Denv_factor')

        # Read the metadata file
        metadata_file = file_filter(self.Denv_dc_filepath, ['metadata.json'])
        if len(metadata_file) == 0:
            raise ValueError('There has no valid sdc or the metadata file of the sdc was missing!')
        elif len(metadata_file) > 1:
            raise ValueError('There has more than one metadata file in the dir')
        else:
            try:
                # Load json metadata
                with open(metadata_file[0]) as js_temp:
                    dc_metadata = json.load(js_temp)

                if not isinstance(dc_metadata, dict):
                    raise Exception('Please make sure the metadata file is a dictionary constructed in python!')
                else:
                    # Determine whether this datacube is Denv or not
                    if 'Denv_factor' not in dc_metadata.keys():
                        raise TypeError('The Denv factor was lost or it is not a Denv datacube!')
                    elif dc_metadata['Denv_factor'] is False:
                        raise TypeError(f'{self.Denv_dc_filepath} is not a Denv datacube!')
                    elif dc_metadata['Denv_factor'] is True:
                        self.Denv_factor = dc_metadata['Denv_factor']
                    else:
                        raise TypeError('The Denv factor was under wrong type!')

                    for dic_name in self._fund_factor:
                        if dic_name not in dc_metadata.keys():
                            if dic_name == 'Nodata_value' and dc_metadata.get('sparse_matrix', False):
                                self.Nodata_value = 0
                            else:
                                raise Exception(f'The {dic_name} is not in the dc metadata, double check!')
                        else:
                            self.__dict__[dic_name] = dc_metadata[dic_name]

            except:
                raise Exception('Something went wrong when reading the metadata!')

        # Check the timescale and timerange is consistency or not
        if self.timescale == 'year' and len(str(self.timerange)) != 4:
            raise Exception('The annual time range should be in YYYY format')
        elif self.timescale == 'month' and len(str(self.timerange)) != 6:
            raise Exception('The monthly time range should be in YYYYMM format')
        elif self.timescale == 'all' and len(str(self.timerange)) != '.TIF':
            raise Exception('The all time range should be in .TIF format')
        elif self.timescale is None or self.timerange is None:
            raise Exception('The timescale and timerange para is not properly assigned!')

        start_time = time.time()
        print(f'Start loading the \033[1;31m{str(self.timerange)}\033[0m Denv dc of \033[1;31m{self.index}\033[0m for the \033[1;34m{self.ROI_name}\033[0m')

        # Read doy or date file of the Datacube
        try:
            if self.sdc_factor is True:
                # Read doylist
                doy_file = file_filter(self.Denv_dc_filepath, ['doy.npy'])
                if len(doy_file) == 0:
                    raise ValueError('There has no valid doy file or file was missing!')
                elif len(doy_file) > 1:
                    raise ValueError('There has more than one doy file in the dc dir')
                else:
                    sdc_doylist = np.load(doy_file[0], allow_pickle=True)
                    self.sdc_doylist = [int(sdc_doy) for sdc_doy in sdc_doylist]
            else:
                raise TypeError('Please input as a sdc')
        except:
            raise Exception('Something went wrong when reading the doy and date list!')
        self.year_domain = set([int(np.floor(temp)) for temp in self.sdc_doylist])

        # Compete doy list
        if self.timescale == 'year':
            date_start = datetime.date(year = self.timerange, month=1, day=1).toordinal()
            date_end = datetime.date(year = self.timerange + 1, month=1, day=1).toordinal()
            compete_doy_list = [datetime.date.fromordinal(date_temp).strftime('%Y%m%d') for date_temp in range(date_start, date_end)]
            self.compete_doy_list = date2doy(compete_doy_list)
        elif self.timescale == 'month':
            year_temp = int(np.floor(self.timerange / 100))
            month_temp = int(np.mod(self.timerange, 100))
            date_start = datetime.date(year=year_temp, month=month_temp, day=1).toordinal()
            date_end = datetime.date(year=year_temp, month=month_temp + 1, day=1).toordinal()
            compete_doy_list = [datetime.date.fromordinal(date_temp).strftime('%Y%m%d') for date_temp in range(date_start, date_end)]
            self.compete_doy_list = date2doy(compete_doy_list)
        elif self.timescale == 'all':
            date_min, date_max = doy2date(min(self.sdc_doylist)), doy2date(max(self.sdc_doylist))
            date_min = datetime.date(year=int(np.floor(date_min / 1000)), month=1, day=1).toordinal() + np.mod(date_min, 1000) - 1
            date_max = datetime.date(year=int(np.floor(date_max / 1000)), month=1, day=1).toordinal() + np.mod(date_max, 1000)
            compete_doy_list = [datetime.date.fromordinal(date_temp).strftime('%Y%m%d') for date_temp in range(date_min, date_max)]
            self.compete_doy_list = date2doy(compete_doy_list)

        # Read the Denv datacube
        try:
            if self.sparse_matrix and self.huge_matrix:
                self.dc_filename = self.Denv_dc_filepath + f'{self.index}_Denv_datacube\\'
                if os.path.exists(self.dc_filename):
                    self.dc = NDSparseMatrix().load(self.dc_filename)
                else:
                    raise Exception('Please double check the code if the sparse huge matrix is generated properly')
            elif not self.huge_matrix:
                self.dc_filename = file_filter(self.Denv_dc_filepath, ['Denv_datacube.npy'])
                if len(self.dc_filename) == 0:
                    raise ValueError('There has no valid dc or the dc was missing!')
                elif len(self.dc_filename) > 1:
                    raise ValueError('There has more than one date file in the dc dir')
                else:
                    self.dc = np.load(self.dc_filename[0], allow_pickle=True)
            elif self.huge_matrix and not self.sparse_matrix:
                self.dc_filename = file_filter(self.Denv_dc_filepath, ['Denv_datacube', '.npy'], and_or_factor='and')
        except:
            raise Exception('Something went wrong when reading the datacube!')

        if autofill is True and len(self.compete_doy_list) > len(self.sdc_doylist):
            self._autofill_Denv_DC()
        elif len(self.compete_doy_list) < len(self.sdc_doylist):
            raise Exception('Code has issues in the Denv autofill procedure!')

        # autotrans sparse matrix
        if self.sparse_matrix and self.dc._matrix_type == sm.coo_matrix:
            self._autotrans_sparse_matrix()

        # Backdoor metadata check
        self._backdoor_metadata_check()

        # Size calculation and shape definition
        self.dc_XSize, self.dc_YSize, self.dc_ZSize = self.dc.shape[1], self.dc.shape[0], self.dc.shape[2]
        if self.dc_ZSize != len(self.sdc_doylist):
            raise TypeError('The Denv datacube is not consistent with the doy list')

        print(f'Finish loading the \033[1;31m{str(self.timerange)}\033[0m Denv dc of \033[1;31m{self.index}\033[0m for the \033[1;34m{self.ROI_name}\033[0m using \033[1;31m{str(time.time() - start_time)}\033[0ms')

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

        if self.ROI_tif is None or self.ROI_array is None or self.ROI is None:
            raise Exception('Please manually change the roi path in the phemetric dc')

        # Problem 2
        if backdoor_issue:
            self.save(self.Denv_dc_filepath)
            self.__init__(self.Denv_dc_filepath)

    def _autofill_Denv_DC(self):
        # Interpolate the denv dc
        autofill_factor = False
        for date_temp in self.compete_doy_list:
            if date_temp not in self.sdc_doylist:
                autofill_factor = True
                if date_temp == self.compete_doy_list[0]:
                    date_merge = [_ for _ in self.compete_doy_list if _ in self.sdc_doylist][0]
                    if self.sparse_matrix:
                        self.dc.add_layer(self.dc.SM_group[date_merge], date_temp, 0)
                    else:
                        self.dc = np.insert(self.dc, 0, values=self.dc[:, :, 0], axis=2)
                    self.sdc_doylist.insert(0, date_temp)
                elif date_temp == self.compete_doy_list[-1]:
                    date_merge = [_ for _ in self.compete_doy_list if _ in self.sdc_doylist][-1]
                    if self.sparse_matrix:
                        self.dc.add_layer(self.dc.SM_group[date_merge], date_temp, -1)
                    else:
                        self.dc = np.insert(self.dc, 0, values=self.dc[:, :, -1], axis=2)
                    self.sdc_doylist.append(date_temp)
                else:
                    date_beg, date_end, _beg, _end = None, None, None, None
                    for _ in range(1, 60):
                        ordinal_date = datetime.date(year=int(np.floor(doy2date(date_temp) / 10000)),
                                                     month=int(np.floor(np.mod(doy2date(date_temp), 10000) / 100)),
                                                     day=int(np.mod(doy2date(date_temp), 100))).toordinal()
                        if date_beg is None:
                            date_out = date2doy(int(datetime.date.fromordinal(ordinal_date - _).strftime('%Y%m%d')))
                            date_beg = date_out if date_out in self.sdc_doylist else None
                            _beg = _ if date_out in self.sdc_doylist else None

                        if date_end is None:
                            date_out = date2doy(int(datetime.date.fromordinal(ordinal_date + _).strftime('%Y%m%d')))
                            date_end = date_out if date_out in self.sdc_doylist else None
                            _end = _ if date_out in self.sdc_doylist else None

                        if date_end is not None and date_beg is not None:
                            break

                    if isinstance(self.dc, NDSparseMatrix):
                        if date_end is None:
                            array_beg = self.dc.SM_group[date_beg]
                            self.dc.add_layer(array_beg, date_temp, self.compete_doy_list.index(date_temp))
                        elif date_beg is None:
                            array_end = self.dc.SM_group[date_end]
                            self.dc.add_layer(array_end, date_temp, self.compete_doy_list.index(date_temp))
                        else:
                            type_temp = type(self.dc.SM_group[date_beg])
                            array_beg = self.dc.SM_group[date_beg].toarray()
                            array_end = self.dc.SM_group[date_end].toarray()
                            dtype_temp = array_end.dtype
                            array_beg = array_beg.astype(np.float32)
                            array_end = array_end.astype(np.float32)
                            array_out = array_beg + (array_end - array_beg) * _beg / (_beg + _end)
                            array_out = array_out.astype(dtype_temp)
                            array_out = type_temp(array_out)
                            self.dc.add_layer(array_out, date_temp, self.compete_doy_list.index(date_temp))
                    else:
                        array_beg = self.dc[:, :, date_beg]
                        array_end = self.dc[:, :, date_end]
                        array_beg = array_beg.astype(np.float32)
                        array_end = array_end.astype(np.float32)
                        array_out = array_beg + (array_end - array_beg) * _beg / (_beg + _end)
                        self.dc = np.insert(self.dc, self.compete_doy_list.index(date_temp),
                                            values=array_out.reshape([array_out.shape[0], array_out.shape[1], 1]),
                                            axis=2)
                    self.sdc_doylist.insert(self.compete_doy_list.index(date_temp), date_temp)

        if self.sdc_doylist != self.compete_doy_list:
            raise Exception('Error occurred during the autofill for the Denv DC!')

        if autofill_factor:
            self.save(self.Denv_dc_filepath)
            self.__init__(self.Denv_dc_filepath)

    def save(self, output_path: str):

        start_time = time.time()
        print(f'Start saving the sdc of \033[1;31m{self.index}\033[0m in the \033[1;34m{self.ROI_name}\033[0m')

        output_path = Path(output_path).path_name
        create_folder(output_path) if not os.path.exists(output_path) else None

        # Save doy list
        doy = self.sdc_doylist
        np.save(f'{output_path}doy.npy', doy)

        # Save the metadata
        metadata_dic = {}
        for _ in self._fund_factor:
            metadata_dic[_] = self.__dict__[_]
        with open(f'{output_path}metadata.json', 'w') as js_temp:
            json.dump(metadata_dic, js_temp)

        # Save the main datacube
        if self.sparse_matrix:
            self.dc.save(f'{output_path}{str(self.index)}_Denv_datacube\\')
        else:
            np.save(f'{output_path}{str(self.index)}_Denv_datacube.npy', self.dc)

        print(f'Finish saving the sdc of \033[1;31m{self.index}\033[0m for the \033[1;34m{self.ROI_name}\033[0m using \033[1;31m{str(time.time() - start_time)}\033[0ms')

    def _autotrans_sparse_matrix(self):

        if not isinstance(self.dc, NDSparseMatrix):
            raise TypeError('The autotrans sparse matrix is specified for the NDsm!')

        for _ in self.dc.SM_namelist:
            if isinstance(self.dc.SM_group[_], sm.coo_matrix):
                self.dc.SM_group[_] = sm.csr_matrix(self.dc.SM_group[_])
                self.dc._update_size_para()
        self.save(self.Denv_dc_filepath)

    def denv_comparison(self):
        pass

    def compute_annual_mean(self, output_path: str = None):

        start_time = time.time()
        print(f"Start calculating annual mean for \033[1;31m{self.index}\033[0m in the \033[1;34m{self.ROI_name}\033[0m")

        if output_path is None:
            output_path = Path(f"{self.Denv_dc_filepath}annual_mean\\").path_name
        else:
            output_path = Path(output_path).path_name

        create_folder(output_path) if not os.path.exists(output_path) else None

        if self.sparse_matrix and isinstance(self.dc, NDSparseMatrix):
            rows, cols, _ = self.dc.shape
            sum_arr = np.zeros((rows, cols), dtype=np.float64)
            count_arr = np.zeros((rows, cols), dtype=np.uint32)

            for sm_name in self.dc.SM_namelist:
                layer = self.dc.SM_group[sm_name]
                if not sm.issparse(layer):
                    raise TypeError('Sparse matrix flag is set, but layer is not sparse')

                layer_coo = layer.tocoo()
                data = layer_coo.data.astype(np.float64)
                row_idx = layer_coo.row
                col_idx = layer_coo.col

                if self.Nodata_value is not None:
                    valid_mask = data != self.Nodata_value
                    data = data[valid_mask]
                    row_idx = row_idx[valid_mask]
                    col_idx = col_idx[valid_mask]

                np.add.at(sum_arr, (row_idx, col_idx), data)
                np.add.at(count_arr, (row_idx, col_idx), 1)

            with np.errstate(invalid='ignore', divide='ignore'):
                annual_mean_arr = sum_arr / count_arr

            if self.Nodata_value is not None:
                annual_mean_arr[count_arr == 0] = self.Nodata_value
        else:
            # Load the full datacube as a numpy array
            dc_array = self.dc[:, :, :]
            if dc_array.ndim == 2:
                dc_array = dc_array[:, :, np.newaxis]

            dc_array = dc_array.astype(np.float32)
            if self.Nodata_value is not None:
                dc_array[dc_array == self.Nodata_value] = np.nan

            annual_mean_arr = np.nanmean(dc_array, axis=2)
            if self.Nodata_value is not None:
                annual_mean_arr[np.isnan(annual_mean_arr)] = self.Nodata_value

        annual_mean_basename = f"{self.timerange}_annual_mean_{self.index}"

        try:
            roi_ds = gdal.Open('G:\A_Landsat_Floodplain_veg\ROI_map\\floodplain_2020_map.TIF')
            write_raster(
                roi_ds,
                annual_mean_arr,
                output_path,
                f"{annual_mean_basename}.tif",
                raster_datatype=gdal.GDT_Float32,
                nodatavalue=self.Nodata_value,
            )
        except Exception as e:
            print(f"Unable to export GeoTIFF for annual mean due to: {e}")

        print(f"Finish calculating annual mean for \033[1;31m{self.index}\033[0m using \033[1;31m{str(time.time() - start_time)}\033[0ms")