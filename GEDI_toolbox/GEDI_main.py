import h5py
import numpy as np
import pandas as pd
import geopandas as gp
from shapely.geometry import Point,LineString, Polygon, shape
import basic_function as bf
import requests as r
import os
import time
import concurrent.futures
import traceback


# def pointVisual(features, vdims):
#     return (gvts.EsriImagery * gv.Points(features, vdims=vdims).options(tools=['hover'], height=500, width=900, size=5,
#                                                                         color='yellow', fontsize={'xticks': 10, 'yticks': 10, 'xlabel':16, 'ylabel': 16}))


def gedi_finder(product, bbox):
    # Define the base CMR granule search url, including LPDAAC provider name and max page size (2000 is the max allowed)
    cmr = "https://cmr.earthdata.nasa.gov/search/granules.json?pretty=true&provider=LPDAAC_ECS&page_size=2000&concept_id="

    # Set up dictionary where key is GEDI shortname + version and value is CMR Concept ID
    concept_ids = {'GEDI01_B.002': 'C1908344278-LPDAAC_ECS',
                   'GEDI02_A.002': 'C1908348134-LPDAAC_ECS',
                   'GEDI02_B.002': 'C1908350066-LPDAAC_ECS'}

    # CMR uses pagination for queries with more features returned than the page size
    page = 1
    bbox = bbox.replace(' ', '')  # Remove any white spaces
    try:
        # Send GET request to CMR granule search endpoint w/ product concept ID, bbox & page number, format return as json
        cmr_response = r.get(f"{cmr}{concept_ids[product]}&bounding_box={bbox}&pageNum={page}").json()['feed']['entry']

        # If 2000 features are returned, move to the next page and submit another request, and append to the response
        while len(cmr_response) % 2000 == 0:
            page += 1
            cmr_response += r.get(f"{cmr}{concept_ids[product]}&bounding_box={bbox}&pageNum={page}").json()['feed'][
                'entry']

        # CMR returns more info than just the Data Pool links, below use list comprehension to return a list of DP links
        return [c['links'][0]['href'] for c in cmr_response]
    except:
        # If the request did not complete successfully, print out the response from CMR
        print(r.get(f"{cmr}{concept_ids[product]}&bounding_box={bbox.replace(' ', '')}&pageNum={page}").json())


class GEDI_df(object):

    def __init__(self, *args):

        """
        This function is used to initialize the GEDI_df class. The GEDI_df class is used to store the GEDI data in a pandas dataframe format.

        Parameters
        ----------
        *args: str
            The file path of the GEDI data in excel or csv format.

        Notes
        -----
        This function will check if the file exists and if it contains all the required information. If not, it will raise an exception.

        Returns
        -------
        GEDI_df: object
            A GEDI_df class object.

        Examples
        --------
        GEDI_df('GEDI_inform.xlsx')
        """
        self.GEDI_inform_DF = None
        # self._GEDI_fund_att = ['Date', 'Shot Number', 'Beam', 'Latitude', 'Longitude', 'Tandem-X DEM', 'Elevation (m)',
        #                        'Canopy Elevation (m)', 'Canopy Height (rh100)', 'RH 98', 'RH 25', 'Quality Flag',
        #                        'Degrade Flag', 'Sensitivity', 'Urban rate', 'Landsat water rate', 'Leaf off flag']
        self._GEDI_fund_att = ['Date', 'Shot Number', 'Beam', 'Latitude', 'Longitude', 'AGBD', 'PFT class']
        self._GEDI_fund_att_simu = ['Date', 'Shot Number', 'Beam', 'EPSG_lat', 'EPSG_lon', 'Tandem-X DEM', 'Elevation (m)',
                                    'Canopy Elevation (m)', 'Canopy Height (rh100)', 'RH 98', 'RH 25', 'Quality Flag',
                                    'Degrade Flag', 'Sensitivity', 'Urban rate', 'Landsat water rate', 'Leaf off flag']

        for GEDI_inform_xlsx in args:
            if not os.path.exists(GEDI_inform_xlsx):
                raise Exception(f'The {GEDI_inform_xlsx} is not a valid file name')
            elif GEDI_inform_xlsx.endswith('.xlsx'):
                GEDI_df = pd.read_excel(GEDI_inform_xlsx)
            elif GEDI_inform_xlsx.endswith('.csv'):
                GEDI_df = pd.read_csv(GEDI_inform_xlsx)
            else:
                raise Exception(f'The {GEDI_inform_xlsx} is not a valid xlsx file')

            # Obtain the size of gedi
            if 'high_quality' in GEDI_inform_xlsx:
                self._shp_name = GEDI_inform_xlsx.split('\\')[-1].split(f'_high_quality')[0]
                fund = self._GEDI_fund_att
            elif 'all' in GEDI_inform_xlsx:
                self._shp_name = GEDI_inform_xlsx.split('\\')[-1].split(f'_all')[0]
                fund = self._GEDI_fund_att
            elif 'simulated' in GEDI_inform_xlsx:
                self._shp_name = GEDI_inform_xlsx.split('\\')[-1].split(f'_simulated')[0]
                fund = self._GEDI_fund_att_simu
            else:
                raise Exception('Not a valid GEDI dataframe file')

            if False in [q in GEDI_df.keys() for q in fund]:
                raise Exception(f'The {GEDI_inform_xlsx} does not contain all the required inform!')

            elif self.GEDI_inform_DF is None:
                self.GEDI_inform_DF = GEDI_df[fund]

            else:
                key_temp = list(GEDI_df.keys())

                _ = 0
                while _ < len(key_temp):
                    if key_temp[_] not in self.GEDI_inform_DF.keys() or 'Unnamed' in key_temp[_] or 'index' in key_temp[_]:
                        key_temp.remove(key_temp[_])
                        _ -= 1
                    _ += 1
                self.GEDI_inform_DF = pd.merge(GEDI_df, self.GEDI_inform_DF, on=key_temp, how='outer')

            self.work_env = os.path.dirname(GEDI_inform_xlsx)
            self.file_name = os.path.basename(GEDI_inform_xlsx).split('.')[0]

        self.df_size = self.GEDI_inform_DF.shape[0]
        self.year_list = list(set([_ // 1000 for _ in self.GEDI_inform_DF['Date']]))

    def save(self, output_filename: str):
        if output_filename.endswith('.csv'):
            self.GEDI_inform_DF.to_csv(output_filename)
        elif output_filename.endswith('.xlsx') or output_filename.endswith('.xls'):
            self.GEDI_inform_DF.to_excel(output_filename)
    
    def GEDI_df2shpfile(self):
        # Take the lat/lon dataframe and convert each lat/lon to a shapely point
        self.GEDI_inform_DF['geometry'] = self.GEDI_inform_DF.apply(lambda row: Point(row.Longitude, row.Latitude), axis=1)

        # Convert to GeoDataframe
        self.GEDI_inform_DF = gp.GeoDataFrame(self.GEDI_inform_DF, crs='EPSG:4326')
        self.GEDI_inform_DF = self.GEDI_inform_DF.drop(columns=['Latitude', 'Longitude'])
        self.GEDI_inform_DF['Shot Number'] = self.GEDI_inform_DF['Shot Number'].astype(str)  # Convert shot number to string

        bf.create_folder(os.path.join(self.work_env, f'shpfile\\'))
        outName = self._shp_name + '.shp'
        self.GEDI_inform_DF.to_file(os.path.join(self.work_env, f'shpfile\\{outName}'), driver='ESRI Shapefile')
    
    def generate_boundary(self, ):
        pass

    def reprojection(self, proj: str, xycolumn_start: str = 'new'):

        if not isinstance(xycolumn_start, str):
            raise TypeError(f'{xycolumn_start} is not a str')

        if xycolumn_start + '_lat' not in self.GEDI_inform_DF.keys() or xycolumn_start + '_lon' not in self.GEDI_inform_DF.keys():
            point_temp = gp.points_from_xy(list(self.GEDI_inform_DF.Longitude), list(self.GEDI_inform_DF.Latitude), crs='epsg:4326')
            point_temp = point_temp.to_crs(crs=proj)
            lon = point_temp.x
            lat = point_temp.y

            for data_temp, name_temp in zip([lat, lon], [xycolumn_start + '_' + temp for temp in ['lat', 'lon']]):
                self.GEDI_inform_DF.insert(len(self.GEDI_inform_DF.columns), name_temp, data_temp)

            # Sort it according to lat and lon
            self.GEDI_inform_DF = self.GEDI_inform_DF.sort_values([f'{xycolumn_start}_lon', f'{xycolumn_start}_lat'], ascending=[True, False])
            self.GEDI_inform_DF = self.GEDI_inform_DF.reset_index(drop=True)
        else:
            self.GEDI_inform_DF = self.GEDI_inform_DF.sort_values([f'{xycolumn_start}_lon', f'{xycolumn_start}_lat'], ascending=[True, False])
            self.GEDI_inform_DF = self.GEDI_inform_DF.reset_index(drop=True)


class GEDI_ds(object):
    """
    The GEDI Dataset
    """
    def __init__(self, ori_folder, work_env=None):

        if not os.path.exists(ori_folder) or os.path.isfile(ori_folder):
            raise Exception('Not a valid path for GEDI DS')
        else:
            ori_folder = bf.Path(ori_folder).path_name
            self.ori_folder = ori_folder

        if work_env is None:
            self.work_env = ''
            for i in self.ori_folder.split('\\'):
                if i != self.ori_folder.split('\\')[-2] and i != '':
                    self.work_env += i + '\\'
        else:
            if os.path.isdir(work_env):
                self.work_env = bf.Path(work_env).path_name
            else:
                print(f'The {work_env} is not a valid path! The default work env will be used!')
                for i in self.ori_folder.split('\\'):
                    if i != self.ori_folder.split('\\')[-2] and i != '':
                        self.work_env += i + '\\'

        # Define the file list
        self.filelist = bf.file_filter(self.ori_folder, ['.hdf', '.h5'])
        self.l4_filelist = []
        self._l4_file_num = len(self.l4_filelist)
        self.l2_filelist = []
        self._l2_file_num = len(self.l2_filelist)

        # Var for metadata
        self.l4_metadata_pd = None
        self.l2_metadata_pd = None

        # Var for GEDI extraction
        self.GEDI_inform_DF = None
        self._shp_name = None
        self._shpfile_gp = None
        self._quality_flag = True
        self._lat_min, self._lat_max, self._lon_min, self._lon_max = None, None, None, None

    def generate_metadata(self):
        """
        Construction of the metadata 4 GEDI L4 and L2 files
        """
        print('--------------Start generate the metadata of GEDI datasets--------------')
        str_time = time.time()
        l4_metadata_temp, l2_metadata_temp = [], []
        for filepath in self.filelist:
            if 'GEDI04' in filepath:
                filename_list = filepath.split('\\')[-1].split('_')
                meta_temp = [filepath, 'GEDI_L4']
                meta_temp.append(filename_list[2][0:7])
                meta_temp.extend([filename_list[3], filename_list[6], filename_list[5], filename_list[9].split('.')[0]])
                l4_metadata_temp.append(meta_temp)
                self.l4_filelist.append(filepath)
            elif 'GEDI02' in filepath:
                filename = filepath.split('\\')[-1]
                filename_list = filename.split('_')
                meta_temp = [filepath, 'GEDI_L2']
                meta_temp.append(filename_list[2][0:7])
                meta_temp.extend([filename_list[3], filename_list[6], filename_list[5], filename_list[9].split('.')[0]])
                l2_metadata_temp.append(meta_temp)
                self.l2_filelist.append(filepath)
            else:
                raise ValueError(f'File {filepath} is not under GEDI L4 file type!')

        if len(l4_metadata_temp) > 0:
            self.l4_metadata_pd = pd.DataFrame(l4_metadata_temp, columns=['Absolute dir', 'Level', 'Sensed DOY', 'Orbit Number', 'PPDS', 'Track Number', 'Product Version'])
            self.l4_metadata_pd.to_excel(self.work_env + 'GEDI_L4_Metadata.xlsx')
            self._l4_file_num = len(self.l4_filelist)

        if len(l2_metadata_temp) > 0:
            self.l2_metadata_pd = pd.DataFrame(l2_metadata_temp, columns=['Absolute dir', 'Level', 'Sensed DOY', 'Orbit Number', 'PPDS', 'Track Number', 'Product Version'])
            self.l2_metadata_pd.to_excel(self.work_env + 'GEDI_L2_Metadata.xlsx')
            self._l2_file_num = len(self.l2_filelist)

        if len(self.filelist) > 0:
            self._file_num = len(self.filelist)
        else:
            raise Exception('No valid GEDI files')
        print(f'--------------Finished in {round(time.time()-str_time)} seconds!--------------')

    def _process_footprint_extraction_args(self, *args, **kwargs):
        # process indicator
        for temp in kwargs.keys():
            if temp not in ['shp_file', 'quality_flag', 'work_env']:
                raise KeyError(f'{temp} is not valid args for shot extraction!')

        # shp_file
        if 'shp_file' not in kwargs.keys() or kwargs['shp_file'] is None:
            print('Please mention that the GEDI dataset is extremely large! It takes times extracting all the shots information!')
            while True:
                continue_flag = str(input("Continue?(y/n):")).lower()
                if continue_flag == 'y':
                    self._shp_name = 'Entire'
                    self._shpfile_gp = None
                    break
                elif continue_flag == 'n':
                    raise Exception('Please input a valid ROI')
                else:
                    print('Invalid input!')
        else:
            if os.path.exists(kwargs['shp_file']) and kwargs['shp_file'].endswith('.shp'):
                try:
                    self._shpfile_gp = gp.read_file(kwargs['shp_file'])
                    self._shp_name = kwargs['shp_file'].split('\\')[-1].split('.')[0]
                except:
                    raise Exception('Something error during the shpfile input!')

                if self._shpfile_gp.crs != 'EPSG:4326':
                    self._shpfile_gp = self._shpfile_gp.to_crs(4326)
            else:
                raise TypeError('Please input a valid shp file path!')

        # quality_flag
        if 'quality_flag' not in kwargs.keys():
            self._quality_flag = None
        else:
            if kwargs['quality_flag'] is None:
                self._quality_flag = None
            elif type(kwargs['quality_flag']) is not bool:
                raise TypeError('The quality flag should under bool type!')
            else:
                self._quality_flag = kwargs['quality_flag']

        # work_env
        if 'work_env' in kwargs.keys():
            if os.path.isdir(kwargs['work_env']):
                self.work_env = bf.Path(kwargs['work_env']).path_name
            else:
                work_env_temp = kwargs['work_env']
                print(f'The input {work_env_temp} is not a valid filepath!')

    def _extract_L2_vegh(self, file_itr):

        start_time = time.time()
        # Check the availability of metadata
        if self.l2_metadata_pd is None:
            raise Exception('Run the metadata generation before extract information')
        elif self._l2_file_num == 0:
            raise Exception('No valid GEDI L2 files')
        else:
            filepath = self.l2_metadata_pd['Absolute dir'][file_itr]

        date, shotNum, dem, zElevation, zHigh, zLat, zLon, rh25, rh75, rh90, rh98, rh100, quality, degrade, sensitivity, beamI, urban_rate, Landsat_water_rate, leaf_off_flag = ([] for _ in range(19))
        file_name = filepath.split('\\')[-1]
        try:
            beam_itr = 0
            beam_lat_lon_time = 0
            detection_time = 0
            append_time = 0
            GEDI_l2a_temp = h5py.File(filepath, 'r')
            GEDI_l2a_objs = []
            GEDI_l2a_temp.visit(GEDI_l2a_objs.append)
            gediSDS = [o for o in GEDI_l2a_objs if isinstance(GEDI_l2a_temp[o], h5py.Dataset)]
            beamNames = [g for g in GEDI_l2a_temp.keys() if g.startswith('BEAM')]
            date_temp = self.l2_metadata_pd['Sensed DOY'][file_itr]
            for beam_temp in beamNames:

                # Open the SDS:
                # print(f'Start process {beam_temp} of {file_name} ({file_itr} of {thalweg_temp._file_num}))')
                start_time = time.time()
                beam_itr += 1
                time_sta = time.time()
                lat_array = np.array(GEDI_l2a_temp[f'{beam_temp}/lat_lowestmode'])
                lon_array = np.array(GEDI_l2a_temp[f'{beam_temp}/lon_lowestmode'])
                ll = []
                for _ in range(len(lat_array)):
                    if ~np.isnan(lat_array[_]) and ~np.isnan(lon_array[_]):
                        ll.append((lon_array[_], lat_array[_]))
                ll = LineString(ll)
                beam_lat_lon_time += time.time()-time_sta
                # print(f'beam lat lon generation consumes ca. {str(beam_lat_lon_time)} seconds.')

                if True in list(self._shpfile_gp.intersects(ll)):
                    time_sta = time.time()
                    shot_number = np.array(GEDI_l2a_temp[f'{beam_temp}/shot_number'])
                    digital_elevation_model = np.array(GEDI_l2a_temp[f'{beam_temp}/digital_elevation_model'])
                    elev_lowestmode = np.array(GEDI_l2a_temp[f'{beam_temp}/elev_lowestmode'])
                    elev_highestreturn = np.array(GEDI_l2a_temp[f'{beam_temp}/elev_highestreturn'])
                    rh = np.array(GEDI_l2a_temp[f'{beam_temp}/rh'])
                    quality_flag_temp = np.array(GEDI_l2a_temp[f'{beam_temp}/quality_flag'])
                    degrade_flag = np.array(GEDI_l2a_temp[f'{beam_temp}/degrade_flag'])
                    sensitivity_temp = np.array(GEDI_l2a_temp[f'{beam_temp}/sensitivity'])
                    urban_proportion = np.array(GEDI_l2a_temp[f'{beam_temp}/land_cover_data/urban_proportion'])
                    water_rate = np.array(GEDI_l2a_temp[f'{beam_temp}/land_cover_data/landsat_water_persistence'])
                    leaf_off_flag_temp = np.array(GEDI_l2a_temp[f'{beam_temp}/land_cover_data/leaf_off_flag'])
                    for h in range(shot_number.shape[0]):
                        if self._lat_min < lat_array[h] < self._lat_max:
                            if self._shpfile_gp.contains(Point(lon_array[h], lat_array[h]))[0]:
                                time_sta_2 = time.time()
                                date.append(date_temp)
                                zLat.append(lat_array[h])
                                zLon.append(lon_array[h])
                                shotNum.append(shot_number[h])
                                dem.append(digital_elevation_model[h])
                                zElevation.append(elev_lowestmode[h])
                                zHigh.append(elev_highestreturn[h])
                                rh25.append(rh[h, 24])
                                rh75.append(rh[h, 74])
                                rh90.append(rh[h, 89])
                                rh98.append(rh[h, 97])
                                rh100.append(rh[h, 99])
                                quality.append(quality_flag_temp[h])
                                degrade.append(degrade_flag[h])
                                sensitivity.append(sensitivity_temp[h])
                                beamI.append(beam_temp)
                                urban_rate.append(urban_proportion[h])
                                Landsat_water_rate.append(water_rate[h])
                                leaf_off_flag.append(leaf_off_flag_temp[h])
                                append_time += time.time() - time_sta_2
                                # print(f'append consumes ca. {str(append_time)} seconds.')
                    detection_time += time.time() - time_sta
                print(f'Finished in {str(time.time()-start_time)} seconds {beam_temp} of {file_name} ({file_itr + 1} of {self._file_num})).')
        except:
            print(traceback.format_exc())
            print(f'The {file_name} has some issues \n')
            date, shotNum, dem, zElevation, zHigh, zLat, zLon, rh25, rh75, rh90, rh98, rh100, quality, degrade, sensitivity, beamI, urban_rate, Landsat_water_rate, leaf_off_flag = ([] for _ in range(19))
            return date, shotNum, dem, zElevation, zHigh, zLat, zLon, rh25, rh75, rh90, rh98, rh100, quality, degrade, sensitivity, beamI, urban_rate, Landsat_water_rate, leaf_off_flag
        print(f'The \033[1;31mGEDI vegetation height\033[0m of \033[1;33m{str(self._shp_name)}\033[0m were extracted in \033[1;34m{str(time.time() - start_time)}s\033[0m ({str(file_itr + 1)} of {str(self.l2_metadata_pd.shape[0])})')

        # Output the extracted information
        if len(zLat) == len(leaf_off_flag):
            return date, shotNum, dem, zElevation, zHigh, zLat, zLon, rh25, rh75, rh90, rh98, rh100, quality, degrade, sensitivity, beamI, urban_rate, Landsat_water_rate, leaf_off_flag
        else:
            date, shotNum, dem, zElevation, zHigh, zLat, zLon, rh25, rh75, rh90, rh98, rh100, quality, degrade, sensitivity, beamI, urban_rate, Landsat_water_rate, leaf_off_flag = ([] for _ in range(19))
            raise Exception('The output list consistency is invalid!')

    def _extract_L4_AGBD(self, file_itr):

        if self.l4_metadata_pd is None:
            raise Exception('Run the metadata generation before extract information')
        else:
            filepath = self.l4_metadata_pd['Absolute dir'][file_itr]
        date, shotNum, zLat, zLon, AGBD, AGBD_se, AGBD_func, AGBD_quality, beamI, L2_quality, pft_class, tree_cover = ([] for _ in range(12))
        file_name = filepath.split('\\')[-1]
        try:
            # beam_itr = 0
            beam_lat_lon_time = 0
            detection_time = 0
            append_time = 0
            GEDI_l4a_temp = h5py.File(filepath, 'r')
            GEDI_l4a_objs = []
            GEDI_l4a_temp.visit(GEDI_l4a_objs.append)
            gediSDS = [o for o in GEDI_l4a_objs if isinstance(GEDI_l4a_temp[o], h5py.Dataset)]
            beamNames = [g for g in GEDI_l4a_temp.keys() if g.startswith('BEAM')]
            date_temp = self.l4_metadata_pd['Sensed DOY'][file_itr]
            for beam_temp in beamNames:
                # Open the SDS:
                # print(f'Start process {beam_temp} of {file_name} ({file_itr} of {thalweg_temp._file_num}))')
                start_time = time.time()
                # beam_itr += 1
                # time_sta = time.time()
                lat_array = np.array(GEDI_l4a_temp[f'{beam_temp}/lat_lowestmode'])
                lon_array = np.array(GEDI_l4a_temp[f'{beam_temp}/lon_lowestmode'])
                ll = []
                for _ in range(len(lat_array)):
                    if ~np.isnan(lat_array[_]) and ~np.isnan(lon_array[_]):
                        ll.append((lon_array[_], lat_array[_]))
                ll = LineString(ll)
                # beam_lat_lon_time += time.time()-time_sta
                # print(f'beam lat lon generation consumes ca. {str(beam_lat_lon_time)} seconds.')
                if self._shpfile_gp.intersects(ll)[0]:
                    time_sta = time.time()
                    shot_number = np.array(GEDI_l4a_temp[f'{beam_temp}/shot_number'])
                    agbd = np.array(GEDI_l4a_temp[f'{beam_temp}/agbd'])
                    agbd_se = np.array(GEDI_l4a_temp[f'{beam_temp}/agbd_se'])
                    alg_run_flag = np.array(GEDI_l4a_temp[f'{beam_temp}/algorithm_run_flag'])
                    quality_flag = np.array(GEDI_l4a_temp[f'{beam_temp}/l4_quality_flag'])
                    tree_cover_comb = np.array(GEDI_l4a_temp[f'{beam_temp}/land_cover_data/landsat_treecover'])
                    pft_class_comb = np.array(GEDI_l4a_temp[f'{beam_temp}/land_cover_data/pft_class'])
                    l2_quality_flag = np.array(GEDI_l4a_temp[f'{beam_temp}/l2_quality_flag'])
                    for h in range(shot_number.shape[0]):
                        if self._lat_min < lat_array[h] < self._lat_max:
                            if self._shpfile_gp.contains(Point(lon_array[h], lat_array[h]))[0]:
                                time_sta_2 = time.time()
                                date.append(date_temp)
                                zLat.append(lat_array[h])
                                zLon.append(lon_array[h])
                                shotNum.append(shot_number[h])
                                AGBD.append(agbd[h])
                                AGBD_se.append(agbd_se[h])
                                AGBD_func.append(alg_run_flag[h])
                                AGBD_quality.append(quality_flag[h])
                                L2_quality.append(l2_quality_flag[h])
                                pft_class.append(pft_class_comb[h])
                                tree_cover.append(tree_cover_comb[h])
                                beamI.append(beam_temp)
                                append_time += time.time() - time_sta_2
                                # print(f'append consumes ca. {str(append_time)} seconds.')
                    detection_time += time.time() - time_sta
                print(f'Finished in {str(time.time()-start_time)} seconds {beam_temp} of {file_name} ({file_itr + 1} of {self._file_num})).')
        except:
            print(traceback.format_exc())
            date, shotNum, zLat, zLon, AGBD, AGBD_se, AGBD_func, AGBD_quality, beamI, L2_quality, pft_class, tree_cover = ([] for _ in range(12))
            print(f'The {file_name} has some issues \n')
            return date, shotNum, zLat, zLon, AGBD, AGBD_se, AGBD_func, AGBD_quality, beamI, L2_quality, pft_class, tree_cover
            # raise Exception(f'{traceback.format_exc()} \n The {file_name} has some issues \n')

        print(f'The \033[1;31mGEDI vegetation AGBD\033[0m of \033[1;33m{str(self._shp_name)}\033[0m were extracted in \033[1;34m{str(time.time() - start_time)}s\033[0m ({str(file_itr + 1)} of {str(self.l4_metadata_pd.shape[0])})')

        # Output the extracted information
        if len(zLat) == len(AGBD_quality):
            return date, shotNum, zLat, zLon, AGBD, AGBD_se, AGBD_func, AGBD_quality, beamI, L2_quality, pft_class, tree_cover
        else:
            date, shotNum, zLat, zLon, AGBD, AGBD_se, AGBD_func, AGBD_quality, beamI, L2_quality, pft_class, tree_cover = ([] for _ in range(12))
            raise Exception('The output list consistency is invalid!')

    def seq_extract_L4_AGBD(self, output_df_factor=True, *args, **kwargs):

        # Process all the args
        self._process_footprint_extraction_args(*args, **kwargs)

        # Generate the initial boundary
        if self._shpfile_gp is None:
            self._lat_min, self._lat_max, self._lon_min, self._lon_max = -360, 360, -360, 360
        elif self._shpfile_gp is not None:
            self._lat_min, self._lat_max, self._lon_min, self._lon_max = self._shpfile_gp.bounds['miny'][0], self._shpfile_gp.bounds['maxy'][0], self._shpfile_gp.bounds['minx'][0], self._shpfile_gp.bounds['maxx'][0]

        # Define inform list
        date_all, shotNum_all, zLat_all, zLon_all, AGBD_all, AGBD_se_all, AGBD_func_all, AGBD_quality_all, BEAM_all, L2_quality_all, pft_class_all, tree_cover_all = ([] for _ in range(12))
        file_itr = range(0, self._file_num)
        for i in file_itr:
            date, shotNum, zLat, zLon, AGBD, AGBD_se, AGBD_func, AGBD_quality, beamI, L2_quality, pft_class, tree_cover = self._extract_L4_AGBD(i)
            date_all.extend(date)
            shotNum_all.extend(shotNum)
            zLat_all.extend(zLat)
            zLon_all.extend(zLon)
            AGBD_all.extend(AGBD)
            AGBD_se_all.extend(AGBD_se)
            AGBD_func_all.extend(AGBD_func)
            AGBD_quality_all.extend(AGBD_quality)
            BEAM_all.extend(beamI)
            L2_quality_all.extend(L2_quality)
            pft_class_all.extend(pft_class)
            tree_cover_all.extend(tree_cover)

        self.GEDI_inform_DF = pd.DataFrame(
            {'Date': date_all, 'Shot Number': shotNum_all, 'Beam': BEAM_all, 'Latitude': zLat_all, 'Longitude': zLon_all,
             'AGBD': AGBD_all, 'AGBD SE': AGBD_se_all, 'AGBD func': AGBD_func_all, 'AGBD quality': AGBD_quality_all,
             'L2 quality': L2_quality_all, 'PFT name': tree_cover_all, 'PFT class': pft_class_all})

        output_folder = os.path.join(f'{self.work_env}', 'L4_AGBD\\')
        bf.create_folder(output_folder)
        if output_df_factor:
            self.GEDI_inform_DF.to_excel(os.path.join(f'{output_folder}', f'{self._shp_name}_all.xlsx'))

            # Remove poor quality returns
            self.GEDI_inform_DF = self.GEDI_inform_DF.where(self.GEDI_inform_DF['AGBD quality'].ne(0))
            # thalweg_temp.GEDI_inform_DF = thalweg_temp.GEDI_inform_DF.where(thalweg_temp.GEDI_inform_DF['Degrade Flag'] < 1)
            # thalweg_temp.GEDI_inform_DF = thalweg_temp.GEDI_inform_DF.where(thalweg_temp.GEDI_inform_DF['Sensitivity'] > 0.95)
            self.GEDI_inform_DF = self.GEDI_inform_DF.dropna().reset_index(drop=True)
            self.GEDI_inform_DF.to_excel(os.path.join(f'{output_folder}', f'{self._shp_name}_high_quality.xlsx'))

    def mp_extract_L4_AGBD(self, output_df_factor=True, *args, **kwargs):

        # Process all the args
        self._process_footprint_extraction_args(*args, **kwargs)

        # Generate the initial boundary
        if self._shpfile_gp is None:
            self._lat_min, self._lat_max, self._lon_min, self._lon_max = -360, 360, -360, 360
        elif self._shpfile_gp is not None:
            self._lat_min, self._lat_max, self._lon_min, self._lon_max = self._shpfile_gp.bounds['miny'][0], self._shpfile_gp.bounds['maxy'][0], self._shpfile_gp.bounds['minx'][0], self._shpfile_gp.bounds['maxx'][0]

        # Define inform list
        date_all, shotNum_all, zLat_all, zLon_all, AGBD_all, AGBD_se_all, AGBD_func_all, AGBD_quality_all, BEAM_all, L2_quality_all, pft_class_all, tree_cover_all = ([] for _ in range(12))
        file_itr = range(0, self._file_num)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            result = executor.map(self._extract_L4_AGBD, file_itr)

        result = list(result)
        for result_temp in result:
            date_all.extend(result_temp[0])
            shotNum_all.extend(result_temp[1])
            zLat_all.extend(result_temp[2])
            zLon_all.extend(result_temp[3])
            AGBD_all.extend(result_temp[4])
            AGBD_se_all.extend(result_temp[5])
            AGBD_func_all.extend(result_temp[6])
            AGBD_quality_all.extend(result_temp[7])
            BEAM_all.extend(result_temp[8])
            L2_quality_all.extend(result_temp[9])
            pft_class_all.extend(result_temp[10])
            tree_cover_all.extend(result_temp[11])

        self.GEDI_inform_DF = pd.DataFrame(
            {'Date': date_all, 'Shot Number': shotNum_all, 'Beam': BEAM_all, 'Latitude': zLat_all, 'Longitude': zLon_all,
             'AGBD': AGBD_all, 'AGBD SE': AGBD_se_all, 'AGBD func': AGBD_func_all, 'AGBD quality': AGBD_quality_all,
             'L2 quality': L2_quality_all, 'PFT name': tree_cover_all, 'PFT class': pft_class_all})

        output_folder = os.path.join(f'{self.work_env}', 'L4_AGBD\\')
        bf.create_folder(output_folder)
        if output_df_factor:
            self.GEDI_inform_DF.to_excel(os.path.join(f'{output_folder}', f'{self._shp_name}_all.xlsx'))

            # Remove poor quality returns
            self.GEDI_inform_DF = self.GEDI_inform_DF.where(self.GEDI_inform_DF['AGBD quality'].ne(0))
            # thalweg_temp.GEDI_inform_DF = thalweg_temp.GEDI_inform_DF.where(thalweg_temp.GEDI_inform_DF['Degrade Flag'] < 1)
            # thalweg_temp.GEDI_inform_DF = thalweg_temp.GEDI_inform_DF.where(thalweg_temp.GEDI_inform_DF['Sensitivity'] > 0.95)
            self.GEDI_inform_DF = self.GEDI_inform_DF.dropna().reset_index(drop=True)
            self.GEDI_inform_DF.to_excel(os.path.join(f'{output_folder}', f'{self._shp_name}_high_quality.xlsx'))

    def seq_extract_L2_vegh(self, output_df_factor=True, *args, **kwargs):

        # Process all the args
        self._process_footprint_extraction_args(*args, **kwargs)

        # Generate the initial boundary
        if self._shpfile_gp is None:
            self._lat_min, self._lat_max, self._lon_min, self._lon_max = -360, 360, -360, 360
        elif self._shpfile_gp is not None:
            self._lat_min, self._lat_max, self._lon_min, self._lon_max = self._shpfile_gp.bounds['miny'][0], self._shpfile_gp.bounds['maxy'][0], self._shpfile_gp.bounds['minx'][0], self._shpfile_gp.bounds['maxx'][0]

        # Define inform list
        date_all, shotNum_all, dem_all, zElevation_all, zHigh_all, zLat_all, zLon_all, rh25_all, rh75_all, rh90_all, rh98_all, rh100_all, quality_all, degrade_all, sensitivity_all, beamI_all, urban_rate_all, Landsat_water_rate_all, leaf_off_flag_all = ([] for _ in range(19))
        file_itr = range(0, self._file_num)
        for i in file_itr:
            date, shotNum, dem, zElevation, zHigh, zLat, zLon, rh25, rh75, rh90, rh98, rh100, quality, degrade, sensitivity, beamI, urban_rate, Landsat_water_rate, leaf_off_flag = self._extract_L2_vegh(i)
            date_all.extend(date)
            shotNum_all.extend(shotNum)
            dem_all.extend(dem)
            zElevation_all.extend(zElevation)
            zHigh_all.extend(zHigh)
            zLat_all.extend(zLat)
            zLon_all.extend(zLon)
            rh25_all.extend(rh25)
            rh75_all.extend(rh75)
            rh90_all.extend(rh90)
            rh98_all.extend(rh98)
            rh100_all.extend(rh100)
            quality_all.extend(quality)
            degrade_all.extend(degrade)
            sensitivity_all.extend(sensitivity)
            beamI_all.extend(beamI)
            urban_rate_all.extend(urban_rate)
            Landsat_water_rate_all.extend(Landsat_water_rate)
            leaf_off_flag_all.extend(leaf_off_flag)

        self.GEDI_inform_DF = pd.DataFrame(
            {'Date': date_all, 'Shot Number': shotNum_all, 'Beam': beamI_all, 'Latitude': zLat_all, 'Longitude': zLon_all,
             'Tandem-X DEM': dem_all, 'Elevation (m)': zElevation_all, 'Canopy Elevation (m)': zHigh_all,
             'Canopy Height (rh100)': rh100_all, 'RH 98': rh98_all, 'RH 90': rh90_all, 'RH 75': rh75_all, 'RH 25': rh25_all, 'Quality Flag': quality_all,
             'Degrade Flag': degrade_all, 'Sensitivity': sensitivity_all, 'Urban rate': urban_rate_all,
             'Landsat water rate': Landsat_water_rate_all, 'Leaf off flag': leaf_off_flag_all})

        output_folder = os.path.join(f'{self.work_env}', 'L2_vegh\\')
        bf.create_folder(output_folder)
        if output_df_factor:
            self.GEDI_inform_DF.to_excel(os.path.join(f'{output_folder}', f'{self._shp_name}_all.xlsx'))

            # Remove poor quality returns
            self.GEDI_inform_DF = self.GEDI_inform_DF.where(self.GEDI_inform_DF['Quality Flag'].ne(0))
            # thalweg_temp.GEDI_inform_DF = thalweg_temp.GEDI_inform_DF.where(thalweg_temp.GEDI_inform_DF['Degrade Flag'] < 1)
            # thalweg_temp.GEDI_inform_DF = thalweg_temp.GEDI_inform_DF.where(thalweg_temp.GEDI_inform_DF['Sensitivity'] > 0.95)
            self.GEDI_inform_DF = self.GEDI_inform_DF.dropna().reset_index(drop=True)
            self.GEDI_inform_DF.to_excel(os.path.join(f'{output_folder}', f'{self._shp_name}_high_quality.xlsx'))

    def mp_extract_L2_vegh(self, output_df_factor=True, *args, **kwargs):

        # Process all the args
        self._process_footprint_extraction_args(*args, **kwargs)

        # Generate the initial boundary
        if self._shpfile_gp is None:
            self._lat_min, self._lat_max, self._lon_min, self._lon_max = -360, 360, -360, 360
        elif self._shpfile_gp is not None:
            self._lat_min, self._lat_max, self._lon_min, self._lon_max = self._shpfile_gp.bounds['miny'][0], self._shpfile_gp.bounds['maxy'][0], self._shpfile_gp.bounds['minx'][0], self._shpfile_gp.bounds['maxx'][0]

        # Define inform list
        date_all, shotNum_all, dem_all, zElevation_all, zHigh_all, zLat_all, zLon_all, rh25_all, rh75_all, rh90_all, rh98_all, rh100_all, quality_all, degrade_all, sensitivity_all, beamI_all, urban_rate_all, Landsat_water_rate_all, leaf_off_flag_all = ([] for _ in range(19))
        file_itr = range(0, self._file_num)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            result = executor.map(self._extract_L2_vegh, file_itr)

        result = list(result)
        for result_temp in result:
            date_all.extend(result_temp[0])
            shotNum_all.extend(result_temp[1])
            dem_all.extend(result_temp[2])
            zElevation_all.extend(result_temp[3])
            zHigh_all.extend(result_temp[4])
            zLat_all.extend(result_temp[5])
            zLon_all.extend(result_temp[6])
            rh25_all.extend(result_temp[7])
            rh75_all.extend(result_temp[8])
            rh90_all.extend(result_temp[9])
            rh98_all.extend(result_temp[10])
            rh100_all.extend(result_temp[11])
            quality_all.extend(result_temp[12])
            degrade_all.extend(result_temp[13])
            sensitivity_all.extend(result_temp[14])
            beamI_all.extend(result_temp[15])
            urban_rate_all.extend(result_temp[16])
            Landsat_water_rate_all.extend(result_temp[17])
            leaf_off_flag_all.extend(result_temp[18])

        self.GEDI_inform_DF = pd.DataFrame(
            {'Date': date_all, 'Shot Number': shotNum_all, 'Beam': beamI_all, 'Latitude': zLat_all, 'Longitude': zLon_all,
             'Tandem-X DEM': dem_all, 'Elevation (m)': zElevation_all, 'Canopy Elevation (m)': zHigh_all,
             'Canopy Height (rh100)': rh100_all, 'RH 98': rh98_all, 'RH 90': rh90_all, 'RH 75': rh75_all, 'RH 25': rh25_all, 'Quality Flag': quality_all,
             'Degrade Flag': degrade_all, 'Sensitivity': sensitivity_all, 'Urban rate': urban_rate_all,
             'Landsat water rate': Landsat_water_rate_all, 'Leaf off flag': leaf_off_flag_all})

        output_folder = os.path.join(f'{self.work_env}', 'L2_vegh\\')
        bf.create_folder(output_folder)
        if output_df_factor:
            self.GEDI_inform_DF.to_excel(os.path.join(f'{output_folder}', f'{self._shp_name}_all.xlsx'))

            # Remove poor quality returns
            self.GEDI_inform_DF = self.GEDI_inform_DF.where(self.GEDI_inform_DF['Quality Flag'].ne(0))
            # thalweg_temp.GEDI_inform_DF = thalweg_temp.GEDI_inform_DF.where(thalweg_temp.GEDI_inform_DF['Degrade Flag'] < 1)
            # thalweg_temp.GEDI_inform_DF = thalweg_temp.GEDI_inform_DF.where(thalweg_temp.GEDI_inform_DF['Sensitivity'] > 0.95)
            self.GEDI_inform_DF = self.GEDI_inform_DF.dropna().reset_index(drop=True)
            self.GEDI_inform_DF.to_excel(os.path.join(f'{output_folder}', f'{self._shp_name}_high_quality.xlsx'))

    # def visualise_shots(thalweg_temp):
    #     vdims = []
    #     for f in thalweg_temp.GEDI_inform_DF:
    #         if f not in ['geometry']:
    #             vdims.append(f)
    #
    #     visual = pointVisual(thalweg_temp.GEDI_inform_DF, vdims=vdims)
    #     # Plot the basemap and geoviews Points, defining the color as the Canopy Height for each shot
    #     (gvts.EsriImagery * gv.Points(thalweg_temp.GEDI_inform_DF, vdims=vdims).options(color='Canopy Height (rh100)', cmap='plasma', size=3,
    #                                                               tools=['hover'],
    #                                                               clim=(0, 102), colorbar=True, clabel='Meters',
    #                                                               title='GEDI Canopy Height over Redwood National Park: June 19, 2019',
    #                                                               fontsize={'xticks': 10, 'yticks': 10, 'xlabel': 16,
    #                                                                         'clabel': 12,
    #                                                                         'cticks': 10, 'title': 16,
    #                                                                         'ylabel': 16})).options(height=500,
    #                                                                                                 width=900)
    #     (gvts.EsriImagery * gv.Points(thalweg_temp.GEDI_inform_DF, vdims=vdims).options(color='Elevation (m)', cmap='terrain', size=3,
    #                                                               tools=['hover'],
    #                                                               clim=(min(thalweg_temp.GEDI_inform_DF['Elevation (m)']),
    #                                                                     max(thalweg_temp.GEDI_inform_DF['Elevation (m)'])),
    #                                                               colorbar=True, clabel='Meters',
    #                                                               title='GEDI Elevation over Redwood National Park: June 19, 2019',
    #                                                               fontsize={'xticks': 10, 'yticks': 10, 'xlabel': 16,
    #                                                                         'clabel': 12,
    #                                                                         'cticks': 10, 'title': 16,
    #                                                                         'ylabel': 16})).options(height=500,
    #                                                                                                 width=900)




