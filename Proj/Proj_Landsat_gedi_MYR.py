import json
import shutil

import numpy as np

from RSDatacube.RSdc import *
from GEDI_toolbox.GEDI_main import *
import basic_function as bf
from RF.VHM import VHM

if __name__ == '__main__':

    for denv_ in ['GST']:
        dc_temp_dic = []
        for year in [str(_) for _ in range(1986, 2024)]:
            dc = Denv_dc(f'G:\\A_Climatology_dataset\\station_dataset\\CMA_dataset\\2400_all_station_1950_2023\CMA_OUTPUT\\floodplain_2020_UTM_Denv_datacube\\{denv_}\\{year}\\')
            bf.create_folder(f'G:\\A_Climatology_dataset\\station_dataset\\CMA_dataset\\2400_all_station_1950_2023\CMA_OUTPUT\\annual_mean\\{denv_}\\')
            dc.compute_annual_mean(f'G:\\A_Climatology_dataset\\station_dataset\\CMA_dataset\\2400_all_station_1950_2023\CMA_OUTPUT\\annual_mean\\{denv_}\\')

    # a = Landsat_dc('G:\A_Landsat_Floodplain_veg\Landsat_floodplain_2020_datacube\Inundation_DT_datacube')
    # b = 0
    # for _ in a.dc.SM_namelist:
    #     b += np.sum(a.dc.SM_group[_] != 0)
    #     print(str(a))

    # Link simulated gedi df with RSdc
    # index_list = [f'{_}_noninun' for _ in ['OSAVI', 'MNDWI', 'TCGREENESS', 'SVVI', 'NIR', 'GREEN', 'RED', 'SWIR', 'SWIR2', 'BLUE']]
    # dc_temp_dic = [Landsat_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\{index_}_datacube\\') for index_ in index_list]
    # rs_dc_temp = RS_dcs(*dc_temp_dic)
    # # rs_dc_temp.CCDC(index_list)
    # rs_dc_temp.process_CCDC_res('G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\CCDC\\output\\pixel_csv\\', [_ for _ in range(1986, 2024)])

    # # Calculate the biomass
    # for _ in range(2000, 2024):
    #     denv_dc = [Denv_dc(f'G:\\A_Climatology_dataset\\gridded_dataset\\MODIS_PAR_V6.2\\MODIS_Output\\floodplain_2020_Denv_datacube\\{str(_)}\\'),
    #                Denv_dc(f'G:\\A_Climatology_dataset\\station_dataset\\CMA_dataset\\2400_all_station_1950_2023\\CMA_OUTPUT\\floodplain_2020_UTM_Denv_datacube\\TEM\\{str(_)}\\')]
    #     denv_dc = RS_dcs(*denv_dc)
    #     denv_dc.simulate_biomass('G:\\A_Climatology_dataset\\station_dataset\\CMA_dataset\\2400_all_station_1950_2023\\CMA_OUTPUT\\floodplain_2020_UTM_Denv_datacube\\AGB\\')

    # # Correct the metadata
    # for index in ['SVVI', 'TCGREENESS', 'BLUE', 'GREEN', 'RED', 'NIR', 'SWIR', 'SWIR2', 'MNDWI', 'OSAVI']:
    #     with open(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\{index}_datacube\\metadata.json') as js_temp:
    #         dc_metadata = json.load(js_temp)
    #         dc_metadata["ROI"] = "G:\\A_Landsat_Floodplain_veg\\ROI_map\\floodplain_2020.shp"
    #         dc_metadata["ROI_array"] = "G:\\A_Landsat_Floodplain_veg\\ROI_map\\floodplain_2020_map.npy"
    #         dc_metadata["ROI_tif"] = "G:\\A_Landsat_Floodplain_veg\\ROI_map\\floodplain_2020_map.TIF"
    #         dc_metadata["oritif_folder"] = f"G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_index\\{index}\\"
    #
    #     with open(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\{index}_datacube\\metadata.json', 'w') as js_temp:
    #         json.dump(dc_metadata, js_temp)

    # # Inundation detection
    # Landsat_WI_temp = Landsat_dc('G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_datacube')
    # Landsat_WI_temp.print_stacked_Zvalue()
    # Landsat_dcs = RS_dcs(Landsat_WI_temp)
    # Landsat_dcs.inundation_detection('DT', 'MNDWI', 'Landsat', DT_std_fig_construction=False)

    # for index in ['SVVI', 'TCGREENESS', 'BLUE', 'GREEN', 'RED', 'NIR', 'SWIR', 'SWIR2', 'OSAVI']:
    #     Landsat_inun_temp = Landsat_dc('G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube')
    #     Landsat_index_temp = Landsat_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\{index}_datacube')
    #     Landsat_dcs = RS_dcs(Landsat_inun_temp, Landsat_index_temp)
    #     Landsat_dcs.inundation_removal(index, 'DT', 'Landsat', append_new_dc=False)
    #

    # pheme_list = []
    # for year in [str(_) for _ in range(1987,2024)]:
    #     pheme_list.append(Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{year}\\'))
    # pheme_all = RS_dcs(*pheme_list)
    # pheme_all.plot_pheme_var('G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\Pheme_fig\\')

    # Landsat_VI_temp = Landsat_dc('G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_datacube\\')
    # Landsat_dcs = RS_dcs(Landsat_VI_temp)
    # Landsat_dcs.curve_fitting('OSAVI_noninun')
    #
    # for year in [str(_) for _ in range(1987, 2024)]:
    #     dc_temp_dic = Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{year}\\')
    #     dc_temp_dic.compute_phemetrics(['EOS', 'SOS'], replace=True)

    # # Vegtype map
    # ds = gdal.Open('G:\\A_GEDI_Floodplain_vegh\\Veg_map\\results\\Classification\\veg_map_ex.v1.tif')
    # arr = ds.GetRasterBand(3).ReadAsArray()
    # arr[arr == 18] = 0
    # arr[arr == 107] = 1
    # arr[arr == 179] = 2
    # arr[arr == 0] = 3
    # arr[arr == 35] = 4
    # arr[arr == 52] = 5
    # arr[arr == 255] = 6
    # driver = gdal.GetDriverByName('GTiff')
    # out_ds = driver.Create('G:\\A_GEDI_Floodplain_vegh\\Veg_map\\results\\Classification\\veg_map_ex_v2.tif', ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_UInt16, options=['COMPRESS=LZW'])
    #
    # # Set the geotransform and projection from the original dataset
    # out_ds.SetGeoTransform(ds.GetGeoTransform())
    # out_ds.SetProjection(ds.GetProjection())
    #
    # # Write the array to the new dataset
    # out_band = out_ds.GetRasterBand(1)
    # out_band.WriteArray(arr)
    #
    # # Flush the cache to ensure all data is written to disk
    # out_band.FlushCache()
    #
    # # Close the datasets
    # del out_ds
    # del ds
    # pass

    #     with open(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\{index}_datacube\\metadata.json', 'w') as js_temp:
    #         json.dump(dc_metadata, js_temp)
    #     inun_dc = Landsat_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\')
    #     dc_temp = Landsat_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\{index}_datacube\\')
    #     Landsat_dcs = RS_dcs(dc_temp, inun_dc)
    #     Landsat_dcs.inundation_removal(index, 'DT', 'Landsat', append_new_dc=False)
    #     Landsat_dcs = None
    #
    # # Link high quality GEDI data with Phedc

    # dc_temp_dic = []
    # c = 10000
    # d=-10000
    # q = []
    # for year in [str(_) for _ in range(1987, 2024)]:
    #     a = Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{year}\\')
    #     B = a.dc.SM_group[f'{str(year)}_MAVI'].toarray()
    #     B[B == 0] = np.nan
    #     aa = B.flatten()
    #     aa = aa[~np.isnan(aa)]
    #     q.extend(aa.tolist())
    #     c = min(c, np.nanmin(B))
    #     d = max(d, np.nanmax(B))
    #     pass
    # pass
    # dc_temp_dic = []
    # for year in [str(_) for _ in range(2018, 2024)]:
    #     dc_temp_dic.append(Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{year}\\'))
    # rsdc_temp = RS_dcs(*dc_temp_dic)
    # rsdc_temp.link_GEDI_Phemedc(GEDI_df('G:\\A_Landsat_Floodplain_veg\\GEDI_L4A\\Result\\floodplain_2020_high_quality.xlsx'), ['MAVI'], spatial_interpolate_method=['area_average', 'focal'],)

    # # Link high quality GEDI data with RSdc
    # dc_temp_dic = []
    # index_list = [f'{_}_noninun' for _ in ['OSAVI', 'GREEN', 'BLUE', 'RED', 'SWIR', 'SWIR2', 'NIR', 'SVVI', 'TCGREENESS']]
    # for index_ in index_list:
    #     dc_temp_dic.append(Landsat_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\{index_}_datacube\\'))
    # for year in [str(_) for _ in range(2018, 2024)]:
    #     dc_temp_dic.append(Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{year}\\'))
    # rs_dc_temp = RS_dcs(*dc_temp_dic)
    # rs_dc_temp.link_GEDI_RS_dc(GEDI_df('G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\floodplain_2020_high_quality.xlsx'), index_list, spatial_interpolate_method=['area_average', 'focal'], temporal_interpolate_method = ['linear_interpolation', '24days_max', '24days_ave'])

    # # Link high quality GEDI data with VegType
    # rsdc_temp = RS_dcs(Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\2018\\'))
    # rsdc_temp.link_GEDI_VegType('G:\A_GEDI_Floodplain_vegh\Veg_map\Veg_classification\\veg_map_30m.tif', GEDI_df('G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\floodplain_2020_high_quality.xlsx'), spatial_interpolate_method=['area_average', 'focal'])

    # Link high quality GEDI data with denvdc and par
    # for denv_ in ['AGB', 'TEM', 'RHU', 'PRE', 'PRS', 'RHU', 'WIN']:
    #     dc_temp_dic = []
    #     for year in [str(_) for _ in range(2018, 2024)]:
    #         dc_temp_dic.append(Denv_dc(f'G:\\A_Climatology_dataset\\station_dataset\\CMA_dataset\\2400_all_station_1950_2023\\CMA_OUTPUT\\floodplain_2020_UTM_Denv_datacube\\{denv_}\\{year}\\'))
    #         dc_temp_dic.append(Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{year}\\'))
    #     rsdc_temp = RS_dcs(*dc_temp_dic)
    #     rsdc_temp.link_GEDI_Denvdc(GEDI_df('G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\floodplain_2020_high_quality.xlsx'), [denv_], spatial_interpolate_method=['area_average', 'focal'])
    #
    # dc_temp_dic = []
    # for year in [str(_) for _ in range(2018, 2024)]:
    #     dc_temp_dic.append(Denv_dc(f'G:\\A_Climatology_dataset\\gridded_dataset\\MODIS_PAR_V6.2\\MODIS_Output\\floodplain_2020_Denv_datacube\\{year}\\'))
    #     dc_temp_dic.append(Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{year}\\'))
    # rsdc_temp = RS_dcs(*dc_temp_dic)
    # rsdc_temp.link_GEDI_Denvdc(GEDI_df('G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\floodplain_2020_high_quality.xlsx'), ['DPAR'], spatial_interpolate_method=['area_average', 'focal'])
    # bf.merge_csv_files("G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\GEDI_link_RS\\", 'G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\floodplain_GEDI_2019_2023_for_model_high_quality.csv', merge_keys = ['Quality Flag', 'Canopy Elevation (m)', 'index', 'EPSG_lat', 'Elevation (m)', 'Beam', 'Landsat water rate', 'Latitude', 'Shot Number', 'Sensitivity', 'EPSG_lon', 'Tandem-X DEM', 'Longitude', 'Degrade Flag', 'Date', 'RH 98', 'RH 25', 'Urban rate', 'Canopy Height (rh100)', 'Leaf off flag'])


    ###############################################################################
    # # # Simulate gedi df
    # for year in [str(_) for _ in range(1987, 2024)]:
    #     dcs_temp = RS_dcs(Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{year}\\'))
    #     dcs_temp.simulate_GEDI_df([f'peak_{year}'], 'G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\')
    #
    # # # Link simulated gedi df with Phedc
    # for year in [str(_) for _ in range(1987, 2024)]:
    #     dc_temp_dic = []
    #     dc_temp_dic.append(Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{year}\\'))
    #     rsdc_temp = RS_dcs(*dc_temp_dic)
    #     rsdc_temp.link_GEDI_Phemedc(GEDI_df(f'G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\simulated_GEDI_df\\peak_{year}_simulated.csv'), ['DR', 'DR2', 'EOS', 'GR', 'SOS', 'peak_vi', 'peak_doy', 'trough_vi', 'MAVI'], spatial_interpolate_method=['nearest_neighbor'],)
    #
    # # # # Link simulated gedi df with RSdc
    # index_list = [f'{_}_noninun' for _ in ['OSAVI', 'GREEN', 'BLUE', 'RED', 'SWIR', 'SWIR2', 'NIR', 'SVVI', 'TCGREENESS']]
    # for index_ in index_list:
    #     dc_temp_dic = []
    #     dc_temp_dic.append(Landsat_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\{index_}_datacube\\'))
    #     for year in [str(_) for _ in range(1987, 2024)]:
    #         dc_temp_dic.append(Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{year}\\'))
    #     rs_dc_temp = RS_dcs(*dc_temp_dic)
    #     for year in [str(_) for _ in range(1987, 2024)]:
    #         rs_dc_temp.link_GEDI_RS_dc(GEDI_df(f'G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\simulated_GEDI_df\\peak_{year}_simulated.csv'), {index_}, spatial_interpolate_method=['nearest_neighbor'])

    # # Link simulated gedi df with denvdc and par
    # for denv_ in ['AGB', 'TEM', 'RHU', 'PRE', 'PRS', 'RHU', 'WIN']:
    #     for year in [str(_) for _ in range(1987, 2024)]:
    #         dc_temp_dic = []
    #         dc_temp_dic.append(Denv_dc(f'G:\\A_Climatology_dataset\\station_dataset\\CMA_dataset\\2400_all_station_1950_2023\\CMA_OUTPUT\\floodplain_2020_UTM_Denv_datacube\\{denv_}\\{year}\\'))
    #         dc_temp_dic.append(Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{year}\\'))
    #         rsdc_temp = RS_dcs(*dc_temp_dic)
    #         rsdc_temp.link_GEDI_Denvdc(GEDI_df(f'G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\simulated_GEDI_df\\peak_{year}_simulated.csv'), [denv_], spatial_interpolate_method=['nearest_neighbor'])
    #
    # for year in [str(_) for _ in range(1987,2024)]:
    #     dc_temp_dic = []
    #     dc_temp_dic.append(Denv_dc(f'G:\\A_Climatology_dataset\\gridded_dataset\\MODIS_PAR_V6.2\\MODIS_Output\\floodplain_2020_Denv_datacube\\{year}\\'))
    #     dc_temp_dic.append(Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{year}\\'))
    #     rsdc_temp = RS_dcs(*dc_temp_dic)
    #     rsdc_temp.link_GEDI_Denvdc(GEDI_df(f'G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\simulated_GEDI_df\\peak_{year}_simulated.csv'), ['DPAR'], spatial_interpolate_method=['nearest_neighbor'])

    # # Link simulated gedi df with VegType
    # for year in [str(_) for _ in range(1987,2024)]:
    #     rsdc_temp = RS_dcs(Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{year}\\'))
    #     rsdc_temp.link_GEDI_VegType('G:\A_GEDI_Floodplain_vegh\Veg_map\Veg_classification\\veg_map_30m.tif', GEDI_df(f'G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\simulated_GEDI_df\\peak_{year}_simulated.csv'), spatial_interpolate_method=['nearest_neighbor'])

    # for _ in range(2023, 2024):
    #     bf.merge_csv_files(f"G:\A_GEDI_Floodplain_vegh\GEDI_MYR\L2_vegh\simu_2023\GEDI_link_RS\peak_2023_simulated\\", f'G:\A_GEDI_Floodplain_vegh\GEDI_MYR\L2_vegh\simu_2023\GEDI_link_RS_comb\\peak_{str(_)}_simulated_all.csv',
    #                        merge_keys = ['Quality Flag', 'Canopy Elevation (m)', 'index', 'EPSG_lat', 'Elevation (m)', 'Beam', 'Landsat water rate', 'Shot Number', 'Sensitivity', 'EPSG_lon', 'Tandem-X DEM', 'Degrade Flag', 'Date', 'RH 98', 'RH 25', 'Urban rate', 'Canopy Height (rh100)', 'Leaf off flag'])

    # for _ in range(2023, 2024):
    #     VHM_ = VHM(f'G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\simu_2023\\GEDI_link_RS_comb\\peak_{str(_)}_simulated_all.csv')
    #     VHM_.predict_VHM('G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\L2_vegh\\simu_2023\\Predict_Vegh\\',
    #                      'G:\A_GEDI_Floodplain_vegh\GEDI_MYR\L2_vegh\VHM_model\VHM_20250505_150818\VHM_res\\res_model\\XGB_mode3_area_average_dep4_minchild3_gamma1.5_subs0.7_lamda50_alpha10_KF1.json')

    # ds = gdal.Open('G:\\A_Climatology_dataset\\gridded_dataset\\MODIS_PAR_V6.2\\MODIS_Output\\ROI_map\\floodplain_2020.TIF')
    # arr = ds.GetRasterBand(1).ReadAsArray()
    # proj = ds.GetGeoTransform()
    # bf.create_folder('G:\A_GEDI_Floodplain_vegh\GEDI_MYR\L2_vegh\simu_2023\Predict_Vegh_tif')
    # for _ in range(2023, 2024):
    #     vegh_pd = pd.read_csv(f'G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\simu_2023\\Predict_Vegh\\peak_{str(_)}_simulated_all_predict.csv')
    #     arr_ = copy.deepcopy(arr)
    #     arr_ = arr_.astype(np.float32)
    #     arr_[:, :] = np.nan
    #     with tqdm(total=vegh_pd.shape[0], desc=f'Reconstruct \033[1;33m{str(_)}\033[0m vegh tif',  bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
    #         for __ in range(vegh_pd.shape[0]):
    #             x_cord, y_cord = int(np.floor((vegh_pd.loc[__]['EPSG_lon'] - proj[0]) / proj[1])), int(np.floor((vegh_pd.loc[__]['EPSG_lat'] - proj[3]) / proj[5]))
    #             arr_[y_cord, x_cord] = vegh_pd.loc[__]['Vegh']
    #             pbar.update()
    #     bf.write_raster(ds, arr_, 'G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\simu_2023\\Predict_Vegh_tif\\', f'peak_{str(_)}_vegh.TIF', raster_datatype=gdal.GDT_Float32, nodatavalue=np.nan)
    # #######################################################################################
    # # TRAIN VHM
    # VHM = VHM('XGB')
    # VHM.input_dataset('G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\floodplain_GEDI_2019_2023_for_model_high_quality.csv', mode_list=[0, 1, 2, 3, 4, 5, 6], cross_validation_factor=False, normalised_dataset=True)
    # VHM.train_VHM(learning_ratio=0.05, bulk_train=True, max_d=[4, 5, 6, 7, 8, 9, 10], child_weight=[0, 0.25, 0.5], gamma=[0], subsample=[0.7, 0.8, 0.85, 0.9, 0.95, 1.0], lamda=[0, 2, 5, 10, 15, 20, 25], alpha=[0, 2, 4, 6, 8, 10])
    # VHM.grid_search_best_para()

    # for year in [str(_) for _ in range(2018, 2024)]:
    #     dc_temp_dic.append(Denv_dc(f'G:\\A_Climatology_dataset\\gridded_dataset\\MODIS_PAR_V6.2\\MODIS_Output\\floodplain_2020_Denv_datacube\\{year}\\'))

# # # Link simulated gedi df with Phedc
#     for year in [str(_) for _ in range(2023, 2024)]:
#         dc_temp_dic = []
#         dc_temp_dic.append(Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{year}\\'))
#         rsdc_temp = RS_dcs(*dc_temp_dic)
#         rsdc_temp.link_GEDI_Phemedc(GEDI_df(f'G:\A_GEDI_Floodplain_vegh\GEDI_MYR\L2_vegh\simu_2023\\peak_{year}_simulated.csv'), ['DR', 'DR2', 'EOS', 'GR', 'SOS', 'peak_vi', 'peak_doy', 'trough_vi', 'MAVI'], spatial_interpolate_method=['nearest_neighbor'],)
#
#     # # Link simulated gedi df with RSdc
#     index_list = [f'{_}_noninun' for _ in ['OSAVI', 'GREEN', 'BLUE', 'RED', 'SWIR', 'SWIR2', 'NIR', 'SVVI', 'TCGREENESS']]
#     for index_ in index_list:
#         dc_temp_dic = []
#         dc_temp_dic.append(Landsat_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\{index_}_datacube\\'))
#         for year in [str(_) for _ in range(2023, 2024)]:
#             dc_temp_dic.append(Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{year}\\'))
#         rs_dc_temp = RS_dcs(*dc_temp_dic)
#         for year in [str(_) for _ in range(2023, 2024)]:
#             rs_dc_temp.link_GEDI_RS_dc(GEDI_df(f'G:\A_GEDI_Floodplain_vegh\GEDI_MYR\L2_vegh\simu_2023\\peak_{year}_simulated.csv'), {index_}, spatial_interpolate_method=['nearest_neighbor'])
#
#     # Link simulated gedi df with denvdc and par
#     for denv_ in ['AGB', 'TEM', 'RHU', 'PRE', 'PRS', 'RHU', 'WIN']:
#         for year in [str(_) for _ in range(2023, 2024)]:
#             dc_temp_dic = []
#             dc_temp_dic.append(Denv_dc(f'G:\\A_Climatology_dataset\\station_dataset\\CMA_dataset\\2400_all_station_1950_2023\\CMA_OUTPUT\\floodplain_2020_UTM_Denv_datacube\\{denv_}\\{year}\\'))
#             dc_temp_dic.append(Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{year}\\'))
#             rsdc_temp = RS_dcs(*dc_temp_dic)
#             rsdc_temp.link_GEDI_Denvdc(GEDI_df(f'G:\A_GEDI_Floodplain_vegh\GEDI_MYR\L2_vegh\simu_2023\\peak_{year}_simulated.csv'), [denv_], spatial_interpolate_method=['nearest_neighbor'])
#
#     for year in [str(_) for _ in range(2023,2024)]:
#         dc_temp_dic = []
#         dc_temp_dic.append(Denv_dc(f'G:\\A_Climatology_dataset\\gridded_dataset\\MODIS_PAR_V6.2\\MODIS_Output\\floodplain_2020_Denv_datacube\\{year}\\'))
#         dc_temp_dic.append(Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{year}\\'))
#         rsdc_temp = RS_dcs(*dc_temp_dic)
#         rsdc_temp.link_GEDI_Denvdc(GEDI_df(f'G:\A_GEDI_Floodplain_vegh\GEDI_MYR\L2_vegh\simu_2023\\peak_{year}_simulated.csv'), ['DPAR'], spatial_interpolate_method=['nearest_neighbor'])
#
#     # Link simulated gedi df with VegType
#     for year in [str(_) for _ in range(2023,2024)]:
#         rsdc_temp = RS_dcs(Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{year}\\'))
#         rsdc_temp.link_GEDI_VegType('G:\A_GEDI_Floodplain_vegh\Veg_map\Veg_classification\\veg_map_30m.tif', GEDI_df(f'G:\A_GEDI_Floodplain_vegh\GEDI_MYR\L2_vegh\simu_2023\\peak_{year}_simulated.csv'), spatial_interpolate_method=['nearest_neighbor'])
#
#
