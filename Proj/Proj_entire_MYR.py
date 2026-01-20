from Landsat_toolbox.Landsat_main_v2 import *
from RSDatacube.RSdc import *
from River_GIS.River_GIS import *
from Crawler.crawler_weatherdata import Qweather_dataset

if __name__ == '__main__':

    # # Water level import
    # wl1 = HydroStationDS()
    # wl1.import_from_standard_files('G:\\A_1Dflow_sed\\Hydrodynamic_model\\Original_water_level\\', 'G:\\A_1Dflow_sed\\Hydrodynamic_model\\Original_water_level\\对应表.csv')
    # wl1.to_csvs()
    #
    # # Process the climate date
    # QW_ds = Qweather_dataset('G:\\A_Climatology_dataset\\station_dataset\\Qweather_dataset\\')
    # QW_ds.to_standard_cma_file()
    #
    # landsat_temp = Landsat_l2_ds('G:\\A_Landsat_Floodplain_veg\\Landsat_YZR_2023\\Ori_zipfile\\')
    # landsat_temp.construct_metadata(unzipped_para=False)
    # landsat_temp.mp_construct_index(['SVVI', 'TCGREENESS'], cloud_removal_para=True, size_control_factor=True, ROI='G:\\A_Landsat_Floodplain_veg\\ROI_map\\floodplain_2020.shp', harmonising_data=True, )
    # landsat_temp.mp_ds2landsatdc(['SVVI'], inherit_from_logfile=True)

    # file_name = 'G:\\A_Landsat_veg\\ROI_map\\floodplain_2020_map.TIF'
    # a = retrieve_correct_filename(file_name)
    #
    # # Inundation dcs
    # args_inundation = [Inunfac_dc(f'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Inundation_indicator\\inundation_dc\\{str(_)}\\') for _ in range(1988, 2021)]
    # args_pheme = [Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{str(_)}\\') for _ in range(1988, 2022)]
    # args_inundation.extend(args_pheme)
    # all_dcs = RS_dcs(*args_inundation)
    # all_dcs.pheme_inun_analysis(['peak_vi'], ['inun_duration'], [1988,2020])

    # pre_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\inun_DT_inundation_frequency_pretgd.TIF')
    # pre_arr = pre_ds.GetRasterBand(1).ReadAsArray()
    # post_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\inun_DT_inundation_frequency_posttgd.TIF')
    # post_arr = post_ds.GetRasterBand(1).ReadAsArray()
    # for sec, coord in zip(['all', 'yz', 'jj', 'ch', 'hh'], [[0, 16537], [0, 5000], [950, 6100], [6100, 10210], [10210, 16537]]):
    #     year_list, inund_list, inunh_list = [], [], []
    #     for year in range(1988, 2021):
    #         inund_ds = gdal.Open(
    #             f'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Inundation_indicator\\inundation_factor\\{str(year)}\\inun_duration.tif')
    #         inunh_ds = gdal.Open(
    #             f'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Inundation_indicator\\inundation_factor\\{str(year)}\\inun_mean_wl.tif')
    #         inund_arr = inund_ds.GetRasterBand(1).ReadAsArray()
    #         inunh_arr = inunh_ds.GetRasterBand(1).ReadAsArray()
    #
    #         if year < 2004:
    #             roi = copy.deepcopy(pre_arr)
    #             roi[roi > 0.4] = np.nan
    #         elif year >= 2004:
    #             roi = copy.deepcopy(post_arr)
    #             roi[roi > 0.4] = np.nan
    #
    #         inunh_arr[np.isnan(roi)] = np.nan
    #         inund_arr[np.isnan(roi)] = np.nan
    #         year_list.append(year)
    #         inund_list.append(np.nanmean(inund_arr[:, coord[0]: coord[1]]))
    #         inunh_list.append(np.nanmean(inunh_arr[:, coord[0]: coord[1]]))
    #     dic_temp = {'year': year_list, 'inun_h': inunh_list, 'inun_d': inund_list}
    #     pd_temp = pd.DataFrame(dic_temp)
    #     pd_temp.to_csv(f'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig10\\v2\\MAVI_var\\inun\\flood_indi_{str(sec)}.csv')

    # veg_pre_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11\\veg_pre_tgd.TIF')
    # veg_post_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11\\veg_post_tgd.TIF')
    # roi_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\ROI_map\\floodplain_2020_map.TIF')
    # veg_pre_arr = veg_pre_ds.GetRasterBand(1).ReadAsArray()
    # veg_post_arr = veg_post_ds.GetRasterBand(1).ReadAsArray()
    # roi_arr = roi_ds.GetRasterBand(1).ReadAsArray()
    # #
    #
    # veg_pre_arr[np.isnan(veg_pre_arr)] = 0
    # veg_post_arr[np.isnan(veg_post_arr)] = 0
    # veg_diff_arr = veg_post_arr - veg_pre_arr
    # veg_diff_arr[roi_arr == -32768] = np.nan
    # veg_diff_arr[veg_diff_arr == 0] = -200
    # bf.write_raster(veg_pre_ds, veg_diff_arr, 'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig10\\peak_vi_var\\', 'veg_diff.TIF')

    # pre_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\ele_DT_inundation_frequency_pretgd.TIF')
    # post_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\ele_DT_inundation_frequency_posttgd.TIF')
    # pre_arr = pre_ds.GetRasterBand(1).ReadAsArray()
    # post_arr = post_ds.GetRasterBand(1).ReadAsArray()
    # mid_arr = (pre_arr + post_arr) / 2
    # bf.write_raster(pre_ds, mid_arr, 'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\', 'ele_mid.TIF')

    # refine dem
    thal1 = Thalweg()
    thal1 = thal1.load_geojson('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\output_geojson\\thelwag.json')
    # thal1.load_smooth_Thalweg_shp('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\output_shpfile\\thelwag_smooth2.shp')
    # for itr_ in np.linspace(10, 60, 11):
    #     thal1.straighten_river_through_thalweg('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11\\veg_diff.TIF', itr=int(itr_))

    # for cs, date in zip(['界Z3+3', 'CZ63', 'CZ63', 'CZ89', 'CZ118', ], [20200816, 20130710, 20190804, 20120915, 20170828]):
    #     wl1.cs_wl(thal1, cs, date)
    #
    # for cs, date in zip(['界Z3+3', 'CZ63', 'CZ63', 'CZ89', 'CZ118', ], [20200811, 20130708, 20190804, 20121001, 20170828]):
    #     wl1.cs_wl(thal1, cs, date)
    #
    # for station in ['螺山', '石首(二)', '调玄口', '监利', '广兴洲', '莲花塘', '汉口', '枝城', '陈家湾', '沙市', '郝穴', '新厂(二)']:
    #     wl1.linear_comparison(station, thal1, [2015,2016,2017,2018,2019,2020])

    # hc = HydroDatacube()
    # hc.merge_hydro_inform(wl1)
    # hc.from_hydromatrix('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\hydrodatacube\\2003\\')
    # hc.hydrodc_csv2matrix('D:\\Hydrodatacube\\',
    #                       'D:\\Hydrodatacube\\hydro_dc_X_16357_Y_4827_posttgd.csv')
    # hc.hydrodc_csv2matrix('D:\\Hydrodatacube\\',
    #                       'D:\\Hydrodatacube\\hydro_dc_X_16357_Y_4827_pretgd.csv')

    # for _ in range(1986, 2023):
    #     pheme_dc = Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{str(_)}')
    #     # pheme_dc.remove_layer(['MAVI', 'peak_vi', 'TSVI'])
    #     pheme_dc.calculate_phemetrics(['TSVI', 'peak_vi', 'MAVI'])
    #     pheme_dc.dc2tif()

    for year in range(1995, 2024):
        year = int(year)
        hc = HydroDatacube()
        hc.from_hydromatrix(f'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\hydrodatacube\\{str(year)}\\')
        hc.simplified_conceptual_inundation_model('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\ele_mid.TIF', thal1, f'G:\A_Landsat_Floodplain_veg\Water_level_python\predamwl_middamdem\\', meta_dic='G:\A_Landsat_Floodplain_veg\Landsat_floodplain_2020_datacube\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\1986\\metadata.json', inun_factor=True)

    # for year in range(1988, 2004):
    #     year = int(year)
    #     hc = HydroDatacube()
    #     hc.from_hydromatrix(f'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\hydrodatacube\\{str(year)}\\')
    #     hc.simplified_conceptual_inundation_model('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\ele_DT_inundation_frequency_pretgd.TIF', thal1, f'G:\A_Landsat_Floodplain_veg\Water_level_python\Inundation_indicator\\', meta_dic='G:\A_Landsat_Floodplain_veg\Landsat_floodplain_2020_datacube\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\1986\\metadata.json', inun_factor=False)
    #
    # for year in range(2004, 2024):
    #     year = int(year)
    #     hc = HydroDatacube()
    #     hc.from_hydromatrix(f'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\hydrodatacube\\{str(year)}\\')
    #     hc.simplified_conceptual_inundation_model('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\ele_DT_inundation_frequency_posttgd.TIF', thal1, f'G:\A_Landsat_Floodplain_veg\Water_level_python\Inundation_indicator\\', meta_dic='G:\A_Landsat_Floodplain_veg\Landsat_floodplain_2020_datacube\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\1986\\metadata.json', inun_factor=False)

    # Landsat_inun_temp = Landsat_dc('G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube')
    # hyspo = Flood_freq_based_hyspometry_method([2003], work_env='G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Annual_refined_dem\\')
    # hyspo.refine_annual_topography(thal1, Landsat_inun_temp, hc, elevation_map='G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\ele_DT_inundation_frequency_posttgd.TIF')
    #
    # #
    # # # Cross section construction POST-TGD
    # cs1 = CrossSection('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\')
    # cs1.from_stdCSfiles(
    #     'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Original_cross_section\\cross_section_DEM_2019_all.csv')
    # cs1.import_CS_coords(
    #     'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Original_cross_section\\cross_section_coordinates_wgs84.csv',
    #     epsg_crs='epsg:32649')
    # cs1.import_CS_tribu(
    #     'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Original_cross_section\\cross_section_tributary.xlsx')
    # cs1.to_geojson()
    # cs1.to_shpfile()
    # cs1.to_csv()
    # # cs1.compare_2ddem('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\ele_DT_inundation_frequency_posttgd.TIF')
    #
    # # Cross section import
    # thal1 = cs1.generate_Thalweg()
    # thal1.to_shapefile()
    # thal1.to_geojson()
    # thal1 = Thalweg()
    # thal1 = thal1.load_geojson('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\output_geojson\\thelwag.json')
    # thal1.load_smooth_Thalweg_shp('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\output_shpfile\\thelwag_smooth.shp')
    # # # #
    # thal1.merged_hydro_inform(wl1)
    # thal1.perform_in_epoch(
    #     'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\inun_factor\\DT_inundation_frequency_posttgd.TIF',
    #     year_range=[2004, 2021])
    # thal1.perform_in_epoch(
    #     'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\inun_factor\\DT_inundation_frequency_pretgd.TIF',
    #     year_range=[1987, 2004])
    #
    # # Cross section construction pre-TGD
    # cs2 = CrossSection('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Pre_TGD\\')
    # cs2.from_standard_cross_profiles('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Original_cross_section\\cross_section_DEM_2003.xlsx')
    # cs2.import_section_coordinates(
    #     'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Original_cross_section\\cross_section_coordinates_wgs84.csv',
    #     epsg_crs='epsg:32649')
    # cs2.import_section_tributary(
    #     'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Original_cross_section\\cross_section_tributary.xlsx')
    # cs2.to_geojson()
    # cs2.to_shpfile()
    # cs2.to_csv()
    # cs2.compare_2ddem('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Pre_TGD\\ele_DT_inundation_frequency_pretgd.TIF')
    #
    # # Differential the cross section
    # cs2.generate_differential_cross_profile(cs1)
    # if not os.path.exists('G:\\A_Landsat_Floodplain_veg\\Water_level_python\Diff_TGD\\ele_diff.TIF'):
    #     ds1 = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\ele_DT_inundation_frequency_posttgd.TIF')
    #     ds2 = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Pre_TGD\\ele_DT_inundation_frequency_pretgd.TIF')
    #     arr1 = ds1.GetRasterBand(1).ReadAsArray()
    #     arr2 = ds2.GetRasterBand(1).ReadAsArray()
    #     bf.write_raster(ds1, arr1-arr2, 'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Diff_TGD\\', 'ele_diff.TIF')
    # cs2.compare_2ddem('G:\\A_Landsat_Floodplain_veg\\Water_level_python\Diff_TGD\\ele_diff.TIF', diff_ele=True)
    #
    # # Compare inundation freq
    # cs1.merged_hydro_inform(wl1)
    # cs1.compare_inundation_frequency(
    #     'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\inun_factor\\DT_inundation_frequency_posttgd.TIF',
    #     year_range=[2004, 2021])
    #
    # # cs2.compare_2ddem('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Pre_TGD\\ele_DT_inundation_frequency_pretgd.TIF')
    # cs2.merged_hydro_inform(wl1)
    # cs2.compare_inundation_frequency(
    #     'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\inun_factor\\DT_inundation_frequency_pretgd.TIF',
    #     year_range=[1987, 2004])
    #
    # # Cross section import
    # thal1 = cs1.generate_Thalweg()
    # thal1.to_shapefile()
    # thal1.to_geojson()
    # thal1 = Thalweg()
    # thal1 = thal1.load_geojson('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\output_geojson\\thelwag.json')
    # thal1.load_smooth_Thalweg_shp('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\output_shpfile\\thelwag_smooth.shp')
    # # #
    # thal1.merged_hydro_inform(wl1)
    # thal1.perform_in_epoch(
    #     'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\inun_factor\\DT_inundation_frequency_posttgd.TIF',
    #     year_range=[2004, 2021])
    # thal1.perform_in_epoch(
    #     'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\inun_factor\\DT_inundation_frequency_pretgd.TIF',
    #     year_range=[1987, 2004])
    #
