from Landsat_toolbox.Landsat_main_v2 import *
from RSDatacube.RSdc import *
from River_GIS.River_GIS import *


if __name__ == '__main__':

    # Water level import
    # wl1 = HydroStationDS()
    # wl1.import_from_standard_files('G:\\A_1Dflow_sed\\Hydrodynamic_model\\Original_water_level\\', 'G:\\A_1Dflow_sed\\Hydrodynamic_model\\Original_water_level\\对应表.csv')

    # # Cross section construction POST-TGD
    # cs1 = CSprofile('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\')
    # cs1.from_stdCSfiles(
    #     'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Original_cross_section\\cross_section_DEM_2019_all.csv')
    # cs1.import_CS_coords(
    #     'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Original_cross_section\\cross_section_coordinates_wgs84.csv',
    #     epsg_crs='epsg:32649')
    # cs1.import_CS_tribu(
    #     'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Original_cross_section\\cross_section_tributary.xlsx')
    # thal1 = cs1.generate_Thalweg()

    # thal1 = Thalweg()
    # thal1 = thal1.load_geojson('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\output_geojson\\thelwag.json')
    # thal1.load_smooth_Thalweg_shp('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\output_shpfile\\thelwag_smooth.shp')
    # thal1.merged_hydro_inform(wl1)
    #
    # bath_method = Flood_freq_based_hyspometry_method([_ for _ in range(2004, 2021)])
    # bath_method.perform_in_epoch(thal1, 'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\inun_factor\\DT_inundation_frequency_posttgd.TIF')

    # # Cross-section construction POST-TGD
    cs1 = CSprofile()
    cs1.from_CSProf41DHM('G:\A_1Dflow_sed\Hydrodynamic_model\MYR_input\MYR_CSProf.csv')
    cs1 = CSprofile('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\')
    cs1.from_stdCSfiles('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Original_cross_section\\cross_section_DEM_2019_all.csv')
    cs1.import_CS_coords('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Original_cross_section\\cross_section_coordinates_wgs84.csv', epsg_crs='epsg:32649')
    cs1.import_CS_tribu('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Original_cross_section\\cross_section_tributary.xlsx')
    # cs1.merge_Hydrods(wl1, )
    # cs1.generate_CSProf41DHM('G:\\A_Landsat_Floodplain_veg\\Hydrodynamic_model\\para\\',ROI_name='MYR')
    # cs1.to_geojson()
    # cs1.to_shpfile()
    # cs1.to_csv()
    #
    # arr_dic = {}
    # for outputpath in ['prewl_predem','prewl_postdem','postwl_predem','postwl_postdem' ]:
    #     dir = 'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\inundation_status\\' + str(outputpath) + '\\inundation_freq.TIF'
    #     ds = gdal.Open(dir)
    #     arr_dic[outputpath] = ds.GetRasterBand(1).ReadAsArray()
    #
    # contribution_dam = ((arr_dic['prewl_postdem'] - arr_dic['prewl_predem'] + arr_dic['postwl_postdem'] - arr_dic['postwl_predem']) /
    #                     ((arr_dic['prewl_postdem'] - arr_dic['prewl_predem'] + arr_dic['postwl_postdem'] - arr_dic['postwl_predem']) + (arr_dic['postwl_postdem'] - arr_dic['prewl_postdem'] + arr_dic['postwl_predem'] - arr_dic['prewl_predem'])))
    # contribution_wl = ((arr_dic['postwl_postdem'] - arr_dic['prewl_postdem'] + arr_dic['postwl_predem'] - arr_dic['prewl_predem']) /
    #                     ((arr_dic['prewl_postdem'] - arr_dic['prewl_predem'] + arr_dic['postwl_postdem'] - arr_dic['postwl_predem']) + (arr_dic['postwl_postdem'] - arr_dic['prewl_postdem'] + arr_dic['postwl_predem'] - arr_dic['prewl_predem'])))
    # bf.write_raster(ds, contribution_dam, 'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\inundation_status\\', 'dam_contribution.TIF', raster_datatype=gdal.GDT_Float32)
    # bf.write_raster(ds, contribution_wl, 'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\inundation_status\\', 'wl_contribution.TIF', raster_datatype=gdal.GDT_Float32)
    #
    # for outputpath in ['prewl_predem','prewl_postdem','postwl_predem','postwl_postdem' ]:
    #     dir = 'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\inundation_status\\' + str(outputpath) + '\\'
    #     arr_all, file_len = None, 0
    #     if not os.path.exists(dir + 'inundation_freq.TIF'):
    #         for _ in bf.file_filter(dir + 'inundation_final\\', ['.tif']):
    #             ds = gdal.Open(_)
    #             arr = ds.GetRasterBand(1).ReadAsArray()
    #             file_len += 1
    #             arr[arr == 255] = 0
    #             if arr_all is None:
    #                 arr_all = arr.astype(np.uint32)
    #             else:
    #                 arr_all += arr.astype(np.uint32)
    #         bf.write_raster(ds, arr_all/file_len, dir, 'inundation_freq.TIF', raster_datatype=gdal.GDT_Float32)
    #
    # if not os.path.exists('G:\A_Landsat_Floodplain_veg\Water_level_python\Post_TGD\\ele_posttgd4model.TIF') or not os.path.exists('G:\A_Landsat_Floodplain_veg\Water_level_python\Post_TGD\\ele_pretgd4model.TIF'):
    #     pre_ele_ds = gdal.Open('G:\A_Landsat_Floodplain_veg\Water_level_python\Post_TGD\\ele_DT_inundation_frequency_pretgd.TIF')
    #     pre_inun_ds = gdal.Open('G:\A_Landsat_Floodplain_veg\Water_level_python\Post_TGD\\inun_DT_inundation_frequency_pretgd.TIF')
    #     post_ele_ds = gdal.Open('G:\A_Landsat_Floodplain_veg\Water_level_python\Post_TGD\\ele_DT_inundation_frequency_posttgd.TIF')
    #     post_inun_ds = gdal.Open('G:\A_Landsat_Floodplain_veg\Water_level_python\Post_TGD\\inun_DT_inundation_frequency_posttgd.TIF')
    #
    #     pre_ele_arr = pre_ele_ds.GetRasterBand(1).ReadAsArray()
    #     pre_inun_arr = pre_inun_ds.GetRasterBand(1).ReadAsArray()
    #     post_ele_arr = post_ele_ds.GetRasterBand(1).ReadAsArray()
    #     post_inun_arr = post_inun_ds.GetRasterBand(1).ReadAsArray()
    #
    #     pre_ele_arr[pre_inun_arr == 1] = -1000
    #     post_ele_arr[post_inun_arr == 1] = -1000
    #
    #     bf.write_raster(pre_ele_ds, pre_ele_arr, 'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\', 'ele_pretgd4model.TIF')
    #     bf.write_raster(post_ele_ds, post_ele_arr, 'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\', 'ele_posttgd4model.TIF')
    #
    thal1 = cs1.to_Thalweg()
    #
    # # Import hydrodatacube
    # # with concurrent.futures.ProcessPoolExecutor() as exe:
    # #     exe.map(multiple_concept_model([_ for _ in range(1987, 2004)], thal1))
    for year in range(1987, 2004):
        hydrodc1 = HydroDC()
        hydrodc1.from_hydromatrix(f'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\hydrodatacube\\{str(year)}\\')
        hydrodc1.seq_simplified_conceptual_inundation_model('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\ele_pretgd4model.TIF', thal1, 'G:\A_Landsat_Floodplain_veg\Water_level_python\inundation_status\\prewl_predem\\')

    for year in range(1987, 2004):
        hydrodc1 = HydroDC()
        hydrodc1.from_hydromatrix(f'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\hydrodatacube\\{str(year)}\\')
        hydrodc1.seq_simplified_conceptual_inundation_model('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\ele_posttgd4model.TIF', thal1, 'G:\A_Landsat_Floodplain_veg\Water_level_python\inundation_status\\prewl_postdem\\')

    for year in range(2004, 2021):
        hydrodc1 = HydroDC()
        hydrodc1.from_hydromatrix(f'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\hydrodatacube\\{str(year)}\\')
        hydrodc1.seq_simplified_conceptual_inundation_model('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\ele_pretgd4model.TIF', thal1, 'G:\A_Landsat_Floodplain_veg\Water_level_python\inundation_status\\postwl_predem\\')

    for year in range(2004, 2021):
        hydrodc1 = HydroDC()
        hydrodc1.from_hydromatrix(f'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\hydrodatacube\\{str(year)}\\')
        hydrodc1.seq_simplified_conceptual_inundation_model('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\ele_posttgd4model.TIF', thal1, 'G:\A_Landsat_Floodplain_veg\Water_level_python\inundation_status\\postwl_postdem\\')
    #
    # # Cross section construction POST-TGD
    # cs1 = CSprofile('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\')
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
    #
    # thal1 = cs1.generate_Thalweg()
    # thal1.to_shapefile()
    # thal1.to_geojson()
    # thal1 = Thalweg()
    # thal1 = thal1.load_geojson('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\output_geojson\\thelwag.json')
    # thal1.load_smooth_Thalweg_shp('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\output_shpfile\\thelwag_smooth.shp')
    # # # #
    # thal1.merge_Hydrods(wl1)
    # thal1.perform_in_epoch(
    #     'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\inun_factor\\DT_inundation_frequency_posttgd.TIF',
    #     year_range=[2004, 2021])
    #
    # cs1.compare_2ddem('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\ele_DT_inundation_frequency_posttgd.TIF', inun_freq='G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\inun_factor\\DT_inundation_frequency_posttgd.TIF')
    #
    # # Cross section construction pre-TGD
    # cs2 = CSprofile('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Pre_TGD\\')
    # cs2.from_stdCSfiles('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Original_cross_section\\cross_section_DEM_2003.xlsx')
    # cs2.import_CS_coords(
    #     'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Original_cross_section\\cross_section_coordinates_wgs84.csv',
    #     epsg_crs='epsg:32649')
    # cs2.import_CS_tribu(
    #     'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Original_cross_section\\cross_section_tributary.xlsx')
    # cs2.to_geojson()
    # cs2.to_shpfile()
    # cs2.to_csv()
    # cs2.compare_2ddem('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Pre_TGD\\ele_DT_inundation_frequency_pretgd.TIF', inun_freq='G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\inun_factor\\DT_inundation_frequency_pretgd.TIF')