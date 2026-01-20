import numpy as np
from Aborted_codes import Landsat_main_v1
from osgeo import gdal
import scipy.stats as stats
import pandas as pd


if '__name__' == '__main__':
    a = pd.read_excel('E:\A_Vegetation_Identification\\ks.xlsx')
    stats.ks_2samp(a[2002][31:], a[2016][31:])
    print(stats.ks_2samp(a['2002_28'], a['2016_28']))

    # starting time
    # now = datetime.datetime.now()
    # print( now.strftime('%Y-%m-%d %H:%M:%S')  )
    #
    #
    # folder_gsw = 'G:\Landsat\Inundation\GSW\\'
    # folder_roi = 'G:\Landsat\Jingjiang_shp\shpfile\Intersect\\'
    # gdal.Warp('G:\Landsat\Inundation\\20130708.TIF', folder_gsw+'water_2013_07.tif', cutlineDSName=folder_roi+'dongcaozhou.shp',
    #                                           cropToCutline=True, dstSRS='EPSG:32650', xRes=30, yRes=30)
    # gdal.Warp('G:\Landsat\Inundation\\20190802.TIF', folder_gsw+'water_2019_08.tif',cutlineDSName=folder_roi+'dongcaozhou.shp',
    #                                           cropToCutline=True, dstSRS='EPSG:32650', xRes=30, yRes=30)
    # gdal.Warp('G:\Landsat\Inundation\\20121001.TIF', folder_gsw+'water_2012_09.tif',cutlineDSName=folder_roi+'guniuzhou.shp',
    #                                           cropToCutline=True, dstSRS='EPSG:32650', xRes=30, yRes=30)
    # gdal.Warp('G:\Landsat\Inundation\\20170828.TIF', folder_gsw+'water_2017_08.tif',cutlineDSName=folder_roi+'shanjiazhou.shp',
    #                                           cropToCutline=True, dstSRS='EPSG:32650', xRes=30, yRes=30)
    # gdal.Warp('G:\Landsat\Inundation\\20200816_water.TIF', 'E:\A_Vegetation_Identification\Wuhan_Landsat_Original\Sample_123039\Google_Earth_Sample\\nmz\output\\20200816_water.TIF',cutlineDSName= 'E:\A_Vegetation_Identification\Wuhan_Landsat_Original\Sample_123039\study_area_shapefile\\nmz.shp',
    #                                           cropToCutline=True, dstSRS='EPSG:32649', xRes=30, yRes=30)
    # gdal.Warp('G:\Landsat\Inundation\\20200816_all.TIF', 'E:\A_Vegetation_Identification\Wuhan_Landsat_Original\Sample_123039\Google_Earth_Sample\\nmz\output\\20200816_all.TIF',cutlineDSName= 'E:\A_Vegetation_Identification\Wuhan_Landsat_Original\Sample_123039\study_area_shapefile\\nmz.shp',
    #                                           cropToCutline=True, dstSRS='EPSG:32649', xRes=30, yRes=30)


    gdal.Warp('G:\Landsat\Inundation\GSW\\water_2020_082.tif', 'G:\Landsat\Inundation\GSW\\water_2020_08.tif', cutlineDSName='G:\Landsat\Inundation\shp\\nmz.shp', cropToCutline=True, dstSRS='EPSG:32649', xRes=30, yRes=30)

    sample_2013_ds = gdal.Open('G:\Landsat\Inundation\dongcaozhou\output\\20130710.TIF')
    sample_2019_ds = gdal.Open('G:\Landsat\Inundation\dongcaozhou\output\\20190804.TIF')
    sample_2012_ds = gdal.Open('G:\Landsat\Inundation\guniuzhou\output\\20120915.TIF')
    sample_2017_ds = gdal.Open('G:\Landsat\Inundation\shanjiazhou\output\\20170828.TIF')
    sample_2020_ds = gdal.Open('G:\Landsat\Inundation\dongcaozhou\output\\20190804.TIF')

    gsw_2013_ds = gdal.Open('G:\Landsat\Inundation\\20130708.TIF')
    gsw_2019_ds = gdal.Open('G:\Landsat\Inundation\\20190802.TIF')
    gsw_2012_ds = gdal.Open('G:\Landsat\Inundation\\20121001.TIF')
    gsw_2017_ds = gdal.Open('G:\Landsat\Inundation\\20170828.TIF')
    gsw_2020_ds = gdal.Open('G:\Landsat\Inundation\\20190802.TIF')

    sample_water = gdal.Open('G:\Landsat\Inundation\\20200816_water1.TIF')
    sample_all = gdal.Open('G:\Landsat\Inundation\\20200816_all1.TIF')
    sample_all_raster = sample_all.GetRasterBand(1).ReadAsArray()
    sample_water_raster = sample_water.GetRasterBand(1).ReadAsArray()

    sample_all_raster[sample_all_raster == 1] = -2
    sample_all_raster[np.logical_and(sample_water_raster == 0, sample_all_raster != -2)] = 1

    landsat_temp_ds = gdal.Open('G:\Landsat\Inundation\\water_2020_081.TIF')
    landsat_temp_raster = landsat_temp_ds.GetRasterBand(1).ReadAsArray()
    landsat_temp_raster [landsat_temp_raster == 1] = 0
    landsat_temp_raster [landsat_temp_raster == 2] = 1
    confusion_matrix_temp = Landsat_main_v1.confusion_matrix_2_raster(landsat_temp_raster, sample_all_raster, nan_value=-2)

    np.seterr(divide='ignore', invalid='ignore')

    # define var
    root_path = 'H:\\RS\\YJ_all\\'
    original_file_path = root_path + 'Zip_YJ\\'
    corrupted_file_path = root_path + 'Corrupted\\'
    unzipped_file_path = root_path + 'Unzipped_Landsat_YJ_TIFF\\'
    Landsat_main_v1.create_folder(unzipped_file_path)
    study_area_shp = root_path + 'SHP_YJ\\domain12_100km.shp'
    # threshold_temp = [a,b,c,d],Dynamic Surface Water Extent (DSWE)
    # MNDWI >0.123(a)
    # or MNDWI <-0.5(b) and rou_NIR <0.2(c) and rou_MIR <0.1(d)
    # [Q] different months maybe need various threshold_temp.
    threshold_temp = [0.123, -0.5, 0.2, 0.1]
    #  time span, coverage = 'month', 'week', or 'year'
    #  'week', 7-day groups from the first day of 1 Jan.
    #  'month', 30.5 days refer to a month
    coverage = 'monsoon'
    monsoon_begin_month = 6
    monsoon_end_month = 10
    inundated_value = 1
    non_inundated_value = 0
    # composition_strat used to determine the composition strategy,
    # all supported strategy include 'first' 'last' and 'mean'
    # while 'first' indicate the output data is first scene within the time coverage and 'last' conversely
    # 'mean' is the mean value of all the non-nan scene
    composition_strat = 'first'
    study_area_name = 'YJ'
    main_coordinate = 'EPSG:32646'

    wet_threshold = [0.15, 0.95]
    # process data
    # all_supported_vi_list = ['NDVI', 'OSAVI', 'MNDWI', 'EVI', 'FVC']
    # 'NDVI', Normalized Difference Vegetation Index
    # 'OSAVI', Optimizing soil adjusted vegetation index
    # 'MNDWI'，INUNDATION DETECTION, Modified Normalized Difference Water Index
    # 'EVI', Enhanced Vegetation Index
    # 'FVC', Fractional Vegetation cover


    Landsat_main_v1.generate_dry_wet_ratio('H:\\RS\\YJ_all\\Landsat_Inundation_Condition\\YJ_global\\individual_tif\\', 1, 0)
    # when Zip file needs unzipped, unzipped_para =True. Otherwise if Zip file had been unzipped, unzipped_para =False
    file_metadata = Landsat_main_v1.generate_landsat_metadata(original_file_path, unzipped_file_path, corrupted_file_path,
                                                              root_path, unzipped_para=False)

    Landsat_main_v1.generate_landsat_vi(root_path, unzipped_file_path, file_metadata, vi_construction_para=True,
                                        construction_overwritten_para=False, cloud_removal_para=True, vi_clipped_para=True,
                                        clipped_overwritten_para=False, construct_dc_para=True, dc_overwritten_para=False,
                                        construct_sdc_para=True, sdc_overwritten_para=False, VI_list=['MNDWI'],
                                        ROI_mask_f=study_area_shp, study_area=study_area_name, main_coordinate_system=main_coordinate)
    Landsat_main_v1.landsat_inundation_detection(root_path, VI_list_f=['MNDWI'], study_area=study_area_name,
                                                 file_metadata_f=file_metadata, unzipped_file_path_f=unzipped_file_path,
                                                 ROI_mask_f=study_area_shp, global_local_factor='global', global_threshold=threshold_temp, main_coordinate_system=main_coordinate)
    Landsat_main_v1.data_composition(root_path + 'Landsat_Inundation_Condition\\' + study_area_name + '_global\\individual_tif\\', wet_threshold, root_path + 'Metadata.xlsx', time_coverage=coverage, composition_strategy=composition_strat, file_format='.TIF', nan_value=-32768, user_defined_monsoon=[monsoon_begin_month, monsoon_end_month], inundated_indicator=[inundated_value, non_inundated_value])


    # ending time
    # import datetime
    # now = datetime.datetime.now()
    # print( now.strftime('%Y-%m-%d %H:%M:%S')  )