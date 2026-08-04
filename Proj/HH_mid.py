from RSDatacube.RSdc import *
from Sentinel2_toolbox.Sentinel_Download import Queried_Sentinel_ds_ODATA
from Sentinel2_toolbox.Sentinel_main_V2 import Sentinel2_ds, Sentinel2_dc


if __name__ == '__main__':

    ## Download Sentinel-2 data with IDM
    # DownPath = 'D:\A_MYR_Sentinel2\\aa'
    # shpfile_path = 'E:\A_Veg_phase2\Sample_Inundation\Floodplain\\Floodplain_2020_simplified3.shp'
    #
    # aacc = [('shixi2ng@gmail.com', 'shi_xi_2nG2nG'), ('sx1998@whu.edu.cn', 'shi_xi_xi_2nG'), ('422424829@qq.com', 'Vera_lu_1997')]
    # S2_MID_YZR = Queried_Sentinel_ds_ODATA('sx1998@outlook.com', 'shi_shi_xi_2nG@', DownPath, additional_account=aacc)
    # S2_MID_YZR.queried_with_ROI(shpfile_path, ('20241101', '20241231'), 'SENTINEL-2', 'S2MSI2A',  (0, 100), overwritten_factor=False)
    # S2_MID_YZR.download_with_request()
    # #
    filepath = 'D:\\A_MYR_Sentinel2\\Original_zipfile\\'
    s2_ds_temp = Sentinel2_ds(filepath)
    s2_ds_temp.construct_metadata()
    s2_ds_temp.sequenced_subset(['all_band', 'OSAVI_20m', 'NDVI_20m', 'MNDWI'], ROI='G:\A_Landsat_Floodplain_veg\ROI_map\\floodplain_2020.shp',
                         ROI_name='MYR', cloud_removal_strategy='QI_all_cloud', size_control_factor=True, combine_band_factor=False)
    # # s2_ds_temp.seq_ds2sdc(['OSAVI_20m', 'NDVI_20m', 'MNDWI'], inherit_from_logfile=True, remove_nan_layer=True)
    #
    # s2_ds_temp.mp_mosaic2seqtif([ 'OSAVI_20m', 'NDVI_20m', 'MNDWI'], inherit_from_logfile=True)
    #
    # s2_ds_temp.seq_ds2sdc(['OSAVI_20m', 'NDVI_20m', 'MNDWI'], inherit_from_logfile=True, remove_nan_layer=True)
    # s2_ds_temp._sdc_consistency_check(['OSAVI_20m', 'NDVI_20m', 'MNDWI'], inherit_from_logfile=True)

    sentinel_MNDWVI_temp = Sentinel2_dc('D:\A_MYR_Sentinel2\Sentinel2_L2A_Output\Sentinel2_MYR_datacube\\MNDWI_sequenced_datacube\\')
    sentinel_dcs = RS_dcs(sentinel_MNDWVI_temp)
    sentinel_dcs.inundation_detection('static_wi_thr', 'MNDWI', 'Sentinel2', static_wi_threshold=0.09)