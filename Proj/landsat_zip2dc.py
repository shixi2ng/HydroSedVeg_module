from Landsat_toolbox.Landsat_main_v2 import *
from RSDatacube.RSdc import *
from River_GIS.River_GIS import *


if __name__ == '__main__':

    # landsat_temp = Landsat_l2_ds('D:\\Landsat_YZR_2023\\Ori_zipfile\\')
    # landsat_temp.construct_metadata(unzipped_para=False)
    # landsat_temp.mp_construct_index(['MNDWI', 'OSAVI'], cloud_removal_para=True, size_control_factor=True, ROI='G:\\A_Landsat_Floodplain_veg\\ROI_map\\floodplain_2020.shp', harmonising_data=True)
    # landsat_temp.mp_ds2landsatdc(['MNDWI', 'OSAVI'], inherit_from_logfile=True)
    # Landsat_WI_temp = Landsat_dc('G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\MNDWI_datacube\\')
    Landsat_VI_temp = Landsat_dc('G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\MNDWI_datacube\\')
    #
    # Landsat_dcs = RS_dcs(Landsat_WI_temp)
    # Landsat_dcs.inundation_detection('DT', 'MNDWI', 'Landsat', DT_std_fig_construction=False)

    Landsat_inun_temp = Landsat_dc('G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube')
    Landsat_dcs = RS_dcs(Landsat_inun_temp, Landsat_VI_temp)
    Landsat_dcs.inundation_removal('MNDWI', 'DT', 'Landsat', append_new_dc=False)

    # for _ in range(1987, 2024):
    #     Pheme2022 = Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{str(_)}\\')
    #     Pheme2022.dc2tif(['SOS', 'EOS', 'trough_vi', 'peak_vi', 'GR', 'DR', 'DR2', 'MAVI', 'peak_doy'], f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\Phemetric_tif\\{str(_)}\\')
        # Pheme2022.calculate_phemetrics(['SOS', 'EOS', 'trough_vi', 'peak_vi', 'GR', 'DR', 'DR2', 'MAVI', 'peak_doy'])

    # Landsat_VI_temp = Landsat_dc('G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_datacube\\')
    # Landsat_dcs = RS_dcs(Landsat_VI_temp)
    # Landsat_dcs.curve_fitting('OSAVI_noninun', 'Landsat')