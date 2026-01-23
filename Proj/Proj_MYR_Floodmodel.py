from Landsat_toolbox.Landsat_main_v2 import *
from RSDatacube.RSdc import *
from River_GIS.River_GIS import *


if __name__ == '__main__':

    # # Water level import
    # wl1 = HydroStationDS()
    # wl1.import_from_standard_files('G:\\A_1Dflow_sed\\Hydrodynamic_model\\Original_water_level\\', 'G:\\A_1Dflow_sed\\Hydrodynamic_model\\Original_water_level\\对应表.csv')
    # wl1.annual_runoff_sediment('宜昌', 'G:\A_PhD_Main_paper\Chap.2\Figure\Fig.2.3\\')
    # wl1.annual_runoff_sediment('枝城', 'G:\A_PhD_Main_paper\Chap.2\Figure\Fig.2.3\\')
    # wl1.annual_runoff_sediment('螺山', 'G:\A_PhD_Main_paper\Chap.2\Figure\Fig.2.3\\')
    # wl1.annual_runoff_sediment('汉口', 'G:\A_PhD_Main_paper\Chap.2\Figure\Fig.2.3\\')
    # wl1.to_csvs()
    # wl1.to_FlwBound41DHM('G:\\A_1Dflow_sed\\Hydrodynamic_model\\para\\MYR_FlwBound.csv', [20190101, 20191231], '宜昌', '九江', 'Z-T')
    #
    # # Process cross-sectional profile
    # cs = CSprofile()
    # cs.from_stdCSfiles('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Original_cross_section\\cross_section_csv\\cross_section_DEM_2019_all.csv')
    # cs.merge_Hydrods(wl1)
    # cs.to_CSProf41DHM('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Original_cross_section\\cross_section_csv\\')

    # Water level import
    wl1 = HydroStationDS()
    wl1.from_std_csvs('G:\A_1Dflow_sed\Hydrodynamic_model\Original_water_level\standard_csv')

    thal1 = Thalweg()
    thal1 = thal1.from_geojson('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\output_geojson\\thelwag.json')
    thal1.load_smooth_Thalweg_shp('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\output_shpfile\\thelwag_smooth.shp')

    # hc = HydroDC()
    # hc.merge_hydro_inform(wl1)
    # hc.hydrodc_csv2matrix('G:\\A_Landsat_veg\\Water_level_python\\hydrodatacube\\', 'G:\\A_Landsat_veg\\Water_level_python\\hydrodatacube\\hydro_dc_X_16357_Y_4827_posttgd.csv')
    # hc.hydrodc_csv2matrix('G:\\A_Landsat_veg\\Water_level_python\\hydrodatacube\\', 'G:\\A_Landsat_veg\\Water_level_python\\hydrodatacube\\hydro_dc_X_16357_Y_4827_pretgd.csv')
