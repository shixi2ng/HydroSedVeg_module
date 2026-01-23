from Landsat_toolbox.Landsat_main_v2 import *
from RSDatacube.RSdc import *
from River_GIS.River_GIS import *
from Crawler.crawler_weatherdata import Qweather_dataset

if __name__ == '__main__':

    # Generate HydroDC
    wl1 = HydroStationDS()
    wl1.from_std_csvs('G:\A_1Dflow_sed\Hydrodynamic_model\Original_water_level\standard_csv\\')

    thal1 = Thalweg()
    thal1.from_geojson('G:\A_Floodplain_topography\\Water_level_python\\Post_TGD\\output_geojson\\thelwag.json')
    thal1.from_smooth_Thalweg_shp('G:\A_Floodplain_topography\\Water_level_python\\Post_TGD\\output_shpfile\\thelwag_smooth2.shp')

    wl1.to_HydroDC(thal1, 'G:\A_Floodplain_topography\ROI_map\\floodplain_2020_map.TIF', [1988, 2020], 'G:\A_Floodplain_topography\HydroDC\\')
    pass