from Landsat_toolbox.Landsat_main_v2 import *
from RSDatacube.RSdc import *
from River_GIS.River_GIS import *


if __name__ == '__main__':

    # Water level import
    wl1 = HydroStationDS()
    file_list = bf.file_filter('D:\\Hydrodatacube\\water_level\\', ['.xls'])
    corr_temp = pd.read_csv('D:\\Hydrodatacube\\water_level\\对应表.csv')
    cs_list, wl_list = [], []
    for file_ in file_list:
        for hs_num in range(corr_temp.shape[0]):
            hs = corr_temp[corr_temp.keys()[1]][hs_num]
            if hs in file_:
                cs_list.append(corr_temp[corr_temp.keys()[0]][hs_num])
                wl_list.append(corr_temp[corr_temp.keys()[2]][hs_num])

    for fn_, cs_, wl_ in zip(file_list, cs_list, wl_list):
        wl1.import_from_standard_excel(fn_, cs_, water_level_offset=wl_)
    wl1.to_csvs()

    hc = HydroDC()
    hc.merge_hydro_inform(wl1)
    hc.hydrodc_csv2matrix('D:\\Hydrodatacube\\',  'D:\\Hydrodatacube\\hydro_dc_X_16357_Y_4827_pretgd.csv')