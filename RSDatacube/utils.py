import basic_function
from NDsm import NDSparseMatrix
import scipy.sparse as sm
from tqdm.auto import tqdm
from osgeo import gdal, ogr, osr
import shapely
from scipy import stats
from scipy.optimize import curve_fit
import time
from datetime import datetime
from basic_function import Path, write_raster, create_folder, file_filter, reassign_sole_pixel, date2doy, raster_ds2bounds, retrieve_srs, doy2date
from CCDC.CCDC import *


def seven_para_logistic_function(x, m1, m2, m3, m4, m5, m6, m7):
    return m1 + (m2 - m7 * x) * ((1 / (1 + np.exp((m3 - x) / m4))) - (1 / (1 + np.exp((m5 - x) / m6))))


def system_recovery_function(t, ymax, yo, b):
    return ymax - (ymax - yo) * np.exp(- b * t)


def linear_f(x, a, b):
    return a * (x + 1985) + b


def two_term_fourier(x, a0, a1, b1, a2, b2, w):
    return a0 + a1 * np.cos(w * x) + b1 * np.sin(w * x) + a2 * np.cos(2 * w * x) + b2 * np.sin(2 * w * x)


def compute_SPDL_pheme(para_dic: dict, pheme_list: list, ul_pos,):

    try:
        # Get the x size and y size of arr
        x_size = [para_dic[_].shape[1] for _ in para_dic.keys()]
        y_size = [para_dic[_].shape[0] for _ in para_dic.keys()]
        if False in [x_size[0] == _ for _ in x_size] or False in [y_size[0] == _ for _ in y_size]:
            raise Exception('The size of the input array is not consistent!')

        # Define the res dic
        res_dic = {}
        for _ in pheme_list:
            res_dic[_] = np.zeros([y_size[0], x_size[0]]) * np.nan

        with tqdm(total=y_size[0] * x_size[0], desc=f'Compute the pheme of {str(ul_pos[0])}_{str(ul_pos[1])}', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
            for y_ in range(y_size[0]):
                for x_ in range(x_size[0]):
                    if True not in [np.isnan(para_dic[_][y_, x_]) for _ in para_dic.keys()]:
                        # Generate annual vi series
                        vi_series = seven_para_logistic_function(np.linspace(1, 365, 365), para_dic[0][y_, x_], para_dic[1][y_, x_], para_dic[2][y_, x_],
                                                                 para_dic[3][y_, x_], para_dic[4][y_, x_], para_dic[5][y_, x_], para_dic[6][y_, x_])

                        # Overfitting detection
                        if np.argwhere(vi_series > para_dic[0][y_, x_] + 0.1 * para_dic[1][y_, x_]).shape[0] == 0:
                            continue
                        elif np.argwhere(vi_series < 0.98 * para_dic[0][y_, x_]).shape[0] > 1:
                            continue

                        # Generate the delta
                        peak_doy = np.argmax(vi_series)
                        if peak_doy > para_dic[2][y_, x_]:
                            delta = (peak_doy - para_dic[2][y_, x_]) / para_dic[3][y_, x_]
                            delta2 = (para_dic[2][y_, x_] - np.argwhere(vi_series > para_dic[0][y_, x_] + 0.1 * para_dic[1][y_, x_]).min()) / para_dic[3][y_, x_]
                            # print(f'delta1: {str(delta)}')
                            # print(f'delta2: {str(delta2)}')
                        else:
                            pbar.update()
                            continue

                        # Compute pheme at the pixel lvl
                        for pheme_temp in pheme_list:
                            if pheme_temp in ['SOS', 'EOS']:
                                pos_arr = np.argwhere(vi_series > para_dic[0][y_, x_] + 0.1 * para_dic[1][y_, x_])
                                res_dic[pheme_temp][y_, x_] = pos_arr.min() if pheme_temp == 'SOS' else pos_arr.max()
                            elif pheme_temp == 'peak_vi':
                                res_dic[pheme_temp][y_, x_] = np.nanmax(vi_series)
                            elif pheme_temp == 'peak_doy':
                                res_dic[pheme_temp][y_, x_] = peak_doy
                            elif pheme_temp == 'EOM':
                                res_dic[pheme_temp][y_, x_] = para_dic[4][y_, x_] - delta * para_dic[5][y_, x_]
                            elif pheme_temp == 'MAVI':
                                res_dic[pheme_temp][y_, x_] = np.mean(vi_series[peak_doy: int(para_dic[4][y_, x_] - delta2 * para_dic[5][y_, x_]) + 1])
                            elif pheme_temp == 'MAVI_std':
                                res_dic[pheme_temp][y_, x_] = np.std(vi_series[peak_doy: int(para_dic[4][y_, x_] - delta * para_dic[5][y_, x_]) + 1])
                            elif pheme_temp == 'TSVI':
                                res_dic[pheme_temp][y_, x_] = vi_series[int(para_dic[4][y_, x_] - delta * para_dic[5][y_, x_])]
                            else:
                                raise Exception('The pheme type is not supported!')
                    pbar.update()
        return res_dic, ul_pos
    except:
        print(traceback.format_exc())
        print('Above error occurred during the pheme calculation!')
    

def shapely_to_ogr_type(shapely_type):
    from osgeo import ogr
    if shapely_type == "Polygon":
        return ogr.wkbPolygon
    elif shapely_type == "LineString":
        return ogr.wkbLineString
    elif shapely_type == "MultiPolygon":
        return ogr.wkbMultiPolygon
    elif shapely_type == "MultiLineString":
        return ogr.wkbLineString
    raise TypeError("shapely type %s not supported" % shapely_type)


def create_circle_polygon(center_coordinate: list, diameter):

    n_points = 3600
    angles = np.linspace(0, 360, n_points)
    x, y = [], []
    for ang_temp in angles:
        x.append(center_coordinate[0] + (diameter / 2) * np.sin(ang_temp * 2 * np.pi / 360))
        y.append(center_coordinate[1] + (diameter / 2) * np.cos(ang_temp * 2 * np.pi / 360))
    x.append(x[0])
    y.append(y[0])
    return shapely.geometry.Polygon(zip(x, y))


def extract_value2shpfile(raster: np.ndarray, raster_gt: tuple, shpfile: shapely.geometry.Polygon, epsg_id: int,
                          factor: int = 10, nodatavalue=-32768):
    # Retrieve vars
    xsize, ysize = raster.shape[1], raster.shape[0]
    ulx, uly, lrx, lry, xres, yres = raster_gt[0], raster_gt[3], raster_gt[0] + raster.shape[1] * raster_gt[1], \
                                                                 raster_gt[3] + raster.shape[0] * raster_gt[5], \
    raster_gt[1], -raster_gt[5]

    xres_min = float(xres / factor)
    yres_min = float(yres / factor)

    if (uly - lry) / yres != ysize or (lrx - ulx) / xres != xsize:
        raise ValueError('The raster and proj is not compatible!')

    # Define the combined raster size
    shp_xmax, shp_xmin, shp_ymax, shp_ymin = shpfile.bounds[2], shpfile.bounds[0], shpfile.bounds[3], shpfile.bounds[1]
    out_xmax, out_xmin, out_ymax, out_ymin = None, None, None, None
    ras_xmax_indi, ras_xmin_indi, ras_ymax_indi, ras_ymin_indi = None, None, None, None

    for i in range(xsize):
        if ulx + i * xres < shp_xmin < ulx + (i + 1) * xres:
            out_xmin = (i * xres) + ulx
            ras_xmin_indi = i
        if ulx + i * xres < shp_xmax < ulx + (i + 1) * xres:
            out_xmax = (i + 1) * xres + ulx
            ras_xmax_indi = i + 1
            break

    for q in range(ysize):
        if uly - q * yres > shp_ymax > uly - (q + 1) * yres:
            out_ymax = uly - (q * yres)
            ras_ymin_indi = q
        if uly - q * yres > shp_ymin > uly - (q + 1) * yres:
            out_ymin = uly - (q + 1) * yres
            ras_ymax_indi = q + 1
            break

    if out_ymin is None or out_ymax is None or out_xmin is None or out_xmax is None:
        return np.nan

    if not isinstance(raster, np.ndarray):
        raster_temp = raster.toarray()[ras_ymin_indi: ras_ymax_indi, ras_xmin_indi: ras_xmax_indi].astype(np.float64)
    else:
        raster_temp = raster[ras_ymin_indi: ras_ymax_indi, ras_xmin_indi: ras_xmax_indi].astype(np.float64)

    rasterize_Xsize, rasterize_Ysize = int((out_xmax - out_xmin) / xres_min), int((out_ymax - out_ymin) / yres_min)
    raster_temp = np.broadcast_to(raster_temp[:, None, :, None],
                                  (raster_temp.shape[0], factor, raster_temp.shape[1], factor)).reshape(
        np.int64(factor * raster_temp.shape[0]), np.int64(factor * raster_temp.shape[1])).copy()

    if [raster_temp == nodatavalue][0].all():
        return np.nan
    elif ~np.isnan(nodatavalue):

        raster_temp[raster_temp == nodatavalue] = np.nan

    if raster_temp.shape[0] != rasterize_Ysize or raster_temp.shape[1] != rasterize_Xsize:
        raise ValueError('The output raster and rasterise raster are not consistent!')

    new_gt = [out_xmin, xres_min, 0, out_ymax, 0, -yres_min]
    ogr_geom_type = shapely_to_ogr_type(shpfile.geom_type)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_id)

    # Create a temporary vector layer in memory
    mem_drv = ogr.GetDriverByName(str("Memory"))
    mem_ds = mem_drv.CreateDataSource(str('out'))
    mem_layer = mem_ds.CreateLayer(str('out'), srs, ogr_geom_type)
    ogr_feature = ogr.Feature(feature_def=mem_layer.GetLayerDefn())
    ogr_geom = ogr.CreateGeometryFromWkt(shpfile.wkt)
    ogr_feature.SetGeometryDirectly(ogr_geom)
    mem_layer.CreateFeature(ogr_feature)

    # Rasterize it
    driver = gdal.GetDriverByName(str('MEM'))
    rvds = driver.Create(str('rvds'), rasterize_Xsize, rasterize_Ysize, 1, gdal.GDT_Byte)
    rvds.SetGeoTransform(new_gt)
    file_temp = gdal.RasterizeLayer(rvds, [1], mem_layer, None, None, burn_values=[1], options=['ALL_TOUCHED=True'])
    info_temp = rvds.GetRasterBand(1).ReadAsArray()
    raster_temp[info_temp == 0] = np.nan

    # Return the value
    if np.sum(~np.isnan(raster_temp)) / np.sum(info_temp == 1) > 0.5:
        return np.nansum(raster_temp) / np.sum(~np.isnan(raster_temp))
    else:
        return np.nan


def process_denv_via_pheme(denv_dc: str, pheme_dc: str, year, pheme, kwargs):
    try:
        st = time.time()
        denvname, cal_method, base_status, size_control_factor, output_path, ROI_tif = kwargs
        start_pheme, end_pheme = pheme

        print(f'Start calculate the \033[1;31m{str(cal_method)}\033[0m \033[1;31m{denvname}\033[0m for the year \033[1;34m{str(year)}\033[0m')

        # Read the roi tif file
        ds_temp = gdal.Open(ROI_tif)
        dc_YSize, dc_XSize = ds_temp.RasterYSize, ds_temp .RasterXSize

        # Read the datacube
        if denv_dc.endswith('\\'):
            denv_dc = NDSparseMatrix().load(denv_dc)
        elif denv_dc.endswith('.npy'):
            denv_dc = np.load(denv_dc)
        else:
            raise TypeError('The type of denv_dc is not supported!')

        if pheme_dc.endswith('\\'):
            pheme_dc = NDSparseMatrix().load(pheme_dc)
        elif pheme_dc.endswith('.npy'):
            pheme_dc = np.load(pheme_dc)
        else:
            raise TypeError('The type of pheme_dc is not supported!')

        # Get the type of the denv and pheme dc
        denv_sparse_factor = isinstance(denv_dc, NDSparseMatrix)
        pheme_sparse_factor = isinstance(pheme_dc, NDSparseMatrix)

        # Get the base status
        if base_status is True:
            if not os.path.exists(os.path.join(output_path, f'{str(cal_method)}_{denvname}_{str(year)}_static_{start_pheme}.TIF')):

                # Determine the base time
                if start_pheme == 'SOY':
                    start_doy = np.zeros([dc_YSize, dc_XSize])
                    end_doy = np.ones([dc_YSize, dc_XSize]) * 10

                elif start_pheme == 'EOY':
                    raise Exception('EOY can not be the start pheme when base status is True')

                elif start_pheme in ['SOS', 'peak_doy', 'EOS']:
                    if pheme_sparse_factor:
                        start_doy = np.round(pheme_dc.SM_group[f'{str(year)}_{start_pheme}'].toarray())
                        start_doy[start_doy == 0] = np.nan
                        start_doy = start_doy - 5
                        end_doy = np.round(pheme_dc.SM_group[f'{str(year)}_{start_pheme}'].toarray())
                        end_doy[end_doy == 0] = np.nan
                        end_doy = end_doy + 5
                    else:
                        raise Exception('Code error')
                else:
                    raise Exception('Code Error')

                # Generate the base value
                start_time = np.nanmin(start_doy).astype(np.int16)
                end_time = np.nanmax(end_doy).astype(np.int16)
                acc_static = np.zeros([dc_YSize, dc_XSize])
                cum_static = np.zeros([dc_YSize, dc_XSize])
                with tqdm(total=end_time + 1 - start_time, desc=f'Get the static value of {str(year)}', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
                    for _ in range(start_time, end_time + 1):
                        within_factor = np.logical_and(np.ones([dc_YSize, dc_XSize]) * _ >= start_doy,
                                                       np.ones([dc_YSize, dc_XSize]) * _ < end_doy)
                        cum_static = cum_static + within_factor
                        if denv_sparse_factor:
                            denv_date = int(year * 1000 + _)
                            denv_arr = denv_dc.SM_group[denv_date].toarray()
                        else:
                            raise Exception('Code Error')
                        denv_doy_arr = denv_arr * within_factor
                        acc_static = acc_static + denv_doy_arr
                        pbar.update()

                acc_static = acc_static / cum_static
                cum_static = None
                write_raster(ds_temp, acc_static, output_path, f'{str(cal_method)}_{denvname}_{str(year)}_static_{start_pheme}.TIF',
                                raster_datatype=gdal.GDT_Float32)
            else:
                ds_temp = gdal.Open(os.path.join(output_path, f'{str(cal_method)}_{denvname}_{str(year)}_static_{start_pheme}.TIF'))
                acc_static = ds_temp.GetRasterBand(1).ReadAsArray()

        if not os.path.exists(os.path.join(output_path, f'{str(cal_method)}_{denvname}_{start_pheme}_{end_pheme}_{str(year)}.TIF')):
            # Get the denv matrix
            start_arr = np.round(pheme_dc.SM_group[f'{str(year)}_{start_pheme}'].toarray())
            end_arr = np.round(pheme_dc.SM_group[f'{str(year)}_{end_pheme}'].toarray())
            start_arr[start_arr == 0] = np.nan
            end_arr[end_arr == 0] = np.nan

            # Get the unique value
            start_doy = np.nanmin(np.unique(start_arr)).astype(np.int16)
            end_doy = np.nanmax(np.unique(end_arr)).astype(np.int16)
            acc_denv = np.zeros([dc_YSize, dc_XSize])
            cum_denv = np.zeros([dc_YSize, dc_XSize])

            with tqdm(total=end_doy + 1 - start_doy, desc=f'Get the static value of {str(year)}', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
                for _ in range(start_doy, end_doy + 1):

                    if denv_sparse_factor:
                        denv_date = int(year * 1000 + _)
                        denv_arr = denv_dc.SM_group[denv_date].toarray()
                    else:
                        raise Exception('Code Error')

                    if base_status:
                        within_factor = np.logical_and(np.logical_and(np.ones([dc_YSize, dc_XSize]) * _ >= start_doy,
                                                       np.ones([dc_YSize, dc_XSize]) * _ < end_doy),  denv_arr >= acc_static)
                        denv_doy_arr = (denv_arr - acc_static) * within_factor
                    else:
                        within_factor = np.logical_and(np.ones([dc_YSize, dc_XSize]) * _ >= start_doy,
                                                       np.ones([dc_YSize, dc_XSize]) * _ < end_doy)
                        denv_doy_arr = denv_arr * within_factor

                    if cal_method != 'max':
                        cum_denv = cum_denv + within_factor
                        acc_denv = acc_denv + denv_doy_arr
                    else:
                        acc_denv = np.nanmax(acc_denv, denv_doy_arr)
                    pbar.update()

            if cal_method == 'mean':
                acc_denv = acc_denv / cum_denv

            if size_control_factor:
                acc_denv = acc_denv / 100

            write_raster(ds_temp, acc_denv, output_path, f'{str(cal_method)}_{denvname}_{start_pheme}_{end_pheme}_{str(year)}.TIF', raster_datatype=gdal.GDT_Float32)
        print(f'Finish calculate the \033[1;31m{str(cal_method)}\033[0m \033[1;31m{denvname}\033[0m for the year \033[1;34m{str(year)}\033[0m in {str(time.time() - st)}s')
    except:
        print(traceback.format_exc())
        raise Exception('Some error occurred during handling the above process!')


def assign_wl(inun_inform: list, wl_arr: np.ndarray):

    min_wl_, pos_y, pos_x, status, wl_refined = inun_inform
    if status == 0 and np.isnan(wl_refined):
        wl_centre = min_wl_
        upper_wl_dis, lower_wl_dis, lower_wl = np.nan, np.nan, np.nan
        for r in range(1, 100):

            pos_y_lower = 0 if pos_y - r < 0 else pos_y - r
            pos_x_lower = 0 if pos_x - r < 0 else pos_x - r
            pos_y_upper = wl_arr.shape[0] if pos_y + r + 1 > wl_arr.shape[0] else pos_y + r + 1
            pos_x_upper = wl_arr.shape[1] if pos_x + r + 1 > wl_arr.shape[1] else pos_x + r + 1

            arr_tt = wl_arr[pos_y_lower: pos_y_upper, pos_x_lower: pos_x_upper]
            if np.isnan(upper_wl_dis):
                upper_wl_dis_list = []
                if (arr_tt == wl_centre).any():
                    for pos_ttt in np.argwhere(arr_tt == wl_centre):
                        upper_wl_dis_list.append(np.sqrt((pos_ttt[0] - r) ** 2 + (pos_ttt[1] - r) ** 2))
                    upper_wl_dis = min(upper_wl_dis_list)

            if np.isnan(lower_wl_dis):
                lower_wl_dis_list = []
                lower_wl_dat_list = []
                lower_wl_dis_list2 = []
                arr_tt[arr_tt < 0] = 100000
                if (arr_tt < wl_centre).any():
                    for pos_ttt in np.argwhere(arr_tt <= wl_centre):
                        lower_wl_dis_list.append(np.sqrt((pos_ttt[0] - r) ** 2 + (pos_ttt[1] - r) ** 2))
                        lower_wl_dat_list.append(arr_tt[pos_ttt[0], pos_ttt[1]])

                    lower_wl = set(lower_wl_dat_list)
                    lower_wl = min(lower_wl)
                    for ___ in range(len(lower_wl_dat_list)):
                        if lower_wl_dat_list[___] == lower_wl:
                            lower_wl_dis_list2.append(lower_wl_dis_list[___])
                    lower_wl_dis = min(lower_wl_dis_list2)

            if ~np.isnan(upper_wl_dis) and ~np.isnan(lower_wl_dis) and ~np.isnan(lower_wl):
                break

        wl_refined = lower_wl + (wl_centre - lower_wl) * lower_wl_dis / (lower_wl_dis + upper_wl_dis)

    elif status == 0:
        print('code error during assignment')
        raise Exception('code error during assignment')

    if np.isnan(wl_refined):
        return [min_wl_, pos_y, pos_x, status, min_wl_]
    else:
        return [min_wl_, pos_y, pos_x, status, wl_refined]


def pre_assign_wl(inun_inform: list, min_inun_wl_arr: np.ndarray, max_noninun_wl_arr: np.ndarray, water_level_data, doy_list):

    min_wl_, pos_y, pos_x, status = inun_inform
    if status == 1:
        inun_inform.append(min_inun_wl_arr[pos_y, pos_x])
    elif status == -1:
        if max_noninun_wl_arr[pos_y, pos_x] > min_inun_wl_arr[pos_y, pos_x]:
            wl_temp = min_inun_wl_arr[pos_y, pos_x]
            wl_pos = np.argwhere(water_level_data == float(str(wl_temp)))
            date, date_pos = [], []
            for wl_pos_temp in wl_pos:
                if water_level_data[wl_pos_temp[0], 0] in doy_list:
                    date.append(water_level_data[wl_pos_temp[0]])
                    date_pos.append(wl_pos_temp[0])

            if len(date) == 1:
                wl_temp_2 = water_level_data[int(date_pos[0]) - 5, 1]
            else:
                date_pos = min(date_pos)
                wl_temp_2 = water_level_data[int(date_pos) - 5, 1]

            if wl_temp_2 <= wl_temp:
                inun_inform.append((wl_temp_2 + wl_temp) / 2)
            else:
                inun_inform.append(wl_temp - 1)

        else:
            inun_inform.append((min_inun_wl_arr[pos_y, pos_x] + max_noninun_wl_arr[pos_y, pos_x]) / 2)
    elif status == 0:
        inun_inform.append(np.nan)
    return inun_inform


def assign_wl_status(wl_arr: np.ndarray,  wl: float):

    inun_inform_list = []
    pos_all = np.argwhere(wl_arr == wl)

    for pos_temp in pos_all:
        arr_temp = wl_arr[pos_temp[0] - 1: pos_temp[0] + 2, pos_temp[1] - 1: pos_temp[1] + 2]
        inun_inform = [wl]
        inun_inform.extend([pos_temp[0], pos_temp[1]])
        if (arr_temp == wl).all():
            inun_inform.append(0)
        elif (arr_temp > wl).any():
            inun_inform.append(1)
        elif np.isnan(arr_temp).any() == 1:
            inun_inform.append(1)
        elif (arr_temp == -1).any():
            inun_inform.append(-1)
        elif (arr_temp < wl).any():
            inun_inform.append(0)
        else:
            raise Exception(str(arr_temp))
        inun_inform_list.append(inun_inform)

    return inun_inform_list


def mp_static_wi_detection(dc: NDSparseMatrix, sz_f, zoffset, nodata, thr):

    inundation_arr_list = []
    name_list = []
    for z_temp in range(len(dc.SM_namelist)):
        inundation_array = dc.SM_group[dc.SM_namelist[z_temp]]
        dtype_temp = type(inundation_array)

        inundation_array = invert_data(inundation_array, sz_f, zoffset, nodata)
        inundation_array[inundation_array >= thr] = 2
        inundation_array[inundation_array < thr] = 1
        inundation_array[np.isnan(inundation_array)] = 0
        inundation_array = inundation_array.astype(np.byte)
        inundation_arr_list.append(dtype_temp(inundation_array))
        name_list.append(dc.SM_namelist[z_temp])
    return inundation_arr_list, name_list


def invert_data(data, size_control, offset_value, nodata_value, original_dtype: bool = False):

    # Convert the data to ndarray
    if isinstance(data, np.ndarray):
        dtype = np.ndarray
    elif isinstance(data, (sm.coo_matrix, sm.csr_matrix, sm.csc_matrix, sm.bsr_matrix, sm.lil_matrix, sm.dok_matrix)):
        dtype = type(data)
        data = data.toarray()
    elif isinstance(data, list):
        dtype = list
        try:
            data = np.ndarray(data)
        except TypeError:
            print(traceback.format_exc())
            raise TypeError('There are str type in the input data during the data inversion!')
    else:
        raise TypeError(f'The input data is not under a proper type!')

    # Convert nodata value to np.nan
    if ~np.isnan(nodata_value):
        data = data.astype(np.float32)
        data[data == nodata_value] = np.nan
    else:
        data = data.astype(np.float32)

    # Data inversion
    if offset_value is not None and ~np.isnan(offset_value) and isinstance(offset_value, (np.integer, int, float)):
        data = data - offset_value

    if isinstance(size_control, bool) and size_control:
        data = data / 10000

    # Return data
    if original_dtype:
        return dtype(data)
    else:
        return data


def create_bimodal_histogram_threshold(data_cube, pos_list: pd.DataFrame, yx_offset_list: list, doy_array, size_control_factor, zoffset, nodata_value, variance_num, bimodal_factor):

    output_list = []
    with tqdm(total=pos_list.shape[0], desc=f'Create bimodal histogram threshold Y{str(yx_offset_list[0])} X{str(yx_offset_list[1])}', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
        for _ in range(pos_list.shape[0]):
            y, x = pos_list['y'][_] - yx_offset_list[0], pos_list['x'][_] - yx_offset_list[1]
            wi_series = data_cube[y, x, :]
            wi_series = invert_data(wi_series, size_control_factor, zoffset, nodata_value)
            wi_series = wi_series.flatten()

            # Create the bimodal histogram threshold
            if bimodal_factor:
                doy_array_pixel = np.mod(doy_array, 1000)
                doy_array_pixel = np.delete(doy_array_pixel, np.argwhere(np.isnan(wi_series)))
                wi_series = np.delete(wi_series, np.argwhere(np.isnan(wi_series)))

                bh_threshold = bimodal_histogram_threshold(wi_series, init_threshold=0.1)
                if np.isnan(bh_threshold):
                    bh_threshold = -2
            else:
                bh_threshold = 0.123

            # Create the dynamic threshold
            wi_series = np.delete(wi_series, np.argwhere(np.logical_and(doy_array_pixel >= 182, doy_array_pixel <= 300)))
            wi_series = np.delete(wi_series, np.argwhere(wi_series < -0.7))
            all_dry_sum = wi_series.shape[0]
            wi_series = np.delete(wi_series, np.argwhere(wi_series > bh_threshold))

            if wi_series.shape[0] < 5:
                output_list.append([y + yx_offset_list[0], x + yx_offset_list[1], np.nan, np.nan])
            elif wi_series.shape[0] < 0.5 * all_dry_sum:
                output_list.append([y + yx_offset_list[0], x + yx_offset_list[1], np.nan, np.nan])
            else:
                mndwi_temp_std = np.nanstd(wi_series)
                mndwi_ave = np.mean(wi_series)
                output_list.append([y + yx_offset_list[0], x + yx_offset_list[1], min(mndwi_ave + variance_num * mndwi_temp_std, bh_threshold), bh_threshold])
            pbar.update()
    return output_list


def bimodal_histogram_threshold(input_temp, method = None, init_threshold=0.123):

    # Turn the input temp as np.ndarray
    if type(input_temp) is list:
        array_temp = np.array(input_temp).flatten()
    else:
        array_temp = input_temp.flatten()

    if array_temp.shape[0] == 0:
        return np.nan
    else:
        # Detect whether the input temp is valid
        if type(input_temp) is not list and type(input_temp) is not np.ndarray:
            raise TypeError('Please input a list for the generation of bimodal histogram threshold!')
        elif False in [isinstance(data_temp, (float, np.float32, np.float64)) or np.isnan(data_temp) for data_temp in input_temp]:
            raise TypeError('Please input the list with all numbers in it!')

        # Define the init_threshold
        if np.isnan(init_threshold):
            init_threshold = (array_temp.max() + array_temp.min()) / 2
        elif type(init_threshold) is not int and type(init_threshold) is not float:
            raise TypeError('Please input the init_threshold as a num!')

        # Check whether the array is bimodal
        list_lower = [q for q in input_temp if q < init_threshold]
        list_greater = [q for q in input_temp if q >= init_threshold]

        if np.sum(np.array([~np.isnan(temp) for temp in array_temp])) < 8:
            return 0.123
        elif len(list_lower) == 0 or len(list_greater) == 0:
            return 0.123
        elif len(list_greater) >= 0.9 * (len(list_lower) + len(list_greater)):
            return 0.123
        else:
            ith_threshold = init_threshold
            while True:
                list_lower = [q for q in input_temp if q < ith_threshold]
                list_greater = [q for q in input_temp if q >= ith_threshold]
                i1th_threshold = (np.array(list_greater).mean() + np.array(list_lower).mean()) / 2
                if i1th_threshold == ith_threshold or np.isnan(ith_threshold) or np.isnan(i1th_threshold):
                    break
                else:
                    ith_threshold = i1th_threshold

            if np.isnan(ith_threshold) or np.isnan(i1th_threshold):
                return init_threshold
            elif i1th_threshold < -0.05:
                return 0.123
            elif i1th_threshold > 0.2:
                return 0.123
            else:
                return i1th_threshold


def slice_datacube(datacube, pos, sort_y_x='x'):

    # Process para
    if sort_y_x not in ['y', 'x']:
        raise ValueError('The index can only be sort through x or y!')

    # Preprocess pos
    cpu_amount = os.cpu_count()
    try:
        pos = pd.DataFrame(pos, columns=['y', 'x'])
        pos = pos.sort_values(sort_y_x, ascending=True).reset_index(drop=True)
    except:
        print(traceback.format_exc())
        raise ValueError('The pos might not under the correct type during the datacube slice!')

    # Slice datacube
    datacube_list, pos_list, yxoffset_list = [], [], []
    indi_len = int(np.floor(pos.shape[0] / cpu_amount))
    for _ in range(cpu_amount):
        pos_ = pos[_ * indi_len: min((_ + 1) * indi_len, pos.shape[0])].reset_index(drop=True)
        pos_list.append(pos_)
        y_min, y_max, x_min, x_max = pos_['y'].min(), pos_['y'].max(), pos_['x'].min(), pos_['x'].max()
        if isinstance(datacube, NDSparseMatrix):
            datacube_list.append(datacube.extract_matrix(([y_min, y_max + 1], [x_min, x_max + 1], ['all'])))
        elif isinstance(datacube, np.ndarray):
            datacube_list.append(datacube[y_min: y_max + 1, x_min: x_max + 1, :])
        else:
            raise TypeError('The slicing datacube is not under NDSM or datacube type!')
        yxoffset_list.append([y_min, x_min])

    return datacube_list, pos_list, yxoffset_list


def create_indi_DT_inundation_map(inundated_arr, doy_array: np.ndarray, date_num: int, DT_threshold: np.ndarray, output_path: str, inundation_overwritten_factor: bool, sz_ctrl: bool, zoffset, nodata_value, ROI_tif: str):

    if not os.path.exists(f'{output_path}\\DT_{str(doy_array[date_num])}.TIF') or inundation_overwritten_factor:
        inundated_arr = invert_data(inundated_arr, sz_ctrl, zoffset, nodata_value)
        inundation_map = inundated_arr - DT_threshold
        inundation_map[inundation_map >= 0] = 2
        inundation_map[inundation_map < 0] = 1
        inundation_map[np.isnan(inundation_map)] = 0
        inundation_map[inundated_arr > 0.16] = 2
        inundation_map = reassign_sole_pixel(inundation_map, Nan_value=0, half_size_window=2)
        inundation_map = inundation_map.astype(np.byte)
        inundation_arr = None
        write_raster(gdal.Open(ROI_tif), inundation_map, output_path, f'DT_{str(doy_array[date_num])}.TIF', raster_datatype=gdal.GDT_Byte, nodatavalue=0)
    else:
        inundated_ds = gdal.Open(f'{output_path}DT_{str(doy_array[date_num])}.TIF')
        inundation_map = inundated_ds.GetRasterBand(1).ReadAsArray()

    if isinstance(inundated_arr, np.ndarray):
        return date_num, inundation_map
    else:
        inundation_map = type(inundated_arr)(inundation_map)
        return date_num, inundation_map


def curfit4bound_annual(pos_df: pd.DataFrame, index_dc_temp, doy_all: list, curfit_dic: dict, sparse_matrix_factor: bool, size_control_factor: bool, xy_offset: list, cache_folder: str, nd_v, zoff, divider: int = 10000):

    try:
        # Set up initial var
        print(str(xy_offset[0]) + '_' + str(xy_offset[1]))
        start_time1 = time.time()
        year_all = np.unique(np.array([temp // 1000 for temp in doy_all]))
        year_range = range(np.min(year_all), np.max(year_all) + 1)
        pos_len = pos_df.shape[0]
        pos_df = pos_df.reset_index()

        # Set up the Cache folder
        cache_folder = Path(cache_folder).path_name
        if os.path.exists(f'{cache_folder}postemp_{str(xy_offset[1])}.csv'):
            pos_df = pd.read_csv(f'{cache_folder}postemp_{str(xy_offset[1])}.csv')
            pos_init = int(np.ceil(max(pos_df.loc[~np.isnan(pos_df.para_ori_0)].index) / divider) * divider) + 1
            pos_init = pos_len - 1 if pos_init > pos_len else pos_init
            q_all = pos_init
            q_temp = np.mod(q_all, 100)
        else:
            q_all, q_temp, pos_init = 0, 0, 0
            # insert columns

            pos_df = pd.concat([pos_df, pd.DataFrame([[np.nan for q in range(curfit_dic['para_num'])] for qq in range(pos_df.shape[0])], index=pos_df.index, columns=[f'para_ori_{str(i)}' for i in range(curfit_dic['para_num'])])], axis=1)
            pos_df = pd.concat([pos_df, pd.DataFrame([[np.nan for q in range(curfit_dic['para_num'])] for qq in range(pos_df.shape[0])], index=pos_df.index, columns=[f'para_bound_min_{str(i)}' for i in range(curfit_dic['para_num'])])], axis=1)
            pos_df = pd.concat([pos_df, pd.DataFrame([[np.nan for q in range(curfit_dic['para_num'])] for qq in range(pos_df.shape[0])], index=pos_df.index, columns=[f'para_bound_max_{str(i)}' for i in range(curfit_dic['para_num'])])], axis=1)

            for year_temp in year_range:
                col = [f'{str(year_temp)}_para_{str(i)}' for i in range(curfit_dic['para_num'])]
                col.append(f'{str(year_temp)}_Rsquare')
                pos_df = pd.concat([pos_df, pd.DataFrame([[np.nan for q in range(curfit_dic['para_num'] + 1)] for qq in range(pos_df.shape[0])], index=pos_df.index, columns=col)], axis=1)

        # Define the fitting curve algorithm
        if curfit_dic['CFM'] == 'SPL':
            para_ori = [0.10, 0.5, 108.2, 7.596, 311.4, 7.473, 0.00225]
            para_upbound = [0.6, 0.8, 180, 20, 330, 20, 0.01]
            para_lowerbound = [0.08, 0, 40, 3, 180, 3, 0.0001]
            curfit_algorithm = seven_para_logistic_function
        elif curfit_dic['CFM'] == 'TTF':
            curfit_algorithm = two_term_fourier
            para_ori = [0, 0, 0, 0, 0, 0.017]
            para_upbound = [1, 0.5, 0.5, 0.05, 0.05, 0.019]
            para_lowerbound = [0, -0.5, -0.5, -0.05, -0.05, 0.015]
        t1, t2, t3, t4, t5 = 0, 0, 0, 0, 0

        # # Create name_dic
        # name_dic = {}
        # col_list = [f'para_bound_max_{str(num)}' for num in range(curfit_dic['para_num'])]
        # col_list.extend([f'para_bound_min_{str(num)}' for num in range(curfit_dic['para_num'])])
        # col_list.extend([f'para_ori_{str(num)}' for num in range(curfit_dic['para_num'])])
        # name_dic['bounds'] = col_list
        # for year_temp in year_range:
        #     col_list = [f'{str(year_temp)}_para_{str(num)}' for num in range(curfit_dic['para_num'])]
        #     col_list.append(f'{str(year_temp)}_Rsquare')
        #     name_dic[year_temp] = col_list

        # Start generate the boundary and paras based on curve fitting
        with tqdm(total=pos_len - pos_init, desc=f'Curve fitting Y{str(xy_offset[0])} X{str(xy_offset[1])}', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
            for pos_len_temp in range(pos_init, pos_len):

                # start_time = time.time()
                # Define the key var
                y_t, x_t = pos_df.loc[pos_len_temp, 'y'], pos_df.loc[pos_len_temp, 'x']
                y_t = int(y_t - xy_offset[0])
                x_t = int(x_t - xy_offset[1])

                if not sparse_matrix_factor:
                    vi_all = index_dc_temp[y_t, x_t, :].flatten()
                elif sparse_matrix_factor:
                    vi_all = index_dc_temp[y_t, x_t, :]
                    vi_all = vi_all.flatten()
                doy_temp = copy.deepcopy(doy_all)
                doy_temp = np.mod(doy_temp, 1000)
                year_doy_all = copy.deepcopy(doy_all)
                vi_all = invert_data(vi_all, size_control_factor, zoff, nd_v)

                vi_pos = np.argwhere(np.logical_or(vi_all < 0, np.isnan(vi_all)))
                doy_temp = np.delete(doy_temp, vi_pos)
                year_doy_all = np.delete(year_doy_all, vi_pos)
                vi_all = np.delete(vi_all, vi_pos)

                paras_max_dic = [np.nan for _ in range(curfit_dic['para_num'])]
                paras_min_dic = [np.nan for _ in range(curfit_dic['para_num'])]

                if doy_temp.shape[0] >= 7:
                    try:
                        paras, extras = curve_fit(curfit_algorithm, doy_temp, vi_all, maxfev=50000, p0=curfit_dic['initial_para_ori'], bounds=curfit_dic['initial_para_boundary'], ftol=0.001)
                        # t2 += time.time() - start_time

                        vi_dormancy, doy_dormancy, vi_max, doy_max = [], [], [], []
                        doy_index_max = np.argmax(curfit_algorithm(np.linspace(0, 366, 365), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]))

                        # Generate the parameter boundary
                        senescence_t = paras[4] - 4 * paras[5]

                        for doy_index in range(doy_temp.shape[0]):
                            if 0 < doy_temp[doy_index] < paras[2] or paras[4] < doy_temp[doy_index] < 366:
                                vi_dormancy.append(vi_all[doy_index])
                                doy_dormancy.append(doy_temp[doy_index])
                            if doy_index_max - 5 < doy_temp[doy_index] < doy_index_max + 5:
                                vi_max.append(vi_all[doy_index])
                                doy_max.append(doy_temp[doy_index])

                        if vi_max == []:
                            vi_max = [np.max(vi_all)]
                            doy_max = [doy_temp[np.argmax(vi_all)]]

                        itr = 5
                        while itr < 10:
                            doy_senescence, vi_senescence = [], []
                            for doy_index in range(doy_temp.shape[0]):
                                if senescence_t - itr < doy_temp[doy_index] < senescence_t + itr:
                                    vi_senescence.append(vi_all[doy_index])
                                    doy_senescence.append(doy_temp[doy_index])
                            if doy_senescence != [] and vi_senescence != []:
                                break
                            else:
                                itr += 1

                        # define the para1
                        if vi_dormancy != []:
                            vi_dormancy_sort = np.sort(vi_dormancy)
                            vi_max_sort = np.sort(vi_max)
                            paras_max_dic[0] = vi_dormancy_sort[int(np.fix(vi_dormancy_sort.shape[0] * 0.95))]
                            paras_min_dic[0] = vi_dormancy_sort[int(np.fix(vi_dormancy_sort.shape[0] * 0.05))]
                            paras_max_dic[0] = min(paras_max_dic[0], 0.5)
                            paras_min_dic[0] = max(paras_min_dic[0], 0)
                        else:
                            paras_max_dic[0], paras_min_dic[0] = 0.5, 0

                        # define the para2
                        paras_max_dic[1] = vi_max[-1] - paras_min_dic[0]
                        paras_min_dic[1] = vi_max[0] - paras_max_dic[0]
                        if paras_min_dic[1] < 0.2:
                            paras_min_dic[1] = 0.2
                        if paras_max_dic[1] > 0.7 or paras_max_dic[1] < 0.2:
                            paras_max_dic[1] = 0.7

                        # define the para3
                        paras_max_dic[2] = 0
                        for doy_index in range(len(doy_temp)):
                            if paras_min_dic[0] < vi_all[doy_index] < paras_max_dic[0] and doy_temp[doy_index] < 180:
                                paras_max_dic[2] = max(float(paras_max_dic[2]), doy_temp[doy_index])

                        paras_min_dic[2] = 180
                        for doy_index in range(len(doy_temp)):
                            if vi_all[doy_index] > paras_max_dic[0]:
                                paras_min_dic[2] = min(paras_min_dic[2], doy_temp[doy_index])

                        if paras_min_dic[2] > paras[2] or paras_min_dic[2] < paras[2] - 15:
                            paras_min_dic[2] = paras[2] - 15

                        if paras_max_dic[2] < paras[2] or paras_max_dic[2] > paras[2] + 15:
                            paras_max_dic[2] = paras[2] + 15

                        # define the para5
                        paras_max_dic[4] = 0
                        for doy_index in range(len(doy_temp)):
                            if vi_all[doy_index] > paras_max_dic[0]:
                                paras_max_dic[4] = max(paras_max_dic[4], doy_temp[doy_index])
                        paras_min_dic[4] = 365
                        for doy_index in range(len(doy_temp)):
                            if paras_min_dic[0] < vi_all[doy_index] < paras_max_dic[0] and doy_temp[doy_index] > 180:
                                paras_min_dic[4] = min(paras_min_dic[4], doy_temp[doy_index])
                        if paras_min_dic[4] > paras[4] or paras_min_dic[4] < paras[4] - 15:
                            paras_min_dic[4] = paras[4] - 15

                        if paras_max_dic[4] < paras[4] or paras_max_dic[4] > paras[4] + 15:
                            paras_max_dic[4] = paras[4] + 15

                        # define the para 4
                        if len(doy_max) != 1:
                            paras_max_dic[3] = (np.nanmax(doy_max) - paras_min_dic[2]) / 4
                            paras_min_dic[3] = (np.nanmin(doy_max) - paras_max_dic[2]) / 4
                        else:
                            paras_max_dic[3] = (np.nanmax(doy_max) + 5 - paras_min_dic[2]) / 4
                            paras_min_dic[3] = (np.nanmin(doy_max) - 5 - paras_max_dic[2]) / 4
                        paras_min_dic[3] = max(3, paras_min_dic[3])
                        paras_max_dic[3] = min(17, paras_max_dic[3])
                        if paras_min_dic[3] > 17:
                            paras_min_dic[3] = 3
                        if paras_max_dic[3] < 3:
                            paras_max_dic[3] = 17
                        paras_max_dic[5] = paras_max_dic[3]
                        paras_min_dic[5] = paras_min_dic[3]
                        if doy_senescence == [] or vi_senescence == []:
                            paras_max_dic[6] = 0.01
                            paras_min_dic[6] = 0.00001
                        else:
                            paras_max_dic[6] = (np.nanmax(vi_max) - np.nanmin(vi_senescence)) / (
                                    doy_senescence[np.argmin(vi_senescence)] - doy_max[np.argmax(vi_max)])
                            paras_min_dic[6] = (np.nanmin(vi_max) - np.nanmax(vi_senescence)) / (
                                    doy_senescence[np.argmax(vi_senescence)] - doy_max[np.argmin(vi_max)])
                        if np.isnan(paras_min_dic[6]):
                            paras_min_dic[6] = 0.00001
                        if np.isnan(paras_max_dic[6]):
                            paras_max_dic[6] = 0.01
                        paras_max_dic[6] = min(paras_max_dic[6], 0.01)
                        paras_min_dic[6] = max(paras_min_dic[6], 0.00001)
                        if paras_max_dic[6] < 0.00001:
                            paras_max_dic[6] = 0.01
                        if paras_min_dic[6] > 0.01:
                            paras_min_dic[6] = 0.00001
                        if paras_min_dic[0] > paras[0]:
                            paras_min_dic[0] = paras[0] - 0.01
                        if paras_max_dic[0] < paras[0]:
                            paras_max_dic[0] = paras[0] + 0.01
                        if paras_min_dic[1] > paras[1]:
                            paras_min_dic[1] = paras[1] - 0.01
                        if paras_max_dic[1] < paras[1]:
                            paras_max_dic[1] = paras[1] + 0.01
                        if paras_min_dic[2] > paras[2]:
                            paras_min_dic[2] = paras[2] - 1
                        if paras_max_dic[2] < paras[2]:
                            paras_max_dic[2] = paras[2] + 1
                        if paras_min_dic[3] > paras[3]:
                            paras_min_dic[3] = paras[3] - 0.1
                        if paras_max_dic[3] < paras[3]:
                            paras_max_dic[3] = paras[3] + 0.1
                        if paras_min_dic[4] > paras[4]:
                            paras_min_dic[4] = paras[4] - 1
                        if paras_max_dic[4] < paras[4]:
                            paras_max_dic[4] = paras[4] + 1
                        if paras_min_dic[5] > paras[5]:
                            paras_min_dic[5] = paras[5] - 0.5
                        if paras_max_dic[5] < paras[5]:
                            paras_max_dic[5] = paras[5] + 0.5
                        if paras_min_dic[6] > paras[6]:
                            paras_min_dic[6] = paras[6] - 0.00001
                        if paras_max_dic[6] < paras[6]:
                            paras_max_dic[6] = paras[6] + 0.00001
                        # t2 += time.time() - start_time
                        # start_time = time.time()
                        for num in range(curfit_dic['para_num']):
                            pos_df.loc[pos_len_temp, f'para_bound_max_{str(num)}'] = paras_max_dic[num]
                            pos_df.loc[pos_len_temp, f'para_bound_min_{str(num)}'] = paras_min_dic[num]
                            pos_df.loc[pos_len_temp, f'para_ori_{str(num)}'] = paras[num]
                        # t3 += time.time() - start_time
                    except:
                        for num in range(curfit_dic['para_num']):
                            pos_df.loc[pos_len_temp, f'para_bound_max_{str(num)}'] = para_upbound[num]
                            pos_df.loc[pos_len_temp, f'para_bound_min_{str(num)}'] = para_lowerbound[num]
                            pos_df.loc[pos_len_temp, f'para_ori_{str(num)}'] = para_ori[num]

                    ori_temp = [pos_df.loc[pos_len_temp, f'para_ori_{str(num)}'] for num in range(curfit_dic['para_num'])]
                    bounds_temp = ([pos_df.loc[pos_len_temp, f'para_bound_min_{str(num)}'] for num in range(curfit_dic['para_num'])],
                                   [pos_df.loc[pos_len_temp, f'para_bound_max_{str(num)}'] for num in range(curfit_dic['para_num'])])

                    for year_temp in year_range:
                        year_pos = np.argwhere(np.floor(year_doy_all / 1000) == year_temp)
                        annual_doy = doy_temp[year_pos].flatten()
                        annual_vi = vi_all[year_pos].flatten()
                        if np.sum(~np.isnan(annual_vi)) >= curfit_dic['para_num']:

                            try:
                                # start_time = time.time()
                                paras, extras = curve_fit(curfit_algorithm, annual_doy, annual_vi, maxfev=50000, p0=ori_temp, bounds=bounds_temp, ftol=0.001)
                                predicted_y_data = curfit_algorithm(annual_doy, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6])
                                R_square = (1 - np.sum((predicted_y_data - annual_vi) ** 2) / np.sum((annual_vi - np.mean(annual_vi)) ** 2))
                                # t4 += time.time()-start_time
                                # start_time = time.time()
                                for num in range(curfit_dic['para_num']):
                                    pos_df.loc[pos_len_temp, f'{str(year_temp)}_para_{str(num)}'] = paras[num]
                                    pos_df.loc[pos_len_temp, f'{str(year_temp)}_Rsquare'] = R_square
                                # t5 += time.time()-start_time
                            except:
                                for num in range(curfit_dic['para_num']):
                                    pos_df.loc[pos_len_temp, f'{str(year_temp)}_para_{str(num)}'] = -1
                                    pos_df.loc[pos_len_temp, f'{str(year_temp)}_Rsquare'] = -1
                        else:
                            for num in range(curfit_dic['para_num']):
                                pos_df.loc[pos_len_temp, f'{str(year_temp)}_para_{str(num)}'] = -1
                                pos_df.loc[pos_len_temp, f'{str(year_temp)}_Rsquare'] = -1
                pbar.update()

                if (np.mod(q_all, divider) == 0 and q_all != 0) or q_all == pos_len - 1:
                    pos_df.to_csv(f'{cache_folder}postemp_{str(xy_offset[1])}.csv')
                q_all += 1
    except:
        print(traceback.format_exc())
    return pos_df


def curfit_pd2tif(tif_output_path: str, df: pd.DataFrame, key: str, ds_path: str):
    if not os.path.exists(tif_output_path + key + '.TIF'):
        if key not in df.keys() or 'y' not in df.keys() or 'x' not in df.keys():
            raise TypeError("The df doesn't contain the essential keys")

        ds_temp = gdal.Open(ds_path)
        ysize, xsize = ds_temp.RasterYSize, ds_temp.RasterXSize
        array_temp = np.zeros([ysize, xsize], dtype=np.float32) * np.nan
        for pos in range(len(df)):
            if df.loc[pos, key] != -1:
                array_temp[df.loc[pos, 'y'], df.loc[pos, 'x']] = df.loc[pos, key]

        write_raster(ds_temp,  array_temp, tif_output_path, key + '.TIF', raster_datatype=gdal.GDT_Float32, nodatavalue=np.nan)


def cf2phemetric_dc(input_path, output_path, year, index, metadata_dic):

    start_time = time.time()
    print(f"Start constructing the {str(year)} {index} Phemetric datacube of {metadata_dic['ROI_name']}.")

    # Create the output path
    yearly_output_path = output_path + str(int(year)) + '\\'
    create_folder(yearly_output_path)

    if not os.path.exists(output_path + f'{str(year)}\\metadata.json') or not not os.path.exists(output_path + f'{str(year)}\\paraname.npy'):

        # Determine the input files
        yearly_input_files = file_filter(input_path, ['.TIF', '\\' + str(year)], exclude_word_list=['aux'], and_or_factor='and')

        if yearly_input_files == []:
            raise Exception('There are no valid input files, double check the temporal division!')

        ds_temp = gdal.Open(yearly_input_files[0])
        rows, cols = ds_temp.RasterYSize, ds_temp.RasterXSize
        nodata_value = ds_temp.GetRasterBand(1).GetNoDataValue()

        # Create the para list
        para_list = [filepath_temp.split('\\')[-1].split('.TIF')[0] for filepath_temp in yearly_input_files]

        if metadata_dic['huge_matrix']:
            if metadata_dic['ROI_name']:
                i, data_cube = 0, NDSparseMatrix()
                while i < len(para_list):
                    try:
                        t1 = time.time()
                        if not os.path.exists(f"{input_path}{str(para_list[i])}.TIF"):
                            raise Exception(f'The {input_path}{str(para_list[i])} is not properly generated!')
                        else:
                            array_temp = gdal.Open(f"{input_path}{str(para_list[i])}.TIF")
                            array_temp = array_temp.GetRasterBand(1).ReadAsArray()
                            array_temp[array_temp == -1] = 0
                            if np.isnan(nodata_value):
                                array_temp[np.isnan(array_temp)] = 0
                            else:
                                array_temp[array_temp == nodata_value] = 0

                        sm_temp = sm.csr_matrix(array_temp.astype(float))
                        array_temp = None
                        data_cube.append(sm_temp, name=para_list[i])
                        print(f'Assemble the {str(para_list[i])} into the Phemetric_datacube using {str(time.time() - t1)[0:5]}s (layer {str(i)} of {str(len(para_list))})')
                        i += 1
                    except:
                        error_inf = traceback.format_exc()
                        print(error_inf)

                # Save the sdc
                np.save(f'{yearly_output_path}paraname.npy', para_list)
                data_cube.save(f'{yearly_output_path}{index}_Phemetric_datacube\\')
                nodata_value = 0
            else:
                pass
        else:
            i = 0
            data_cube = np.zeros([rows, cols, len(para_list)])
            while i < len(para_list):

                try:
                    t1 = time.time()
                    if not os.path.exists(f"{input_path}{str(para_list[i])}.TIF"):
                        raise Exception(f'The {input_path}{str(para_list[i])} is not properly generated!')
                    else:
                        array_temp = gdal.Open(f"{input_path}{str(para_list[i])}.TIF")
                        array_temp = array_temp.GetRasterBand(1).ReadAsArray()
                        array_temp[array_temp == -1] = nodata_value

                    data_cube[:, :, i] = array_temp
                    print(f'Assemble the {str(para_list[i])} into the Phemetric_datacube using {str(time.time() - t1)[0:5]}s (layer {str(i)} of {str(len(para_list))})')
                    i += 1
                except:
                    error_inf = traceback.format_exc()
                    print(error_inf)

            np.save(f'{yearly_output_path}paraname.npy', para_list)
            np.save(f'{yearly_output_path}{index}_Phemetric_datacube.npy', data_cube)

        # Save the metadata dic
        metadata_dic['pheyear'], metadata_dic['Nodata_value'] = year, nodata_value
        with open(f'{yearly_output_path}metadata.json', 'w') as js_temp:
            json.dump(metadata_dic, js_temp)

    print(f"Finish constructing the {str(year)} {index} Phemetric datacube of {metadata_dic['ROI_name']} in \033[1;31m{str(time.time() - start_time)} s\033[0m.")


def process_itr_wl(water_level_: list, veg_arr: np.ndarray, output_file: str):

    if not os.path.exists(output_file):
        inund_arr = np.zeros_like(veg_arr, dtype=np.int16)
        water_level_unique = list(set(water_level_))
        for _ in water_level_unique:
            inund_arr = inund_arr + (_ > veg_arr).astype(np.int16) * water_level_.count(_)

        np.save(output_file, inund_arr)


def retrieve_correct_filename(file_name: str):
    file_list = file_name.split('\\')
    sep = '\\'
    if len(file_list) == 1:
        raise Exception('filename cannot be a folder!')
    else:
        for _ in range(1, len(file_list)):
            folder = [__ for __ in file_list[: _]]
            folder_name = sep.join(folder) + sep
            up_folder = [__ for __ in file_list[: _ - 1]]
            up_folder_name = sep.join(up_folder) + sep
            if os.path.exists(folder_name):
                pass
            else:
                dir_all = os.listdir(up_folder_name)
                dir_all = [up_folder_name + __ for __ in dir_all]
                dir_all_ = copy.copy(dir_all)
                try:
                    for __ in dir_all:
                        if os.path.isdir(__):
                            dir_all_.extend([os.path.join(__, file) for file in os.listdir(__)])
                except:
                    pass

                new_path = None
                file_temp = [__ for __ in file_list[_:]]
                for file_ in file_temp:
                    for dir_ in dir_all_:
                        if dir_.endswith(file_) and '.' in file_:
                            return os.path.join(dir_, file_)
                        elif dir_.endswith(file_) and '.' not in file_:
                            file_name = os.path.join(dir_, sep.join(file_temp[file_temp.index(file_) + 1:]))
                            if os.path.isfile(file_name):
                                return file_name
                        else:
                            pass
                return None


def link_GEDI_Phedc_inform(Phemedc, Pheme_year_list, Pheme_index, Phemedc_GeoTransform, gedi_df, furname,
                           GEDI_link_Pheme_spatial_interpolate_method_list, GEDI_circle_diameter: int = 25):

    df_size = gedi_df.shape[0]
    furlat, furlon = furname + '_' + 'lat', furname + '_' + 'lon'
    for spatial_method in GEDI_link_Pheme_spatial_interpolate_method_list:
        gedi_df.insert(loc=len(gedi_df.columns), column=f'Pheme_{Pheme_index}_{spatial_method}', value=np.nan)
        gedi_df.insert(loc=len(gedi_df.columns), column=f'Pheme_{Pheme_index}_{spatial_method}_reliability', value=np.nan)
    gedi_df = gedi_df.reset_index()
    sparse_matrix = True if isinstance(Phemedc, NDSparseMatrix) else False

    # itr through the gedi_df
    for i in range(df_size):
        # Timing
        t1 = time.time()
        print(f'Start linking the Pheme {Pheme_index} value with the GEDI dataframe!({str(i)} of {str(df_size)})')
        try:
            lat, lon, date_temp, year_temp = gedi_df[furlat][i], gedi_df[furlon][i], gedi_df['Date'][i], int(np.floor(gedi_df['Date'][i]/1000))
            point_coords = [lon, lat]

            if year_temp in Pheme_year_list:
                centre_pixel_xy = [int(np.floor((point_coords[0] - Phemedc_GeoTransform[0]) / Phemedc_GeoTransform[1])),
                                   int(np.floor((point_coords[1] - Phemedc_GeoTransform[3]) / Phemedc_GeoTransform[5]))]
                if 0 <= centre_pixel_xy[1] <= Phemedc.shape[0] and 0 <= centre_pixel_xy[0] <= Phemedc.shape[1]:
                    spatial_dic, spatial_weight = {}, {}
                    for spatial_method in GEDI_link_Pheme_spatial_interpolate_method_list:
                        if spatial_method == 'nearest_neighbor':
                            spatial_dic['nearest_neighbor'] = [centre_pixel_xy[1], centre_pixel_xy[1] + 1,
                                                               centre_pixel_xy[0], centre_pixel_xy[0] + 1]
                            spatial_weight['nearest_neighbor'] = np.ones([1, 1])
                        elif spatial_method == 'focal':
                            spatial_dic['focal'] = [max(centre_pixel_xy[1] - 1, 0), min(centre_pixel_xy[1] + 2, Phemedc.shape[0]),
                                                    max(centre_pixel_xy[0] - 1, 0), min(centre_pixel_xy[0] + 2, Phemedc.shape[1])]
                            spatial_weight['focal'] = np.ones([3, 3]) / 9
                        elif spatial_method == 'area_average':
                            GEDI_circle = create_circle_polygon(point_coords, GEDI_circle_diameter)
                            min_x, min_y, max_x, max_y = GEDI_circle.bounds
                            ul_corner = [int(np.floor((min_x - Phemedc_GeoTransform[0]) / Phemedc_GeoTransform[1])),
                                         int(np.floor((max_y - Phemedc_GeoTransform[3]) / Phemedc_GeoTransform[5]))]
                            lr_corner = [int(np.ceil((max_x - Phemedc_GeoTransform[0]) / Phemedc_GeoTransform[1])),
                                         int(np.ceil((min_y - Phemedc_GeoTransform[3]) / Phemedc_GeoTransform[5]))]
                            spatial_dic['area_average'] = [max(ul_corner[1], 0), min(lr_corner[1] + 1, Phemedc.shape[0]),
                                                           max(ul_corner[0], 0), min(lr_corner[0] + 1, Phemedc.shape[1])]
                            spatial_weight['area_average'] = np.zeros([spatial_dic['area_average'][1] - spatial_dic['area_average'][0],
                                                                       spatial_dic['area_average'][3] - spatial_dic['area_average'][2]])
                            entire_area = GEDI_circle.intersection(shapely.geometry.Polygon([(Phemedc_GeoTransform[0], Phemedc_GeoTransform[3]),
                                                                                             (Phemedc_GeoTransform[0] + Phemedc_GeoTransform[1] * Phemedc.shape[1], Phemedc_GeoTransform[3]),
                                                                                             (Phemedc_GeoTransform[0] + Phemedc_GeoTransform[1] * Phemedc.shape[1],  Phemedc_GeoTransform[3] + Phemedc_GeoTransform[5] * Phemedc.shape[0]),
                                                                                             (Phemedc_GeoTransform[0], Phemedc_GeoTransform[3] + Phemedc_GeoTransform[5] * Phemedc.shape[0])])).area
                            for y_ in range(spatial_weight['area_average'].shape[0]):
                                for x_ in range(spatial_weight['area_average'].shape[1]):
                                    square_ulcorner_x, square_ulcorner_y = Phemedc_GeoTransform[0] + Phemedc_GeoTransform[1] * (ul_corner[0] + x_),\
                                                                           Phemedc_GeoTransform[3] + Phemedc_GeoTransform[5] * (ul_corner[1] + y_)
                                    square_polygon = shapely.geometry.Polygon([(square_ulcorner_x, square_ulcorner_y),
                                                                               (square_ulcorner_x + Phemedc_GeoTransform[1], square_ulcorner_y),
                                                                               (square_ulcorner_x + Phemedc_GeoTransform[1], square_ulcorner_y + Phemedc_GeoTransform[5]),
                                                                               (square_ulcorner_x, square_ulcorner_y + Phemedc_GeoTransform[5])])
                                    intersection = square_polygon.intersection(GEDI_circle)
                                    spatial_weight['area_average'][y_, x_] = intersection.area / entire_area
                        else:
                            raise Exception('The spatial interpolation method is not supported!')
                else:
                    spatial_dic = None

                # Get the maximum range in spatial and temporal
                if spatial_dic is None:
                    pass
                else:
                    max_spatial_range = [Phemedc.shape[0], 0, Phemedc.shape[1], 0]
                    for spatial_ in spatial_dic.keys():
                        max_spatial_range = [min(max_spatial_range[0], spatial_dic[spatial_][0]),
                                             max(max_spatial_range[1], spatial_dic[spatial_][1]),
                                             min(max_spatial_range[2], spatial_dic[spatial_][2]),
                                             max(max_spatial_range[3], spatial_dic[spatial_][3])]
                    for spatial_ in spatial_dic.keys():
                        spatial_dic[spatial_] = [spatial_dic[spatial_][0] - max_spatial_range[0],
                                                 spatial_dic[spatial_][1] - max_spatial_range[0],
                                                 spatial_dic[spatial_][2] - max_spatial_range[2],
                                                 spatial_dic[spatial_][3] - max_spatial_range[2]]

                # Extract the RSdc
                if spatial_dic is None:
                    RSdc_temp = None
                elif isinstance(Phemedc, NDSparseMatrix):
                    RSdc_temp = Phemedc[max_spatial_range[0]: max_spatial_range[1], max_spatial_range[2]: max_spatial_range[3], Pheme_year_list.index(year_temp): Pheme_year_list.index(year_temp) + 1].reshape(max_spatial_range[1]-max_spatial_range[0], max_spatial_range[3]-max_spatial_range[2])
                    RSdc_temp = RSdc_temp.astype(np.float32)
                    RSdc_temp[RSdc_temp == 0] = np.nan
                elif isinstance(Phemedc, np.ndarray):
                    RSdc_temp = Phemedc[max_spatial_range[0]: max_spatial_range[1], max_spatial_range[2]: max_spatial_range[3], Pheme_year_list.index(year_temp)].reshape(max_spatial_range[1]-max_spatial_range[0], max_spatial_range[3]-max_spatial_range[2])
                else:
                    raise Exception('The RSdc is not under the right type!')

                if RSdc_temp is not None and ~np.all(np.isnan(RSdc_temp)):
                    for spatial_method in GEDI_link_Pheme_spatial_interpolate_method_list:
                        spatial_temp = spatial_dic[spatial_method]
                        dc_temp = RSdc_temp[spatial_temp[0]: spatial_temp[1], spatial_temp[2]: spatial_temp[3]]
                        if ~np.all(np.isnan(RSdc_temp)):
                            inform_value = np.nansum(spatial_weight[spatial_method] * dc_temp.reshape(dc_temp.shape[0], dc_temp.shape[1]))
                            reliability_value = np.nansum(spatial_weight[spatial_method] * ~np.isnan(dc_temp[:, :].reshape(dc_temp.shape[0], dc_temp.shape[1])))
                            inform_value = inform_value / reliability_value if reliability_value != 0 else np.nan
                        else:
                            inform_value = np.nan
                            reliability_value = np.nan
                        gedi_df.loc[i, f'Pheme_{Pheme_index}_{spatial_method}'] = inform_value
                        gedi_df.loc[i, f'Pheme_{Pheme_index}_{spatial_method}_reliability'] = reliability_value
                    print(f'Finish linking the Pheme {Pheme_index} value with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')
                else:
                    print(f'Invalid value for {Pheme_index} linking with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')
            else:
                print(f'Unsupported year not import in the Pheme dc! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')
        except:
            print(traceback.format_exc())
            print(f'Failed linking the Pheme {Pheme_index} value with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')

    return gedi_df


def link_GEDI_Denvdc_inform(Denvdc, Denv_index, Denv_doy_list, Denvdc_GeoTransform, Phemedc, gedi_df, furname,
                            GEDI_link_Denvdc_spatial_interpolate_method_list: list, Denv_factor, GEDI_circle_diameter: int = 25):

    df_size = gedi_df.shape[0]
    furlat, furlon = furname + '_' + 'lat', furname + '_' + 'lon'
    denv_nodata_value, denv_zoffset, denv_size_control = Denv_factor
    for spatial_method in GEDI_link_Denvdc_spatial_interpolate_method_list:
        if Denv_index in ['DPAR', 'AGB', 'PRE', 'TEM']:
            for measure in ['fast-growing', 'maturity', 'GEDI-acq', 'ratio_gedi_fast-growing']:
                gedi_df.insert(loc=len(gedi_df.columns), column=f'Denv_{Denv_index}_{spatial_method}_{measure}', value=np.nan)

        elif Denv_index in ['WIN', 'PRS', 'RHU']:
            for measure in ['max', 'min', 'mean', 'std']:
                gedi_df.insert(loc=len(gedi_df.columns), column=f'Denv_{Denv_index}_{spatial_method}_{measure}', value=np.nan)

        gedi_df.insert(loc=len(gedi_df.columns), column=f'Denv_{Denv_index}_{spatial_method}_reliability', value=np.nan)

    sparse_matrix = True if isinstance(Denvdc, NDSparseMatrix) else False
    gedi_df = gedi_df.reset_index()

    # itr through the gedi_df
    for i in range(df_size):

        # Timing
        t1 = time.time()
        print(f'Start linking the Denv {str(Denv_index)} value with the GEDI dataframe!({str(i)} of {str(df_size)})')

        try:
            lat, lon, year_temp, GEDI_doy = gedi_df[furlat][i], gedi_df[furlon][i], int(np.floor(gedi_df['Date'][i]/1000)), gedi_df['Date'][i]
            point_coords = [lon, lat]

            # Process the value for each method
            centre_pixel_xy = [int(np.floor((point_coords[0] - Denvdc_GeoTransform[0]) / Denvdc_GeoTransform[1])),
                               int(np.floor((point_coords[1] - Denvdc_GeoTransform[3]) / Denvdc_GeoTransform[5]))]
            if 0 <= centre_pixel_xy[1] <= Denvdc.shape[0] and 0 <= centre_pixel_xy[0] <= Denvdc.shape[1]:

                # Get the spatial range for each method
                spatial_dic, spatial_weight = {}, {}
                for spatial_method in GEDI_link_Denvdc_spatial_interpolate_method_list:
                    if spatial_method == 'nearest_neighbor':
                        spatial_dic['nearest_neighbor'] = [centre_pixel_xy[1], centre_pixel_xy[1] + 1,
                                                           centre_pixel_xy[0], centre_pixel_xy[0] + 1]
                        spatial_weight['nearest_neighbor'] = np.ones([1, 1])
                    elif spatial_method == 'focal':
                        spatial_dic['focal'] = [max(centre_pixel_xy[1] - 1, 0),
                                                min(centre_pixel_xy[1] + 2, Denvdc.shape[0]),
                                                max(centre_pixel_xy[0] - 1, 0),
                                                min(centre_pixel_xy[0] + 2, Denvdc.shape[1])]
                        x_size = spatial_dic['focal'][3] - spatial_dic['focal'][2]
                        y_size = spatial_dic['focal'][1] - spatial_dic['focal'][0]
                        spatial_weight['focal'] = np.ones([y_size, x_size]) / (y_size * x_size)
                    elif spatial_method == 'area_average':
                        GEDI_circle = create_circle_polygon(point_coords, GEDI_circle_diameter)
                        min_x, min_y, max_x, max_y = GEDI_circle.bounds
                        ul_corner = [int(np.floor((min_x - Denvdc_GeoTransform[0]) / Denvdc_GeoTransform[1])),
                                     int(np.floor((max_y - Denvdc_GeoTransform[3]) / Denvdc_GeoTransform[5]))]
                        lr_corner = [int(np.ceil((max_x - Denvdc_GeoTransform[0]) / Denvdc_GeoTransform[1])),
                                     int(np.ceil((min_y - Denvdc_GeoTransform[3]) / Denvdc_GeoTransform[5]))]
                        spatial_dic['area_average'] = [max(ul_corner[1], 0), min(lr_corner[1] + 1, Denvdc.shape[0]),
                                                       max(ul_corner[0], 0), min(lr_corner[0] + 1, Denvdc.shape[1])]
                        spatial_weight['area_average'] = np.zeros(
                            [spatial_dic['area_average'][1] - spatial_dic['area_average'][0],
                             spatial_dic['area_average'][3] - spatial_dic['area_average'][2]])
                        entire_area = GEDI_circle.intersection(shapely.geometry.Polygon([(Denvdc_GeoTransform[0], Denvdc_GeoTransform[3]),
                                                                                         (Denvdc_GeoTransform[0] + Denvdc_GeoTransform[1] * Denvdc.shape[1], Denvdc_GeoTransform[3]),
                                                                                         (Denvdc_GeoTransform[0] + Denvdc_GeoTransform[1] * Denvdc.shape[1], Denvdc_GeoTransform[3] + Denvdc_GeoTransform[5] * Denvdc.shape[0]),
                                                                                         (Denvdc_GeoTransform[0], Denvdc_GeoTransform[3] + Denvdc_GeoTransform[5] * Denvdc.shape[0])])).area
                        for y_ in range(spatial_weight['area_average'].shape[0]):
                            for x_ in range(spatial_weight['area_average'].shape[1]):
                                square_ulcorner_x, square_ulcorner_y = (Denvdc_GeoTransform[0] + Denvdc_GeoTransform[1] * (ul_corner[0] + x_),
                                                                        Denvdc_GeoTransform[3] + Denvdc_GeoTransform[5] * (ul_corner[1] + y_))
                                square_polygon = shapely.geometry.Polygon([(square_ulcorner_x, square_ulcorner_y), (square_ulcorner_x + Denvdc_GeoTransform[1], square_ulcorner_y),
                                                                           (square_ulcorner_x + Denvdc_GeoTransform[1], square_ulcorner_y + Denvdc_GeoTransform[5]),
                                                                           (square_ulcorner_x, square_ulcorner_y + Denvdc_GeoTransform[5])])
                                intersection = square_polygon.intersection(GEDI_circle)
                                spatial_weight['area_average'][y_, x_] = intersection.area / entire_area
                    else:
                        raise Exception('The spatial interpolation method is not supported!')

                # Get the maximum range in spatial
                max_spatial_range = [Denvdc.shape[0], 0, Denvdc.shape[1], 0]
                for spatial_ in spatial_dic.keys():
                    max_spatial_range = [min(max_spatial_range[0], spatial_dic[spatial_][0]), max(max_spatial_range[1], spatial_dic[spatial_][1]),
                                         min(max_spatial_range[2], spatial_dic[spatial_][2]), max(max_spatial_range[3], spatial_dic[spatial_][3])]
                    spatial_dic[spatial_] = [spatial_dic[spatial_][0] - max_spatial_range[0], spatial_dic[spatial_][1] - max_spatial_range[0],
                                             spatial_dic[spatial_][2] - max_spatial_range[2], spatial_dic[spatial_][3] - max_spatial_range[2]]

                # Get the key pheme metrics
                phe_dic = {}
                for _ in ['peak_doy', 'SOS', 'EOM']:
                    # Extract the Pheme datacube
                    if f'{str(year_temp)}_{_}' in Phemedc.SM_namelist:
                        if isinstance(Denvdc, NDSparseMatrix):
                            phe_arr = Phemedc.SM_group[f'{str(year_temp)}_{_}'][max_spatial_range[0]: max_spatial_range[1],
                                      max_spatial_range[2]: max_spatial_range[3]].toarray()
                            phe_arr = phe_arr.astype(np.float32)
                            phe_arr[phe_arr == 0] = np.nan
                        else:
                            pass
                    else:
                        raise Exception(f'The {_} is not in the Pheme datacube!')

                    for __ in GEDI_link_Denvdc_spatial_interpolate_method_list:
                        phe_arr_method = phe_arr[spatial_dic[__][0]: spatial_dic[__][1], spatial_dic[__][2]: spatial_dic[__][3]]
                        if np.isnan(phe_arr_method).all():
                            phe_dic[f'{__}_{_}'] = np.nan
                        else:
                            phe_dic[f'{__}_{_}'] = np.nansum(phe_arr_method * spatial_weight[__]) / np.nansum(~np.isnan(phe_arr_method) * spatial_weight[__])

                # gET THE TEMPORAL RANGE
                lower_range, upper_range = np.argwhere(np.array(Denv_doy_list) > year_temp * 1000).min(), np.argwhere(np.array(Denv_doy_list) < year_temp * 1000 + 1000).max()
                doy_list = Denv_doy_list[lower_range: upper_range + 1]

                # Extract the Denvdc
                if spatial_dic is None:
                    Denvdc_temp = None
                elif isinstance(Denvdc, NDSparseMatrix):
                    Denvdc_temp = Denvdc[max_spatial_range[0]: max_spatial_range[1], max_spatial_range[2]: max_spatial_range[3], lower_range: upper_range + 1]
                    Denvdc_temp = Denvdc_temp.astype(np.float32)
                    Denvdc_temp[Denvdc_temp == 0] = np.nan
                elif isinstance(Denvdc, np.ndarray):
                    Denvdc_temp = Denvdc[max_spatial_range[0]: max_spatial_range[1], max_spatial_range[2]: max_spatial_range[3], lower_range: upper_range + 1]
                else:
                    raise Exception('The Denvdc is not under the right type!')

                if Denvdc_temp is not None and ~np.all(np.isnan(Denvdc_temp)):
                    for spatial_method in GEDI_link_Denvdc_spatial_interpolate_method_list:
                        rs_arr_spatial = Denvdc_temp[spatial_dic[spatial_method][0]: spatial_dic[spatial_method][1], spatial_dic[spatial_method][2]: spatial_dic[spatial_method][3], :]
                        if denv_zoffset is not None:
                            rs_arr_spatial = rs_arr_spatial - denv_zoffset
                        if denv_size_control:
                            rs_arr_spatial = rs_arr_spatial / 100
                        spatial_weight_arr = np.broadcast_to(spatial_weight[spatial_method][:, :, np.newaxis], (spatial_weight[spatial_method].shape[0], spatial_weight[spatial_method].shape[1], Denvdc_temp.shape[2]))
                        reliability_arr = np.nansum(~np.isnan(rs_arr_spatial) * spatial_weight_arr, axis=(0, 1))
                        inform_arr = np.nansum(rs_arr_spatial * spatial_weight_arr, axis=(0, 1)) / reliability_arr
                        gedi_df.loc[i, f'Denv_{Denv_index}_{spatial_method}_reliability'] = np.nanmean(reliability_arr)

                        if np.isnan(inform_arr).all():
                            pass
                        elif Denv_index in ['SSD', 'DPAR', 'AGB', 'PRE', 'TEM']:
                            thr = [0, 0,0,0,15][['DPAR', 'AGB', 'PRE', 'TEM'].index(Denv_index)]
                            for measure in ['fast-growing', 'maturity', 'GEDI-acq', 'ratio_gedi_fast-growing']:
                                if measure == 'fast-growing' and ~np.isnan(phe_dic[f'{spatial_method}_peak_doy']):
                                    inform_arr_ = inform_arr[0: np.argwhere(np.array(doy_list) > year_temp * 1000 + phe_dic[f'{spatial_method}_peak_doy']).min()] - thr
                                    inform_arr_[inform_arr_ < 0] = 0
                                    inform_value = np.nansum(inform_arr_)
                                elif measure == 'maturity' and ~np.isnan(phe_dic[f'{spatial_method}_EOM']):
                                    inform_arr_ = inform_arr[0: np.argwhere(np.array(doy_list) > year_temp * 1000 + phe_dic[f'{spatial_method}_EOM']).min()] - thr
                                    inform_arr_[inform_arr_ < 0] = 0
                                    inform_value = np.nansum(inform_arr_)
                                elif measure == 'GEDI-acq':
                                    inform_arr_ = inform_arr[0: np.argwhere(np.array(doy_list) <= GEDI_doy).max() + 1] - thr
                                    inform_arr_[inform_arr_ < 0] = 0
                                    inform_value = np.nansum(inform_arr_)
                                elif measure == 'ratio_gedi_fast-growing' and ~np.isnan(phe_dic[f'{spatial_method}_peak_doy']):
                                    inform_arr_ = inform_arr[0: np.argwhere(np.array(doy_list) > year_temp * 1000 + phe_dic[f'{spatial_method}_peak_doy']).min()] - thr
                                    inform_arr_[inform_arr_ < 0] = 0
                                    fs_value = np.nansum(inform_arr_)
                                    inform_arr_ = inform_arr[0: np.argwhere(np.array(doy_list) <= GEDI_doy).max() + 1] - thr
                                    inform_arr_[inform_arr_ < 0] = 0
                                    gedi_value = np.nansum(inform_arr_)
                                    inform_value = gedi_value / fs_value
                                else:
                                    inform_value = np.nan
                                gedi_df.loc[i, f'Denv_{Denv_index}_{spatial_method}_{measure}'] = inform_value
                        elif Denv_index in ['WIN', 'PRS', 'RHU']:
                            inform_arr_ = inform_arr[0: np.argwhere(np.array(doy_list) > GEDI_doy).min()]
                            for measure in ['max', 'min', 'mean', 'std']:
                                if measure == 'max':
                                    inform_value = np.nanmax(inform_arr_)
                                elif measure == 'min':
                                    inform_value = np.nanmin(inform_arr_)
                                elif measure == 'mean':
                                    inform_value = np.nanmean(inform_arr_)
                                else:
                                    inform_value = np.nanstd(inform_arr_)
                                gedi_df.loc[i, f'Denv_{Denv_index}_{spatial_method}_{measure}'] = inform_value
                        else:
                            raise Exception('Unsupport denv index')

                    # elif Denv_index in ['WIN', 'PRS', 'RHU']:
                    # for measure in ['fast-growing', 'maturity', 'GEDI-acq', 'ratio_gedi_fast-growing']:
                    #     for measure in ['max', 'min', 'mean', 'std']:
                    #         pass

                    # for accumulated_period_ in accumulated_period:
                    #     temporal_dc = temporal_dic[f'{accumulated_period_}_dc']
                    #     Denvdc_ = Denvdc_temp * temporal_dc
                    #     Denvdc_[Denvdc_ < thr_dc] = np.nan
                    #     Denvdc_ = np.nanmean(Denvdc_, axis=2)
                    #     for spatial_method in GEDI_link_Denvdc_spatial_interpolate_method_list:
                    #         spatial_temp = spatial_dic[spatial_method]
                    #         dc_temp = RSdc_[spatial_temp[0]: spatial_temp[1], spatial_temp[2]: spatial_temp[3]]
                    #         inform_value, reliability_value = np.nan, np.nan
                    #
                    #         if ~np.all(np.isnan(dc_temp)):
                    #             inform_value = np.nansum(spatial_weight[spatial_method] * dc_temp)
                    #             reliability_value = np.nansum(spatial_weight[spatial_method] * ~np.isnan(dc_temp))
                    #             inform_value = inform_value / reliability_value if reliability_value != 0 else np.nan
                    #             if reliability_value == 0:
                    #                 reliability_value = np.nan
                    #         else:
                    #             inform_value = np.nan
                    #             reliability_value = np.nan
                    #
                    #         if ~np.isnan(inform_value):
                    #             gedi_df.loc[i, f'Denv_{Denv_index}_{spatial_method}_{accumulated_period_}_{threshold_method[0]}'] = inform_value
                    #             gedi_df.loc[i, f'Denv_{Denv_index}_{spatial_method}_{accumulated_period_}_{threshold_method[0]}_reliability'] = reliability_value

                print(f'Finish linking the Denv {Denv_index} value with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')
            else:
                print(f'Invalid value for {Denv_index} linking with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')
            
        except:
            print(traceback.format_exc())
            print(f'Failed linking the Denv {Denv_index} value with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')
    return gedi_df


def link_GEDI_VegType_inform(VegType_arr, VegType_GeoTransform, gedi_df, furname, GEDI_link_VegType_spatial_interpolate_method_list: list, GEDI_circle_diameter: int = 25):

    df_size = gedi_df.shape[0]
    furlat, furlon = furname + '_' + 'lat', furname + '_' + 'lon'
    gedi_df = gedi_df.reset_index()

    # Insert new column for GEDI_df based on spatial methods and temporal methods
    for spatial_method in GEDI_link_VegType_spatial_interpolate_method_list:
        gedi_df.insert(loc=len(gedi_df.columns), column=f'VegType_{spatial_method}', value=np.nan)
        gedi_df.insert(loc=len(gedi_df.columns), column=f'VegType_{spatial_method}_reliability', value=np.nan)

    # itr through the gedi_df
    for i in range(df_size):

        # Timing
        t1 = time.time()
        print(f'Start linking the VegType value with the GEDI dataframe!({str(i)} of {str(df_size)})')
        try:
            # Draw a circle around the central point
            lat, lon, GEDI_doy = gedi_df[furlat][i], gedi_df[furlon][i], gedi_df['Date'][i]
            point_coords = [lon, lat]

            # Process the spatial range
            centre_pixel_xy = [int(np.floor((point_coords[0] - VegType_GeoTransform[0]) / VegType_GeoTransform[1])),
                               int(np.floor((point_coords[1] - VegType_GeoTransform[3]) / VegType_GeoTransform[5]))]

            if 0 <= centre_pixel_xy[1] <= VegType_arr.shape[0] and 0 <= centre_pixel_xy[0] <= VegType_arr.shape[1]:
                for spatial_method in GEDI_link_VegType_spatial_interpolate_method_list:
                    if spatial_method == 'nearest_neighbor':
                        type_value = VegType_arr[centre_pixel_xy[1]: centre_pixel_xy[1] + 1, centre_pixel_xy[0]: centre_pixel_xy[0] + 1]
                        type_value = np.nan if type_value == 0 else type_value
                    elif spatial_method == 'focal':
                        type_value = VegType_arr[max(centre_pixel_xy[1] - 1, 0): min(centre_pixel_xy[1] + 2, VegType_arr.shape[0]), max(centre_pixel_xy[0] - 1, 0): min(centre_pixel_xy[0] + 2, VegType_arr.shape[1])]
                        type_value = type_value.flatten()
                        type_value = type_value[type_value != 0]
                        if type_value.size == 0:
                            type_value = np.nan
                        else:
                            type_value, _ = stats.mode(type_value)
                    elif spatial_method == 'area_average':
                        GEDI_circle = create_circle_polygon(point_coords, GEDI_circle_diameter)
                        min_x, min_y, max_x, max_y = GEDI_circle.bounds
                        ul_corner = [int(np.floor((min_x - VegType_GeoTransform[0]) / VegType_GeoTransform[1])),
                                     int(np.floor((max_y - VegType_GeoTransform[3]) / VegType_GeoTransform[5]))]
                        lr_corner = [int(np.ceil((max_x - VegType_GeoTransform[0]) / VegType_GeoTransform[1])),
                                     int(np.ceil((min_y - VegType_GeoTransform[3]) / VegType_GeoTransform[5]))]
                        type_value = VegType_arr[max(ul_corner[1], 0): min(lr_corner[1] + 1, VegType_arr.shape[0]), max(ul_corner[0], 0): min(lr_corner[0] + 1, VegType_arr.shape[1])]
                        spatial_weight = np.zeros_like(type_value)
                        entire_area = GEDI_circle.intersection(shapely.geometry.Polygon([(VegType_GeoTransform[0], VegType_GeoTransform[3]),
                                                                                         (VegType_GeoTransform[0] + VegType_GeoTransform[1] * VegType_arr.shape[1], VegType_GeoTransform[3]),
                                                                                         (VegType_GeoTransform[0] + VegType_GeoTransform[1] * VegType_arr.shape[1],  VegType_GeoTransform[3] + VegType_GeoTransform[5] * VegType_arr.shape[0]),
                                                                                         (VegType_GeoTransform[0], VegType_GeoTransform[3] + VegType_GeoTransform[5] * VegType_arr.shape[0])])).area

                        weight_dic = {}
                        for y_ in range(type_value.shape[0]):
                            for x_ in range(type_value.shape[1]):
                                square_ulcorner_x, square_ulcorner_y = (VegType_GeoTransform[0] + VegType_GeoTransform[1] * (ul_corner[0] + x_), VegType_GeoTransform[3] + VegType_GeoTransform[5] * (ul_corner[1] + y_))
                                square_polygon = shapely.geometry.Polygon([(square_ulcorner_x, square_ulcorner_y), (square_ulcorner_x + VegType_GeoTransform[1], square_ulcorner_y),
                                                                           (square_ulcorner_x + VegType_GeoTransform[1], square_ulcorner_y + VegType_GeoTransform[5]),
                                                                           (square_ulcorner_x, square_ulcorner_y + VegType_GeoTransform[5])])
                                intersection = square_polygon.intersection(GEDI_circle)
                                if type_value[y_, x_] not in weight_dic.keys():
                                    weight_dic[type_value[y_, x_]] = intersection.area / entire_area
                                else:
                                    weight_dic[type_value[y_, x_]] += intersection.area / entire_area

                        max_weight, type_value = 0, 0
                        for _ in weight_dic.keys():
                            if weight_dic[_] > max_weight:
                                max_weight = weight_dic[_]
                                type_value = _
                        type_value = np.nan if type_value == 0 else type_value

                    else:
                        raise Exception('The spatial interpolation method is not supported!')

                    if ~np.isnan(type_value):
                        gedi_df.loc[i, f'VegType_{spatial_method}'] = type_value
                        gedi_df.loc[i, f'VegType_{spatial_method}_reliability'] = 0

                print(f'Finish linking the Veg Type  with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')
        except:
            print(traceback.format_exc())
            print(f'Failed linking the Veg Type   with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')
    return gedi_df


def link_GEDI_RSdc_inform(RSdc, RSdc_GeoTransform, RSdc_doy_list, RSdc_index, Phemedc, gedi_df, furname,
                          GEDI_link_RS_spatial_interpolate_method_list, temporal_search_window: int = 96, GEDI_circle_diameter: int = 25):

    """
    This function is used to link GEDI and RSdc inform. It will draw a circle around the central point and extract the corresponding RSdc value using the given spatial and temporal interpolation methods.

    Parameters:
    RSdc (NDSparseMatrix/np.ndarray): The RSdc datacube
    RSdc_GeoTransform (list): The geotransform of the RSdc datacube
    RSdc_doy_list (list): The list of doy for the RSdc datacube
    RSdc_index (str): The index of the RSdc datacube
    gedi_df (pd.DataFrame): The GEDI dataframe
    furname (str): The name of the firmware
    GEDI_link_RS_spatial_interpolate_method_list (list): The list of spatial interpolation methods
    temporal_search_window (int): The search window for the temporal interpolation
    GEDI_circle_diameter (int): The diameter of the circle

    Returns:
    pd.DataFrame: The linked GEDI dataframe
    """
    df_size = gedi_df.shape[0]
    furlat, furlon = furname + '_' + 'lat', furname + '_' + 'lon'
    gedi_df = gedi_df.reset_index()
    temporal_search_window = 16 if temporal_search_window <= 16 else temporal_search_window

    # Insert new column for GEDI_df based on spatial methods and temporal methods
    for spatial_method in GEDI_link_RS_spatial_interpolate_method_list:
        for date in ['GEDI_acq', 'EOM', 'peak_doy']:
            gedi_df.insert(loc=len(gedi_df.columns), column=f'{RSdc_index}_{spatial_method}_{date}', value=np.nan)
            gedi_df.insert(loc=len(gedi_df.columns), column=f'{RSdc_index}_{spatial_method}_{date}_reliability', value=np.nan)

        for statistic_measure in ['growth_mean', 'growth_max', 'growth_min', 'growth_std']:
            gedi_df.insert(loc=len(gedi_df.columns), column=f'{RSdc_index}_{spatial_method}_{statistic_measure}',  value=np.nan)
            gedi_df.insert(loc=len(gedi_df.columns), column=f'{RSdc_index}_{spatial_method}_{statistic_measure}_reliability', value=np.nan)

    # itr through the gedi_df
    for i in range(df_size):

        # Timing
        t1 = time.time()
        print(f'Start linking the {RSdc_index} value with the GEDI dataframe!({str(i)} of {str(df_size)})')
        try:
            # Draw a circle around the central point
            lat, lon, GEDI_doy = gedi_df[furlat][i], gedi_df[furlon][i], gedi_df['Date'][i]
            point_coords = [lon, lat]
            year_temp = int(np.floor(gedi_df['Date'][i]/1000))

            # Process the value for each method
            centre_pixel_xy = [int(np.floor((point_coords[0] - RSdc_GeoTransform[0]) / RSdc_GeoTransform[1])), int(np.floor((point_coords[1] - RSdc_GeoTransform[3]) / RSdc_GeoTransform[5]))]
            if 0 <= centre_pixel_xy[1] <= RSdc.shape[0] and 0 <= centre_pixel_xy[0] <= RSdc.shape[1]:

                # Get the spatial range for each method
                spatial_dic, spatial_weight = {}, {}
                for spatial_method in GEDI_link_RS_spatial_interpolate_method_list:
                    if spatial_method == 'nearest_neighbor':
                        spatial_dic['nearest_neighbor'] = [centre_pixel_xy[1], centre_pixel_xy[1] + 1, centre_pixel_xy[0], centre_pixel_xy[0] + 1]
                        spatial_weight['nearest_neighbor'] = np.ones([1, 1])
                    elif spatial_method == 'focal':
                        spatial_dic['focal'] = [max(centre_pixel_xy[1] - 1, 0), min(centre_pixel_xy[1] + 2, RSdc.shape[0]), max(centre_pixel_xy[0] - 1, 0), min(centre_pixel_xy[0] + 2, RSdc.shape[1])]
                        x_size = spatial_dic['focal'][3] - spatial_dic['focal'][2]
                        y_size = spatial_dic['focal'][1] - spatial_dic['focal'][0]
                        spatial_weight['focal'] = np.ones([y_size, x_size]) / (y_size * x_size)
                    elif spatial_method == 'area_average':
                        GEDI_circle = create_circle_polygon(point_coords, GEDI_circle_diameter)
                        min_x, min_y, max_x, max_y = GEDI_circle.bounds
                        ul_corner = [int(np.floor((min_x - RSdc_GeoTransform[0]) / RSdc_GeoTransform[1])), int(np.floor((max_y - RSdc_GeoTransform[3]) / RSdc_GeoTransform[5]))]
                        lr_corner = [int(np.ceil((max_x - RSdc_GeoTransform[0]) / RSdc_GeoTransform[1])), int(np.ceil((min_y - RSdc_GeoTransform[3]) / RSdc_GeoTransform[5]))]
                        spatial_dic['area_average'] = [max(ul_corner[1], 0), min(lr_corner[1] + 1, RSdc.shape[0]), max(ul_corner[0], 0), min(lr_corner[0] + 1, RSdc.shape[1])]
                        spatial_weight['area_average'] = np.zeros([spatial_dic['area_average'][1] - spatial_dic['area_average'][0], spatial_dic['area_average'][3] - spatial_dic['area_average'][2]])
                        entire_area = GEDI_circle.intersection(shapely.geometry.Polygon([(RSdc_GeoTransform[0],  RSdc_GeoTransform[3]), (RSdc_GeoTransform[0] + RSdc_GeoTransform[1] * RSdc.shape[1],  RSdc_GeoTransform[3]),
                                                                                         (RSdc_GeoTransform[0] + RSdc_GeoTransform[1] * RSdc.shape[1],  RSdc_GeoTransform[3] + RSdc_GeoTransform[5] * RSdc.shape[0]), (RSdc_GeoTransform[0], RSdc_GeoTransform[3] + RSdc_GeoTransform[5] * RSdc.shape[0])])).area
                        for y_ in range(spatial_weight['area_average'].shape[0]):
                            for x_ in range(spatial_weight['area_average'].shape[1]):
                                square_ulcorner_x, square_ulcorner_y = RSdc_GeoTransform[0] + RSdc_GeoTransform[1] * (ul_corner[0] + x_), RSdc_GeoTransform[3] + RSdc_GeoTransform[5] * (ul_corner[1] + y_)
                                square_polygon = shapely.geometry.Polygon([(square_ulcorner_x, square_ulcorner_y), (square_ulcorner_x + RSdc_GeoTransform[1], square_ulcorner_y),
                                                                           (square_ulcorner_x + RSdc_GeoTransform[1], square_ulcorner_y + RSdc_GeoTransform[5]), (square_ulcorner_x, square_ulcorner_y + RSdc_GeoTransform[5])])
                                intersection = square_polygon.intersection(GEDI_circle)
                                spatial_weight['area_average'][y_, x_] = intersection.area / entire_area
                    else:
                        raise Exception('The spatial interpolation method is not supported!')

                # Get the maximum range in spatial
                max_spatial_range = [RSdc.shape[0], 0, RSdc.shape[1], 0]
                for spatial_ in spatial_dic.keys():
                    max_spatial_range = [min(max_spatial_range[0], spatial_dic[spatial_][0]), max(max_spatial_range[1], spatial_dic[spatial_][1]),
                                         min(max_spatial_range[2], spatial_dic[spatial_][2]), max(max_spatial_range[3], spatial_dic[spatial_][3])]
                for spatial_ in spatial_dic.keys():
                    spatial_dic[spatial_] = [spatial_dic[spatial_][0] - max_spatial_range[0], spatial_dic[spatial_][1] - max_spatial_range[0],
                                             spatial_dic[spatial_][2] - max_spatial_range[2], spatial_dic[spatial_][3] - max_spatial_range[2]]

                # Get the temporal range of this year
                lower_range, upper_range = np.argwhere(np.array(RSdc_doy_list) > year_temp * 1000).min(), np.argwhere(np.array(RSdc_doy_list) < year_temp * 1000 + 1000).max()
                doy_arr = np.array(RSdc_doy_list[lower_range: upper_range + 1])

                # Extract the RSdc
                if spatial_dic is None:
                    RSdc_temp = None
                elif isinstance(RSdc, NDSparseMatrix):
                    RSdc_temp = RSdc[max_spatial_range[0]: max_spatial_range[1], max_spatial_range[2]: max_spatial_range[3], lower_range: upper_range + 1]
                    RSdc_temp = RSdc_temp.astype(np.float32)
                    RSdc_temp[RSdc_temp == 0] = np.nan
                elif isinstance(RSdc, np.ndarray):
                    RSdc_temp = RSdc[max_spatial_range[0]: max_spatial_range[1], max_spatial_range[2]: max_spatial_range[3], lower_range: upper_range + 1]
                else:
                    raise Exception('The RSdc is not under the right type!')

                # Get the key pheme metrics
                phe_dic = {}
                for _ in ['peak_doy', 'SOS', 'EOS', 'EOM']:
                    # Extract the Pheme datacube
                    if f'{str(year_temp)}_{_}' in Phemedc.SM_namelist:
                        if isinstance(RSdc, NDSparseMatrix):
                            phe_arr = Phemedc.SM_group[f'{str(year_temp)}_{_}'][max_spatial_range[0]: max_spatial_range[1], max_spatial_range[2]: max_spatial_range[3]].toarray()
                            phe_arr = phe_arr.astype(np.float32)
                            phe_arr[phe_arr == 0] = np.nan
                        else:
                            pass
                    else:
                        raise Exception(f'The {_} is not in the Pheme datacube!')

                    for __ in GEDI_link_RS_spatial_interpolate_method_list:
                        phe_arr_method = phe_arr[spatial_dic[__][0]: spatial_dic[__][1], spatial_dic[__][2]: spatial_dic[__][3]]
                        if np.isnan(phe_arr_method).all():
                            phe_dic[f'{__}_{_}'] = np.nan
                        else:
                            phe_dic[f'{__}_{_}'] = np.nansum(phe_arr_method * spatial_weight[__]) / np.nansum(~np.isnan(phe_arr_method) * spatial_weight[__])

                # Execute the weight
                if RSdc_temp is not None and ~np.all(np.isnan(RSdc_temp)):
                    for spatial_method in GEDI_link_RS_spatial_interpolate_method_list:
                        rs_arr_spatial = RSdc_temp[spatial_dic[spatial_method][0]: spatial_dic[spatial_method][1], spatial_dic[spatial_method][2]: spatial_dic[spatial_method][3], :]
                        spatial_weight_arr = np.broadcast_to(spatial_weight[spatial_method][:, :, np.newaxis], (spatial_weight[spatial_method].shape[0], spatial_weight[spatial_method].shape[1], RSdc_temp.shape[2]))
                        reliability_arr = np.nansum(~np.isnan(rs_arr_spatial) * spatial_weight_arr, axis=(0, 1))
                        inform_arr = np.nansum(rs_arr_spatial * spatial_weight_arr, axis=(0, 1)) / reliability_arr

                        # Generate growth value
                        if ~np.isnan(phe_dic[f'{spatial_method}_SOS']) and ~np.isnan(phe_dic[f'{spatial_method}_EOS']):
                            lower_range, upper_range = np.argwhere(np.array(doy_arr) >= year_temp * 1000 + phe_dic[f'{spatial_method}_SOS']).min(), np.argwhere(np.array(doy_arr) < year_temp * 1000 + phe_dic[f'{spatial_method}_EOS']).max()

                            for statistic_measure in ['growth_mean', 'growth_max', 'growth_min', 'growth_std']:
                                if statistic_measure == 'growth_mean':
                                    inform_value = np.nanmean(inform_arr[lower_range: upper_range + 1])
                                elif statistic_measure == 'growth_max':
                                    inform_value = np.nanmax(inform_arr[lower_range: upper_range + 1])
                                elif statistic_measure == 'growth_min':
                                    inform_value = np.nanmin(inform_arr[lower_range: upper_range + 1])
                                elif statistic_measure == 'growth_std':
                                    inform_value = np.nanstd(inform_arr[lower_range: upper_range + 1])
                                else:
                                    raise Exception('Not supported statistic measure!')

                                if ~np.isnan(inform_value):
                                    gedi_df.loc[i, f'{RSdc_index}_{spatial_method}_{statistic_measure}'] = inform_value
                                    gedi_df.loc[i, f'{RSdc_index}_{spatial_method}_{statistic_measure}_reliability'] = np.nanmean(reliability_arr[lower_range: upper_range + 1])
                                else:
                                    gedi_df.loc[i, f'{RSdc_index}_{spatial_method}_{statistic_measure}'] = np.nan
                                    gedi_df.loc[i, f'{RSdc_index}_{spatial_method}_{statistic_measure}_reliability'] = 0
                        else:
                            gedi_df.loc[i, f'{RSdc_index}_{spatial_method}_growth_mean'] = np.nan
                            gedi_df.loc[i, f'{RSdc_index}_{spatial_method}_growth_mean_reliability'] = 0

                        # Generate interpolated value
                        for date in ['GEDI_acq', 'EOM', 'peak_doy']:
                            if date == 'GEDI_acq':
                                doy_ = GEDI_doy
                            else:
                                doy_ = phe_dic[f'{spatial_method}_{date}'] + year_temp * 1000
                                print(str(doy_))

                            if doy_ in doy_arr and ~np.isnan(inform_arr[np.argwhere(doy_arr == doy_)[0][0]]):
                                gedi_df.loc[i, f'{RSdc_index}_{spatial_method}_{date}'] = inform_arr[np.argwhere(doy_arr == doy_)[0][0]]
                                gedi_df.loc[i, f'{RSdc_index}_{spatial_method}_{date}_reliability'] = reliability_arr[np.argwhere(doy_arr == doy_)[0][0]]
                            elif np.isnan(doy_):
                                gedi_df.loc[i, f'{RSdc_index}_{spatial_method}_{date}'] = np.nan
                                gedi_df.loc[i, f'{RSdc_index}_{spatial_method}_{date}_reliability'] = 0
                            elif doy_ <= doy_arr.min() or doy_ >= doy_arr.max():
                                gedi_df.loc[i, f'{RSdc_index}_{spatial_method}_{date}'] = np.nan
                                gedi_df.loc[i, f'{RSdc_index}_{spatial_method}_{date}_reliability'] = 0
                            else:
                                doy_positive_pos = np.argwhere(doy_arr > doy_).min()
                                doy_negative_pos = np.argwhere(doy_arr < doy_).max()
                                doy_positive_value = np.nan
                                doy_negative_value = np.nan
                                reliability_positive = np.nan
                                reliability_negative = np.nan
                                for date_t in doy_arr[doy_positive_pos:]:
                                    date_pos = np.argwhere(doy_arr == date_t)[0][0]
                                    if ~np.isnan(inform_arr[date_pos]):
                                        doy_positive_value = inform_arr[date_pos]
                                        reliability_positive = reliability_arr[date_pos]
                                        break
                                for date_t in doy_arr[doy_negative_pos:0:-1]:
                                    date_pos = np.argwhere(doy_arr == date_t)[0][0]
                                    if ~np.isnan(inform_arr[date_pos]):
                                        doy_negative_value = inform_arr[date_pos]
                                        reliability_negative = reliability_arr[date_pos]
                                        break
                                if np.isnan(doy_positive_value) or np.isnan(doy_negative_value):
                                    gedi_df.loc[i, f'{RSdc_index}_{spatial_method}_{date}'] = np.nan
                                    gedi_df.loc[i, f'{RSdc_index}_{spatial_method}_{date}_reliability'] = 0
                                else:
                                    gedi_df.loc[i, f'{RSdc_index}_{spatial_method}_{date}'] = doy_negative_value + (doy_positive_value - doy_negative_value) * (doy_ - doy_arr[doy_negative_pos]) / (doy_arr[doy_positive_pos] - doy_arr[doy_negative_pos])
                                    gedi_df.loc[i, f'{RSdc_index}_{spatial_method}_{date}_reliability'] = reliability_negative + (reliability_positive - reliability_negative) * (doy_ - doy_arr[doy_negative_pos]) / (doy_arr[doy_positive_pos] - doy_arr[doy_negative_pos])
                else:
                    pass
                print(f'Finish linking the {RSdc_index} with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')
            else:
                print(f'Invalid value for {RSdc_index} linking with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')
                raise Exception('Invalid value in the GEDI df')
        except:
            print(traceback.format_exc())
            print(f'Failed linking the {RSdc_index} value with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')

    return gedi_df

def process_ccdc_csv(csv_list_chunk:list):
    result_chunk = []
    for csv_file in csv_list_chunk:
        try:
            x = int([_[1:] for _ in csv_file.split('.csv')[0].split('_') if _.startswith('x')][-1])
            y = int([_[1:] for _ in csv_file.split('.csv')[0].split('_') if _.startswith('y')][-1])
            print(f'Start_{str(x)}_{str(y)}')
            df = pd.read_csv(csv_file)

            if df.empty or 't_break' not in df.columns:
                result_chunk.append((x, y, 0, 0, 0, (0)))
                continue

            valid = df['t_break'][df['t_break'] > 0].to_numpy()
            count = len(valid)
            if count > 0:
                tyears = np.floor(valid / 365.25).astype(int)
            else:
                tyears = (0)
            tmin = int(valid.min()) if count > 0 else 0
            tmax = int(valid.max()) if count > 0 else 0
            result_chunk.append((x, y, count, tmin, tmax, tyears))
            print(f'End_{str(x)}_{str(y)}')
        except:
           print(traceback.format_exc())
    return result_chunk


def run_CCDC(dc_list: list, date_list: list, pos_list: pd.DataFrame, xy_offset_list: list, index_list: list, nodata_list: list, offset_list:list, resize_list:list, output_folder, min_year: int):

    # Date 2 doy
    try:
        dates = np.array([datetime.strptime(str(d), "%Y%m%d") for d in date_list])
        origin = datetime(min_year, 1, 1)
        doy_arr = np.array([(d - origin).days for d in dates])
        years = np.ceil(max(doy_arr) / 365.25)

        # Create folder
        fig_folder = os.path.join(output_folder, 'pixel_fig\\')
        csv_folder = os.path.join(output_folder, 'pixel_csv\\')
        raw_data_folder = os.path.join(output_folder, 'raw_input')
        basic_function.create_folder(fig_folder)
        basic_function.create_folder(csv_folder)
        basic_function.create_folder(raw_data_folder)

        pos_list = pos_list.reset_index(drop=True)

        for _ in tqdm(range(pos_list.shape[0]), desc="Running CCDC", unit="pixel"):
            try:
                if not os.path.exists(os.path.join(raw_data_folder, f"RawData_x{str(pos_list['x'][_])}_y{str(pos_list['y'][_])}.csv")):
                    x_ = pos_list['x'][_]
                    y_ = pos_list['y'][_]

                    # Gey x and y
                    y_ = y_ - xy_offset_list[0]
                    x_ = x_ - xy_offset_list[1]

                    # define trend index and
                    trend_index = np.empty([doy_arr.shape[0], len(index_list)])
                    nodata_mask = np.full_like(trend_index, False, dtype=bool)

                    for index_num in range(len(index_list)):
                        nodata_val = nodata_list[index_num]
                        offset = offset_list[index_num]
                        resize = 10000 if resize_list[index_num] else 1
                        data_ = dc_list[index_num][y_, x_, :].reshape(-1)
                        nodata_mask[:, index_num] = data_ == nodata_val
                        data_ = (data_ - offset)/ resize
                        trend_index[:, index_num] = data_

                    valid_row_mask = ~np.any(nodata_mask, axis=1)
                    trend_index = trend_index[valid_row_mask, :]
                    doy_arr_temp = doy_arr[valid_row_mask]

                    # 保存 trend_index 和 doy_arr_temp
                    raw_data_dict = {'doy': doy_arr_temp}
                    for i, band in enumerate(index_list):
                        raw_data_dict[f'{band}'] = trend_index[:, i]

                    raw_df = pd.DataFrame(raw_data_dict)
                    raw_df.to_csv(os.path.join(raw_data_folder, f"RawData_x{str(pos_list['x'][_])}_y{str(pos_list['y'][_])}.csv"), index=False)

                else:
                    try:
                        raw_df = pd.read_csv(os.path.join(raw_data_folder, f"RawData_x{str(pos_list['x'][_])}_y{str(pos_list['y'][_])}.csv"))
                        doy_arr_temp = np.array(raw_df['doy'].values)
                        trend_index = np.zeros((len(doy_arr_temp), len(index_list)), dtype=float)

                        for i, band in enumerate(index_list):
                            trend_index[:, i] = np.array(raw_df[band].values, dtype=float)
                    except:
                        doy_arr_temp = np.array([])
                        trend_index = np.array([])

                if len(doy_arr_temp) > 20 and not os.path.exists(os.path.join(csv_folder, f"CCDC_result_x{str(pos_list['x'][_])}_y{str(pos_list['y'][_])}.csv")) and not os.path.exists(os.path.join(fig_folder, f"CCDC_result_x{str(pos_list['x'][_])}_y{str(pos_list['y'][_])}.png")):

                    # Run ccdc
                    ccdc_para = TrendSeasonalFit_v12_30Line(doy_arr_temp, trend_index)

                    # save ccdc csv
                    df = pd.DataFrame(ccdc_para)
                    for col in ['coefs', 'magnitude']:
                        if col in df.columns:
                            df[col] = df[col].apply(lambda x: ','.join(map(str, x)) if isinstance(x, (list, np.ndarray)) else x)

                    # 保存为 CSV
                    df.to_csv(os.path.join(csv_folder, f"CCDC_result_x{str(pos_list['x'][_])}_y{str(pos_list['y'][_])}.csv"), index=False)

                    # 保存图片
                    plot_ccdc_segments(doy_arr_temp, trend_index, ccdc_para, os.path.join(fig_folder, f"CCDC_result_x{str(pos_list['x'][_])}_y{str(pos_list['y'][_])}.png"), min_year)
            except:
                print(traceback.format_exc())
    except:
        print(traceback.format_exc())




