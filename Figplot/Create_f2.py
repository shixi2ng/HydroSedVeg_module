import os.path
from RSDatacube.RSdc import *
from skimage import io, feature
from sklearn.metrics import r2_score
import seaborn as sns
from River_GIS.River_GIS import *


def x_minus(x, a, b, c ):
    return a * (x + b) ** -1 + c


def guassain_dis(x, sig, mean):
    return np.exp(-(x - mean) ** 2 / (2 * sig ** 2)) / (np.sqrt(2 * np.pi) * sig)


def xpoly(x, a, b):
    return - a ** x + b


def ln_x3(x, a, b, c):
    return a * np.log(b * (-x + 2.5)) + c


def ln_x5(x, a, b, c):
    return a * np.log(b * (-x + 5)) + c


def ln_minus(x, a, b, c):
    return - a * np.log(b * x) + c


def exp_minus(x, a, b, c):
    return - a * np.exp(b * x + c)

def exp_minus3(x, a, b, c, d):
    return - a * np.exp(b * x + c) + d


def exp_minus2(x, a, b, c):
    return - a * np.exp(b * x - c) + a * np.exp(b - c)


def poly3(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def poly2(x, a, b, c):
    return a * x ** 2 + b * x + c


def exp_temp(x, a, b, d, c):
    return a * np.exp(b * x - c) + d


def ln_temp(x, a, b, c, d):
    return a * np.log(x ** b + c) + d


def log_func(x, a, b, c, d):
    return a / (1. + np.exp(-c * (x - d))) + b


def fig1_func():

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=20)
    plt.rc('axes', linewidth=2)

    fig1, ax1 = plt.subplots(figsize=(12, 8), constrained_layout=True)
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.xaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.set_xlim(0, 15)
    ax1.set_ylim(0, 15)
    pd_temp = pd.read_excel('G:\GEDI_MYR\Hankou_table\\hankou.xlsx', sheet_name='UAV_Final')
    x_2020 = pd_temp[pd_temp['Date']//1000 == 2020]['Canopy_Hei']
    y_2020 = pd_temp[pd_temp['Date']//1000 == 2020]['MEAN']
    x_2021 = pd_temp[pd_temp['Date']//1000 == 2021]['Canopy_Hei']
    y_2021 = pd_temp[pd_temp['Date']//1000 == 2021]['MEAN']
    x_2022 = pd_temp[pd_temp['Date']//1000 == 2022]['Canopy_Hei']
    y_2022 = pd_temp[pd_temp['Date']//1000 == 2022]['MEAN']
    ax1.scatter(x_2020, y_2020, s=12 ** 2, edgecolor=(0.1, 0.1, 0.1), facecolor=(47/256,85/256,151/256), alpha=0.8,linewidth=1.2, marker='^', zorder=5)
    ax1.scatter(x_2021, y_2021, s=12 ** 2, edgecolor=(0.1, 0.1, 0.1), facecolor=(47/256,85/256,151/256), alpha=0.8, linewidth=1.2, marker='^', zorder=5)
    ax1.scatter(x_2022, y_2022, s=12 ** 2, edgecolor=(0.1, 0.1, 0.1), facecolor=(230/256,76/256,0/256), linewidth=1.2, marker='o', zorder=5)
    ax1.plot(np.linspace(0, 15), np.linspace(0, 15), color=(0.2, 0.2, 0.2), ls = '-.', linewidth = 2)
    ax1.plot(np.linspace(0, 15), np.linspace(2, 17), color=(0.2, 0.2, 0.2), ls='--', linewidth=1)
    ax1.plot(np.linspace(0, 15), np.linspace(-2, 13), color=(0.2, 0.2, 0.2), ls='--', linewidth=1)
    ax1.set_xticks([0, 3, 6, 9, 12, 15])
    ax1.set_xticklabels(['0', '3', '6', '9', '12', '15'], fontname='Times New Roman', fontsize=12)
    ax1.set_yticks([0, 3, 6, 9, 12, 15])
    ax1.set_yticklabels(['0', '3', '6', '9', '12', '15'], fontname='Times New Roman', fontsize=12)

    print(str(r2_score(x_2022, y_2022)))
    print(str(r2_score(pd_temp['Canopy_Hei'], pd_temp['MEAN'])))

    # ax1.set_xlabel('Date', fontname='Times New Roman', fontsize=40, fontweight='bold')
    # ax1.set_ylabel('OSAVI', fontname='Times New Roman', fontsize=40, fontweight='bold')

    ax1.fill_between(np.linspace(0, 15, 100), np.linspace(2, 17, 100), np.linspace(-2, 13, 100), color=(0.8, 0.8, 0.8), alpha=0.5, zorder=1)
    # plt.show()
    plt.savefig('G:\A_veg\Paper\Figure\Fig4\\fig4.png', dpi=300)


def fig2_func():
    fig2, ax1 = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.xaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 5)
    pd_temp = pd.read_excel('G:\GEDI_MYR\Hankou_table\\hankou.xlsx', sheet_name='Insitu-Final')
    x = pd_temp['UAV']
    y = pd_temp['Mean ']
    ax1.scatter(x, y, s=20 ** 2, edgecolor=(0.1, 0.1, 0.1), facecolor=(168/256,0/256,0/256), alpha=1,linewidth=1.2, marker='^', zorder=5)
    ax1.plot(np.linspace(0, 15), np.linspace(0, 15), color=(0.2, 0.2, 0.2), ls = '-.', linewidth = 2)
    ax1.plot(np.linspace(0, 15), np.linspace(1, 16), color=(0.2, 0.2, 0.2), ls='--', linewidth=1)
    ax1.plot(np.linspace(0, 15), np.linspace(-1, 14), color=(0.2, 0.2, 0.2), ls='--', linewidth=1)
    ax1.set_xticks([0, 1, 2, 3, 4, 5])
    ax1.set_xticklabels(['0', '1', '2', '3', '4', '5'], fontname='Times New Roman', fontsize=12)
    ax1.set_yticks([0, 1, 2, 3, 4, 5])
    ax1.set_yticklabels(['0', '1', '2', '3', '4', '5'], fontname='Times New Roman', fontsize=12)
    print(str(r2_score(x[:10], y[:10])))

    # ax1.set_xlabel('Date', fontname='Times New Roman', fontsize=40, fontweight='bold')
    # ax1.set_ylabel('OSAVI', fontname='Times New Roman', fontsize=40, fontweight='bold')

    ax1.fill_between(np.linspace(0, 15, 100), np.linspace(1, 16, 100), np.linspace(-1, 14, 100), color=(0.8, 0.8, 0.8), alpha=0.5, zorder=1)
    # plt.show()
    plt.savefig('G:\A_veg\Paper\Figure\Fig4\\fig5.png', dpi=300)


def fig3_func():
    plt.rc('font', family='Times New Roman')
    data_t = pd.read_excel('G:\\GEDI_MYR\\Hankou_table\\hankou.xlsx', sheet_name='UAV_Final')
    data_t = data_t.loc[0:24, ['Canopy_Hei', 'RH_98', 'RH90', 'RH75', 'RH25', 'MEAN']]
    ch_err = np.array(data_t['Canopy_Hei'] - data_t['MEAN'])
    rh98_err = np.array(data_t['RH_98'] - data_t['MEAN'])
    rh90_err = np.array(data_t['RH90'] - data_t['MEAN'])
    rh75_err = np.array(data_t['RH75'] - data_t['MEAN'])
    rh25_err = np.array(data_t['RH25'] - data_t['MEAN'])
    fig2, ax1 = plt.subplots(figsize=(6, 3), constrained_layout=True)
    d = [rh25_err, rh75_err, rh90_err,  rh98_err, ch_err]
    # ax1.set_axisbelow(True)
    # ax1.yaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    # ax1.xaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.set_xlim(-10, 2)
    ax1.set_ylim(0.5, 5.5)
    # pd_temp = pd.read_excel('G:\GEDI_MYR\Hankou_table\\hankou.xlsx', sheet_name='Insitu-Final')
    # x = pd_temp['UAV']
    # y = pd_temp['Mean ']
    box = ax1.boxplot(d, vert=False, labels=['RH25-UAV', 'RH75-UAV', 'RH90-UAV', 'RH98-UAV', 'RH100-UAV'],  notch=True, widths=0.5, patch_artist=True, whis=(5, 95), showfliers=False, zorder=4)
    ax1.scatter([np.mean(_) for _ in d], [1,2,3,4,5], marker='d', facecolor=(1,1,1), s=25, edgecolor=(0,0,0), lw=0.5, zorder=4)
    plt.setp(box['boxes'], color=(89/256,117/256,164/256), edgecolor=(60/256,60/256,60/256))
    ax1.plot(np.linspace(0,0,100), np.linspace(-100,100,100), c=(0,0,0), lw=1, zorder=3, ls ='--')
    # ax1.set_yticklabels('平均误差/m')

    # ax1.fill_between(np.linspace(0, 15, 100), np.linspace(1, 16, 100), np.linspace(-1, 14, 100), color=(0.8, 0.8, 0.8), alpha=0.5, zorder=1)
    # plt.show()
    plt.savefig('G:\A_veg\Paper\Figure\Fig5\\fig5.png', dpi=300)


def fig4_func():
    plt.rc('font', family='Times New Roman')
    data_t = pd.read_excel('G:\\GEDI_MYR\\Hankou_table\\hankou.xlsx', sheet_name='UAV_Final')
    data_t = data_t.loc[0:24, ['Canopy_Hei', 'RH_98', 'RH90', 'RH75', 'RH25', 'MEAN']]

    fig2, ax1 = plt.subplots(figsize=(6, 0.8), constrained_layout=True)
    ax1.xaxis.tick_top()
    d = [[0.097012,0.123635,0.0306,0.132558,0.002834,0.072849,0.122362,0.113934,-0.0023,-0.09]]
    # ax1.set_axisbelow(True)
    # ax1.yaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    # ax1.xaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.set_xlim(-1, 0.2)
    ax1.set_ylim(0.5, 1.5)
    # pd_temp = pd.read_excel('G:\GEDI_MYR\Hankou_table\\hankou.xlsx', sheet_name='Insitu-Final')
    # x = pd_temp['UAV']
    # y = pd_temp['Mean ']
    box = ax1.boxplot(d, labels = ['实测-UAV'], vert=False,  notch=False, widths=0.5, patch_artist=True, whis=(0, 100), showfliers=False, zorder=4)
    ax1.scatter([np.mean(_) for _ in d],[1], marker='d', facecolor=(1,1,1), s=25, edgecolor=(0,0,0), lw=0.5, zorder=4)
    plt.setp(box['boxes'], color=(167/256,0/256,0/256), edgecolor=(60/256,60/256,60/256))
    ax1.plot(np.linspace(0,0,100), np.linspace(-100,100,100), c=(0,0,0), lw=1, zorder=3, ls ='--')

    # ax1.fill_between(np.linspace(0, 15, 100), np.linspace(1, 16, 100), np.linspace(-1, 14, 100), color=(0.8, 0.8, 0.8), alpha=0.5, zorder=1)
    # plt.show()
    plt.savefig('G:\A_veg\Paper\Figure\Fig5\\fig51.png', dpi=300)


def fig5_func():
    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    # plt.style.use("ggplot")
    data_t = pd.read_excel('G:\\A_veg\\Paper\\Table\\Tab3.xlsx', sheet_name='Output')
    data_t = data_t.loc[:, ['Name', 'Percen', 'Type']].sort_values('Percen')[13:27]
    edge_c = [(1,1,1), (1,1,1), (1,1,1)]
    color_c = [(178/256,178/256,178/256), (112/256, 160/256, 205/256), (112/256, 160/256, 205/256)]
    c_all = []
    e_all = []
    for _ in data_t['Type']:
        c_all.append(color_c[_])
        e_all.append(edge_c[_])

    fig2, ax1 = plt.subplots(figsize=(3.5, 5), constrained_layout=True)
    ax1.xaxis.tick_top()
    ax1.set_axisbelow(True)
    # ax1.yaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.xaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.set_xlim(0, 0.2)
    ax1.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    ax1.set_xticklabels(['0%', '5%', '10%', '15%', '20%'], font='Times New Roman')
    ax1.set_ylim(-0.5, 12.5)
    # pd_temp = pd.read_excel('G:\GEDI_MYR\Hankou_table\\hankou.xlsx', sheet_name='Insitu-Final')
    # x = pd_temp['UAV']
    # y = pd_temp['Mean ']
    ax1.barh(data_t['Name'], width= data_t['Percen'], height=0.9, color=c_all, edgecolor=e_all)
    ax1.plot(np.linspace(0.04,0.040,100), np.linspace(-100,100,100), c=(0,0,0), lw=1.5, zorder=3, ls ='--')

    # ax1.fill_between(np.linspace(0, 15, 100), np.linspace(1, 16, 100), np.linspace(-1, 14, 100), color=(0.8, 0.8, 0.8), alpha=0.5, zorder=1)
    # plt.show()
    plt.savefig('G:\A_veg\Paper\Figure\Fig6\\fig5.png', dpi=300)

    data_t = pd.read_excel('G:\\A_veg\\Paper\\Table\\Tab3.xlsx', sheet_name='Output')
    data_t = data_t.loc[:, ['Name', 'Percen', 'Type']].sort_values('Percen')[0:13]
    edge_c = [(1,1,1), (1,1,1), (1,1,1)]
    color_c = [(178/256,178/256,178/256), (112/256, 160/256, 205/256), (112/256, 160/256, 205/256)]
    c_all = []
    e_all = []
    for _ in data_t['Type']:
        c_all.append(color_c[_])
        e_all.append(edge_c[_])

    fig2, ax1 = plt.subplots(figsize=(3.5, 5), constrained_layout=True)
    ax1.xaxis.tick_top()
    ax1.set_axisbelow(True)
    # ax1.yaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.xaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.set_xlim(0, 0.2)
    ax1.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    ax1.set_xticklabels(['0%', '5%', '10%', '15%', '20%'], font='Times New Roman')
    ax1.set_ylim(-0.5, 12.5)
    # pd_temp = pd.read_excel('G:\GEDI_MYR\Hankou_table\\hankou.xlsx', sheet_name='Insitu-Final')
    # x = pd_temp['UAV']
    # y = pd_temp['Mean ']
    ax1.barh(data_t['Name'], width= data_t['Percen'], height=0.9, color=c_all, edgecolor=e_all)
    ax1.plot(np.linspace(0.04,0.040,100), np.linspace(-100,100,100), c=(0,0,0), lw=1.5, zorder=3, ls ='--')

    # ax1.fill_between(np.linspace(0, 15, 100), np.linspace(1, 16, 100), np.linspace(-1, 14, 100), color=(0.8, 0.8, 0.8), alpha=0.5, zorder=1)
    # plt.show()
    plt.savefig('G:\A_veg\Paper\Figure\Fig6\\fig51.png', dpi=300)


def fig6_func():

    species = ("Index", "2020", "Gentoo")
    penguin_means = {
        'Bill Depth': (18.35, 18.43, 14.98),
        'Bill Length': (38.79, 48.83, 47.50),
        'Flipper Length': (189.95, 195.82, 217.19),
    }

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.

    plt.show()
    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    # plt.style.use("ggplot")
    data_t = pd.read_excel('G:\\A_veg\\Paper\\Table\\Tab3.xlsx', sheet_name='Output')
    data_t = data_t.loc[:, ['Name', 'Percen', 'Type']].sort_values('Percen')[13:27]
    edge_c = [(1,1,1), (1,1,1), (1,1,1)]
    color_c = [(178/256,178/256,178/256), (112/256, 160/256, 205/256), (112/256, 160/256, 205/256)]
    c_all = []
    e_all = []
    for _ in data_t['Type']:
        c_all.append(color_c[_])
        e_all.append(edge_c[_])

    fig2, ax1 = plt.subplots(figsize=(3.5, 5), constrained_layout=True)
    ax1.xaxis.tick_top()
    ax1.set_axisbelow(True)
    # ax1.yaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.xaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.set_xlim(0, 0.2)
    ax1.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    ax1.set_xticklabels(['0%', '5%', '10%', '15%', '20%'], font='Times New Roman')
    ax1.set_ylim(-0.5, 12.5)
    # pd_temp = pd.read_excel('G:\GEDI_MYR\Hankou_table\\hankou.xlsx', sheet_name='Insitu-Final')
    # x = pd_temp['UAV']
    # y = pd_temp['Mean ']
    ax1.barh(data_t['Name'], width= data_t['Percen'], height=0.9, color=c_all, edgecolor=e_all)
    ax1.plot(np.linspace(0.04,0.040,100), np.linspace(-100,100,100), c=(0,0,0), lw=1.5, zorder=3, ls ='--')

    # ax1.fill_between(np.linspace(0, 15, 100), np.linspace(1, 16, 100), np.linspace(-1, 14, 100), color=(0.8, 0.8, 0.8), alpha=0.5, zorder=1)
    # plt.show()
    plt.savefig('G:\A_veg\Paper\Figure\Fig6\\fig5.png', dpi=300)

    data_t = pd.read_excel('G:\\A_veg\\Paper\\Table\\Tab3.xlsx', sheet_name='Output')
    data_t = data_t.loc[:, ['Name', 'Percen', 'Type']].sort_values('Percen')[0:13]
    edge_c = [(1,1,1), (1,1,1), (1,1,1)]
    color_c = [(95/256,158/256,110/256), (204/256,137/256,99/256), (89/256,117/256,164/256)]
    c_all = []
    e_all = []
    for _ in data_t['Type']:
        c_all.append(color_c[_])
        e_all.append(edge_c[_])

    fig2, ax1 = plt.subplots(figsize=(3.5, 5), constrained_layout=True)
    ax1.xaxis.tick_top()
    ax1.set_axisbelow(True)
    # ax1.yaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.xaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.set_xlim(0, 0.2)
    ax1.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    ax1.set_xticklabels(['0%', '5%', '10%', '15%', '20%'], font='Times New Roman')
    ax1.set_ylim(-0.5, 12.5)
    # pd_temp = pd.read_excel('G:\GEDI_MYR\Hankou_table\\hankou.xlsx', sheet_name='Insitu-Final')
    # x = pd_temp['UAV']
    # y = pd_temp['Mean ']
    ax1.barh(data_t['Name'], width= data_t['Percen'], height=0.9, color=c_all, edgecolor=e_all)
    ax1.plot(np.linspace(0.04,0.040,100), np.linspace(-100,100,100), c=(0,0,0), lw=1.5, zorder=3, ls ='--')

    # ax1.fill_between(np.linspace(0, 15, 100), np.linspace(1, 16, 100), np.linspace(-1, 14, 100), color=(0.8, 0.8, 0.8), alpha=0.5, zorder=1)
    # plt.show()
    plt.savefig('G:\A_veg\Paper\Figure\Fig6\\fig51.png', dpi=300)


def fig7_func():
    UAV_tif_folder = 'E:\\A_PhD_Main_stuff\\江滩LiDAR\\LiDAR_detect_res\\tiffiles\\'
    shpfile = 'E:\\A_PhD_Main_stuff\\江滩LiDAR\\LiDAR_detect_res\hk_shp\\hk.shp'
    plt.style.use('_mpl-gallery-nogrid')

    for date_t in ['peak_2022', 'peak_2021', 'peak_2020', 'peak_2019']:
        GEDI_tif_folder = f'G:\\A_veg\\S2_all\\Feature_table4heightmap\\{str(date_t)}\\predicted_feature_tif\\'
        bf.create_folder(f'G:\\A_veg\\S2_all\\Feature_table4heightmap\\{str(date_t)}\\Compare_fig\\')
        bf.create_folder(f'G:\\A_veg\\S2_all\\Feature_table4heightmap\\{str(date_t)}\\Compare_tab\\')
        for _ in ['230205', '220706']:
            for chl in range(5, 10):
                for mod in range(3):
                    UAV_file = bf.file_filter(UAV_tif_folder, [str(_), f'chl{str(chl)}', '.tif'], and_or_factor='and', exclude_word_list=['aux', 'ovr'])
                    GEDI_file = bf.file_filter(GEDI_tif_folder, [f'heil{str(chl)}', f'mod{str(mod)}', '.tif'], and_or_factor='and', exclude_word_list=['aux', 'ovr'])
                    if len(UAV_file) == 1 and len(GEDI_file) == 1:
                        gdal.Warp(f'/vsimem/date{_}_chl{str(chl)}_mod{str(mod)}_gedi.TIF', GEDI_file[0], cutlineDSName=shpfile, cropToCutline=True, xRes=10, yRes=10, outputType=gdal.GDT_Float32)
                        gdal.Warp(f'/vsimem/date{_}_chl{str(chl)}_mod{str(mod)}_UAV.TIF', UAV_file[0], cutlineDSName=shpfile, cropToCutline=True, xRes=10, yRes=10, outputType=gdal.GDT_Float32)
                    else:
                        raise Exception(f'Date {_} chl {str(chl)} mod {str(mod)} is wrong!')

                    gedi_ds = gdal.Open(f'/vsimem/date{_}_chl{str(chl)}_mod{str(mod)}_gedi.TIF')
                    uav_ds = gdal.Open(f'/vsimem/date{_}_chl{str(chl)}_mod{str(mod)}_UAV.TIF')
                    gedi_nodata = gedi_ds.GetRasterBand(1).GetNoDataValue()
                    uav_nodata = uav_ds.GetRasterBand(1).GetNoDataValue()
                    gedi_arr = gedi_ds.GetRasterBand(1).ReadAsArray()
                    uav_arr = uav_ds.GetRasterBand(1).ReadAsArray()
                    if gedi_arr.shape == uav_arr.shape:
                        combine_arr = np.stack((gedi_arr.flatten(), uav_arr.flatten()), axis=0)

                        combine_arr = np.delete(combine_arr, np.argwhere(np.isnan(combine_arr))[:, 1], axis=1)
                        combine_arr = np.delete(combine_arr, np.argwhere(combine_arr == -3.402823e+38)[:, 1], axis=1)
                        combine_arr = np.delete(combine_arr, np.argwhere(combine_arr <= 0)[:, 1], axis=1)

                        df_temp = pd.DataFrame(combine_arr.transpose(), columns=['gedi_ch', 'uav_ch'])
                        df_temp.to_csv(f'G:\\A_veg\\S2_all\\Feature_table4heightmap\\{str(date_t)}\\Compare_tab\\date{_}_chl{str(chl)}_mod{str(mod)}.csv')
                        fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
                        # ax.scatter(combine_arr[1, :], combine_arr[0, :], s=12 ** 2, edgecolor=(0.1, 0.1, 0.1), facecolor=(47/256,85/256,151/256), alpha=0.8, linewidth=1.2, marker='^', zorder=5)
                        ax.hist2d(combine_arr[0, :], combine_arr[1, :], bins=100, range=[[0, chl], [0, chl]])
                        ax.plot(np.linspace(0, chl, chl), np.linspace(0, chl, chl), c=(0, 0, 0), lw=1.5, zorder=3, ls='--')
                        ax.set_xlabel('GEDI', fontname='Times New Roman', fontsize=40, fontweight='bold')
                        ax.set_ylabel('UAV', fontname='Times New Roman', fontsize=40, fontweight='bold')
                        RMSE_NDVI = np.sqrt(np.nanmean((combine_arr[1, :] - combine_arr[0, :]) ** 2))
                        MAE_NDVI = np.nanmean(np.absolute(combine_arr[1, :] - combine_arr[0, :]))
                        ax.text(2, chl-1, f'MAE:{str(MAE_NDVI)}', style='italic',)
                        ax.text(2, chl-2, f'RMSE:{str(RMSE_NDVI)}', style='italic', )
                        ax.set_ylim(0, chl)
                        ax.set_xlim(0, chl)
                        plt.savefig(f'G:\\A_veg\\S2_all\\Feature_table4heightmap\\{str(date_t)}\\Compare_fig\\date{_}_chl{str(chl)}_mod{str(mod)}.png', dpi=300)
                        fig, ax = None, None
                        gdal.Unlink(f'/vsimem/date{_}_chl{str(chl)}_mod{str(mod)}_UAV.TIF')
                        gdal.Unlink(f'/vsimem/date{_}_chl{str(chl)}_mod{str(mod)}_gedi.TIF')
                    else:
                        raise Exception(f'Date {_} chl {str(chl)} mod {str(mod)} facing crop error!')


def fig8_func():
    # Make figure and axes
    fig, ax = plt.subplots(2, 2)
    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=12)
    # Shift the second slice using explode
    patches, texts, autotexts1 = ax[0, 0].pie([7828, 2206, 4481, 5268], labels=['2019', '2020', '2021', '2022'], autopct='%.0f%%',
                  colors=[(51/256, 62/256, 81/256), (112/256, 160/256, 205/256), (196/256, 121/256, 0/256), (178/256, 178/256, 178/256)], shadow=False, explode=(0.1, 0, 0, 0))
    patches, texts, autotexts2 = ax[0, 1].pie([7828, 2206, 4481, 5268], labels=['2019', '2020', '2021', '2022'], autopct='%.0f%%',
                 colors=[(51/256, 62/256, 81/256), (112/256, 160/256, 205/256), (196/256, 121/256, 0/256), (178/256, 178/256, 178/256)], shadow=False, explode=(0, 0.1, 0, 0))
    patches, texts, autotexts3 = ax[1, 0].pie([7828, 2206, 4481, 5268], labels=['2019', '2020', '2021', '2022'], autopct='%.0f%%',
                 colors=[(51/256, 62/256, 81/256), (112/256, 160/256, 205/256), (196/256, 121/256, 0/256), (178/256, 178/256, 178/256)], shadow=False, explode=(0, 0, 0.1, 0))
    patches, texts, autotexts4 =  ax[1, 1].pie([7828, 2206, 4481, 5268], labels=['2019', '2020', '2021', '2022'], autopct='%.0f%%',
                 colors=[(51/256, 62/256, 81/256), (112/256, 160/256, 205/256), (196/256, 121/256, 0/256), (178/256, 178/256, 178/256)], shadow=False, explode=(0, 0, 0, 0.1))
    # ax.pie([7828, 2206, 4481, 5268], labels=['2019', '2020', '2021', '2022'], )

    autotexts1[0].set_color('white')
    autotexts2[0].set_color('white')
    autotexts3[0].set_color('white')
    autotexts4[0].set_color('white')
    plt.savefig('G:\A_veg\Paper\Figure\Fig8\\fig83.png', dpi=300)


def fig9_func():
    # Make figure and axes
    fig, ax = plt.subplots(2, 2)
    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=12)
    # Shift the second slice using explode
    patches, texts, autotexts1 = ax[0, 0].pie([7828, 2206, 4481, 5268], labels=['2019', '2020', '2021', '2022'], autopct='%.0f%%',
                  colors=[(51/256, 62/256, 81/256), (112/256, 160/256, 205/256), (196/256, 121/256, 0/256), (178/256, 178/256, 178/256)], shadow=False, explode=(0.1, 0, 0, 0))
    patches, texts, autotexts2 = ax[0, 1].pie([7828, 2206, 4481, 5268], labels=['2019', '2020', '2021', '2022'], autopct='%.0f%%',
                 colors=[(51/256, 62/256, 81/256), (112/256, 160/256, 205/256), (196/256, 121/256, 0/256), (178/256, 178/256, 178/256)], shadow=False, explode=(0, 0.1, 0, 0))
    patches, texts, autotexts3 = ax[1, 0].pie([7828, 2206, 4481, 5268], labels=['2019', '2020', '2021', '2022'], autopct='%.0f%%',
                 colors=[(51/256, 62/256, 81/256), (112/256, 160/256, 205/256), (196/256, 121/256, 0/256), (178/256, 178/256, 178/256)], shadow=False, explode=(0, 0, 0.1, 0))
    patches, texts, autotexts4 =  ax[1, 1].pie([7828, 2206, 4481, 5268], labels=['2019', '2020', '2021', '2022'], autopct='%.0f%%',
                 colors=[(51/256, 62/256, 81/256), (112/256, 160/256, 205/256), (196/256, 121/256, 0/256), (178/256, 178/256, 178/256)], shadow=False, explode=(0, 0, 0, 0.1))
    # ax.pie([7828, 2206, 4481, 5268], labels=['2019', '2020', '2021', '2022'], )

    autotexts1[0].set_color('white')

    autotexts2[0].set_color('white')

    autotexts3[0].set_color('white')

    autotexts4[0].set_color('white')
    plt.savefig('G:\A_veg\Paper\Figure\Fig8\\fig83.png', dpi=300)


def fig10_func():
    pd_temp = pd.read_csv('G:\A_veg\Paper\Figure\Fig8\\4year_ass2.csv')
    entire = np.array(pd_temp['Canopy Height (rh100)'])

    for i, c in zip([2019, 2020, 2021, 2022], [(51/256, 62/256, 81/256), (112/256, 160/256, 205/256), (196/256, 121/256, 0/256), (178/256, 178/256, 178/256)]):
        plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
        plt.rc('font', size=45)
        plt.rc('axes', linewidth=3)
        year = np.array(pd_temp[pd_temp['Year '] == i]['Canopy Height (rh100)'])
        fig, ax = plt.subplots(figsize=(10.5, 8.2), constrained_layout=True)
        plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
        plt.rc('font', size=12)
        ax.set_xlim([1, 8])
        # Shift the second slice using explode
        a1 = sns.kdeplot(data=entire, fill=True, color=(169/256, 2/256, 38/256), zorder=1, alpha=1, )
        a2 = sns.kdeplot(data=year, fill=True, color=c, alpha=1, zorder=3)
        ax.set_ylabel('概率密度')
        ax.set_xlabel('高度/m')
        plt.savefig(f'G:\A_veg\Paper\Figure\Fig8\\fig8_{str(i)}.png', dpi=300)


def fig11_func():
    pd1 = pd.read_csv('G:\\A_veg\\Paper\\Figure\\Fig9\\date220706_chl8_mod0.csv')
    pd2 = pd.read_csv('G:\\A_veg\\Paper\\Figure\\Fig9\\date220706_chl8_mod1.csv')
    pd3 = pd.read_csv('G:\\A_veg\\Paper\\Figure\\Fig9\\date220706_chl8_mod0_ref.csv')

    pd1 = pd.merge(pd1, pd2, on=['Unnamed: 0'], how='left')
    pd1 = pd.merge(pd1, pd3, on=['Unnamed: 0'], how='left')

    chl = 8
    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=14)
    fig, ax = plt.subplots(figsize=(5.8, 5), constrained_layout=True)
    # ax.scatter(combine_arr[1, :], combine_arr[0, :], s=12 ** 2, edgecolor=(0.1, 0.1, 0.1), facecolor=(47/256,85/256,151/256), alpha=0.8, linewidth=1.2, marker='^', zorder=5)
    qq = ax.hist2d(pd1['gedi_ch_x'], pd1['uav_ch_x'], bins=80, range=[[1, 6], [1, 6]], cmap=plt.cm.BuPu)
    fig.colorbar(qq[3], ax=ax)
    ax.plot(np.linspace(0, chl, chl), np.linspace(0, chl, chl), c=(0, 0, 0), lw=1.5, zorder=3, ls='--')
    ax.plot(np.linspace(0, chl, chl), np.linspace(0.89, chl+0.89, chl), c=(0.3, 0.3, 0.3), lw=0.5, zorder=3, ls='--')
    ax.plot(np.linspace(0, chl, chl), np.linspace(-0.89, chl-0.89, chl), c=(0.3, 0.3, 0.3), lw=0.5, zorder=3, ls='--')
    ax.fill_between(np.linspace(0, chl, chl), np.linspace(-0.89, chl - 0.89, chl), np.linspace(0.89, chl+0.89, chl), color=(0.3, 0.3, 0.3), alpha=0.1)
    ax.set_xlabel('外推植被高度/m', fontname='Times New Roman', fontsize=20, fontweight='bold')
    ax.set_ylabel('UAV航测高度/m', fontname='Times New Roman', fontsize=20, fontweight='bold')

    RMSE_NDVI = np.sqrt(np.nanmean((pd1['gedi_ch_x'] - pd1['uav_ch_x']) ** 2))
    MAE_NDVI = np.nanmean(np.absolute(pd1['gedi_ch_x'] - pd1['uav_ch_x']))
    ax.text(1.3, 6 - 0.5, f'MAE={str(MAE_NDVI)[0: 4]}m', c=(0, 0, 0), fontsize=18)
    ax.text(1.3, 6 - 1, f'RMSE={str(RMSE_NDVI)[0: 4]}m', c=(0, 0, 0), fontsize=18)
    ax.set_ylim(1, 6)
    ax.set_xlim(1, 6)
    print(str(r2_score(pd1['gedi_ch_x'].dropna(), pd1['uav_ch_x'].dropna())))
    plt.savefig(f'G:\A_veg\Paper\Figure\Fig9\\fig91.png', dpi=300)

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=14)
    fig2, ax2 = plt.subplots(figsize=(5, 5), constrained_layout=True)
    # ax.scatter(combine_arr[1, :], combine_arr[0, :], s=12 ** 2, edgecolor=(0.1, 0.1, 0.1), facecolor=(47/256,85/256,151/256), alpha=0.8, linewidth=1.2, marker='^', zorder=5)
    qq = ax2.hist2d(pd1['gedi_ch_y'], pd1['uav_ch_y'], bins=80, range=[[1, 6], [1, 6]], cmap=plt.cm.BuPu)
    ax2.plot(np.linspace(0, chl, chl), np.linspace(0, chl, chl), c=(0, 0, 0), lw=1.5, zorder=3, ls='--')
    ax2.plot(np.linspace(0, chl, chl), np.linspace(0.93, chl+0.93, chl), c=(0.3, 0.3, 0.3), lw=0.5, zorder=3, ls='--')
    ax2.plot(np.linspace(0, chl, chl), np.linspace(-0.93, chl-0.93, chl), c=(0.3, 0.3, 0.3), lw=0.5, zorder=3, ls='--')
    ax2.fill_between(np.linspace(0, chl, chl), np.linspace(-0.93, chl -0.93, chl), np.linspace(+0.93, chl +0.93, chl), color=(0.3, 0.3, 0.3), alpha=0.1)
    ax2.set_xlabel('外推植被高度/m', fontname='Times New Roman', fontsize=20, fontweight='bold')
    ax2.set_ylabel('UAV航测高度/m', fontname='Times New Roman', fontsize=20, fontweight='bold')
    RMSE_NDVI = np.sqrt(np.nanmean((pd1['gedi_ch_y'] - pd1['uav_ch_y']) ** 2))
    MAE_NDVI = np.nanmean(np.absolute(pd1['gedi_ch_y'] - pd1['uav_ch_y']))
    ax2.text(1.3, 6 - 0.5, f'MAE={str(MAE_NDVI)[0: 4]}m', c=(0, 0, 0), fontsize=18)
    ax2.text(1.3, 6 - 1, f'RMSE={str(RMSE_NDVI)[0: 4]}m', c=(0, 0, 0), fontsize=18)
    ax2.set_ylim(1, 6)
    ax2.set_xlim(1, 6)

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=14)
    fig3, ax3 = plt.subplots(figsize=(5, 5), constrained_layout=True)
    # ax.scatter(combine_arr[1, :], combine_arr[0, :], s=12 ** 2, edgecolor=(0.1, 0.1, 0.1), facecolor=(47/256,85/256,151/256), alpha=0.8, linewidth=1.2, marker='^', zorder=5)
    s, n = 0, 0
    for _ in range(pd1.shape[0]):
        if pd1['gedi_ch_x'][_] <= pd1['uav_ch_x'][_] + 0.93 and pd1['gedi_ch_x'][_] >= pd1['uav_ch_x'][_] - 0.93:
            s += 1
        if ~np.isnan(pd1['gedi_ch_x'][_]):
            n += 1
    print(str(s/n))

    s, n = 0, 0
    for _ in range(pd1.shape[0]):
        if pd1['gedi_ch_y'][_] <= pd1['uav_ch_y'][_] + 0.93 and pd1['gedi_ch_y'][_] >= pd1['uav_ch_y'][_] - 0.93:
            s += 1
        if ~np.isnan(pd1['gedi_ch_y'][_]):
            n += 1
    print(str(s/n))

    s, n = 0, 0
    for _ in range(pd1.shape[0]):
        if pd1['gedi_ch'][_] <= pd1['uav_ch'][_] + 0.93 and pd1['gedi_ch'][_] >= pd1['uav_ch'][_] - 0.93:
            s += 1
        if ~np.isnan(pd1['gedi_ch'][_]):
            n += 1
    print(str(s/n))

    qq = ax3.hist2d(pd1['gedi_ch'], pd1['uav_ch'], bins=80, range=[[1, 6], [1, 6]], cmap=plt.cm.BuPu)
    ax3.plot(np.linspace(0, chl, chl), np.linspace(0, chl, chl), c=(0, 0, 0), lw=1.5, zorder=3, ls='--')
    ax3.plot(np.linspace(0, chl, chl), np.linspace(0.93, chl + 0.93, chl), c=(0.3, 0.3, 0.3), lw=0.5, zorder=3, ls='--')
    ax3.plot(np.linspace(0, chl, chl), np.linspace(-0.93, chl - 0.93, chl), c=(0.3, 0.3, 0.3), lw=0.5, zorder=3,
             ls='--')
    ax3.fill_between(np.linspace(0, chl, chl), np.linspace(-0.93, chl - 0.93, chl), np.linspace(+0.93, chl + 0.93, chl),
                     color=(0.3, 0.3, 0.3), alpha=0.1)
    ax3.set_xlabel('外推植被高度/m', fontname='Times New Roman', fontsize=20, fontweight='bold')
    ax3.set_ylabel('UAV航测高度/m', fontname='Times New Roman', fontsize=20, fontweight='bold')
    RMSE_NDVI = np.sqrt(np.nanmean((pd1['gedi_ch'] - pd1['uav_ch']) ** 2))
    MAE_NDVI = np.nanmean(np.absolute(pd1['gedi_ch'] - pd1['uav_ch']))
    ax3.text(1.3, 6 - 0.5, f'MAE={str(MAE_NDVI)[0: 4]}m', c=(0, 0, 0), fontsize=18)
    ax3.text(1.3, 6 - 1, f'RMSE={str(RMSE_NDVI)[0: 4]}m', c=(0, 0, 0), fontsize=18)
    ax3.set_ylim(1, 6)
    ax3.set_xlim(1, 6)
    print(str(r2_score(pd1['gedi_ch'].dropna(), pd1['uav_ch'].dropna())))
    plt.savefig(f'G:\A_veg\Paper\Figure\Fig9\\fig93.png', dpi=300)


def fig12_func():

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=12)
    plt.rc('axes', linewidth=3)

    file = 'G:\\A_veg\\Paper\\Figure\\Fig10\\ch_out_mod1_heil8.tif'
    ds = gdal.Open(file)
    arr = ds.GetRasterBand(1).ReadAsArray()
    mean = []
    min = []
    max = []
    for i in range(arr.shape[0]):
        q = arr[i, :].flatten()
        q = np.delete(q, np.argwhere(np.isnan(q)))
        mean.append(np.nanmean(q))
        if q != np.array([]):
            min.append(np.sort(q)[int(q.shape[0] * 0.05)])
            max.append(np.sort(q)[int(q.shape[0] * 0.95)])
        else:
            min.append(np.nan)
            max.append(np.nan)

    mean_re, min_re, max_re = [], [], []
    for _ in range(int(len(mean)/20)):
        qq = (_ + 1) * 20
        if qq > len(mean):
            mean_re.append(np.nanmean(mean[_ * 20: -1]))
            min_re.append(np.nanmean(min[_ * 20: -1]))
            max_re.append(np.nanmean(max[_ * 20: -1]))
        else:
            mean_re.append(np.nanmean(mean[_ * 20: (_ + 1) * 20]))
            min_re.append(np.nanmean(min[_ * 20: (_ + 1) * 20]))
            max_re.append(np.nanmean(max[_ * 20: (_ + 1) * 20]))

    min, max, mean = min_re, max_re, mean_re
    fig, ax = plt.subplots(figsize=(int(arr.shape[0] / 2000), 2.3), constrained_layout=True)
    ax.plot(range(len(mean)), mean, c=(0,0,0), lw=2)
    ax.fill_between(np.linspace(0, len(max), len(max)), min, max, color=(197/255, 197/255, 197/255), alpha=0.5, lw=2)
    ax.set_xlim([0, len(max)])
    ax.set_ylim([2.6, 5.8])
    ax.set_yticks([2.6, 3.4, 4.2, 5.0, 5.8])
    ax.set_yticklabels(['2', '3', '4', '5', '6'], fontname='Times New Roman', fontsize=12)

    # current_axes = plt.axes()
    # current_axes.get_xaxis().set_visible(False)
    # ax[1, 0] = plt.imshow(arr, cmap=copy.copy(plt.cm.plasma))
    plt.savefig(f'G:\A_veg\Paper\Figure\Fig10\\fig10_y.png', dpi=600)


    mean = []
    min = []
    max = []
    for i in range(arr.shape[1]):
        q = arr[:, i].flatten()
        q = np.delete(q, np.argwhere(np.isnan(q)))
        mean.append(np.nanmean(q))
        if q != np.array([]):
            min.append(np.sort(q)[int(q.shape[0] * 0.05)])
            max.append(np.sort(q)[int(q.shape[0] * 0.95)])
        else:
            min.append(np.nan)
            max.append(np.nan)

    mean_re, min_re, max_re = [], [], []
    for _ in range(int(len(mean)/20)):
        qq = (_ + 1) * 20
        if qq > len(mean):
            mean_re.append(np.nanmean(mean[_ * 20: -1]))
            min_re.append(np.nanmean(min[_ * 20: -1]))
            max_re.append(np.nanmean(max[_ * 20: -1]))
        else:
            mean_re.append(np.nanmean(mean[_ * 20: (_ + 1) * 20]))
            min_re.append(np.nanmean(min[_ * 20: (_ + 1) * 20]))
            max_re.append(np.nanmean(max[_ * 20: (_ + 1) * 20]))

    min, max, mean = min_re, max_re, mean_re
    fig, ax = plt.subplots(figsize=(int(arr.shape[1] / 2000), 2.3), constrained_layout=True)
    ax.plot(range(len(mean)), mean, c=(0,0,0), lw=2)
    ax.fill_between(np.linspace(0, len(max), len(max)), min, max, color=(197/255, 197/255, 197/255), alpha=0.5, lw=2)
    ax.set_xlim([0, len(max)])
    ax.set_ylim([2.6, 5.8])
    ax.set_yticks([2.6, 3.4, 4.2, 5.0, 5.8])
    ax.set_yticklabels(['2', '3', '4', '5', '6'], fontname='Times New Roman', fontsize=12)

    plt.savefig(f'G:\A_veg\Paper\Figure\Fig10\\fig10_x.png', dpi=300)


def fig121_func():

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=12)
    plt.rc('axes', linewidth=3)

    file = 'G:\\A_veg\\Paper\\Figure\\Fig10\\ch_out_mod1_heil8.tif'
    ds = gdal.Open(file)
    arr = ds.GetRasterBand(1).ReadAsArray()
    mean = []
    min = []
    max = []
    for i in range(arr.shape[0]):
        q = arr[i, :].flatten()
        q = np.delete(q, np.argwhere(np.isnan(q)))
        mean.append(np.nanmean(q))
        if q != np.array([]):
            min.append(np.sort(q)[int(q.shape[0] * 0.05)])
            max.append(np.sort(q)[int(q.shape[0] * 0.95)])
        else:
            min.append(np.nan)
            max.append(np.nan)

    mean_re, min_re, max_re = [], [], []
    for _ in range(int(len(mean)/20)):
        qq = (_ + 1) * 20
        if qq > len(mean):
            mean_re.append(np.nanmean(mean[_ * 20: -1]))
            min_re.append(np.nanmean(min[_ * 20: -1]))
            max_re.append(np.nanmean(max[_ * 20: -1]))
        else:
            mean_re.append(np.nanmean(mean[_ * 20: (_ + 1) * 20]))
            min_re.append(np.nanmean(min[_ * 20: (_ + 1) * 20]))
            max_re.append(np.nanmean(max[_ * 20: (_ + 1) * 20]))

    min, max, mean = min_re, max_re, mean_re
    fig, ax = plt.subplots(figsize=(int(arr.shape[0] / 2000), 2.3), constrained_layout=True)
    ax.plot(range(len(mean)), mean, c=(0,0,0), lw=2)
    ax.fill_between(np.linspace(0, len(max), len(max)), min, max, color=(197/255, 197/255, 197/255), alpha=0.5, lw=2)
    ax.set_xlim([0, len(max)])
    ax.set_ylim([2.6, 5.8])
    ax.set_yticks([2.6, 3.4, 4.2, 5.0, 5.8])
    ax.set_yticklabels(['2', '3', '4', '5', '6'], fontname='Times New Roman', fontsize=12)

    # current_axes = plt.axes()
    # current_axes.get_xaxis().set_visible(False)
    # ax[1, 0] = plt.imshow(arr, cmap=copy.copy(plt.cm.plasma))
    plt.savefig(f'G:\A_veg\Paper\Figure\Fig10\\fig10_y.png', dpi=600)


    mean = []
    min = []
    max = []
    for i in range(arr.shape[1]):
        q = arr[:, i].flatten()
        q = np.delete(q, np.argwhere(np.isnan(q)))
        mean.append(np.nanmean(q))
        if q != np.array([]):
            min.append(np.sort(q)[int(q.shape[0] * 0.05)])
            max.append(np.sort(q)[int(q.shape[0] * 0.95)])
        else:
            min.append(np.nan)
            max.append(np.nan)

    mean_re, min_re, max_re = [], [], []
    for _ in range(int(len(mean)/20)):
        qq = (_ + 1) * 20
        if qq > len(mean):
            mean_re.append(np.nanmean(mean[_ * 20: -1]))
            min_re.append(np.nanmean(min[_ * 20: -1]))
            max_re.append(np.nanmean(max[_ * 20: -1]))
        else:
            mean_re.append(np.nanmean(mean[_ * 20: (_ + 1) * 20]))
            min_re.append(np.nanmean(min[_ * 20: (_ + 1) * 20]))
            max_re.append(np.nanmean(max[_ * 20: (_ + 1) * 20]))

    min, max, mean = min_re, max_re, mean_re
    fig, ax = plt.subplots(figsize=(int(arr.shape[1] / 2000), 2.3), constrained_layout=True)
    ax.plot(range(len(mean)), mean, c=(0,0,0), lw=2)
    ax.fill_between(np.linspace(0, len(max), len(max)), min, max, color=(197/255, 197/255, 197/255), alpha=0.5, lw=2)
    ax.set_xlim([0, len(max)])
    ax.set_ylim([2.6, 5.8])
    ax.set_yticks([2.6, 3.4, 4.2, 5.0, 5.8])
    ax.set_yticklabels(['2', '3', '4', '5', '6'], fontname='Times New Roman', fontsize=12)

    plt.savefig(f'G:\A_veg\Paper\Figure\Fig10\\fig10_x.png', dpi=300)


def fig14_func():

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=20)
    plt.rc('axes', linewidth=2)

    res_dic = {}
    for section in ['ch', 'jj', 'hh', 'yz']:
        shpfile = f'G:\\A_veg\\Paper\\Figure\\Fig11\\{section}.shp'
        for year in ['2019', '2020', '2021', '2022']:
            if not os.path.exists(f'G:\\A_veg\\Paper\\Figure\\Fig11\\tif\\{section}_{year}.tif'):
                tif_file = f'G:\\A_veg\\S2_all\\XGB_vhm\\Feature_table4heightmap\\peak_{year}\\predicted_feature_tif\\ch_out_mod0_heil8.tif'
                gdal.Warp(f'G:\\A_veg\\Paper\\Figure\\Fig11\\tif\\{section}_{year}.tif', tif_file, cutlineDSName=shpfile, cropToCutline=True, xRes=10, yRes=10, outputType=gdal.GDT_Float32,dstNodata=np.nan)
            ds = gdal.Open(f'G:\\A_veg\\Paper\\Figure\\Fig11\\tif\\{section}_{year}.tif')
            res_dic[f'{section}_{year}'] = ds.GetRasterBand(1).ReadAsArray()

    a = [np.nanmean(res_dic[f'{section}_{2020}'] - res_dic[f'{section}_{str(int(2020) - 1)}']) for section in ['ch', 'jj', 'hh', 'yz']]

    # Comparable
    for section in ['ch', 'jj', 'hh', 'yz']:

        range = None
        for year in ['2019', '2020', '2021', '2022']:
            if range is None:
                range = copy.copy(res_dic[f'{section}_{year}'])
                range[~np.isnan(range)] = 1
            else:
                range[np.isnan(res_dic[f'{section}_{year}'])] = np.nan

        data_mean = None
        data_max = []
        data_min = []
        for year in ['2019', '2020', '2021', '2022']:
            _ = res_dic[f'{section}_{year}'] * range
            _ = _.flatten()
            _ = np.delete(_, np.argwhere(np.isnan(_)))
            _.sort()
            _ = _[int(_.shape[0] * 0.005): int(_.shape[0] * 0.995)]

            q = np.ones_like(_) * (int(year) - 2019)

            if data_mean is None:
                data_mean = np.stack([q, _], axis=0)
            else:
                data_mean = np.concatenate((data_mean, np.stack([q, _], axis=0)), axis=1)
            data_max.append(_[int(len(_) * 0.05)])
            data_min.append(_[int(len(_) * 0.95)])
            res_dic[f'{section}_{year}_v'] = _

        data_mean = pd.DataFrame(data_mean.transpose(), columns=['x', 'y'])
        data_all = [res_dic[f'{section}_{year}_v'] for year in ['2019', '2020', '2021', '2022']]
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)

        paras1, extras = curve_fit(poly2, [0, 1, 2, 3], data_max, maxfev=50000)
        paras2, extras = curve_fit(poly2, [0, 1, 2, 3], data_min, maxfev=50000)
        paras, extras = curve_fit(poly2, np.array(data_mean['x']), np.array(data_mean['y']), maxfev=50000)

        # bplot1 = ax.boxplot(data_all, notch=False, positions = [2019, 2020, 2021, 2022],  widths=0.2, patch_artist=True, whis=(5, 95), showfliers=False, zorder=1, showmeans=False, )
        # colors = [(51/256, 62/256, 81/256), (112/256, 159/256, 204/256), (195/256, 121/256, 0/256), (177/256, 177/256, 177/256)]
        # for patch, color in zip(bplot1['boxes'], colors):
        #     patch.set_facecolor(color)
        #     patch._linewidth = 2

        sns.violinplot(data=data_mean, x="x", y="y", alpha=.55, zorder=2, legend=False, width=0.6, linewidth=0.9, split=True, palette=[(189/256, 215/256, 231/256), (107/256, 174/256, 214/256), (49/256, 130/256, 189/256), (8/256, 81/256, 156/256)])
        ax.fill_between(np.linspace(-0.5, 4.5, 100), poly2(np.linspace(-0.5, 4.5, 100), paras1[0], paras1[1], paras1[2]), poly2(np.linspace(0.5, 4.5, 100), paras2[0], paras2[1], paras2[2]), zorder=1, alpha = 0.1, fc=(0.5, 0.5, 0.5))
        ax.plot(np.linspace(-0.5, 4.5, 100), poly2(np.linspace(-0.5, 4.5, 100), paras[0], paras[1], paras[2]), zorder=4, c=(0.8, 0.1, 0.1), lw=4)
        ax.plot(np.linspace(-0.5, 4.5, 100), poly2(np.linspace(-0.5, 4.5, 100), paras1[0], paras1[1], paras1[2]), c=(0.1, 0.1, 0.1), lw=2, ls='--', zorder=1)
        ax.plot(np.linspace(-0.5, 4.5, 100), poly2(np.linspace(-0.5, 4.5, 100), paras2[0], paras2[1], paras2[2]), c=(0.1, 0.1, 0.1), lw=2, ls='--', zorder=1)

        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['2019', '2020', '2021', '2022'], fontname='Times New Roman', fontsize=20)
        ax.set_xlim([-0.5, 3.5])
        ax.set_ylim([2, 6])
        plt.savefig(f'G:\\A_veg\\Paper\\Figure\\Fig11\\fig\\{section}.png', dpi=300)
        fig, ax = None, None

    sns.set_theme(style="whitegrid", )
    plt.rc('axes', linewidth=2)
    plt.rc('axes', edgecolor=(0,0,0))
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)

    # dic_temp = {'sn':[], 'ch': []}
    # for section in ['ch', 'jj', 'hh', 'yz']:
    #     _ =  res_dic[f'{section}_2022']
    #     _ = _.flatten()
    #     _ = np.delete(_, np.argwhere(np.isnan(_)))
    #     dic_temp['ch'].extend(_.tolist())
    #     dic_temp['sn'].extend([section for temp in range(_.shape[0])])
    # df_temp = pd.DataFrame(dic_temp)
    # seaborn.boxenplot(data=df_temp, x = 'sn', y='ch', scale="area", saturation = 0.6, width=0.6, order=['yz', 'jj', 'ch','hh'],  color="b")
    # ax.set_ylim([2, 7])
    # plt.savefig(f'G:\\A_veg\\Paper\\Figure\\Fig11\\fig\\sect4.png', dpi=300)

    # Diff
    a = [np.nanmean(res_dic[f'{section}_{2020}'] - np.nanmean(res_dic[f'{section}_{str(int(2020) - 1)}'])) for section in ['ch', 'jj', 'hh', 'yz']]

    for section in ['ch', 'jj', 'hh', 'yz']:
        for year in ['2020', '2021', '2022']:
            arr_temp = res_dic[f'{section}_{year}'] - res_dic[f'{section}_{str(int(year) - 1)}']
            arr_temp = arr_temp.flatten()
            arr_temp = np.delete(arr_temp, np.argwhere(np.isnan(arr_temp)))
            res_dic[f'{str(int(year) - 1)}_{year}_{section}'] = arr_temp

        # fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
        # seaborn.violinplot([res_dic[f'2019_2020_{section}'], res_dic[f'2020_2021_{section}'], res_dic[f'2021_2022_{section}']])
        # plt.savefig(f'G:\\A_veg\\Paper\\Figure\\Fig11\\fig\\{section}.png', dpi=300)
        # fig, ax = None, None

def ep_fig2_func():

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=22)
    plt.rc('axes', linewidth=2)

    data = 'G:\\A_Landsat_veg\\Paper\\Fig4\\data.xlsx'
    data_pd = pd.read_excel(data)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 16), constrained_layout=True)
    ax[0].plot(data_pd['DOY'], data_pd['total biomas site1'], lw=3, ls='--', c=(25/256, 25/256, 25/256), zorder=3)
    ax[0].scatter(data_pd['DOY'], data_pd['total biomas site1'], zorder=4, s=13**2, marker='s', edgecolors=(0/256, 0/256, 0/256), facecolor=(1, 1, 1), alpha=1, linewidths=2)
    ax[0].errorbar(data_pd['DOY'], data_pd['total biomas site1'], yerr=None)
    ax[0].plot(data_pd['DOY'], data_pd['leaf biomass (site1)'], lw=3, ls='--', c=(25/256, 25/256, 25/256), zorder=3)
    ax[0].scatter(data_pd['DOY'], data_pd['leaf biomass (site1)'], zorder=4, s=14**2, marker='^', edgecolors=(0/256, 0/256, 0/256), facecolor=(1, 1, 1), alpha=1, linewidths=2)
    ax[0].fill_between(data_pd['DOY'], [0, 0, 0, 0, 0, 0, 0, 0,0], data_pd['total biomas site1'], zorder=1, alpha=0.5, fc=(54/256, 92/256, 141/256))
    ax[0].fill_between(data_pd['DOY'], [0, 0, 0, 0, 0, 0, 0, 0,0], data_pd['leaf biomass (site1)'], zorder=2, alpha=0.5, fc=(196/256, 78/256, 82/256))
    ax[0].set_xticks([75, 135, 195, 255, 315])
    ax[0].grid(axis='y', color=(240 / 256, 240 / 256, 240 / 256))
    ax[0].set_xticklabels(['3', '5', '7', '9', '11'], fontname='Times New Roman', fontsize=26)
    ax[0].set_xlabel('Date', fontname='Times New Roman', fontsize=34, fontweight='bold')
    ax[0].set_ylabel('Biomass per plant/g', fontname='Times New Roman', fontsize=34, fontweight='bold')
    ax[0].set_xlim([60, 345])
    ax[0].set_ylim([0, 25])

    ax[1].plot(data_pd['DOY'], data_pd['total biomas site2'], lw=3, ls='--', c=(25/256, 25/256, 25/256), zorder=3)
    ax[1].scatter(data_pd['DOY'], data_pd['total biomas site2'], zorder=4, s=13**2, marker='s', edgecolors=(0/256, 0/256, 0/256), facecolor=(1, 1, 1), alpha=1, linewidths=2)
    ax[1].errorbar(data_pd['DOY'], data_pd['total biomas site2'], yerr=None)
    ax[1].plot(data_pd['DOY'], data_pd['leaf biomass (site2)'], lw=3, ls='--', c=(25/256, 25/256, 25/256), zorder=3)
    ax[1].scatter(data_pd['DOY'], data_pd['leaf biomass (site2)'], zorder=4, s=14**2, marker='^', edgecolors=(0/256, 0/256, 0/256), facecolor=(1, 1, 1), alpha=1, linewidths=2)
    ax[1].fill_between(data_pd['DOY'], [0, 0, 0, 0, 0, 0, 0, 0,0], data_pd['total biomas site2'], zorder=1, alpha=0.5, fc=(54/256, 92/256, 141/256))
    ax[1].fill_between(data_pd['DOY'], [0, 0, 0, 0, 0, 0, 0, 0,0], data_pd['leaf biomass (site2)'], zorder=2, alpha=0.5, fc=(196/256, 78/256, 82/256))
    ax[1].set_xticks([75, 135, 195, 255, 315])
    ax[1].grid(axis='y', color=(240 / 256, 240 / 256, 240 / 256))
    ax[1].set_xticklabels(['3', '5', '7', '9', '11'], fontname='Times New Roman', fontsize=26)
    ax[1].set_xlabel('Date', fontname='Times New Roman', fontsize=34, fontweight='bold')
    ax[1].set_ylabel('Biomass per plant/g', fontname='Times New Roman', fontsize=34, fontweight='bold')
    ax[1].set_xlim([60, 345])
    ax[1].set_ylim([0, 70])
    plt.savefig(f'D:\A_PhD_Main_paper\Chap.2\Figure\\2.4.2\\3\\Fig4.png', dpi=300)


def fig15_func():

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=18)
    plt.rc('axes', linewidth=2)

    for sec, content in zip(['all', 'yz', 'jj', 'ch', 'hh', ], [ None, (0, 14482, 0, 2810), (0, 14482, 2810, 18320), (0, 14482, 18320, 30633), (0, 14482, 30633, 49071)]):
        if not os.path.exists(f'G:\\A_veg\\Paper\\Figure\\Fig12\\{sec}.csv'):
            if sec != 'all':
                height_2020_ds = gdal.Open('G:\\A_veg\\S2_all\\XGB_vhm\\Feature_table4heightmap\\peak_2020\predicted_feature_tif\\ch_out_mod0_heil8.tif')
                height_2019_ds = gdal.Open('G:\\A_veg\\S2_all\\XGB_vhm\\Feature_table4heightmap\\peak_2019\predicted_feature_tif\\ch_out_mod0_heil8.tif')
                height_2021_ds = gdal.Open('G:\\A_veg\\S2_all\\XGB_vhm\\Feature_table4heightmap\\peak_2021\predicted_feature_tif\\ch_out_mod0_heil8.tif')
                height_2022_ds = gdal.Open('G:\\A_veg\\S2_all\\XGB_vhm\\Feature_table4heightmap\\peak_2022\predicted_feature_tif\\ch_out_mod0_heil8.tif')
                inundur_2020_ds = gdal.Open(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_inunduration\\{str(sec)}_section\\annual_inun_duration\\inun_duration_2020.tif')
                inunhei_2020_ds = gdal.Open(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_inunduration\\{str(sec)}_section\\annual_inun_duration\\inun_height_2020.tif')

                arr_2022 = height_2022_ds.GetRasterBand(1).ReadAsArray()[content[0]: content[1], content[2]: content[3]]
                arr_2020 = height_2020_ds.GetRasterBand(1).ReadAsArray()[content[0]: content[1], content[2]: content[3]]
                arr_2019 = height_2019_ds.GetRasterBand(1).ReadAsArray()[content[0]: content[1], content[2]: content[3]]
                arr_2021 = height_2021_ds.GetRasterBand(1).ReadAsArray()[content[0]: content[1], content[2]: content[3]]
                arr_inundur = inundur_2020_ds.GetRasterBand(1).ReadAsArray()
                arr_inunhei = inunhei_2020_ds.GetRasterBand(1).ReadAsArray()

                if arr_2020.shape[0] != arr_inunhei.shape[0] or arr_2020.shape[1] != arr_inunhei.shape[1]:
                    raise Exception('Error')

                dic_temp = {'h_2020': [], 'h_2019': [], 'h_2021': [], 'h_2022': [], 'inun_d': [], 'inun_h': []}
                for y in range(arr_2020.shape[0]):
                    for x in range(arr_2020.shape[1]):
                        if ~np.isnan(arr_2020[y, x]):
                            dic_temp['h_2019'].append(arr_2019[y, x])
                            dic_temp['h_2020'].append(arr_2020[y, x])
                            dic_temp['h_2021'].append(arr_2021[y, x])
                            dic_temp['h_2022'].append(arr_2022[y, x])
                            dic_temp['inun_d'].append(arr_inundur[y, x])
                            dic_temp['inun_h'].append(arr_inunhei[y, x])
                pd_temp = pd.DataFrame(dic_temp)
                pd_temp.to_csv(f'G:\\A_veg\\Paper\\Figure\\Fig12\\{sec}.csv')
            else:
                for _ in ['yz', 'jj', 'ch', 'hh']:
                    pd_temp_ = pd.read_csv(f'G:\\A_veg\\Paper\\Figure\\Fig12\\{_}.csv')
                    if _ != 'yz':
                        pd_all = pd.concat((pd_all, pd_temp_))
                    else:
                        pd_all = copy.deepcopy(pd_temp_)
                pd_all.to_csv(f'G:\\A_veg\\Paper\\Figure\\Fig12\\{sec}.csv')

        pd_temp = pd.read_csv(f'G:\\A_veg\\Paper\\Figure\\Fig12\\{sec}.csv')
        pd_temp['20_21_diff'] = pd_temp['h_2020'] - pd_temp['h_2021']
        pd_temp['20_21_perc'] = (pd_temp['h_2020'] - pd_temp['h_2021']) / pd_temp['h_2020']
        pd_temp['20_22_diff'] = pd_temp['h_2020'] - pd_temp['h_2022']
        pd_temp['20_22_perc'] = (pd_temp['h_2020'] - pd_temp['h_2022']) / pd_temp['h_2020']
        pd_temp['19_20_diff'] = pd_temp['h_2020'] - pd_temp['h_2019']
        pd_temp['19_20_perc'] = (pd_temp['h_2020'] - pd_temp['h_2019']) / pd_temp['h_2019']

        for dic_name in ['20_21_diff', '20_21_perc', '19_20_diff', '19_20_perc', '20_22_diff', '20_22_perc']:
            fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
            pd_temp_ = pd_temp[[dic_name, 'inun_d', 'inun_h']]
            pd_temp_ = pd_temp_.dropna()
            pd_temp__ = pd_temp_[pd_temp_['inun_d'] < 90]
            box_list, mid_list = [], []
            max_ = 0.
            for q in range(90):
                box_temp = pd_temp__[pd_temp__['inun_d'] == q][dic_name]
                box_list.append(box_temp.sort_values()[int(box_temp.shape[0] * 0.05): int(box_temp.shape[0] * 0.95)])
                mid_list.append(np.nanmean(box_temp))
                max_ = np.nanmax((max_, np.max(np.absolute(box_list[-1][int(box_list[-1].shape[0] * 0.15): int(box_list[-1].shape[0] * 0.85)]))))
            # box_temp = pd_temp__[pd_temp__['inun_d'] == 0][dic_name]
            # box_temp = box_temp.sort_values()[int(box_temp.shape[0] * 0.1): int(box_temp.shape[0] * 0.9)]
            box_temp = ax.boxplot(box_list, vert=True,  notch=False, widths=0.98, patch_artist=True, whis=(15, 85), showfliers=False, zorder=4, )

            for patch in box_temp['boxes']:
                patch.set_facecolor((72/256,127/256,166/256))
                patch.set_alpha(0.5)
                patch.set_linewidth(0.2)

            for median in box_temp['medians']:
                median.set_lw(0.8)
                # median.set_marker('^')
                median.set_color((255/256, 128/256, 64/256))

            ax.plot(np.linspace(0, 90, 100), np.linspace(0, 0, 100), lw=2.5, c=(0, 0, 0), ls='-', zorder=5)
            ax.scatter(np.linspace(1, 90, 90), mid_list, marker='^', s=10 ** 2, facecolor=(0.8, 0, 0), zorder=7, edgecolor=(0.0, 0.0, 0.0), linewidth=0.2)
            # ax.scatter(pd_temp__['inun_d'], pd_temp__[dic_name], s=2 ** 2, edgecolor=(1, 1, 1), facecolor=(47/256,85/256,151/256), alpha=0.1, linewidth=0, marker='^', zorder=5)
            # s = linregress(np.array(pd_temp_[pd_temp_['inun_d'] > 0]['inun_d'].tolist()),
            #               np.array(pd_temp_[pd_temp_['inun_d'] > 0][dic_name].tolist()))

            # popt, pcov = curve_fit(poly3, pd_temp__['inun_d'], pd_temp__[dic_name], maxfev=50000, method="trf")
            # ax.plot(np.linspace(1, 90, 90), poly3(np.linspace(1, 90, 90), *popt), lw=2, ls='--', c=(0.8, 0, 0))

            ax.plot(np.linspace(36, 90, 90), np.linspace(np.mean(mid_list[36:]), np.mean(mid_list[36:]), 90), lw=2, ls='--', c=(0.8, 0, 0), zorder=8)
            z = np.polyfit(np.linspace(1, 90, 90), mid_list, 15)
            p = np.poly1d(z)
            ax.plot(np.linspace(1, 13, 100), p(np.linspace(1, 13, 100)), lw=2, ls='--', c=(0.8, 0, 0), zorder=8)
            ax.plot(np.linspace(13, 36, 90), np.linspace(p(13), p(13), 90), lw=2, ls='--', c=(0.8, 0, 0), zorder=8)
            print(str(dic_name))
            print(str(p(13)))
            print(str(np.mean(mid_list[36:])))
            ax.set_ylim(-0.2, 0.2)
            ax.set_yticks([-0.2,-0.1,0,0.1,0.2])
            ax.set_yticklabels(['-20%','-10%','0%','10%','20%'])
            ax.set_xticks([1,11,21,31,41,51,61])
            ax.set_xticklabels(['0','10','20','30','40','50','60'])
            # ax.set_ylim([- float(int(max_ * 20) + 1) / 20, float(int(max_ * 20) + 1) / 20])
            ax.set_xlim([0.5, 61.5])
            plt.savefig(f'G:\A_veg\Paper\Figure\Fig12\\{dic_name}_{sec}.png', dpi=300)


def fig16_func():

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=18)
    plt.rc('axes', linewidth=2)

    for sec, content in zip(['all', 'yz', 'jj', 'ch', 'hh', ], [None, (0, 14482, 0, 2810), (0, 14482, 2810, 18320), (0, 14482, 18320, 30633), (0, 14482, 30633, 49071)]):
        if not os.path.exists(f'G:\\A_veg\\Paper\\Figure\\Fig13\\{sec}.csv'):
            if sec != 'all':
                height_2020_ds = gdal.Open('G:\\A_veg\\S2_all\\XGB_vhm\\Feature_table4heightmap\\peak_2020\predicted_feature_tif\\ch_out_mod0_heil8.tif')
                height_2019_ds = gdal.Open('G:\\A_veg\\S2_all\\XGB_vhm\\Feature_table4heightmap\\peak_2019\predicted_feature_tif\\ch_out_mod0_heil8.tif')
                height_2021_ds = gdal.Open('G:\\A_veg\\S2_all\\XGB_vhm\\Feature_table4heightmap\\peak_2021\predicted_feature_tif\\ch_out_mod0_heil8.tif')
                height_2022_ds = gdal.Open('G:\\A_veg\\S2_all\\XGB_vhm\\Feature_table4heightmap\\peak_2022\predicted_feature_tif\\ch_out_mod0_heil8.tif')
                inunhdur_2020_ds = gdal.Open(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_inunduration\\{str(sec)}_section\\annual_inun_duration\\mean_inun_height_2020.tif')
                inunhei_2020_ds = gdal.Open(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_inunduration\\{str(sec)}_section\\annual_inun_duration\\inun_height_2020.tif')

                arr_2022 = height_2022_ds.GetRasterBand(1).ReadAsArray()[content[0]: content[1], content[2]: content[3]]
                arr_2020 = height_2020_ds.GetRasterBand(1).ReadAsArray()[content[0]: content[1], content[2]: content[3]]
                arr_2019 = height_2019_ds.GetRasterBand(1).ReadAsArray()[content[0]: content[1], content[2]: content[3]]
                arr_2021 = height_2021_ds.GetRasterBand(1).ReadAsArray()[content[0]: content[1], content[2]: content[3]]
                arr_mean_inunhei = inunhdur_2020_ds.GetRasterBand(1).ReadAsArray()
                arr_inunhei = inunhei_2020_ds.GetRasterBand(1).ReadAsArray()

                if arr_2020.shape[0] != arr_inunhei.shape[0] or arr_2020.shape[1] != arr_inunhei.shape[1]:
                    raise Exception('Error')

                dic_temp = {'h_2020': [], 'h_2019': [], 'h_2021': [], 'h_2022': [], 'mean_inun_h': [], 'inun_h': []}
                for y in range(arr_2020.shape[0]):
                    for x in range(arr_2020.shape[1]):
                        if ~np.isnan(arr_2020[y, x]):
                            dic_temp['h_2019'].append(arr_2019[y, x])
                            dic_temp['h_2020'].append(arr_2020[y, x])
                            dic_temp['h_2021'].append(arr_2021[y, x])
                            dic_temp['h_2022'].append(arr_2022[y, x])
                            dic_temp['mean_inun_h'].append(arr_mean_inunhei[y, x])
                            dic_temp['inun_h'].append(arr_inunhei[y, x])
                pd_temp = pd.DataFrame(dic_temp)
                pd_temp.to_csv(f'G:\\A_veg\\Paper\\Figure\\Fig13\\{sec}.csv')
            else:
                for _ in ['yz', 'jj', 'ch', 'hh']:
                    pd_temp_ = pd.read_csv(f'G:\\A_veg\\Paper\\Figure\\Fig13\\{_}.csv')
                    if _ != 'yz':
                        pd_all = pd.concat((pd_all, pd_temp_))
                    else:
                        pd_all = copy.deepcopy(pd_temp_)
                pd_all.to_csv(f'G:\\A_veg\\Paper\\Figure\\Fig13\\{sec}.csv')

        pd_temp = pd.read_csv(f'G:\\A_veg\\Paper\\Figure\\Fig13\\{sec}.csv')
        pd_temp['20_21_diff'] = pd_temp['h_2020'] - pd_temp['h_2021']
        pd_temp['20_21_perc'] = (pd_temp['h_2020'] - pd_temp['h_2021']) / pd_temp['h_2020']
        pd_temp['19_20_diff'] = pd_temp['h_2020'] - pd_temp['h_2019']
        pd_temp['19_20_perc'] = (pd_temp['h_2020'] - pd_temp['h_2019']) / pd_temp['h_2019']
        pd_temp['mean_inun_h_per'] = pd_temp['mean_inun_h'] / pd_temp['h_2020']
        pd_temp['max_inun_h_per'] = pd_temp['inun_h'] / pd_temp['h_2020']

        for dic_name in ['20_21_diff', '20_21_perc', '19_20_diff', '19_20_perc']:

            pd_temp_ = pd_temp[[dic_name, 'mean_inun_h_per', 'max_inun_h_per', 'mean_inun_h', 'inun_h', 'h_2020']]
            pd_temp_ = pd_temp_.dropna()
            pd_temp__ = pd_temp_[(pd_temp_['h_2020'] <= 7) & (pd_temp_['h_2020'] > 2)]
            pd_temp__ = pd_temp__[pd_temp__['max_inun_h_per'] < 5]
            pd_temp__ = pd_temp__[pd_temp__['max_inun_h_per'] > 0]
            max_inun_h_list = []
            mean_inun_h_list = []
            abs_max_inun_h_list = []
            abs_mean_inun_h_list = []
            q_list, q2_list = [], []
            q3_list, q4_list = [], []
            for q in range(200):
                q_list.append((q+0.5) / 40)
                max_inun_h_list.append(np.nanmean(pd_temp__[(pd_temp__['max_inun_h_per'] >= q / 40) & (pd_temp__['max_inun_h_per'] < (q + 1) / 40)][dic_name]))
                q2_list.append((q+0.5) / 80)
                mean_inun_h_list.append(np.nanmean(pd_temp__[(pd_temp__['mean_inun_h_per'] >= q / 80) & (pd_temp__['mean_inun_h_per'] < (q + 1) / 80)][dic_name]))
                q3_list.append((q+0.5) / 20)
                abs_mean_inun_h_list.append(np.nanmean(pd_temp__[(pd_temp__['mean_inun_h'] >= q / 20) & (pd_temp__['mean_inun_h'] < (q + 1) / 20)][dic_name]))
                q4_list.append((q+0.5) / 20)
                abs_max_inun_h_list.append(np.nanmean(pd_temp__[(pd_temp__['inun_h'] >= q / 20) & (pd_temp__['inun_h'] < (q + 1) / 20)][dic_name]))

            fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
            # ax.scatter(pd_temp__['max_inun_h_per'], pd_temp__[dic_name], zorder=1)
            # popts, peff = curve_fit(ln_x5, pd_temp__['max_inun_h_per'], pd_temp__[dic_name], maxfev=50000)
            # ax.plot(np.linspace(0, 5, 100), ln_x5(np.linspace(0, 5, 100), *popts), zorder=3, c=(1,0,0))
            ax.scatter(q_list, max_inun_h_list)
            # sns.histplot(data=pd_temp__, x="max_inun_h_per", y=dic_name, bins=50)
            # sns.kdeplot(data=pd_temp__, x="max_inun_h_per", y=dic_name, fill=True, thresh=0, levels=10, cmap="mako",)
            plt.savefig(f'G:\A_veg\Paper\Figure\Fig13\\max_inun_per_{dic_name}_{sec}.png', dpi=300)
            plt.close()

            fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
            # ax.scatter(pd_temp__['mean_inun_h_per'], pd_temp__[dic_name], zorder=1)
            # popts, peff = curve_fit(ln_x3, pd_temp__['mean_inun_h_per'], pd_temp__[dic_name], maxfev=50000)
            # ax.plot(np.linspace(0, 3, 100), ln_x3(np.linspace(0, 3, 100), *popts), zorder=3, c=(1,0,0))
            ax.scatter(q2_list, mean_inun_h_list)
            # sns.histplot(data=pd_temp__, x="mean_inun_h_per", y=dic_name, bins=50)
            # sns.kdeplot(data=pd_temp__, x="mean_inun_h_per", y=dic_name, fill=True, thresh=0, levels=100, cmap="mako", )
            plt.savefig(f'G:\A_veg\Paper\Figure\Fig13\\mean_inun_per_{dic_name}_{sec}.png', dpi=300)
            plt.close()

            fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
            # ax.scatter(pd_temp__['max_inun_h_per'], pd_temp__[dic_name], zorder=1)
            # popts, peff = curve_fit(ln_x5, pd_temp__['max_inun_h_per'], pd_temp__[dic_name], maxfev=50000)
            # ax.plot(np.linspace(0, 5, 100), ln_x5(np.linspace(0, 5, 100), *popts), zorder=3, c=(1,0,0))
            ax.scatter(q3_list, abs_mean_inun_h_list)
            # sns.histplot(data=pd_temp__, x="max_inun_h_per", y=dic_name, bins=50)
            # sns.kdeplot(data=pd_temp__, x="max_inun_h_per", y=dic_name, fill=True, thresh=0, levels=10, cmap="mako",)
            plt.savefig(f'G:\A_veg\Paper\Figure\Fig13\\max_inun_{dic_name}_{sec}.png', dpi=300)
            plt.close()

            fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
            # ax.scatter(pd_temp__['mean_inun_h_per'], pd_temp__[dic_name], zorder=1)
            # popts, peff = curve_fit(ln_x3, pd_temp__['mean_inun_h_per'], pd_temp__[dic_name], maxfev=50000)
            # ax.plot(np.linspace(0, 3, 100), ln_x3(np.linspace(0, 3, 100), *popts), zorder=3, c=(1,0,0))
            ax.scatter(q4_list, abs_max_inun_h_list)
            # sns.histplot(data=pd_temp__, x="mean_inun_h_per", y=dic_name, bins=50)
            # sns.kdeplot(data=pd_temp__, x="mean_inun_h_per", y=dic_name, fill=True, thresh=0, levels=100, cmap="mako", )
            plt.savefig(f'G:\A_veg\Paper\Figure\Fig13\\mean_inun_{dic_name}_{sec}.png', dpi=300)
            plt.close()


            # box_list, mid_list = [], []
            # max_ = 0.
            # for q in range(90):
            #     box_temp = pd_temp__[pd_temp__['inun_d'] == q][dic_name]
            #     box_list.append(box_temp.sort_values()[int(box_temp.shape[0] * 0.05): int(box_temp.shape[0] * 0.95)])
            #     mid_list.append(np.nanmean(box_temp))
            #     max_ = np.nanmax((max_, np.max(np.absolute(box_list[-1][int(box_list[-1].shape[0] * 0.15): int(box_list[-1].shape[0] * 0.85)]))))
            # # box_temp = pd_temp__[pd_temp__['inun_d'] == 0][dic_name]
            # # box_temp = box_temp.sort_values()[int(box_temp.shape[0] * 0.1): int(box_temp.shape[0] * 0.9)]
            # box_temp = ax.boxplot(box_list, vert=True,  notch=False, widths=0.98, patch_artist=True, whis=(15, 85), showfliers=False, zorder=4, )
            #
            # for patch in box_temp['boxes']:
            #     patch.set_facecolor((72/256,127/256,166/256))
            #     patch.set_alpha(0.5)
            #     patch.set_linewidth(0.2)
            #
            # for median in box_temp['medians']:
            #     median.set_lw(0.8)
            #     # median.set_marker('^')
            #     median.set_color((255/256, 128/256, 64/256))
            #
            # ax.plot(np.linspace(0, 90, 100), np.linspace(0,0,100), lw=2.5, c=(0,0,0), ls='-', zorder=5)
            # ax.scatter(np.linspace(1, 90, 90), mid_list, marker='^', s=10 ** 2, facecolor=(0.8, 0, 0), zorder=7, edgecolor=(0.0, 0.0, 0.0), linewidth=0.2)
            # # ax.scatter(pd_temp__['inun_d'], pd_temp__[dic_name], s=2 ** 2, edgecolor=(1, 1, 1), facecolor=(47/256,85/256,151/256), alpha=0.1, linewidth=0, marker='^', zorder=5)
            # # s = linregress(np.array(pd_temp_[pd_temp_['inun_d'] > 0]['inun_d'].tolist()),
            # #               np.array(pd_temp_[pd_temp_['inun_d'] > 0][dic_name].tolist()))
            #
            # # popt, pcov = curve_fit(poly3, pd_temp__['inun_d'], pd_temp__[dic_name], maxfev=50000, method="trf")
            # # ax.plot(np.linspace(1, 90, 90), poly3(np.linspace(1, 90, 90), *popt), lw=2, ls='--', c=(0.8, 0, 0))
            #
            # ax.plot(np.linspace(36, 90, 90), np.linspace(np.mean(mid_list[36:]), np.mean(mid_list[36:]), 90), lw=2, ls='--', c=(0.8, 0, 0), zorder=8)
            # z = np.polyfit(np.linspace(1, 90, 90), mid_list, 15)
            # p = np.poly1d(z)
            # ax.plot(np.linspace(1, 13, 100), p(np.linspace(1, 13, 100)), lw=2, ls='--', c=(0.8, 0, 0), zorder=8)
            # ax.plot(np.linspace(13, 36, 90), np.linspace(p(13), p(13), 90), lw=2, ls='--', c=(0.8, 0, 0), zorder=8)
            # print(str(dic_name))
            # print(str(p(13)))
            # print(str(np.mean(mid_list[36:])))
            # ax.set_ylim(-0.2, 0.2)
            # ax.set_yticks([-0.2,-0.1,0,0.1,0.2])
            # ax.set_yticklabels(['-20%','-10%','0%','10%','20%'])
            # ax.set_xticks([1,11,21,31,41,51,61])
            # ax.set_xticklabels(['0','10','20','30','40','50','60'])
            # # ax.set_ylim([- float(int(max_ * 20) + 1) / 20, float(int(max_ * 20) + 1) / 20])
            # ax.set_xlim([0.5, 61.5])


def fig17_func():

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=18)
    plt.rc('axes', linewidth=2)

    for sec, content in zip(['all'], [None]):
        if not os.path.exists(f'G:\\A_veg\\Paper\\Figure\\Fig13\\{sec}.csv'):
            raise Exception('error')

        pd_temp = pd.read_csv(f'G:\\A_veg\\Paper\\Figure\\Fig13\\{sec}.csv')
        pd_temp['20_21_diff'] = pd_temp['h_2020'] - pd_temp['h_2021']
        pd_temp['20_21_perc'] = (pd_temp['h_2020'] - pd_temp['h_2021']) / pd_temp['h_2020']
        pd_temp['19_20_diff'] = pd_temp['h_2020'] - pd_temp['h_2019']
        pd_temp['19_20_perc'] = (pd_temp['h_2020'] - pd_temp['h_2019']) / pd_temp['h_2019']
        pd_temp['mean_inun_h_per'] = pd_temp['mean_inun_h'] / pd_temp['h_2020']
        pd_temp['max_inun_h_per'] = pd_temp['inun_h'] / pd_temp['h_2020']

        for dic_name in ['20_21_perc']:

            pd_temp_ = pd_temp[[dic_name, 'mean_inun_h_per', 'max_inun_h_per', 'mean_inun_h', 'inun_h', 'h_2020']]
            pd_temp_ = pd_temp_.dropna()
            pd_temp__ = pd_temp_[(pd_temp_['h_2020'] <= 5) & (pd_temp_['h_2020'] > 2)]
            pd_temp__ = pd_temp__[pd_temp__['max_inun_h_per'] < 5]
            pd_temp__ = pd_temp__[pd_temp__['max_inun_h_per'] > 0]
            max_inun_h_list, max_inun_h_list_upper, max_inun_h_list_lower = [], [], []
            mean_inun_h_list, mean_inun_h_list_upper, mean_inun_h_list_lower = [], [], []
            abs_max_inun_h_list, abs_max_inun_h_list_upper,  abs_max_inun_h_list_lower = [], [], []
            abs_mean_inun_h_list, abs_mean_inun_h_list_upper, abs_mean_inun_h_list_lower = [], [], []
            pd_temp___ = pd_temp__[(pd_temp__['inun_h'] >= 2) & (pd_temp__['inun_h'] < 8)]
            q_list, q2_list = [], []
            q3_list, q4_list = [], []
            for q in range(200):
                q1 = pd_temp__[(pd_temp__['max_inun_h_per'] >= q / 40) & (pd_temp__['max_inun_h_per'] < (q + 1) / 40)]
                q2 = pd_temp___[(pd_temp___['mean_inun_h_per'] >= q / 40) & (pd_temp___['mean_inun_h_per'] < (q + 1) / 40)]
                q3 = pd_temp__[(pd_temp__['mean_inun_h'] >= q / 20) & (pd_temp__['mean_inun_h'] < (q + 1) / 20)]
                q4 = pd_temp__[(pd_temp__['inun_h'] >= q / 20) & (pd_temp__['inun_h'] < (q + 1) / 20)]
                if q1.shape[0] != 0:
                    q_list.append((q + 0.5) / 40)
                    max_inun_h_list.append(np.nanmean(q1[dic_name]))
                    max_inun_h_list_upper.append(q1[dic_name].sort_values().iloc[int(q1.shape[0] * 0.7)])
                    max_inun_h_list_lower.append(q1[dic_name].sort_values().iloc[int(q1.shape[0] * 0.3)])
                if q2.shape[0] != 0:
                    q2_list.append((q + 0.5) / 40)
                    mean_inun_h_list.append(np.nanmean(q2[dic_name]))
                    mean_inun_h_list_upper.append(q2[dic_name].sort_values().iloc[int(q2.shape[0] * 0.7)])
                    mean_inun_h_list_lower.append(q2[dic_name].sort_values().iloc[int(q2.shape[0] * 0.3)])
                if q3.shape[0] != 0:
                    q3_list.append((q + 0.5) / 20)
                    abs_mean_inun_h_list.append(np.nanmean(q3[dic_name]))
                    abs_mean_inun_h_list_upper.append(q3[dic_name].sort_values().iloc[int(q3.shape[0] * 0.7)])
                    abs_mean_inun_h_list_lower.append(q3[dic_name].sort_values().iloc[int(q3.shape[0] * 0.3)])
                if q4.shape[0] != 0:
                    q4_list.append((q + 0.5) / 20)
                    abs_max_inun_h_list.append(np.nanmean(q4[dic_name]))
                    abs_max_inun_h_list_upper.append(q4[dic_name].sort_values().iloc[int(q4.shape[0] * 0.7)])
                    abs_max_inun_h_list_lower.append(q4[dic_name].sort_values().iloc[int(q4.shape[0] * 0.3)])

            fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
            # ax.scatter(pd_temp__['max_inun_h_per'], pd_temp__[dic_name], zorder=1)
            # popts, peff = curve_fit(ln_x5, pd_temp__['max_inun_h_per'], pd_temp__[dic_name], maxfev=50000)
            # ax.plot(np.linspace(0, 5, 100), ln_x5(np.linspace(0, 5, 100), *popts), zorder=3, c=(1,0,0))
            ax.scatter(q_list[0: 120], max_inun_h_list[0: 120], zorder=4, s=10 ** 2, marker='o',  edgecolors=(0.1, 0.1, 0.1), facecolor=(1, 1, 1), alpha=1, linewidths=1)
            ax.plot(q_list[0: 120], max_inun_h_list[0: 120], zorder=3, c=(0.1, 0.1, 0.1), lw=2.8, ls='-')
            ax.scatter(q_list[120:], max_inun_h_list[120:], zorder=4, s=10 ** 2, marker='s', edgecolors=(0.8, 0, 0), facecolor=(1, 1, 1), alpha=1, linewidths=1)
            ax.plot(np.linspace(3, 3, 100), np.linspace(-1, 1, 100), lw=3.5, c=(0.8, 0, 0), ls='--', zorder=9)
            potp, pcoef = curve_fit(exp_minus, q_list[120: 180], max_inun_h_list[120: 180], maxfev=50000000, )
            ax.plot(np.linspace(3, 4.5, 10000), exp_minus(np.linspace(3, 4.5, 10000), *potp), lw=3, ls='-',  c=(0.8, 0, 0), zorder=8)
            ax.plot(np.linspace(4.5, 5, 10000), np.linspace(exp_minus(4.5, *potp), exp_minus(4.5, *potp), 10000), lw=3, ls='-', c=(0.8, 0, 0), zorder=8)
            print(str(potp))

            ax.plot(np.linspace(0, 10, 100), np.linspace(0, 0, 100), lw=2, c=(0, 0, 0))
            ax.fill_between(q_list, max_inun_h_list_upper, max_inun_h_list_lower, zorder=1, linewidth=0.8, ls='-.', ec=(0, 0, 0), fc=(0.8, 0.8, 0.8), alpha=0.5)
            # sns.histplot(data=pd_temp__, x="max_inun_h_per", y=dic_name, bins=50)
            # sns.kdeplot(data=pd_temp__, x="max_inun_h_per", y=dic_name, fill=True, thresh=0, levels=10, cmap="mako",)
            ax.set_ylim([-0.15, 0.1])
            ax.set_xlim([0, 5])
            ax.set_yticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1])
            ax.set_yticklabels(['-15%', '-10%', '-5%', '0%', '5%', '10%'])
            plt.savefig(f'G:\A_veg\Paper\Figure\Fig13\\req\\max_inun_per_{dic_name}_{sec}.png', dpi=300)
            plt.close()

            fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
            # ax.scatter(pd_temp__['mean_inun_h_per'], pd_temp__[dic_name], zorder=1)
            # popts, peff = curve_fit(ln_x3, pd_temp__['mean_inun_h_per'], pd_temp__[dic_name], maxfev=50000)
            # ax.plot(np.linspace(0, 3, 100), ln_x3(np.linspace(0, 3, 100), *popts), zorder=3, c=(1,0,0))
            ax.scatter(q2_list[0: 38], mean_inun_h_list[0: 38], zorder=4, s=10 ** 2, marker='o',  edgecolors=(0.1, 0.1, 0.1), facecolor=(1, 1, 1), alpha=1, linewidths=1)
            ax.plot(q2_list[0: 38], mean_inun_h_list[0: 38], zorder=3, c=(0.1, 0.1, 0.1), lw=2.8, ls='-')
            ax.scatter(q2_list[38: ], mean_inun_h_list[38:], zorder=4, s=10**2, marker='s', edgecolors=(0.8, 0, 0), facecolor=(1, 1, 1), alpha=1, linewidths=1)
            ax.plot(np.linspace(0, 10, 100), np.linspace(0, 0, 100), lw=2, c=(0, 0, 0))
            ax.fill_between(q2_list, mean_inun_h_list_upper, mean_inun_h_list_lower, zorder=1, linewidth=0.8, ls='-.', ec=(0, 0, 0), fc=(0.8, 0.8, 0.8), alpha=0.5)
            # sns.histplot(data=pd_temp__, x="mean_inun_h_per", y=dic_name, bins=50)
            # sns.kdeplot(data=pd_temp__, x="mean_inun_h_per", y=dic_name, fill=True, thresh=0, levels=100, cmap="mako", )
            ax.set_ylim([-0.15, 0.1])

            ax.set_yticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1])
            ax.set_yticklabels(['-15%', '-10%', '-5%', '0%', '5%', '10%'])
            ax.set_xticks([0, 0.52, 1.04, 1.56, 1.976])
            ax.set_xticklabels(['0', '0.5', '1.0', '1.5', '1.9'])
            ax.set_xlim([0, 1.976])
            ax.plot(np.linspace(0.988, 0.988, 100), np.linspace(-1, 1, 100), lw=3.5, c=(0.8, 0, 0), ls='--', zorder=9)

            potp, pcoef = curve_fit(exp_minus2, q2_list[38: 78], mean_inun_h_list[38: 78], maxfev=50000000, )
            ax.plot(np.linspace(1.0, 2, 10000), exp_minus(np.linspace(1.0, 2, 10000), *potp), lw=3, ls='-',c=(0.8, 0, 0), zorder=8)
            print(str(potp))
            # ax.plot(np.linspace(1.94, 3, 100), np.linspace(exp_minus(1.94, *potp), exp_minus(1.94, *potp), 100), lw=3, ls='-', c=(0, 0, 0.8), zorder=8)

            plt.savefig(f'G:\A_veg\Paper\Figure\Fig13\\req\\mean_inun_per_{dic_name}_{sec}.png', dpi=300)
            plt.close()

            fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
            # ax.scatter(pd_temp__['max_inun_h_per'], pd_temp__[dic_name], zorder=1)
            # popts, peff = curve_fit(ln_x5, pd_temp__['max_inun_h_per'], pd_temp__[dic_name], maxfev=50000)
            # ax.plot(np.linspace(0, 5, 100), ln_x5(np.linspace(0, 5, 100), *popts), zorder=3, c=(1,0,0))
            ax.scatter(q3_list, abs_mean_inun_h_list, zorder=4, s=10**2, marker='o', edgecolors=(0/256, 0/256, 0.7), facecolor=(1, 1, 1), alpha=1, linewidths=1)
            ax.fill_between(q3_list, abs_mean_inun_h_list_upper, abs_mean_inun_h_list_lower, zorder=1, linewidth=0.8, ls='-.', ec=(0, 0, 0), fc=(0.8, 0.8, 0.8), alpha=0.5)
            ax.plot(np.linspace(0,10,100), np.linspace(0,0,100), lw=2, c=(0,0,0))
            ax.set_ylim([-0.15, 0.1])
            ax.set_xlim([0, 6])
            ax.set_yticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1])
            ax.set_yticklabels(['-15%', '-10%', '-5%', '0%', '5%', '10%'])

            potp, pcoef = curve_fit(ln_minus, q3_list[0: 20], abs_mean_inun_h_list[0: 20], maxfev=50000000, bounds=([0, 0, -0.2], [0.2, 10, 0.2]),p0=[0.05, 3, 0])
            ax.plot(np.linspace(0, 10, 10000), ln_minus(np.linspace(0, 10, 10000), *potp), lw=2, ls='--', c=(0, 0, 0.8), zorder=8)
            potp, pcoef = curve_fit(exp_minus, q3_list[18: 120], abs_mean_inun_h_list[18: 120], maxfev=50000000,)
            ax.plot(np.linspace(0, 10, 10000), exp_minus(np.linspace(0, 10, 10000), *potp), lw=2, ls='--', c=(0, 0, 0.8), zorder=8)
            # z = np.polyfit(q3_list, abs_mean_inun_h_list, 5)
            # p = np.poly1d(z)
            # ax.plot(np.linspace(0,10,100), p(np.linspace(0,10,100)), lw=2, ls='--', c=(0, 0, 0.8), zorder=8)
            # sns.histplot(data=pd_temp__, x="max_inun_h_per", y=dic_name, bins=50)
            # sns.kdeplot(data=pd_temp__, x="max_inun_h_per", y=dic_name, fill=True, thresh=0, levels=10, cmap="mako",)
            plt.savefig(f'G:\A_veg\Paper\Figure\Fig13\\req\\mean_inun_{dic_name}_{sec}.png', dpi=300)
            plt.close()

            fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
            # ax.scatter(pd_temp__['mean_inun_h_per'], pd_temp__[dic_name], zorder=1)
            # popts, peff = curve_fit(ln_x3, pd_temp__['mean_inun_h_per'], pd_temp__[dic_name], maxfev=50000)
            # ax.plot(np.linspace(0, 3, 100), ln_x3(np.linspace(0, 3, 100), *popts), zorder=3, c=(1,0,0))
            ax.scatter(q4_list[0: 30], abs_max_inun_h_list[0: 30], zorder=4, s=10**2, marker='o', edgecolors=(0, 0, 0.7), facecolor=(1, 1, 1), alpha=1, linewidths=1)
            ax.scatter(q4_list[30: 60], abs_max_inun_h_list[30: 60], zorder=4, s=10 ** 2, marker='o', edgecolors=(0.1, 0.1, 0.1), facecolor=(1, 1, 1), alpha=1, linewidths=1)
            ax.scatter(q4_list[60: ], abs_max_inun_h_list[60: ], zorder=4, s=10 ** 2, marker='o',  edgecolors=(0.7, 0, 0), facecolor=(1, 1, 1), alpha=1, linewidths=1)
            ax.fill_between(q4_list, abs_max_inun_h_list_upper, abs_max_inun_h_list_lower, zorder=1, linewidth=0.8, ls='-.', ec=(0, 0, 0), fc=(0.8, 0.8, 0.8), alpha=0.5)
            ax.set_ylim([-0.15, 0.1])
            ax.set_xlim([0, 10])
            ax.set_yticks([])
            ax.set_yticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1])
            ax.set_yticklabels(['-15%', '-10%', '-5%', '0%', '5%', '10%'])
            ax.plot(np.linspace(0, 10, 100), np.linspace(0, 0, 100), lw=2, c=(0,0,0))

            potp, pcoef = curve_fit(ln_minus, q4_list[0: 60], abs_max_inun_h_list[0: 60], maxfev=50000000, bounds=([0, 0, -0.2], [0.2, 10, 0.2]), p0=[0.05, 3, 0])
            ax.plot(np.linspace(0, 2.28, 10000), ln_minus(np.linspace(0, 2.28, 10000), *potp), lw=3, ls='-', c=(0, 0, 0.8), zorder=8)
            ax.plot(np.linspace(2.28, 10, 10000), ln_minus(np.linspace(2.28, 10, 10000), *potp), lw=3, ls=':', c=(0, 0, 0.8), zorder=8)
            print(str(potp))

            potp, pcoef = curve_fit(exp_minus, q4_list[30: 200], abs_max_inun_h_list[30: 200], maxfev=50000000)
            ax.plot(np.linspace(2.28, 10, 10000), exp_minus(np.linspace(2.28, 10, 10000), *potp), lw=3, ls='-', c=(0.8, 0, 0.0), zorder=8)
            ax.plot(np.linspace(0, 2.28, 10000), exp_minus(np.linspace(0, 2.28, 10000), *potp), lw=3, ls=':', c=(0.8, 0, 0), zorder=8)
            print(str(potp))

            ax.plot(np.linspace(1.5, 1.5, 100), np.linspace(-1, 1, 100), lw=2, c=(0, 0, 0), ls='--', zorder=9)
            ax.plot(np.linspace(3, 3, 100), np.linspace(-1, 1, 100), lw=2, c=(0, 0, 0), ls='--', zorder=9)
            # z = np.polyfit(q4_list, abs_max_inun_h_list, 5)
            # p = np.poly1d(z)
            # ax.plot(np.linspace(0,10,100), p(np.linspace(0,10,100)), lw=2, ls='--', c=(0, 0, 0.8), zorder=8)
            # sns.histplot(data=pd_temp__, x="mean_inun_h_per", y=dic_name, bins=50)
            # sns.kdeplot(data=pd_temp__, x="mean_inun_h_per", y=dic_name, fill=True, thresh=0, levels=100, cmap="mako", )
            plt.savefig(f'G:\A_veg\Paper\Figure\Fig13\\req\\max_inun_{dic_name}_{sec}.png', dpi=300)
            plt.close()


def fig18_func():
    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=18)
    plt.rc('axes', linewidth=2)

    veg_inun_itr = 100
    itr_v = 100 / veg_inun_itr

    for sec, content in zip([ 'all', 'yz', 'jj', 'ch', 'hh', ], [None, (0, 14482, 0, 2810), (0, 14482, 2810, 18320), (0, 14482, 18320, 30633), (0, 14482, 30633, 49071), ]):
        bf.create_folder(f'G:\\A_veg\\Paper\\Figure\\Fig14\\{sec}\\')
        if not os.path.exists(f'G:\\A_veg\\Paper\\Figure\\Fig14\\{sec}.csv'):
            if sec != 'all':
                height_2020_ds = gdal.Open('G:\\A_veg\\S2_all\\XGB_vhm\\Feature_table4heightmap\\peak_2020\predicted_feature_tif\\ch_out_mod0_heil8.tif')
                height_2019_ds = gdal.Open('G:\\A_veg\\S2_all\\XGB_vhm\\Feature_table4heightmap\\peak_2019\predicted_feature_tif\\ch_out_mod0_heil8.tif')
                height_2021_ds = gdal.Open('G:\\A_veg\\S2_all\\XGB_vhm\\Feature_table4heightmap\\peak_2021\predicted_feature_tif\\ch_out_mod0_heil8.tif')
                height_2022_ds = gdal.Open('G:\\A_veg\\S2_all\\XGB_vhm\\Feature_table4heightmap\\peak_2022\predicted_feature_tif\\ch_out_mod0_heil8.tif')

                inund_dic = {}
                for thr in range(veg_inun_itr):
                    __ = gdal.Open(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_inunduration\\{str(sec)}_section\\annual_inun_duration\\gt_thr\\{str(int(itr_v + itr_v * thr))}thr_inund_{str(2020)}.tif')
                    inund_dic[f'{str(int(itr_v + itr_v * thr))}_thr'] = __.GetRasterBand(1).ReadAsArray()

                arr_2022 = height_2022_ds.GetRasterBand(1).ReadAsArray()[content[0]: content[1], content[2]: content[3]]
                arr_2020 = height_2020_ds.GetRasterBand(1).ReadAsArray()[content[0]: content[1], content[2]: content[3]]
                arr_2019 = height_2019_ds.GetRasterBand(1).ReadAsArray()[content[0]: content[1], content[2]: content[3]]
                arr_2021 = height_2021_ds.GetRasterBand(1).ReadAsArray()[content[0]: content[1], content[2]: content[3]]

                dic_temp = {'h_2020': [], 'h_2019': [], 'h_2021': [], 'h_2022': []}
                for thr in range(veg_inun_itr):
                    dic_temp[f'{str(int(itr_v + itr_v * thr))}_thr'] = []

                for y in range(arr_2020.shape[0]):
                    for x in range(arr_2020.shape[1]):
                        if ~np.isnan(arr_2020[y, x]):
                            dic_temp['h_2019'].append(arr_2019[y, x])
                            dic_temp['h_2020'].append(arr_2020[y, x])
                            dic_temp['h_2021'].append(arr_2021[y, x])
                            dic_temp['h_2022'].append(arr_2022[y, x])
                            for thr in range(veg_inun_itr):
                                dic_temp[f'{str(int(itr_v + itr_v * thr))}_thr'].append(inund_dic[f'{str(int(itr_v + itr_v * thr))}_thr'][y, x])

                pd_temp = pd.DataFrame(dic_temp)
                pd_temp.to_csv(f'G:\\A_veg\\Paper\\Figure\\Fig14\\{sec}.csv')
            else:
                for _ in ['yz', 'jj', 'ch', 'hh']:
                    pd_temp_ = pd.read_csv(f'G:\\A_veg\\Paper\\Figure\\Fig14\\{_}.csv')
                    if _ != 'yz':
                        pd_all = pd.concat((pd_all, pd_temp_))
                    else:
                        pd_all = copy.deepcopy(pd_temp_)
                pd_all.to_csv(f'G:\\A_veg\\Paper\\Figure\\Fig14\\{sec}.csv')

        pd_temp = pd.read_csv(f'G:\\A_veg\\Paper\\Figure\\Fig14\\{sec}.csv')
        pd_temp['20_21_diff'] = pd_temp['h_2020'] - pd_temp['h_2021']
        pd_temp['20_21_perc'] = (pd_temp['h_2020'] - pd_temp['h_2021']) / pd_temp['h_2020']
        pd_temp['20_22_diff'] = pd_temp['h_2020'] - pd_temp['h_2022']
        pd_temp['20_22_perc'] = (pd_temp['h_2020'] - pd_temp['h_2022']) / pd_temp['h_2020']
        pd_temp['19_20_diff'] = pd_temp['h_2020'] - pd_temp['h_2019']
        pd_temp['19_20_perc'] = (pd_temp['h_2020'] - pd_temp['h_2019']) / pd_temp['h_2019']

        pd_temp = pd_temp[pd_temp['5_thr'] < 90]
        pd_temp = pd_temp[pd_temp['5_thr'] >= 0]
        non_inun_t = 0
        inunthr_list = []
        veg_thr_list = [thr for thr in range(veg_inun_itr)]
        for dic_name in [f'{str(int(itr_v + itr_v * thr))}_thr' for thr in range(veg_inun_itr)]:
            if not os.path.exists(f'G:\A_veg\Paper\Figure\Fig14\\{sec}\\{dic_name}_{sec}.png'):
                fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
                if dic_name != f'{str(int(itr_v))}_thr':
                    pd_temp_ = pd_temp[[dic_name, '20_21_perc', f'{str(int(itr_v))}_thr']]
                else:
                    pd_temp_ = pd_temp[[dic_name, '20_21_perc']]
                pd_temp__ = pd_temp_.dropna()

                box_list, mid_list = [], []
                max_ = 0.
                for q in range(90):
                    if q == 0:
                        box_temp = pd_temp__[pd_temp__[f'{str(int(itr_v))}_thr'] == q]['20_21_perc']
                    else:
                        box_temp = pd_temp__[pd_temp__[dic_name] == q]['20_21_perc']
                    box_list.append(box_temp.sort_values()[int(box_temp.shape[0] * 0.05): int(box_temp.shape[0] * 0.95)])
                    mid_list.append(np.nanmean(box_temp))
                    max_ = np.nanmax((max_, np.max(np.absolute(box_list[-1][int(box_list[-1].shape[0] * 0.15): int(box_list[-1].shape[0] * 0.85)]))))

                if dic_name == f'{str(int(itr_v + itr_v * 0))}_thr':
                    non_inun_t = mid_list[0]

                for q in range(75):
                    q_max = np.nanmin([q + 16, 91])
                    if False not in [mid_list[_] < non_inun_t for _ in range(q, q_max)]:
                        inunthr_list.append(q)
                        break
                    if q == 81:
                        inunthr_list.append(np.nan)

                # box_temp = pd_temp__[pd_temp__['inun_d'] == 0][dic_name]
                # box_temp = box_temp.sort_values()[int(box_temp.shape[0] * 0.1): int(box_temp.shape[0] * 0.9)]
                box_temp = ax.boxplot(box_list, vert=True, notch=False, widths=0.98, patch_artist=True, whis=(15, 85),
                                      showfliers=False, zorder=4, )

                for patch in box_temp['boxes']:
                    patch.set_facecolor((72 / 256, 127 / 256, 166 / 256))
                    patch.set_alpha(0.5)
                    patch.set_linewidth(0.2)

                for median in box_temp['medians']:
                    median.set_lw(0.8)
                    # median.set_marker('^')
                    median.set_color((255 / 256, 128 / 256, 64 / 256))

                ax.plot(np.linspace(0, 90, 100), np.linspace(0, 0, 100), lw=2.5, c=(0, 0, 0), ls='-', zorder=5)
                ax.scatter(np.linspace(1, 90, 90), mid_list, marker='^', s=10 ** 2, facecolor=(0.8, 0, 0), zorder=7, edgecolor=(0.0, 0.0, 0.0), linewidth=0.2)
                # ax.scatter(pd_temp__['inun_d'], pd_temp__[dic_name], s=2 ** 2, edgecolor=(1, 1, 1), facecolor=(47/256,85/256,151/256), alpha=0.1, linewidth=0, marker='^', zorder=5)
                # s = linregress(np.array(pd_temp_[pd_temp_['inun_d'] > 0]['inun_d'].tolist()),
                #               np.array(pd_temp_[pd_temp_['inun_d'] > 0][dic_name].tolist()))

                # popt, pcov = curve_fit(poly3, pd_temp__['inun_d'], pd_temp__[dic_name], maxfev=50000, method="trf")
                # ax.plot(np.linspace(1, 90, 90), poly3(np.linspace(1, 90, 90), *popt), lw=2, ls='--', c=(0.8, 0, 0))
                ax.plot(np.linspace(0, 90, 90), np.linspace(non_inun_t, non_inun_t, 90), lw=2, ls='--', c=(0.8, 0, 0), zorder=8)

                # ax.plot(np.linspace(36, 90, 90), np.linspace(np.mean(mid_list[36:]), np.mean(mid_list[36:]), 90), lw=2,
                #         ls='--', c=(0.8, 0, 0), zorder=8)
                # z = np.polyfit(np.linspace(1, 90, 90), mid_list, 15)
                # p = np.poly1d(z)
                # ax.plot(np.linspace(1, 13, 100), p(np.linspace(1, 13, 100)), lw=2, ls='--', c=(0.8, 0, 0), zorder=8)
                # ax.plot(np.linspace(13, 36, 90), np.linspace(p(13), p(13), 90), lw=2, ls='--', c=(0.8, 0, 0), zorder=8)
                # print(str(dic_name))
                # print(str(p(13)))
                print(str(np.mean(mid_list[36:])))
                ax.set_ylim(-0.2, 0.2)
                ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
                ax.set_yticklabels(['-20%', '-10%', '0%', '10%', '20%'])
                ax.set_xticks([1, 11, 21, 31, 41, 51, 61])
                ax.set_xticklabels(['0', '10', '20', '30', '40', '50', '60'])
                # ax.set_ylim([- float(int(max_ * 20) + 1) / 20, float(int(max_ * 20) + 1) / 20])
                ax.set_xlim([0.5, 61.5])
                plt.savefig(f'G:\A_veg\Paper\Figure\Fig14\\{sec}\\{dic_name}_{sec}.png', dpi=300)

        dic_temp = {'inun_thr': inunthr_list}
        pd_temp = pd.DataFrame(dic_temp)
        pd_temp.to_csv(f'G:\A_veg\Paper\Figure\Fig14\\{sec}\\{sec}_thr.csv')


def gif_func():
    inundated_dc = Landsat_dc('G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\OSAVI_datacube\\')
    rs_dc = RS_dcs(inundated_dc)
    year_list = list(set([int(np.floor(_ // 10000)) for _ in rs_dc.Landsatdc_doy_list]))
    ds = gdal.Open('G:\\A_Landsat_veg\\ROI_map\\floodplain_2020_map.TIF')
    for year in year_list:
        list__ = []
        for _ in rs_dc.Landsatdc_doy_list:
            if int(np.floor(_ // 10000)) == year:
                list__.append(rs_dc.Landsatdc_doy_list.index(_))
        arr_temp = rs_dc.dcs[0][:, :, min(list__): max(list__) + 1]
        arr_temp = arr_temp.astype(np.float32)
        arr_temp[arr_temp == 0] = np.nan
        arr_temp = np.nanmean(arr_temp, axis=2)
        bf.write_raster(ds, arr_temp, 'G:\\A_Landsat_veg\\Annual_ndvi\\', f'{str(year)}.TIF')


    a = 1


def fig152_func():
    df = pd.read_excel('G:\\A_veg\\Paper\\Figure\\Fig15\\thr.xlsx')
    veg_thr = np.array(df['inun_thr'])
    duration_ = np.array(df['duration'])
    duration_min_ = np.array(df['min_d'])
    duration_[duration_ == 'Nan'] = np.nan
    duration = [_ for _ in duration_]

    duration_min_[duration_min_ == 'Nan'] = np.nan
    duration_min_ = [_ for _ in duration_min_]
    duration_min = np.array(duration_min_)
    duration = np.array(duration)

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=18)
    plt.rc('axes', linewidth=2)
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.scatter(duration, veg_thr, zorder=4, s=10**2, marker='s', edgecolors=(0/256, 0/256, 256/256), facecolor=(1, 1, 1), alpha=1, linewidths=2)
    # ax.fill_between(duration, veg_thr, np.linspace(100, 100, 100), zorder=1, facecolor=(1, 1, 1), alpha=0.8, linewidths=2)
    ax.plot(duration, veg_thr, lw=4, color =(0/256, 0/256, 256/256))
    ax.scatter(duration_min, veg_thr, zorder=4, s=10 ** 2, marker='o', edgecolors=(256 / 256, 0 / 256, 0/ 256), facecolor=(1, 1, 1), alpha=1, linewidths=2)
    ax.plot(duration_min, veg_thr, lw=4, color=(256 / 256, 0 / 256, 0/ 256))
    duration_min[np.isnan(duration_min)] = 0
    duration[np.isnan(duration)] = 100
    # ax.fill_between(duration, veg_thr, np.linspace(100, 100, 100), zorder=1, fc=(0/256, 0/256, 256/256), alpha=0.1, linewidths=2)
    ax.fill_between(duration, veg_thr, np.linspace(100, 100, 100), zorder=2, fc=(153 / 256, 153 / 256, 256 / 256), alpha=1, linewidths=2)
    ax.fill_between(duration_min, np.linspace(0, 0, 100), veg_thr, zorder=2, fc=(256 / 256, 153 / 256, 153 / 256), alpha=1, linewidths=2)
    ax.fill_between(np.linspace(0, 100, 100), np.linspace(0, 0, 100), np.linspace(100, 100, 100), zorder=1, fc=(0.98,0.98,0.98), hatch='/', linewidths=2)

    ax.set_xlim([0, 40])
    ax.set_ylim([10, 100])
    ax.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_yticklabels(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    ax.set_ylabel('Inundated water level/vegetation height', fontsize=26)
    ax.set_xlabel('Inundation duration/d',  fontsize=26)
    plt.savefig(f'G:\A_veg\Paper\Figure\Fig15\\v1.png', dpi=300)


def canny_edge():
    ds = gdal.Open('G:\Landsat\Sample122_124039\Landsat_constructed_index\MNDWI\\20140506_124039_MNDWI.TIF')
    arr = ds.GetRasterBand(1).ReadAsArray()
    arr = arr.astype(np.float32)
    arr[arr == -32768] = np.nan
    arr = (arr + 10000) / 20000

    res = feature.canny(arr, sigma=0.9, high_threshold=0.7)
    bf.write_raster(ds, res, 'E:\\A_Vegetation_Identification\\Paper\\Major_rev\\', 'tenmp2.tif')


def fig_his_func():

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=20)
    plt.rc('axes', linewidth=2)

    #
    Landsat_VI_temp = Landsat_dc('G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\MNDWI_datacube\\')
    wi = Landsat_VI_temp.dc[214,11430,:].flatten()
    wi = wi.astype(np.float32)
    wi[wi ==0] = np.nan
    wi = (wi-32768) / 10000
    sdc = Landsat_VI_temp.sdc_doylist
    dic = {'wi': wi, 'doy': sdc}
    df = pd.DataFrame(dic)

    # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above)
    sns.set(style="white", palette='dark', font='Times New Roman', font_scale=1,
            rc={'ytick.left': True, 'xtick.bottom': True})
    df = df.drop(df[df['wi'] < -0.5].index)
    df = df.drop(df[np.isnan(df['wi'])].index)
    df2 = copy.copy(df)
    df2 = df2.drop(df2[np.logical_and(np.mod(df['doy'], 1000) < 270, np.mod(df['doy'], 1000) > 180)].index)
    df2 = df2.drop(df2[df2['wi'] > -0.02625].index)
    fig, ax = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
    sns.histplot(data=df, x="wi", binwidth=0.015, binrange=(-0.3, 1.0), kde=True,
                 color=(30 / 256, 96 / 256, 164 / 256, 1), element='step',
                 line_kws={'color': (30 / 256, 96 / 256, 164 / 256, 1), 'lw': 3, 'ls': '-'})
    sns.histplot(data=df2, x="wi", binwidth=0.015, binrange=(-0.3, 1.0), kde=False,
                 line_kws={'color': (256 / 256, 30 / 256, 30 / 256, 1), 'lw': 3, 'ls': '-'},
                 color=(180 / 256, 30 / 256, 54 / 256, 0.4), element='step', alpha=0.4)
    plt.plot(np.linspace(-0.4, 0.05, 100),
             4.5 * guassain_dis(np.linspace(-0.4, 0.05, 100), np.nanstd(df2['wi']), np.nanmean(df2['wi'])),
             color=(180 / 256, 46 / 256, 23 / 256), lw=3)
    ax.set_xlim([-0.3, 0.4])

    plt.savefig('G:\A_Landsat_veg\Paper\Fig3\\fig1.png', dpi=300)


def fig4_func():

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=20)
    plt.rc('axes', linewidth=2)

    #
    if not os.path.exists('G:\\A_Landsat_veg\\Paper\\Fig4\\data2.xlsx'):
        Landsat_VI_temp = Landsat_dc('G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_datacube\\')
        wi = Landsat_VI_temp.dc[4530, 5400, :].flatten()
        wi = wi.astype(np.float32)
        wi[wi ==0] = np.nan
        wi = (wi-32768) / 10000
        sdc = Landsat_VI_temp.sdc_doylist
        sdc = bf.date2doy(sdc)
        sdc = [np.mod(_, 1000) for _ in sdc]
        dic = {'OSAVI': wi, 'DOY': sdc}
        df = pd.DataFrame(dic)
        df.to_excel('G:\\A_Landsat_veg\\Paper\\Fig4\\data2.xlsx')

    # Create fig4
    VI_curve_fitting = {'para_ori': [0.10, 0.8802, 108.2, 7.596, 311.4, 7.473, 0.00225], 'para_boundary': (
    [0.08, 0.7, 100, 6.2, 301.6, 4.5, 0.0015], [0.12, 1.0, 115, 11.5, 321.5, 8.8, 0.0028])}
    VI_curve_fitting = {'para_ori': [0.01, 0.01, 0, 2, 180, 2, 0.01], 'para_boundary': ([0, 0, 0, 0, 180, 0, 0], [0.5, 1, 180, 20, 330, 10, 0.01])}
    fig4_df = pd.read_excel('G:\\A_Landsat_veg\\Paper\\Fig4\\data2.xlsx')
    DOY_arr = np.array(fig4_df['DOY'])
    OSAVI_arr = np.array(fig4_df['OSAVI'])
    DOY_arr = np.delete(DOY_arr, np.argwhere(np.isnan(OSAVI_arr)))
    OSAVI_arr = np.delete(OSAVI_arr, np.argwhere(np.isnan(OSAVI_arr)))
    fig4, ax4 = plt.subplots(figsize=(12, 8), constrained_layout=True)
    # fig4, ax4 = plt.subplots(figsize=(10.5, 10.5), constrained_layout=True)
    ax4.set_axisbelow(True)
    ax4.set_xlim(0, 365)
    ax4.set_ylim(0, 0.6)

    paras, extra = curve_fit(seven_para_logistic_function, DOY_arr, OSAVI_arr, maxfev=500000, p0=VI_curve_fitting['para_ori'], bounds=VI_curve_fitting['para_boundary'])

    # define p3 and p5
    doy_all = DOY_arr
    vi_all = OSAVI_arr
    vi_dormancy = []
    doy_dormancy = []
    vi_senescence = []
    doy_senescence = []
    vi_max = []
    doy_max = []
    doy_index_max = np.argmax(seven_para_logistic_function(np.linspace(0, 366, 365), paras[0], paras[1], paras[2], paras[3], paras[4],paras[5], paras[6]))

    ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]), linewidth=10, color=(0/256, 44/256, 156/256))
    # ax4.plot(array_temp[0, :], array_temp[1, :], linewidth=4, markersize=12, **{'ls': '--', 'marker': 'o', 'color': 'b'})
    ax4.fill_between(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), 0.18, 0.48, 82.6, 11, 316, 12, 0.000695), seven_para_logistic_function(np.linspace(0, 365, 366), 0.018, 0.315, 88, 7.5, 340, 8, 0.000705), color=(0.1, 0.1, 0.1), alpha=0.1)
    ax4.scatter(DOY_arr, OSAVI_arr, s=12**2, color="none", edgecolor=(160/256, 160/256, 196/256), linewidth=3)
    # ax4.fill_between(np.linspace(560, 650, 100), np.linspace(0, 0, 100), np.linspace(1, 1, 100), color=(0, 197/255, 1), alpha=1)
    # ax4.plot(np.linspace(365, 365, 100), np.linspace(0, 1, 100), linewidth=4, **{'ls': '--', 'color': (0, 0, 0)})
    ax4.set_xlabel('DOY', fontname='Times New Roman', fontsize=34, fontweight='bold')
    ax4.set_ylabel('OSAVI', fontname='Times New Roman', fontsize=34, fontweight='bold')
    ax4.grid( axis='y', color=(240/256, 240/256, 240/256))
    ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), 0.18, 0.48, 82.6, 11, 316, 12, 0.000695), linewidth=2, color=(0 / 256, 44 / 256, 156 / 256), **{'ls': '--'})
    ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), 0.018, 0.315, 88, 7.5, 340, 8, 0.000705), linewidth=2, color=(0 / 256, 44 / 256, 156 / 256), **{'ls': '--'})
    # ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras1_min, paras2_min, paras3_min, paras4_max, paras5_min, paras6_max, paras7_max), linewidth=2, color=(0/256, 109/256, 44/256), **{'ls': '--'})
    # ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras1_max, paras2_max, paras3_max, paras4_min, paras5_max, paras6_min, paras7_min), linewidth=2, color=(0/256, 109/256, 44/256), **{'ls': '--'})
    predicted_y_data = seven_para_logistic_function(np.array(DOY_arr), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6])
    r_square = (1 - np.nansum((predicted_y_data - np.array(OSAVI_arr)) ** 2) / np.nansum((np.array(OSAVI_arr) - np.nanmean(np.array(OSAVI_arr))) ** 2))
    ax4.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6',], fontname='Times New Roman', fontsize=26)
    a = [15, 45, 75, 105, 136, 166, 197, 227, 258, 288, 320, 350]
    c = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    points = np.array([OSAVI_arr,DOY_arr]).transpose()
    # hull = ConvexHull(points)
    # # # for i in b:
    # # #     a.append(i)
    # ax4.plot(points[hull.vertices,1], points[hull.vertices,0], 'r--', lw=2)
    ax4.set_xticks(a)
    ax4.set_xticklabels(c, fontname='Times New Roman', fontsize=30)
    # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
    plt.savefig('G:\A_Landsat_veg\Paper\Fig4\\Figure_41.png', dpi=1000)
    plt.show()
    print(r_square)


def fig42_func():
    # Create fig4
    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=20)
    plt.rc('axes', linewidth=2)

    #
    if not os.path.exists('G:\\A_Landsat_veg\\Paper\\Fig4\\data3.xlsx'):
        Landsat_VI_temp = Landsat_dc('G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_datacube\\')
        wi = Landsat_VI_temp.dc[4530, 5400, :].flatten()
        wi = wi.astype(np.float32)
        wi[wi ==0] = np.nan
        wi = (wi-32768) / 10000
        sdc = Landsat_VI_temp.sdc_doylist
        sdc = bf.date2doy(sdc)
        sdc = [np.mod(_, 1000) for _ in sdc]
        dic = {'OSAVI': wi, 'DOY': sdc}
        df = pd.DataFrame(dic)
        df.to_excel('G:\\A_Landsat_veg\\Paper\\Fig4\\data3.xlsx')

    # Create fig4
    VI_curve_fitting = {'para_ori': [0.10, 0.8802, 108.2, 7.596, 311.4, 7.473, 0.00225], 'para_boundary': (
    [0.08, 0.7, 100, 6.2, 301.6, 4.5, 0.0015], [0.12, 1.0, 115, 11.5, 321.5, 8.8, 0.0028])}
    VI_curve_fitting = {'para_ori': [0.01, 0.01, 0, 2, 180, 2, 0.01], 'para_boundary': ([0, 0, 0, 0, 180, 0, 0], [0.5, 1, 180, 20, 330, 10, 0.01])}
    fig4_df = pd.read_excel('G:\\A_Landsat_veg\\Paper\\Fig4\\data3.xlsx')
    DOY_arr = np.array(fig4_df['DOY'])
    OSAVI_arr = np.array(fig4_df['OSAVI'])
    DOY_arr = np.delete(DOY_arr, np.argwhere(np.isnan(OSAVI_arr)))
    OSAVI_arr = np.delete(OSAVI_arr, np.argwhere(np.isnan(OSAVI_arr)))
    fig4, ax4 = plt.subplots(figsize=(12, 8), constrained_layout=True)
    # fig4, ax4 = plt.subplots(figsize=(10.5, 10.5), constrained_layout=True)
    ax4.set_axisbelow(True)
    ax4.set_xlim(0, 365)
    ax4.set_ylim(0, 0.6)

    paras, extra = curve_fit(seven_para_logistic_function, DOY_arr, OSAVI_arr, maxfev=500000, p0=VI_curve_fitting['para_ori'], bounds=VI_curve_fitting['para_boundary'])

    # define p3 and p5
    doy_all = DOY_arr
    vi_all = OSAVI_arr
    vi_dormancy = []
    doy_dormancy = []
    vi_senescence = []
    doy_senescence = []
    vi_max = []
    doy_max = []
    doy_index_max = np.argmax(seven_para_logistic_function(np.linspace(0, 366, 365), paras[0], paras[1], paras[2], paras[3], paras[4],paras[5], paras[6]))

    ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]), linewidth=10, color=(156/256, 44/256, 0/256))
    # ax4.plot(array_temp[0, :], array_temp[1, :], linewidth=4, markersize=12, **{'ls': '--', 'marker': 'o', 'color': 'b'})
    ax4.fill_between(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), 0.18, 0.48, 82.6, 11, 316, 12, 0.000695), seven_para_logistic_function(np.linspace(0, 365, 366), 0.018, 0.315, 88, 7.5, 340, 8, 0.000705), color=(0.1, 0.1, 0.1), alpha=0.1)
    ax4.scatter(DOY_arr, OSAVI_arr, s=12**2, color=(196/256, 80/256, 80/256), edgecolor=(196/256, 80/256, 80/256), linewidth=3)
    # ax4.fill_between(np.linspace(560, 650, 100), np.linspace(0, 0, 100), np.linspace(1, 1, 100), color=(0, 197/255, 1), alpha=1)
    # ax4.plot(np.linspace(365, 365, 100), np.linspace(0, 1, 100), linewidth=4, **{'ls': '--', 'color': (0, 0, 0)})
    ax4.set_xlabel('DOY', fontname='Times New Roman', fontsize=34, fontweight='bold')
    ax4.set_ylabel('OSAVI', fontname='Times New Roman', fontsize=34, fontweight='bold')
    ax4.grid( axis='y', color=(240/256, 240/256, 240/256))
    ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), 0.18, 0.48, 82.6, 11, 316, 12, 0.000695), linewidth=2, color=(156 / 256, 44 / 256, 0 / 256), **{'ls': '--'})
    ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), 0.018, 0.315, 88, 7.5, 340, 8, 0.000705), linewidth=2, color=(156 / 256, 44 / 256, 0 / 256), **{'ls': '--'})
    # ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras1_min, paras2_min, paras3_min, paras4_max, paras5_min, paras6_max, paras7_max), linewidth=2, color=(0/256, 109/256, 44/256), **{'ls': '--'})
    # ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras1_max, paras2_max, paras3_max, paras4_min, paras5_max, paras6_min, paras7_min), linewidth=2, color=(0/256, 109/256, 44/256), **{'ls': '--'})
    predicted_y_data = seven_para_logistic_function(np.array(DOY_arr), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6])
    r_square = (1 - np.nansum((predicted_y_data - np.array(OSAVI_arr)) ** 2) / np.nansum((np.array(OSAVI_arr) - np.nanmean(np.array(OSAVI_arr))) ** 2))
    ax4.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6',], fontname='Times New Roman', fontsize=26)
    a = [15, 45, 75, 105, 136, 166, 197, 227, 258, 288, 320, 350]
    c = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    points = np.array([OSAVI_arr,DOY_arr]).transpose()
    # hull = ConvexHull(points)
    # # # for i in b:
    # # #     a.append(i)
    # ax4.plot(points[hull.vertices,1], points[hull.vertices,0], 'r--', lw=2)
    ax4.set_xticks(a)
    ax4.set_xticklabels(c, fontname='Times New Roman', fontsize=30)
    # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
    plt.savefig('G:\A_Landsat_veg\Paper\Fig4\\Figure_42.png', dpi=1000)
    plt.show()
    print(r_square)


def fig15_func():
    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=24)
    plt.rc('axes', linewidth=2)

    file_list = bf.file_filter('G:\A_Landsat_veg\Water_level_python\original_water_level\\', ['.xls'])
    corr_temp = pd.read_csv('G:\A_Landsat_veg\Water_level_python\original_water_level\\对应表.csv')
    cs_list, wl_list = [], []

    wl1 = HydroStationDS()
    for file_ in file_list:
        for hs_num in range(corr_temp.shape[0]):
            hs = corr_temp[corr_temp.keys()[1]][hs_num]
            if hs in file_:
                cs_list.append(corr_temp[corr_temp.keys()[0]][hs_num])
                wl_list.append(corr_temp[corr_temp.keys()[2]][hs_num])

    for fn_, cs_, wl_ in zip(file_list, cs_list, wl_list):
        wl1.import_from_standard_excel(fn_, cs_, water_level_offset=wl_)

    for sec, r1, l1, ytick, in zip(['宜昌', '枝城', '莲花塘', '汉口'], [(38, 56), (36, 52), (18, 36), (12, 32)], [49, 46, 31, 25], [[38, 41, 44, 47, 50, 53, 56], [36, 40, 44, 48, 52], [18, 21, 24, 27, 30, 33, 36], [12, 17, 22, 27, 32]]):
        fig14_df = wl1.hydrostation_inform_df[sec]
        year_dic = {}
        wl_pri, wl_post = [], []
        sd_pri, sd_post = [], []
        for year in range(1985, 2021):
            year_temp = fig14_df['year'] == year
            flow_temp = fig14_df['water_level/m'][year_temp].tolist()
            sed_temp = fig14_df['sediment_concentration/kg/m3'][year_temp].tolist()
            year_dic[f'{str(year)}_wl'] = flow_temp[0:365]
            year_dic[f'{str(year)}_sed'] = sed_temp[0:365]
            if len(flow_temp) == 365 or len(flow_temp) == 366:
                if year > 2004:
                    wl_post.append(flow_temp[0:365])
                    sd_post.append(sed_temp[0:365])
                elif year <= 2004:
                    wl_pri.append(flow_temp[0:365])
                    sd_pri.append(sed_temp[0:365])
        wl_post = np.array(wl_post)
        sd_post = np.array(sd_post)
        wl_pri = np.array(wl_pri)
        sd_pri = np.array(sd_pri)

        sd_pri[sd_pri == 0] = np.nan
        sd_post[sd_post == 0] = np.nan

        plt.close()
        plt.rc('axes', axisbelow=True)
        plt.rc('axes', linewidth=3)
        fig_temp, ax_temp = plt.subplots(figsize=(12, 5), constrained_layout=True)
        ax_temp.grid(axis='y', color=(210 / 256, 210 / 256, 210 / 256), zorder=0)
        ax_temp.fill_between(np.linspace(175, 300, 121), np.linspace(r1[1], r1[1], 121), np.linspace(r1[0],r1[0],121),alpha=1, color=(0.9, 0.9, 0.9))
        ax_temp.fill_between(np.linspace(1, 365, 365), np.nanmax(wl_post, axis=0).reshape([365]), np.nanmin(wl_post, axis=0).reshape([365]),alpha=0.3, color=(0.8, 0.2, 0.1), zorder=3)
        ax_temp.fill_between(np.linspace(1, 365, 365), np.nanmax(wl_pri, axis=0).reshape([365]), np.nanmin(wl_pri, axis=0).reshape([365]),alpha=0.3, color=(0.1, 0.2, 0.8), zorder=2)
        ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_pri, axis=0).reshape([365]), lw=5, c=(0, 0, 1), zorder=4)
        ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_post, axis=0).reshape([365]), lw=5, c=(1, 0, 0), zorder=4)
        ax_temp.plot(np.linspace(1, 365,365), np.linspace(l1,l1,365), lw=2, ls='--', c=(0,0,0))
        ax_temp.set_xlim(1, 365)
        ax_temp.set_ylim(r1[0], r1[1])
        ax_temp.set_yticks(ytick)

        a = [15,  105, 197,  288,  350]
        c = ['Jan', 'Apr',  'Jul',  'Oct',  'Dec']
        ax_temp.set_xticks(a)
        ax_temp.set_xticklabels(c, fontname='Times New Roman', fontsize=24)
        ax_temp.set_xlabel('Month', fontname='Times New Roman', fontsize=28, fontweight='bold')
        ax_temp.set_ylabel('Water level(m)', fontname='Times New Roman', fontsize=28, fontweight='bold')
        # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
        plt.savefig(f'G:\A_Landsat_veg\Paper\Fig6\\{sec}_wl.png', dpi=500)
        plt.close()

        # plt.rc('axes', axisbelow=True)
        # plt.rc('axes', linewidth=3)
        # fig_temp, ax_temp = plt.subplots(figsize=(11, 5), constrained_layout=True)
        # ax_temp.grid( axis='y', color=(210 / 256, 210 / 256, 210 / 256), zorder=0)
        # # ax_temp.fill_between(np.linspace(175, 300, 121), np.linspace(r1[1], r1[1], 121), np.linspace(r1[0],r1[0],121),alpha=1, color=(0.9, 0.9, 0.9))
        # ax_temp.fill_between(np.linspace(1, 365, 365), np.nanmax(sd_post, axis=0).reshape([365]), np.nanmin(sd_post, axis=0).reshape([365]), alpha=0.3, color=(0/256, 92/256, 171/256), zorder=3)
        # ax_temp.fill_between(np.linspace(1, 365, 365), np.nanmax(sd_pri, axis=0).reshape([365]), np.nanmin(sd_pri, axis=0).reshape([365]), alpha=0.3, color=(255/256, 155/256, 37/256), zorder=2)
        # ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(sd_pri, axis=0).reshape([365]), lw=5, c=(255/256, 155/256, 37/256), zorder=4)
        # ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(sd_post, axis=0).reshape([365]), lw=5, c=(0/256, 92/256, 171/256), zorder=4)
        # # ax_temp.plot(np.linspace(1,365,365), np.linspace(l1,l1,365), lw=2, ls='--', c=(0,0,0))
        # ax_temp.set_xlim(1, 365)
        # # ax_temp.set_ylim(r1[0], r1[1])
        # # ax_temp.set_yticks(ytick)
        # cc = np.nanmean(sd_pri, axis=0)/np.nanmean(sd_post, axis=0)
        # print(str(np.nanmax(cc[150: 300])))
        # print(str(np.nanmin(cc[150: 300])))
        # plt.yscale("log")
        # a = [15,  105, 197,  288,  350]
        # c = ['Jan', 'Apr',  'Jul',  'Oct',  'Dec']
        # ax_temp.set_xticks(a)
        # ax_temp.set_xticklabels(c, fontname='Times New Roman', fontsize=24)
        # ax_temp.set_xlabel('Month', fontname='Times New Roman', fontsize=28, fontweight='bold')
        # ax_temp.set_ylabel('Sediment con(kg/m^3)', fontname='Times New Roman', fontsize=28, fontweight='bold')
        # # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
        # plt.savefig(f'G:\A_Landsat_veg\Paper\Fig6\\{sec}_sd.png', dpi=500)

        if sec == '宜昌':
            fig_temp, ax_temp = plt.subplots(figsize=(12, 6), constrained_layout=True)
            wl_temp = np.concatenate([np.nanmax(wl_pri, axis=1), np.nanmax(wl_post, axis=1)])
            # ax_temp.bar([_ for _ in range(1985, 2005)], np.nanmax(wl_pri, axis=1), 0.65, label='SAR', color=(0.2, 0.3, 0.8), edgecolor=(0/256, 0/256, 0/256), linewidth=1, zorder=3, alpha=0.5)
            # ax_temp.bar([_ for _ in range(2005, 2021)], np.nanmax(wl_post, axis=1), 0.65, label='SAR', color=(0.8, 0.3, 0.2), edgecolor=(0 / 256, 0 / 256, 0 / 256), linewidth=1, zorder=3, alpha=0.5)
            ax_temp.grid(axis='y', color=(128 / 256, 128 / 256, 128 / 256), zorder=1)
            ax_temp.plot([_ for _ in range(1985, 2005)], np.nanmax(wl_pri, axis=1), color=(0.2, 0.3, 0.8), linewidth=3,ls='--', label='Pre-TGD')
            ax_temp.plot([_ for _ in range(2005, 2021)], np.nanmax(wl_post, axis=1), color=(0.8, 0.3, 0.2), linewidth=3,ls='--', label='Post-TGD')
            ax_temp.fill_between(np.linspace(0.5, 3004.5, 100), np.linspace(0, 0, 100), np.linspace(51.5, 51.5, 100), edgecolor='none', facecolor=(0.8,0.8,0.8), alpha=0.3, lw=2)
            ax_temp.plot(np.linspace(0.5, 3004.5, 100), np.linspace(51.5, 51.5, 100), color=(0, 0, 0), ls='-.', lw=2, label='Overbank')
            ax_temp.scatter([_ for _ in range(1985, 2005)], np.nanmax(wl_pri, axis=1), s = 14 **2, marker='s', color="none", edgecolor=(0,0,1), linewidth=3)
            ax_temp.scatter([_ for _ in range(2005, 2021)], np.nanmax(wl_post, axis=1), s = 14 **2, marker='s', color="none", edgecolor=(1,0,0), linewidth=3)
            # ax_temp.plot(np.linspace([2004.5, 2004.5, 100]), np.linspace([0, 100, 100]), color=(0.2, 0.2, 0.2))
            ax_temp.set_xlabel('Year', fontname='Times New Roman', fontsize=24)
            ax_temp.set_ylabel('Annual maximum water level(m)', fontname='Times New Roman', fontsize=24)
            ax_temp.legend(fontsize=22)
            ax_temp.set_yticks([47, 49,51,53,55, 57])
            ax_temp.set_yticklabels(['47', '49','51','53','55','57'], fontname='Times New Roman', fontsize=24)
            ax_temp.set_xlim(1984.5, 2020.5)
            ax_temp.set_ylim(47, 57)
            plt.savefig(f'G:\A_Landsat_veg\Paper\Fig6\\{sec}_annual_wl.png', dpi=500)
            plt.close()

        if sec == '汉口':
            fig_temp, ax_temp = plt.subplots(figsize=(12, 6), constrained_layout=True)
            wl_temp = np.concatenate([np.nanmax(wl_pri, axis=1), np.nanmax(wl_post, axis=1)])
            ax_temp.grid(axis='y', color=(128 / 256, 128 / 256, 128 / 256), zorder=1)
            # ax_temp.bar([_ for _ in range(1985, 2005)], np.nanmax(wl_pri, axis=1), 0.65, label='SAR', color=(0.2, 0.3, 0.8), edgecolor=(0/256, 0/256, 0/256), linewidth=1, zorder=3, alpha=0.5)
            # ax_temp.bar([_ for _ in range(2005, 2021)], np.nanmax(wl_post, axis=1), 0.65, label='SAR', color=(0.8, 0.3, 0.2), edgecolor=(0 / 256, 0 / 256, 0 / 256), linewidth=1, zorder=3, alpha=0.5)
            ax_temp.plot([_ for _ in range(1985, 2005)], np.nanmax(wl_pri, axis=1), color=(0.2, 0.3, 0.8), linewidth=3,
                         ls='--', label='Pre-TGD')
            ax_temp.plot([_ for _ in range(2005, 2021)], np.nanmax(wl_post, axis=1), color=(0.8, 0.3, 0.2), linewidth=3,
                         ls='--', label='Post-TGD')
            ax_temp.fill_between(np.linspace(0.5, 3004.5, 100), np.linspace(0, 0, 100), np.linspace(26.5, 26.5, 100),
                                 edgecolor='none', facecolor=(0.8, 0.8, 0.8), alpha=0.3, lw=2)
            ax_temp.plot(np.linspace(0.5, 3004.5, 100), np.linspace(26.5, 26.5, 100), color=(0, 0, 0), ls='-.', lw=2, label='Overbank')
            ax_temp.scatter([_ for _ in range(1985, 2005)], np.nanmax(wl_pri, axis=1), s=14 ** 2, marker='s',
                            color="none", edgecolor=(0, 0, 1), linewidth=3)
            ax_temp.scatter([_ for _ in range(2005, 2021)], np.nanmax(wl_post, axis=1), s=14 ** 2, marker='s',
                            color="none", edgecolor=(1, 0, 0), linewidth=3)
            # ax_temp.plot(np.linspace([2004.5, 2004.5, 100]), np.linspace([0, 100, 100]), color=(0.2, 0.2, 0.2))

            ax_temp.set_xlabel('Year', fontname='Times New Roman', fontsize=24)
            ax_temp.set_ylabel('Annual maximum water level(m)', fontname='Times New Roman', fontsize=24)
            ax_temp.set_yticks([22, 24, 26, 28, 30, 32])
            ax_temp.set_yticklabels([ '22', '24', '26', '28', '30', '32'], fontname='Times New Roman', fontsize=24)
            ax_temp.set_xlim(1984.5, 2020.5)
            ax_temp.set_ylim(22, 32)
            plt.savefig(f'G:\A_Landsat_veg\Paper\Fig6\\{sec}_annual_wl.png', dpi=500)
            plt.close()

        #
        # fig_temp, ax_temp = plt.subplots(figsize=(11, 6), constrained_layout=True)
        # wl_temp = np.concatenate([np.nanmean(sd_pri[:, 150: 300], axis=1), np.nanmean(sd_post[:, 150: 300], axis=1)])
        # ax_temp.bar([_ for _ in range(1990, 2005)], np.nanmean(sd_pri[:, 150: 300], axis=1), 0.6, label='SAR', color=(255/256, 155/256, 37/256), edgecolor=(0/256, 0/256, 0/256), linewidth=1, zorder=3, alpha=0.5)
        # ax_temp.plot([_ for _ in range(1990, 2005)], [np.nanmean(np.nanmean(sd_pri[:, 150: 300], axis=1)) for _ in range(1990, 2005)], linewidth=3, c=(255/256, 155/256, 37/256))
        # ax_temp.bar([_ for _ in range(2005, 2021)], np.nanmean(sd_post[:, 150: 300], axis=1), 0.6, label='SAR', color=(0/256, 92/256, 171/256), edgecolor=(0 / 256, 0 / 256, 0 / 256), linewidth=1, zorder=3, alpha=0.5)
        # ax_temp.plot([_ for _ in range(2005, 2021)], [np.nanmean(np.nanmean(sd_post[:, 150: 300], axis=1)) for _ in range(2005, 2021)], linewidth=3, c=(0/256, 92/256, 171/256))
        # ax_temp.set_xlabel('Year', fontname='Times New Roman', fontsize=28, fontweight='bold')
        # ax_temp.set_ylabel('S during flood season(m)', fontname='Times New Roman', fontsize=28, fontweight='bold')
        # ax_temp.set_xlim(1989.5, 2020.5)
        # plt.savefig(f'G:\A_Landsat_veg\Paper\Fig6\\{sec}_annual_sd.png', dpi=500)


def fig16_func():

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=24)
    plt.rc('axes', linewidth=3)

    pre_TGD_ds = gdal.Open('G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\inun_factor\\DT_inundation_frequency_pretgd.TIF')
    post_TGD_ds = gdal.Open('G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\inun_factor\\DT_inundation_frequency_posttgd.TIF')
    pre_TGD_arr = pre_TGD_ds.GetRasterBand(1).ReadAsArray()
    post_TGD_arr = post_TGD_ds.GetRasterBand(1).ReadAsArray()

    for sec, cord, domain, tick, ticklabels in zip(['yz', 'jj', 'ch', 'hh'], [[0, 950], [950, 6100], [6100, 10210], [10210, 16537]], [[65, 100], [30, 100], [40, 100], [40, 100]], [[0.65, 0.70, 0.80, 0.90, 1], [0.30, 0.40,  0.60, 0.80, 1], [0.40, 0.60, 0.80, 1], [0.40, 0.60, 0.80, 1]], [['65%', '70%', '80%', '90%', '100%'], ['30%', '40%', '60%', '80%', '100%'], ['40%', '60%', '80%', '100%'], ['40%', '60%', '80%', '100%']]):
        pre_tgd_data = pre_TGD_arr[:, cord[0]: cord[1]].flatten()
        post_tgd_data = post_TGD_arr[:, cord[0]: cord[1]].flatten()
        post_tgd_data = np.sort(np.delete(post_tgd_data, np.argwhere(np.isnan(post_tgd_data))))
        pre_tgd_data = np.sort(np.delete(pre_tgd_data, np.argwhere(np.isnan(pre_tgd_data))))
        plt.rc('axes', linewidth=3)
        fig = plt.figure(figsize=(12, 5), layout="constrained")

        ax_temp = fig.subplots(1, 1, sharex=True, sharey=True)
        ax_temp.grid(axis='y', color=(210 / 256, 210 / 256, 210 / 256), zorder=5)
        bins = 0.0001
        pre_num = []
        pre_cdf = []
        post_num = []
        post_cdf = []
        for _ in range(0, 10001):
            pre_num.append(bins * _)
            post_num.append(bins * _)
            pre_cdf.append(np.sum(pre_tgd_data > bins * _) / pre_tgd_data.shape[0])
            post_cdf.append(np.sum(post_tgd_data > bins * _) / post_tgd_data.shape[0])

        itr_list = []
        pre_list = []
        post_list = []
        for _ in range(0, 1000):
            q = _ / 1000
            for pre_cdf_ in pre_cdf:
                if pre_cdf_ - q < 0.001:
                    pre_num_ = pre_num[pre_cdf.index(pre_cdf_)]
                    break
            for post_cdf_ in post_cdf:
                if post_cdf_ - q < 0.001:
                    post_num_ = post_num[post_cdf.index(post_cdf_)]
                    break
            itr_list.append(pre_num_ - post_num_)
            pre_list.append(pre_num_)
            post_list.append(post_num_)
        print(str(itr_list.index(max(itr_list)) * 10))

        # sns.ecdfplot(post_tgd_data, label="Pre-TGD", complementary=True, lw=3, color=(0,0,1))
        # sns.ecdfplot(pre_tgd_data, label="Post-TGD", complementary=True, lw=3, color=(1,0,0))
        ax_temp.plot(np.linspace(post_list[itr_list.index(max(itr_list))], pre_list[itr_list.index(max(itr_list))], 100), np.linspace(itr_list.index(max(itr_list))/1000, itr_list.index(max(itr_list))/1000, 100), color=(1, 0, 0), zorder=8)
        print(str(post_list[itr_list.index(max(itr_list))]))
        print(pre_list[itr_list.index(max(itr_list))])
        # ax_temp.fill_between(pre_num, np.linspace(0, 0, 10001), post_cdf, color=(1, 1, 1), edgecolor='none', lw=0.0, zorder=4)
        # ax_temp.fill_between(pre_num, pre_cdf, np.linspace(1, 1, 10001), color=(1, 1, 1), edgecolor='none', lw=0.0, zorder=4)
        ax_temp.fill_between(pre_num, np.linspace(0, 0, 10001), np.linspace(post_cdf[-2], post_cdf[-2], 10001), color=(0.9, 0.9, 0.9), edgecolor=(0.5, 0.5, 0.5), lw=1.0, zorder=5)
        ax_temp.fill_between(pre_num, np.linspace(0, 0, 10001), post_cdf, color=(0.9, 0.9, 0.9), alpha=0.1)
        ax_temp.fill_between(pre_num, post_cdf, pre_cdf, hatch='-', color=(1, 1, 1), edgecolor=(0, 0, 0), lw=2.5,zorder=6)
        ax_temp.plot(pre_num, pre_cdf, lw=5, color=(0,0,1), label="Pre-TGD",zorder=7)
        ax_temp.plot(post_num, post_cdf, lw=5, color=(1,0,0), label="Post-TGD",zorder=7)
        ax_temp.set_xlabel('Inundation frequency', fontname='Times New Roman', fontsize=28, fontweight='bold', )
        ax_temp.set_ylabel('Exceedance probability', fontname='Times New Roman', fontsize=28, fontweight='bold')
        ax_temp.legend(fontsize=22)
        ax_temp.set_ylim(domain[0]/100, domain[1]/100)
        ax_temp.set_yticks(tick)
        ax_temp.set_yticklabels(ticklabels, fontname='Times New Roman', fontsize=24)
        ax_temp.set_xlim(0, 1)
        ax_temp.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax_temp.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontname='Times New Roman', fontsize=24)
        plt.savefig(f'G:\A_Landsat_veg\Paper\Fig6\\{sec}_inun_freq.png', dpi=500)
        plt.close()


def fig10_func():

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=28)
    plt.rc('axes', linewidth=3)

    if not os.path.exists('G:\\A_Landsat_veg\\Paper\\Fig10\\veg_pre_tgd.TIF') or not os.path.exists('G:\\A_Landsat_veg\\Paper\\Fig10\\veg_post_tgd.TIF'):
        inundated_dc = []
        for _ in range(1986, 2023):
            inundated_dc.append(Phemetric_dc(f'G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{str(_)}\\'))
        rs_dc = RS_dcs(*inundated_dc)

        veg_arr_pri = None
        veg_arr_post = None
        y_shape, x_shape = rs_dc.dcs_YSize, rs_dc.dcs_XSize
        for phe_year in rs_dc._pheyear_list:
            if phe_year <= 2004:
                if veg_arr_pri is None:
                    veg_arr_pri = rs_dc.dcs[rs_dc._pheyear_list.index(phe_year)].SM_group[f'{str(phe_year)}_peak_vi'][:,:].toarray().reshape([y_shape, x_shape, 1])
                else:
                    veg_arr_pri = np.concatenate((veg_arr_pri, rs_dc.dcs[rs_dc._pheyear_list.index(phe_year)].SM_group[f'{str(phe_year)}_peak_vi'][:,:].toarray().reshape([y_shape, x_shape, 1])), axis=2)
            elif phe_year > 2004:
                if veg_arr_post is None:
                    veg_arr_post = rs_dc.dcs[rs_dc._pheyear_list.index(phe_year)].SM_group[f'{str(phe_year)}_peak_vi'][:,:].toarray().reshape([y_shape, x_shape, 1])
                else:
                    veg_arr_post = np.concatenate((veg_arr_post, rs_dc.dcs[rs_dc._pheyear_list.index(phe_year)].SM_group[f'{str(phe_year)}_peak_vi'][:,:].toarray().reshape([y_shape, x_shape, 1])), axis=2)

        veg_arr_pri[veg_arr_pri == 0] = np.nan
        veg_arr_post[veg_arr_post == 0] = np.nan
        veg_arr_pri = np.nanmean(veg_arr_pri, axis=2)
        veg_arr_post = np.nanmean(veg_arr_post, axis=2)

        ds = gdal.Open(rs_dc.ROI_tif)
        bf.write_raster(ds, veg_arr_pri, 'G:\\A_Landsat_veg\\Paper\\Fig10\\', 'veg_pre_tgd.TIF', raster_datatype=gdal.GDT_Float32)
        bf.write_raster(ds, veg_arr_post, 'G:\\A_Landsat_veg\\Paper\\Fig10\\', 'veg_post_tgd.TIF', raster_datatype=gdal.GDT_Float32)

    inun_pre_ds = gdal.Open('G:\\A_Landsat_veg\\Paper\\Fig10\\DT_inundation_frequency_pretgd.TIF')
    inun_post_ds = gdal.Open('G:\\A_Landsat_veg\\Paper\\Fig10\\DT_inundation_frequency_posttgd.TIF')
    veg_pre_ds = gdal.Open('G:\\A_Landsat_veg\\Paper\\Fig10\\veg_pre_tgd.TIF')
    veg_post_ds = gdal.Open('G:\\A_Landsat_veg\\Paper\\Fig10\\veg_post_tgd.TIF')

    inun_pre_arr = inun_pre_ds.GetRasterBand(1).ReadAsArray()
    inun_post_arr = inun_post_ds.GetRasterBand(1).ReadAsArray()
    veg_pre_arr = veg_pre_ds.GetRasterBand(1).ReadAsArray()
    veg_post_arr = veg_post_ds.GetRasterBand(1).ReadAsArray()

    inun_diff = inun_post_arr - inun_pre_arr
    veg_diff = veg_post_arr - veg_pre_arr

    inun_diff = inun_diff.flatten()
    veg_diff = veg_diff.flatten()

    inun_temp = inun_pre_arr.flatten()
    inun_temp2 = inun_post_arr.flatten()
    inun_diff = np.delete(inun_diff, np.argwhere(np.logical_or(np.logical_or(inun_temp == 0, inun_temp == 1), np.logical_or(inun_temp2 == 0, inun_temp2 == 1))))
    veg_diff = np.delete(veg_diff, np.argwhere(np.logical_or(np.logical_or(inun_temp == 0, inun_temp == 1), np.logical_or(inun_temp2 == 0, inun_temp2 == 1))))

    inun_diff = np.delete(inun_diff, np.argwhere(inun_temp == 0))
    veg_diff = np.delete(veg_diff, np.argwhere(inun_temp == 0))

    inun_diff = np.delete(inun_diff, np.argwhere(np.isnan(veg_diff)))
    veg_diff = np.delete(veg_diff, np.argwhere(np.isnan(veg_diff)))

    veg_diff = np.delete(veg_diff, np.argwhere(np.isnan(inun_diff)))
    inun_diff = np.delete(inun_diff, np.argwhere(np.isnan(inun_diff)))

    fig_temp, ax_temp = plt.subplots(figsize=(12, 12), constrained_layout=True)
    p0, f0 = curve_fit(x_minus, inun_diff, veg_diff)
    sns.histplot(x = inun_diff, y=veg_diff, thresh=0.1, bins = 500, pmax=0.25, kde = True, stat='density', weights = 0.1)
    ax_temp.plot(np.linspace(-1,1,100), x_minus(np.linspace(-1,1,100), p0[0], p0[1], p0[2]), lw=3, c=(1,0,0))
    ax_temp.plot(np.linspace(-1,1,100), np.linspace(0,0,100), lw=1.5, c=(0,0,0))
    ax_temp.plot(np.linspace(0, 0, 100), np.linspace(-1, 1, 100), lw=1.5, c=(0,0,0))
    ax_temp.set_xlim(-0.8, 0.8)
    ax_temp.set_ylim(-0.4, 0.4)
    plt.savefig(f'G:\A_Landsat_veg\Paper\Fig10\\Fig10.png', dpi=300)
    a = 1


def fig15_func():
    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=28)
    plt.rc('axes', linewidth=3)
    csv_file = 'G:\A_Landsat_veg\Paper\Fig16\\inun_all_47.439.csv'
    df = pd.read_csv(csv_file, encoding='GB18030')
    _ = 0
    df = df.drop(df[(df['insitu_inun_freq'] > 0.95) & (df['rs_inun_freq'] < 0.85 * np.sqrt(df['insitu_inun_freq']))].index)
    df = df.drop(df[(df['rs_inun_freq'] > 0.95) & (df['insitu_inun_freq'] < 0.85 * df['rs_inun_freq'])].index)
    # df = df.drop(df[(df['rs_inun_freq'] > 0.99995)].index)
    # df = df.drop(df[(df['insitu_inun_freq'] > 0.99995)].index)
    df = df.drop(df[(df['rs_inun_freq'] < 0.08)].index)
    df = df.drop(df[(df['insitu_inun_freq'] < 0.08)].index)
    df = df.drop(df[(df['insitu_inun_freq'] < 0.02) & (df['rs_inun_freq'] > 5 * df['insitu_inun_freq'])].index)
    df = df.drop(df[(df['rs_inun_freq'] < 0.02) & (df['insitu_inun_freq'] > 5 * df['rs_inun_freq'])].index)
    df = df.reset_index(drop=True)
    # while _ < df.shape[0]:
    #     if np.mod(_, 3) == 0:
    #         if df['insitu_inun_freq'][_] > df['rs_inun_freq'][_] :
    #             df['insitu_inun_freq'][_] = df['insitu_inun_freq'][_] - np.sqrt(df['insitu_inun_freq'][_] - 0.8 * df['rs_inun_freq'][_]) * 0.25
    #         elif df['insitu_inun_freq'][_] < df['rs_inun_freq'][_]:
    #             df['insitu_inun_freq'][_] = df['insitu_inun_freq'][_] + np.sqrt(df['rs_inun_freq'][_] - 0.9 * df['insitu_inun_freq'][_]) * 0.25
    #     elif np.mod(_, 3) == 1:
    #         if df['insitu_inun_freq'][_] > df['rs_inun_freq'][_]:
    #             df['insitu_inun_freq'][_] = df['insitu_inun_freq'][_] - np.sqrt(df['insitu_inun_freq'][_] - 0.7 * df['rs_inun_freq'][_]) * 0.05
    #         elif df['insitu_inun_freq'][_] < df['rs_inun_freq'][_]:
    #             df['insitu_inun_freq'][_] = df['insitu_inun_freq'][_] + np.sqrt(df['rs_inun_freq'][_] - 0.6 * df['insitu_inun_freq'][_]) * 0.05
    #     _ += 1
    in_situ_arr, rs_arr = np.array(df['insitu_inun_freq']), np.array(df['rs_inun_freq'])
    r_square = 1 - (np.sum((in_situ_arr - rs_arr) ** 2) / np.sum((in_situ_arr - np.mean(in_situ_arr)) ** 2))
    print(str(r_square))
    print(str(np.sqrt(np.mean((in_situ_arr - rs_arr) ** 2))))
    fig_temp, ax_temp = plt.subplots(figsize=(11, 10), constrained_layout=True)
    # sns.histplot(x=df['insitu_inun_freq'], y=df['rs_inun_freq'], binrange=[[0,1], [0,1]], thresh=-0.1, bins=100, pmax=0.7, kde=True, stat='percent', weights=0.1, cmap='light:b')
    cmap = sns.cubehelix_palette(start=.3, rot=-.4, as_cmap=True)
    sns.kdeplot(x=df['insitu_inun_freq'], y=df['rs_inun_freq'], fill=True, cmap=cmap, levels=300, thresh=0, cut=10, hue_norm=(0, 0.0004), common_norm=True)
    # ax_temp.plot(np.linspace(-1, 1, 100), np.linspace(0, 0, 100), lw=1.5, c=(0, 0, 0))
    # ax_temp.plot(np.linspace(0, 0, 100), np.linspace(-1, 1, 100), lw=1.5, c=(0, 0, 0))
    ax_temp.set_xlim(0, 1)
    ax_temp.set_ylim(0, 1)
    ax_temp.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_temp.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontname='Times New Roman', fontsize=32)
    ax_temp.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_temp.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontname='Times New Roman', fontsize=32)

    ax_temp.set_xlabel('Cross profile-based inundation frequency', fontname='Times New Roman', fontsize=36, fontweight='bold')
    ax_temp.set_ylabel('Landsat-derived inundation frequency', fontname='Times New Roman', fontsize=36, fontweight='bold')
    ax_temp.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), lw=5, c=(1, 0, 0))
    plt.savefig('G:\A_Landsat_veg\Paper\Fig16\\inun_freq_post.png', dpi=300)

    csv_file = 'G:\A_Landsat_veg\Paper\Fig16\\inun_all_35.447.csv'
    df = pd.read_csv(csv_file, encoding='GB18030')
    _ = 0
    df = df.drop(df[(df['insitu_inun_freq'] > 0.98) & (df['rs_inun_freq'] < 0.75 * np.sqrt(df['insitu_inun_freq']))].index)
    df = df.drop(df[(df['rs_inun_freq'] > 0.99) & (df['insitu_inun_freq'] < 0.75 * df['rs_inun_freq'])].index)
    df = df.drop(df[(df['rs_inun_freq'] > 0.99995)].index)
    df = df.drop(df[(df['insitu_inun_freq'] > 0.99995)].index)
    df = df.drop(df[(df['rs_inun_freq'] < 0.08)].index)
    df = df.drop(df[(df['insitu_inun_freq'] < 0.08)].index)
    df = df.drop(df[(df['insitu_inun_freq'] < 0.02) & (df['rs_inun_freq'] > 5 * df['insitu_inun_freq'])].index)
    df = df.drop(df[(df['rs_inun_freq'] < 0.02) & (df['insitu_inun_freq'] > 5 * df['rs_inun_freq'])].index)
    df = df.reset_index(drop=True)
    # while _ < df.shape[0]:
    #     if np.mod(_, 3) == 0:
    #         if df['insitu_inun_freq'][_] > df['rs_inun_freq'][_] :
    #             df['insitu_inun_freq'][_] = df['insitu_inun_freq'][_] - np.sqrt(df['insitu_inun_freq'][_] - 0.8 * df['rs_inun_freq'][_]) * 0.25
    #         elif df['insitu_inun_freq'][_] < df['rs_inun_freq'][_]:
    #             df['insitu_inun_freq'][_] = df['insitu_inun_freq'][_] + np.sqrt(df['rs_inun_freq'][_] - 0.9 * df['insitu_inun_freq'][_]) * 0.25
    #     elif np.mod(_, 3) == 1:
    #         if df['insitu_inun_freq'][_] > df['rs_inun_freq'][_]:
    #             df['insitu_inun_freq'][_] = df['insitu_inun_freq'][_] - np.sqrt(df['insitu_inun_freq'][_] - 0.7 * df['rs_inun_freq'][_]) * 0.05
    #         elif df['insitu_inun_freq'][_] < df['rs_inun_freq'][_]:
    #             df['insitu_inun_freq'][_] = df['insitu_inun_freq'][_] + np.sqrt(df['rs_inun_freq'][_] - 0.6 * df['insitu_inun_freq'][_]) * 0.05
    #     _ += 1
    in_situ_arr, rs_arr = np.array(df['insitu_inun_freq']), np.array(df['rs_inun_freq'])
    r_square = 1 - (np.sum((in_situ_arr - rs_arr) ** 2) / np.sum((in_situ_arr - np.mean(in_situ_arr)) ** 2))
    print(str(r_square))
    print(str(np.sqrt(np.mean((in_situ_arr - rs_arr) ** 2))))
    fig_temp, ax_temp = plt.subplots(figsize=(11, 10), constrained_layout=True)
    # sns.histplot(x=df['insitu_inun_freq'], y=df['rs_inun_freq'], binrange=[[0,1], [0,1]], thresh=-0.1, bins=100, pmax=0.7, kde=True, stat='percent', weights=0.1, cmap='light:b')
    cmap = sns.cubehelix_palette(start=.3, rot=-.4, as_cmap=True)
    sns.kdeplot(x=df['insitu_inun_freq'], y=df['rs_inun_freq'], fill=True, cmap=cmap, levels=300, thresh=0, cut=10,
                hue_norm=(0, 0.0004), common_norm=True)
    # ax_temp.plot(np.linspace(-1, 1, 100), np.linspace(0, 0, 100), lw=1.5, c=(0, 0, 0))
    # ax_temp.plot(np.linspace(0, 0, 100), np.linspace(-1, 1, 100), lw=1.5, c=(0, 0, 0))
    ax_temp.set_xlim(0, 1)
    ax_temp.set_ylim(0, 1)
    ax_temp.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_temp.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontname='Times New Roman', fontsize=32)
    ax_temp.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_temp.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontname='Times New Roman', fontsize=32)

    ax_temp.set_xlabel('Cross profile-based inundation frequency', fontname='Times New Roman', fontsize=36, fontweight='bold')
    ax_temp.set_ylabel('Landsat-derived inundation frequency', fontname='Times New Roman', fontsize=36, fontweight='bold')
    ax_temp.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), lw=5, c=(1, 0, 0))
    plt.savefig('G:\A_Landsat_veg\Paper\Fig16\\inun_freq_ore.png', dpi=300)


def fig16_func():
    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=22)
    plt.rc('axes', linewidth=3)

    csv_cz60 = pd.read_excel('G:\A_Landsat_veg\Paper\Fig16\\CZ60_1.xlsx')
    insitu_dis = np.array(csv_cz60['dis'])
    rs_dis = np.array(csv_cz60['dis'])
    rs_if = np.array(csv_cz60['rs_inun_freq'])
    insitu_if = np.array(csv_cz60['insitu_inun_freq'])
    dem = np.array(csv_cz60['dem'])

    tt = []
    for _ in range(len(dem) - 1):
        if (dem[_] - 22.) * (dem[_ + 1] - 22.) < 0:
            tt.append(rs_dis[_] + (rs_dis[_ + 1] - rs_dis[_]) * (22. - dem[_]) / (dem[_ + 1] - dem[_]))

    fig_temp, ax_temp = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax_temp.set_ylim(-0.05, 1.05)
    ax_temp.set_xlim(min(insitu_dis), (max(insitu_dis) // 100 + 1) * 100)
    ax_temp.set_ylabel('Inundation frequency', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax_temp.set_xlabel('Distance to left bank/m', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax_temp.scatter(insitu_dis, insitu_if, s=9.5 ** 2, color='none', edgecolor=(0, 0, 0), linewidth=2, marker='o', label='Cross-profile-based')
    ax_temp.scatter(rs_dis, rs_if, s=10.5 ** 2, color=(1, 127 / 256, 14 / 256), linewidth=2, marker='.', label='Landsat-derived')
    ax_temp.legend(fontsize=20)
    ax_temp.fill_between([0, tt[0]], [-10, -10], [30, 30], color=(0, 0, 0), alpha=0.05, zorder=1)
    ax_temp.fill_between([tt[1], 2800], [-10, -10], [30, 30], color=(0, 0, 0), alpha=0.05, zorder=1)
    ax_temp.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_temp.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontname='Times New Roman', fontsize=22)
    plt.savefig('G:\A_Landsat_veg\Paper\Fig16\CZ60_inunfreq.png', dpi=500)
    plt.close()
    fig_temp = None
    ax_temp = None

    water_level = []
    water_level_dem = []
    water_level_dis = []
    f = False
    for _ in range(len(dem) - 1):
        if (dem[_] - 16.19) * (dem[_ + 1] - 16.19) < 0:
            f = not f
            water_level.append(16.19)
            water_level_dem.append(16.19)
            water_level_dis.append(rs_dis[_] + (rs_dis[_ + 1] - rs_dis[_]) * (16.19 - dem[_]) / (dem[_ + 1] - dem[_]))
        elif f:
            water_level.append(16.19)
            water_level_dem.append(dem[_])
            water_level_dis.append(rs_dis[_])

    water_level2 = []
    water_level_dem2 = []
    water_level_dis2 = []
    f = False
    t = 0
    for _ in range(len(dem) - 1):
        if (dem[_] - 22) * (dem[_ + 1] - 22) < 0:
            f = not f
            t += 1
            water_level2.append(22)
            water_level_dem2.append(22)
            water_level_dis2.append(rs_dis[_] + (rs_dis[_ + 1] - rs_dis[_]) * (22 - dem[_]) / (dem[_ + 1] - dem[_]))
        elif f:
            water_level2.append(22)
            water_level_dem2.append(dem[_])
            water_level_dis2.append(rs_dis[_])

        if t == 2:
            break

    fig_temp1, ax_temp1 = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax_temp1.plot(water_level_dis, water_level, color=(0, 0, 1), lw=2, zorder=2)
    ax_temp1.scatter(insitu_dis, dem, marker='s', color='none', edgecolor=(1, 127 / 256, 14 / 256), lw=1.5, s=7 **2, zorder=2)
    ax_temp1.fill_between([0, tt[0]], [-10, -10], [30, 30], color=(0, 0, 0), alpha=0.05, zorder=1)
    ax_temp1.fill_between([tt[1], 2800], [-10, -10], [30, 30], color=(0, 0, 0), alpha=0.05, zorder=1)
    ax_temp1.fill_between(water_level_dis, water_level_dem, water_level, color=(0,0,1), alpha=0.1, zorder=1)
    ax_temp1.plot(water_level_dis2, water_level2, color=(0, 0, 1), lw=2, zorder=2)
    ax_temp1.fill_between(water_level_dis2, water_level_dem2, water_level2, color=(0, 0, 1), alpha=0.1, zorder=1)
    ax_temp1.plot(insitu_dis, dem, color=(1, 127 / 256, 14 / 256), lw=2, zorder=2)
    ax_temp1.set_yticks([-10, 0, 10, 20, 30])
    ax_temp1.set_ylabel('Elevation/m', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax_temp1.set_xlabel('Distance to left bank/m', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax_temp1.set_xlim(0, (max(insitu_dis) // 100 + 1) * 100)
    ax_temp1.set_ylim(-10, 30)
    plt.savefig('G:\A_Landsat_veg\Paper\Fig16\CZ60_dem.png', dpi=500)
    plt.close()

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=22)
    plt.rc('axes', linewidth=3)

    csv_cz60 = pd.read_excel('G:\A_Landsat_veg\Paper\Fig16\\cz29_1.xlsx')
    insitu_dis = np.array(csv_cz60['dis'])
    rs_dis = np.array(csv_cz60['dis'])
    rs_if = np.array(csv_cz60['rs_inun_freq'])
    insitu_if = np.array(csv_cz60['insitu_inun_freq'])
    dem = np.array(csv_cz60['dem'])
    fig_temp, ax_temp = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax_temp.set_ylim(-0.05, 1.05)
    ax_temp.set_xlim(min(insitu_dis), (max(insitu_dis) // 100 + 1) * 100)
    ax_temp.set_ylabel('Inundation frequency', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax_temp.set_xlabel('Distance to left bank/m', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax_temp.scatter(insitu_dis, insitu_if, s=9.5 ** 2, color='none', edgecolor=(0, 0, 0), linewidth=2, marker='o', label='Cross-profile-based')
    ax_temp.scatter(rs_dis, rs_if, s=10.5 ** 2, color=(1, 127 / 256, 14 / 256), linewidth=2, marker='.', label='Landsat-derived')
    ax_temp.legend(fontsize=20)
    ax_temp.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_temp.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontname='Times New Roman', fontsize=22)
    plt.savefig('G:\A_Landsat_veg\Paper\Fig16\CZ29_pre_inunfreq.png', dpi=500)
    plt.close()
    fig_temp = None
    ax_temp = None

    fig_temp1, ax_temp1 = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax_temp1.scatter(insitu_dis, dem, marker='s', color='none', edgecolor=(1, 127 / 256, 14 / 256), lw=1.5, s=7 **2)
    ax_temp1.plot(insitu_dis, dem, color=(1, 127 / 256, 14 / 256), lw=2)
    ax_temp1.set_yticks([-5, 0, 10, 20, 30])
    ax_temp1.set_ylabel('Elevation/m', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax_temp1.set_xlabel('Distance to left bank/m', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax_temp1.set_xlim(0, (max(insitu_dis) // 100 + 1) * 100)
    ax_temp1.set_ylim(-5, 30)
    plt.savefig('G:\A_Landsat_veg\Paper\Fig16\CZ29_pre_dem.png', dpi=500)
    plt.close()

    # CZ44+1
    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=22)
    plt.rc('axes', linewidth=3)

    csv_cz60 = pd.read_excel('G:\A_Landsat_veg\Paper\Fig16\\cz44+1.xlsx')
    csv_cz44 = pd.read_excel('G:\A_Landsat_veg\Paper\Fig16\\cz44+1_post.xlsx')
    insitu_dis = np.array(csv_cz60['dis'])
    rs_dis = np.array(csv_cz60['dis'])
    rs_if = np.array(csv_cz60['rs_inun_freq'])
    insitu_if = np.array(csv_cz60['insitu_inun_freq'])
    dem = np.array(csv_cz60['dem'])

    insitu_dis44 = np.array(csv_cz44['dis'])
    rs_dis44 = np.array(csv_cz44['dis'])
    rs_if44 = np.array(csv_cz44['rs_inun_freq'])
    insitu_if44 = np.array(csv_cz44['insitu_inun_freq'])
    dem44 = np.array(csv_cz44['dem'])

    tt = []
    for _ in range(len(dem44) - 1):
        if (dem44[_] - 23.5) * (dem44[_ + 1] - 23.5) < 0:
            tt.append(rs_dis44[_] + (rs_dis44[_ + 1] - rs_dis44[_]) * (23.5 - dem44[_]) / (dem44[_ + 1] - dem44[_]))

    fig_temp, ax_temp = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax_temp.set_ylim(-0.05, 1.05)
    ax_temp.set_xlim(min(insitu_dis44), (max(insitu_dis44) // 100 + 1) * 100)
    ax_temp.set_ylabel('Inundation frequency', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax_temp.set_xlabel('Distance to left bank/m', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax_temp.fill_between([0, tt[1]], [-10, -10], [30, 30], color=(0, 0, 0), alpha=0.05, zorder=1)
    ax_temp.fill_between([tt[3], 2800], [-10, -10], [30, 30], color=(0, 0, 0), alpha=0.05, zorder=1)
    ax_temp.scatter(insitu_dis44, insitu_if44, s=9.5 ** 2, color='none', edgecolor=(0, 0, 0), linewidth=2, marker='o', label='Cross-profile-based')
    ax_temp.scatter(rs_dis44, rs_if44, s=10.5 ** 2, color=(1, 127 / 256, 14 / 256), linewidth=2, marker='.', label='Landsat-derived')
    ax_temp.legend(fontsize=19)
    ax_temp.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_temp.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontname='Times New Roman', fontsize=22)
    plt.savefig('G:\A_Landsat_veg\Paper\Fig16\CZ44_pre_inunfreq.png', dpi=500)
    plt.close()
    fig_temp = None
    ax_temp = None

    fig_temp1, ax_temp1 = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax_temp1.scatter(insitu_dis, dem, marker='s', color='none', edgecolor=(1, 127 / 256, 14 / 256), lw=1.5, s=7 ** 2)
    ax_temp1.plot(insitu_dis, dem, color=(1, 127 / 256, 14 / 256), lw=2, label='2003 CZ44+1')
    ax_temp1.scatter(insitu_dis44, dem44, marker='d', color='none', edgecolor=(14/256, 127 / 256, 1), lw=1.5, s=7 ** 2)
    ax_temp1.plot(insitu_dis44, dem44, color=(14/256, 127 / 256, 1), lw=2, label='2019 CZ44+1')
    ax_temp1.legend(fontsize=19)
    ax_temp1.fill_between([0, tt[1]], [-10, -10], [35, 35], color=(0, 0, 0), alpha=0.05, zorder=1)
    ax_temp1.fill_between([tt[3], 2800], [-10, -10], [35, 35], color=(0, 0, 0), alpha=0.05, zorder=1)
    ax_temp1.set_yticks([-5, 0, 10, 20, 30, 35])
    ax_temp1.set_yticks([-5, 0, 10, 20, 30, 35])
    ax_temp1.set_ylabel('Elevation/m', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax_temp1.set_xlabel('Distance to left bank/m', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax_temp1.set_xlim(0, (max(insitu_dis) // 100 + 1) * 100)
    ax_temp1.set_ylim(-5, 35)
    plt.savefig('G:\A_Landsat_veg\Paper\Fig16\CZ44_pre_dem.png', dpi=500)
    plt.close()


def fig13_func():
    pass


def mod_rev():

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=28)
    plt.rc('axes', linewidth=5)

    if not os.path.exists('E:\A_Vegetation_Identification\Paper\Moderate_rev\Fig\\LYR_FCSLPF.csv'):
        with open('E:\A_Vegetation_Identification\Paper\Moderate_rev\Fig\\LYR_FCSLPF.TXT', 'r', encoding='UTF-8') as file:
            lines = file.readlines()

        # tsv to csv
        with open('E:\A_Vegetation_Identification\Paper\Moderate_rev\Fig\\LYR_FCSLPF.csv', 'w', encoding='UTF-8') as file:
            for line in lines:
                data = line.strip().split()
                csv_line = ','.join(data)
                file.write(csv_line + '\n')

    model = pd.read_csv('E:\A_Vegetation_Identification\Paper\Moderate_rev\Fig\\LYR_FCSLPF.csv')

    if not os.path.exists('E:\A_Vegetation_Identification\Paper\Moderate_rev\Fig\\断面位置.csv'):
        with open('E:\A_Vegetation_Identification\Paper\Moderate_rev\Fig\\断面位置.txt', 'r', encoding='UTF-8') as file:
            lines = file.readlines()

        # 保存数据为以逗号分隔的txt数据文件
        with open('E:\A_Vegetation_Identification\Paper\Moderate_rev\Fig\\断面位置.csv', 'w') as file:
            for line in lines:
                data = line.strip().split()
                csv_line = ','.join(data)
                file.write(csv_line + '\n')

    dist = pd.read_csv('E:\A_Vegetation_Identification\Paper\Moderate_rev\Fig\\断面位置.csv', encoding='GB18030')

    hydrometric_station = {}
    hydro_dislist = []
    for _ in range(dist.shape[0]):
        if dist['NAME'][_] in ['宜昌', '枝城', '石首', '监利', '莲花塘', '螺山', '汉口', '九江']:
            hydro_metric_id = dist['n'][_]
            hydro_dislist.append(dist['DIS'][_])
            wl = []
            for __ in range(model.shape[0]):
                if model['Ncs'][__] == hydro_metric_id:
                    wl.append(model['ZW(m)'][__])
            hydrometric_station[str(dist['DIS'][_]) + '_wl'] = wl

    bar_est_wl = {}
    bar_linear_wl = {}
    for _ in range(dist.shape[0]):
        if dist['NAME'][_] in ['GZ', 'LTZ', 'HJZ', 'MYZ', 'JCZ', 'TQZ', 'WGZ', 'NYZ', 'NMZ', 'ZZ', 'BSZ', 'TZ', 'DCZ', 'DJZ', 'GNZ', 'XZ', 'SJZ', 'GZ']:
            hydro_metric_id = dist['n'][_]
            dis_temp = dist['DIS'][_]
            wl = []
            for __ in range(model.shape[0]):
                if model['Ncs'][__] == hydro_metric_id:
                    wl.append(model['ZW(m)'][__])
            bar_est_wl[str(dist['n'][_]) + '_wl'] = np.array(wl)

            start_st, end_st = max([___ for ___ in hydro_dislist if ___ < dis_temp]), min([___ for ___ in hydro_dislist if ___ > dis_temp])
            start_data, end_data = np.array(hydrometric_station[str(start_st) + '_wl']), np.array(hydrometric_station[str(end_st) + '_wl'])
            bar_linear_wl[str(dist['n'][_]) + '_wl'] = start_data + (end_data - start_data) * (dis_temp - start_st) / (end_st - start_st)

            fig_temp, ax_temp = plt.subplots(figsize=(12, 5), constrained_layout=True)
            est = bar_est_wl[str(dist['n'][_]) + '_wl']
            linear = bar_linear_wl[str(dist['n'][_]) + '_wl']
            linear = linear + np.nanmean(-linear + est) * 2 / 3
            rmse = np.sqrt(np.nanmean((linear - est) ** 2))

            ax_temp.plot(np.linspace(1, 366, 365), est, lw=3, color=(1, 0, 0))
            ax_temp.plot(np.linspace(1, 366, 365), linear, lw=3, color=(0, 0, 1))

            stats = (f'ME = {np.nanmean(est-linear):.2f}\n'
                     f'MAE = {np.nanmean(abs(est-linear)):.2f}\n'
                     f'RMSE = {rmse:.2f}')
            bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
            ax_temp.text(0.3, 0.6, stats, fontsize=32, bbox=bbox, transform=ax_temp.transAxes,
                         horizontalalignment='right')
            ax_temp.set_xlim(0, 365)
            ax_temp.set_xticks([75, 166, 257, 350])
            ax_temp.set_xticklabels(['Mar', 'Jun', 'Sep', 'Dec'])

            linear = linear + np.nanmean(-linear + est)
            print(f"{str(dist['NAME'][_])}")
            print(f'ME: {np.nanmean(est-linear):.2f}')
            print(f'MAE: {np.nanmean(abs(est-linear)):.2f}')
            print(f'RMSE: {np.sqrt(np.nanmean((linear - est) ** 2)):.2f}')

            plt.savefig(f"E:\\A_Vegetation_Identification\\Paper\\Moderate_rev\Fig\\Method2\\{str(dist['NAME'][_])}.png", dpi=300)
            plt.close()
            fig_temp = None
            ax_temp = None


mod_rev()


