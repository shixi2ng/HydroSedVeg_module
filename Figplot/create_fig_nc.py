
import os.path

import scipy.io

from RSDatacube.RSdc import *

import seaborn as sns

from River_GIS.River_GIS import *
from scipy.stats import pearsonr, kurtosis, variation, cramervonmises_2samp, wasserstein_distance

import matplotlib.pyplot as plt



def figS3nc_func():
    plt.rcParams['font.family'] = ['Arial', 'SimHei']
    plt.rc('font', size=22)
    plt.rc('axes', linewidth=2)

    wl1 = HydroStationDS()
    wl1.import_from_standard_files('G:\\A_1Dflow_sed\\Hydrodynamic_model\\Original_water_level\\', 'G:\\A_1Dflow_sed\\Hydrodynamic_model\\Original_water_level\\对应表.csv')

    sec_wl_diff, sec_ds_diff = [], []
    sec_dis = [0, 63.83, 153.87, 306.77, 384.16, 423.15, 653.115, 955]
    sec_name = ['宜昌', '枝城', '沙市', '监利', '莲花塘', '螺山', '汉口', '九江']
    for sec in sec_name:
        fig14_df = wl1.hydrostation_inform_df[sec]
        wl_pri, wl_post = [], []
        sd_pri, sd_post = [], []
        ds_pri, ds_post = [], []
        year_dic = {}
        for year in range(1985, 2021):
            year_temp = fig14_df['year'] == year
            discharge = fig14_df['flow/m3/s'][year_temp].tolist()
            flow_temp = fig14_df['water_level/m'][year_temp].tolist() - wl1.waterlevel_offset_list[wl1.hydrostation_name_list.index(sec)]
            sed_temp = fig14_df['sediment_concentration/kg/m3'][year_temp].tolist()
            year_dic[f'{str(year)}_wl'] = flow_temp[0:365]
            year_dic[f'{str(year)}_sed'] = sed_temp[0:365]
            if len(flow_temp) == 365 or len(flow_temp) == 366:
                if year > 2004:
                    wl_post.append(flow_temp[0:365])
                    sd_post.append(sed_temp[0:365])
                    ds_post.append(discharge[0:365])

                if 1998 <= year <= 2004:
                    wl_pri.append(flow_temp[0:365])
                    ds_pri.append(discharge[0:365])
                    sd_pri.append(sed_temp[0:365])

        diff_dis = np.array(np.nanmean(wl_post, axis=0)) - np.array(np.nanmean(wl_pri, axis=0))
        sec_wl_diff.append(diff_dis[152: 304].tolist())
        diff_dis = np.array(np.nanmean(ds_post, axis=0)) - np.array(np.nanmean(ds_pri, axis=0))
        sec_ds_diff.append(diff_dis[152: 304].tolist())

    plt.close()
    plt.rcParams['font.family'] = ['Arial', 'SimHei']
    plt.rc('font', size=18)
    plt.rc('axes', axisbelow=True)
    plt.rc('axes', linewidth=3)
    fig_temp, ax_temp = plt.subplots(figsize=(12, 5), constrained_layout=True)
    # ax_temp.grid(axis='y', color=(210 / 256, 210 / 256, 210 / 256), zorder=0)
    ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_pri, axis=0).reshape([365]), lw=5, c=(0, 0, 1), zorder=4)
    # ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_post, axis=0).reshape([365]), lw=5, c=(1, 0, 0), zorder=4)
    # ax_temp.plot(np.linspace(1, 365, 365), np.linspace(l1, l1, 365), lw=2, ls='--', c=(0, 0, 0))
    ax_temp.set_xlim(-50, 975)
    ax_temp.set_ylim(-4, 1)
    # ax_temp.set_yticks(ytick)
    bplot = ax_temp.boxplot(sec_wl_diff, widths=30, positions=sec_dis, notch=True,  showfliers=False, whis=(5, 95), patch_artist=True, medianprops={"color": "blue", "linewidth": 2.8}, boxprops={"linewidth": 1.8}, whiskerprops={ "linewidth": 1.8}, capprops={"color": "C0", "linewidth": 1.8})

    ax_temp.set_xticks([0,100,200,300,400,500,600,700,800,900])
    ax_temp.set_xticklabels(['0', '100', '200', '300', '400', '500', '600', '700', '800', '900'])
    # ax_temp.set_xticklabels(['Yichang', 'Zhicheng', 'Jianli', 'Lianhuatang', 'Luoshan', 'Hankou', 'Jiujiang'], fontname='Arial', fontsize=24)
    ax_temp.set_ylabel('Water level difference/m', fontname='Arial', fontsize=28, fontweight='bold')
    colors = []

    for patch in bplot['boxes']:
        patch.set_facecolor((208/256, 156/256, 44/256))
    # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
    # plt.savefig(f'G:\\B_papers_patents\\RA_Dam operations enhance floodplain vegetation resistance and resilience but compress lateral heterogeneity\\A_fig_nc\\S_NC_FigS3\\along_wl_nc.png', dpi=500)
    plt.close()

    plt.close()
    plt.rcParams['font.family'] = ['Arial', 'SimHei']
    plt.rc('font', size=18)
    plt.rc('axes', axisbelow=True)
    plt.rc('axes', linewidth=3)
    fig_temp, ax_temp = plt.subplots(figsize=(12, 5), constrained_layout=True)
    # ax_temp.grid(axis='y', color=(210 / 256, 210 / 256, 210 / 256), zorder=0)
    ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_pri, axis=0).reshape([365]), lw=5, c=(0, 0, 1), zorder=4)
    # ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_post, axis=0).reshape([365]), lw=5, c=(1, 0, 0), zorder=4)
    # ax_temp.plot(np.linspace(1, 365, 365), np.linspace(l1, l1, 365), lw=2, ls='--', c=(0, 0, 0))
    ax_temp.set_xlim(-50, 975)
    # ax_temp.set_yticks(ytick)
    bplot = ax_temp.boxplot(sec_ds_diff, widths=20, positions=sec_dis, notch=True, showfliers=False, patch_artist = True, medianprops={"color": "blue", "linewidth": 2.8}, boxprops={"linewidth": 1.8}, whiskerprops={ "linewidth": 1.8}, capprops={"color": "C0", "linewidth": 1.8})
    ax_temp.set_xticks([0,100,200,300,400,500,600,700,800,900])
    ax_temp.set_xticklabels(['0', '100', '200', '300', '400', '500', '600', '700', '800', '900'])
    # ax_temp.set_xticklabels(['Yichang', 'Zhicheng', 'Jianli', 'Lianhuatang', 'Luoshan', 'Hankou', 'Jiujiang'], fontname='Arial', fontsize=24)
    ax_temp.set_ylabel('Discharge difference/m', fontname='Arial', fontsize=28, fontweight='bold')
    colors = []

    for patch in bplot['boxes']:
        patch.set_facecolor((208/256, 156/256, 44/256))
    # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
    # plt.savefig(f'G:\\B_papers_patents\\RA_Dam operations enhance floodplain vegetation resistance and resilience but compress lateral heterogeneity\\A_fig_nc\\S_NC_FigS3\\along_ds_nc.png', dpi=500)
    plt.close()
    wl_pri2, wl_post2 = [], []
    wl_pri3, wl_post3 = [], []

    for sec, r1, l1, ytick, in zip(['宜昌', '枝城', '莲花塘', '汉口'], [(36, 54), (34, 50), (16, 34), (10, 30)], [50, 44, 30, 24], [[36, 39, 42, 45, 48, 51, 54], [34, 38, 42, 46, 50], [17, 20, 23, 26, 29, 32], [10, 15, 20, 25, 30]]):
        fig14_df = wl1.hydrostation_inform_df[sec]
        year_dic = {}
        wl_pri, wl_post = [], []
        sd_pri, sd_post = [], []
        ds_pri, ds_post = [], []

        for year in range(1985, 2021):
            year_temp = fig14_df['year'] == year
            discharge = fig14_df['flow/m3/s'][year_temp].tolist()
            flow_temp = fig14_df['water_level/m'][year_temp].tolist() - wl1.waterlevel_offset_list[wl1.hydrostation_name_list.index(sec)]
            sed_temp = fig14_df['sediment_concentration/kg/m3'][year_temp].tolist()
            year_dic[f'{str(year)}_wl'] = flow_temp[0:365]
            year_dic[f'{str(year)}_sed'] = sed_temp[0:365]
            if len(flow_temp) == 365 or len(flow_temp) == 366:
                if year > 2002:
                    wl_post.append(flow_temp[0:365])
                    sd_post.append(sed_temp[0:365])
                    ds_post.append(discharge[0:365])
                    if sec == '宜昌':
                        wl_post2.extend(flow_temp[0: 365])
                    elif sec == '汉口':
                        wl_post3.extend(flow_temp[0: 365])
                elif year <= 2002:
                    wl_pri.append(flow_temp[0:365])
                    if sec == '宜昌':
                        wl_pri2.extend(flow_temp[0: 365])
                    elif sec == '汉口':
                        wl_pri3.extend(flow_temp[0: 365])

                if 1998 <= year <= 2004:
                    ds_pri.append(discharge[0:365])
                    sd_pri.append(sed_temp[0:365])

        wl_post = np.array(wl_post)
        sd_post = np.array(sd_post)
        wl_pri = np.array(wl_pri)
        sd_pri = np.array(sd_pri)
        ds_pri = np.array(ds_pri)
        ds_post = np.array(ds_post)

        sd_pri[sd_pri == 0] = np.nan
        sd_post[sd_post == 0] = np.nan

        plt.rcParams['font.family'] = ['Arial', 'SimHei']
        plt.rc('font', size=22)
        plt.rc('axes', linewidth=2)
        fig_temp, ax_temp = plt.subplots(figsize=(10, 3.75), constrained_layout=True)
        ax_temp.grid(axis='y', color=(210 / 256, 210 / 256, 210 / 256), zorder=0)
        ax_temp.fill_between(np.linspace(152, 290, 121), np.linspace(r1[1], r1[1], 121), np.linspace(r1[0],r1[0],121),alpha=1, color=(0.9, 0.9, 0.9))
        ax_temp.fill_between(np.linspace(1, 365, 365), np.nanmax(wl_post, axis=0).reshape([365]), np.nanmin(wl_post, axis=0).reshape([365]),alpha=0.3, color=(0.8, 0.2, 0.1), zorder=3)
        ax_temp.fill_between(np.linspace(1, 365, 365), np.nanmax(wl_pri, axis=0).reshape([365]), np.nanmin(wl_pri, axis=0).reshape([365]),alpha=0.3, color=(0.1, 0.2, 0.8), zorder=2)
        ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_pri, axis=0).reshape([365]), lw=5, c=(0, 0, 1), zorder=4)
        ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_post, axis=0).reshape([365]), lw=5, c=(1, 0, 0), zorder=4)
        ax_temp.plot(np.linspace(1, 365,365), np.linspace(l1,l1,365), lw=2, ls='-.', c=(0,0,0))
        ax_temp.set_xlim(1, 365)
        ax_temp.set_ylim(r1[0], r1[1])
        ax_temp.set_yticks(ytick)

        a = [45,  106, 167, 228, 289, 350]
        c = ['Feb', 'Apr',  'Jun',  'Aug',  'Oct', 'Dec']
        ax_temp.set_xticks(a)
        ax_temp.set_xticklabels(c, fontname='Arial', fontsize=24)
        ax_temp.set_ylabel('Water level/m', fontname='Arial', fontsize=28, fontweight='bold')
        # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
        # plt.savefig(f'G:\\B_papers_patents\\RA_Dam operations enhance floodplain vegetation resistance and resilience but compress lateral heterogeneity\\A_fig_nc\\S_NC_FigS3\\{sec}_wl_nc.png', dpi=500)
        plt.close()

        # fig_temp, ax_temp = plt.subplots(figsize=(11, 6), constrained_layout=True)
        # wl_temp = np.concatenate([np.nanmean(sd_pri[:, 150: 300], axis=1), np.nanmean(sd_post[:, 150: 300], axis=1)])
        # ax_temp.bar([_ for _ in range(1990, 2003)], np.nanmean(sd_pri[:, 150: 300], axis=1), 0.6, label='SAR', color=(255/256, 155/256, 37/256), edgecolor=(0/256, 0/256, 0/256), linewidth=1.5, zorder=3, alpha=0.5)
        # ax_temp.plot([_ for _ in range(1990, 2003)], [np.nanmean(np.nanmean(sd_pri[:, 150: 300], axis=1)) for _ in range(1990, 2003)], linewidth=5, c=(255/256, 155/256, 37/256))
        # ax_temp.bar([_ for _ in range(2003, 2021)], np.nanmean(sd_post[:, 150: 300], axis=1), 0.6, label='SAR', color=(0/256, 92/256, 171/256), edgecolor=(0 / 256, 0 / 256, 0 / 256), linewidth=1.5, zorder=3, alpha=0.5)
        # ax_temp.plot([_ for _ in range(2003, 2021)], [np.nanmean(np.nanmean(sd_post[:, 150: 300], axis=1)) for _ in range(2003, 2021)], linewidth=5, c=(0/256, 92/256, 171/256))
        # ax_temp.set_xlabel('Year', fontname='Arial', fontsize=24, fontweight='bold')
        # ax_temp.set_ylabel('Sediment concentration', fontname='Arial', fontsize=24, fontweight='bold')
        # ax_temp.set_xlim(1989.5, 2020.5)
        # plt.savefig(f'G:\\B_papers_patents\\RA_Dam operations enhance floodplain vegetation resistance and resilience but compress lateral heterogeneity\\A_fig_nc\\S_NC_FigS3\\{sec}_annual_sd_nc.png', dpi=500)

        # fig_temp, ax_temp = plt.subplots(figsize=(15, 6), constrained_layout=True)
        # wl_temp = np.concatenate([np.nanmean(ds_pri[:, 150: 300], axis=1), np.nanmean(sd_post[:, 150: 300], axis=1)])
        # ax_temp.bar([_ for _ in range(1990, 2003)], np.nanmean(ds_pri[:, 150: 300], axis=1), 0.6, label='SAR', color=(256/256, 200/256, 87/256), edgecolor=(0/256, 0/256, 0/256), linewidth=1.5, zorder=3, alpha=0.5)
        # ax_temp.plot([_ for _ in range(1990, 2003)], [np.nanmean(np.nanmean(ds_pri[:, 150: 300], axis=1)) for _ in range(1990, 2003)], linewidth=4, c=(255/256, 200/256, 87/256))
        # ax_temp.bar([_ for _ in range(2003, 2021)], np.nanmean(ds_post[:, 150: 300], axis=1), 0.6, label='SAR', color=(0/256, 72/256, 151/256), edgecolor=(0 / 256, 0 / 256, 0 / 256), linewidth=1.5, zorder=3, alpha=0.5)
        # ax_temp.plot([_ for _ in range(2003, 2021)], [np.nanmean(np.nanmean(ds_post[:, 150: 300], axis=1)) for _ in range(2003, 2021)], linewidth=3, c=(0/256, 72/256, 151/256))
        # ax_temp.set_xlabel('Year', fontname='Arial', fontsize=28, fontweight='bold')
        # ax_temp.set_ylabel('Sediment concentration', fontname='Arial', fontsize=28, fontweight='bold')
        # ax_temp.set_xlim(1989.5, 2020.5)
        # plt.savefig(f'G:\\B_papers_patents\\RA_Dam operations enhance floodplain vegetation resistance and resilience but compress lateral heterogeneity\\A_fig_nc\\S_NC_FigS3\\{sec}_annual_ds_nc.png', dpi=500)

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
        # plt.savefig(f'G:\\B_papers_patents\\RA_Dam operations enhance floodplain vegetation resistance and resilience but compress lateral heterogeneity\\A_fig_nc\\S_NC_FigS3\\{sec}_sd.png', dpi=500)

        if sec == '宜昌':
            fig_temp, ax_temp = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
            wl_temp = np.concatenate([np.nanmax(wl_pri, axis=1), np.nanmax(wl_post, axis=1)])
            # ax_temp.bar([_ for _ in range(1985, 2003)], np.nanmax(wl_pri, axis=1), 0.65, label='SAR', color=(0.2, 0.3, 0.8), edgecolor=(0/256, 0/256, 0/256), linewidth=1, zorder=3, alpha=0.5)
            # ax_temp.bar([_ for _ in range(2003, 2021)], np.nanmax(wl_post, axis=1), 0.65, label='SAR', color=(0.8, 0.3, 0.2), edgecolor=(0 / 256, 0 / 256, 0 / 256), linewidth=1, zorder=3, alpha=0.5)
            ax_temp.grid(axis='y', color=(128 / 256, 128 / 256, 128 / 256), zorder=1)
            ax_temp.plot([_ for _ in range(1985, 2003)], np.nanmax(wl_pri, axis=1), color=(0.2, 0.3, 0.8), linewidth=3,ls='-', label='Pre-TGD')
            ax_temp.plot([_ for _ in range(2003, 2021)], np.nanmax(wl_post, axis=1), color=(0.8, 0.3, 0.2), linewidth=3,ls='-', label='Post-TGD')
            ax_temp.fill_between(np.linspace(0.5, 3004.5, 100), np.linspace(0, 0, 100), np.linspace(47.7, 47.7, 100), edgecolor='none', facecolor=(0.4,0.4,0.4), alpha=0.3, lw=2)
            ax_temp.plot(np.linspace(0.5, 3004.5, 100), np.linspace(47.7, 47.7, 100), color=(0, 0, 0), ls='-.', lw=2, label='Overbank')
            ax_temp.fill_between(np.linspace(0.5, 3004.5, 100), np.linspace(47.7, 47.7, 100), np.linspace(49.7, 49.7, 100), edgecolor='none', facecolor=(0.8,0.8,0.8), alpha=0.3, lw=2)
            ax_temp.plot(np.linspace(0.5, 3004.5, 100), np.linspace(49.7, 49.7, 100), color=(0, 0, 0), ls='--', lw=2, label='Extreme')
            ax_temp.scatter([_ for _ in range(1985, 2003)], np.nanmax(wl_pri, axis=1), s = 15 **2, marker='s', color="none", edgecolor=(0,0,1), linewidth=3)
            ax_temp.scatter([_ for _ in range(2003, 2021)], np.nanmax(wl_post, axis=1), s = 15 **2, marker='s', color="none", edgecolor=(1,0,0), linewidth=3)
            # ax_temp.plot(np.linspace([2004.5, 2004.5, 100]), np.linspace([0, 100, 100]), color=(0.2, 0.2, 0.2))
            ax_temp.set_xlabel('Year', fontname='Arial', fontsize=28, fontweight='bold')
            ax_temp.set_ylabel('Annual maximum water level/m', fontname='Arial', fontsize=28, fontweight='bold')
            ax_temp.legend(fontsize=20, ncol=2)
            ax_temp.set_yticks([45, 47, 49, 51, 53, 55])
            ax_temp.set_yticklabels(['45', '47','49','51','53','55'], fontname='Arial', fontsize=24)
            ax_temp.set_xlim(1984.5, 2020.5)
            ax_temp.set_ylim(45, 55)
            plt.savefig(f'D:\A_PhD_Main_paper\Chap.2\Figure\Fig.2.5\\{sec}_annual_wl_nc.png', dpi=500)
            plt.close()

        if sec == '汉口':
            fig_temp, ax_temp = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
            wl_temp = np.concatenate([np.nanmax(wl_pri, axis=1), np.nanmax(wl_post, axis=1)])
            ax_temp.grid(axis='y', color=(128 / 256, 128 / 256, 128 / 256), zorder=1)
            # ax_temp.bar([_ for _ in range(1985, 2003)], np.nanmax(wl_pri, axis=1), 0.65, label='SAR', color=(0.2, 0.3, 0.8), edgecolor=(0/256, 0/256, 0/256), linewidth=1, zorder=3, alpha=0.5)
            # ax_temp.bar([_ for _ in range(2003, 2021)], np.nanmax(wl_post, axis=1), 0.65, label='SAR', color=(0.8, 0.3, 0.2), edgecolor=(0 / 256, 0 / 256, 0 / 256), linewidth=1, zorder=3, alpha=0.5)
            ax_temp.plot([_ for _ in range(1985, 2003)], np.nanmax(wl_pri, axis=1), color=(0.2, 0.3, 0.8), linewidth=3,
                         ls='-', label='Pre-TGD')
            ax_temp.plot([_ for _ in range(2003, 2021)], np.nanmax(wl_post, axis=1), color=(0.8, 0.3, 0.2), linewidth=3,
                         ls='-', label='Post-TGD')
            ax_temp.fill_between(np.linspace(0.5, 3004.5, 100), np.linspace(0, 0, 100), np.linspace(21.2, 21.2, 100),
                                 edgecolor='none', facecolor=(0.4, 0.4, 0.4), alpha=0.3, lw=2)
            ax_temp.plot(np.linspace(0.5, 3004.5, 100), np.linspace(21.2, 21.2, 100), color=(0, 0, 0), ls='-.', lw=2, label='Overbank')
            ax_temp.fill_between(np.linspace(0.5, 3004.5, 100), np.linspace(21.2, 21.2, 100), np.linspace(24.2, 24.2, 100), edgecolor='none', facecolor=(0.8,0.8,0.8), alpha=0.3, lw=2)
            ax_temp.plot(np.linspace(0.5, 3004.5, 100), np.linspace(24.2, 24.2, 100), color=(0, 0, 0), ls='--', lw=2, label='Extreme')
            ax_temp.scatter([_ for _ in range(1985, 2003)], np.nanmax(wl_pri, axis=1), s=15 ** 2, marker='s',
                            color="none", edgecolor=(0, 0, 1), linewidth=3)
            ax_temp.scatter([_ for _ in range(2003, 2021)], np.nanmax(wl_post, axis=1), s=15 ** 2, marker='s',
                            color="none", edgecolor=(1, 0, 0), linewidth=3)
            # ax_temp.plot(np.linspace([2004.5, 2004.5, 100]), np.linspace([0, 100, 100]), color=(0.2, 0.2, 0.2))

            ax_temp.set_xlabel('Year', fontname='Arial', fontsize=28, fontweight='bold')
            ax_temp.set_ylabel('Annual maximum water level/m', fontname='Arial', fontsize=28, fontweight='bold')
            ax_temp.set_yticks([20, 22, 24, 26, 28, 30])
            ax_temp.set_yticklabels(['20', '22', '24', '26', '28', '30'], fontname='Arial', fontsize=24)
            ax_temp.set_xlim(1984.5, 2020.5)
            ax_temp.set_ylim(20, 30)
            plt.savefig(f'D:\A_PhD_Main_paper\Chap.2\Figure\Fig.2.5\\{sec}_annual_wl_nc.png', dpi=500)
            plt.close()

    # fig_temp, ax_temp = plt.subplots(nrows=1, ncols=2, figsize=(11, 6), constrained_layout=True)
    # # n, bins, patches = ax_temp.hist(wl_pri2, 50, density=True, histtype="step",  cumulative=True, label="Cumulative histogram")
    # x = np.linspace(min(wl_pri2), max(wl_pri2))
    # y = ((1 / (np.sqrt(2 * np.pi) * 50)) * np.exp(-0.5 * (1 / 50 * (x - 50)) ** 2))
    # y = y.cumsum()
    # y /= y[-1]
    #
    # # # Complementary cumulative distributions.
    # # b = plt.ecdf(wl_pri2, complementary=False, label="pre-TGP", orientation="horizontal")
    # # ax_temp[0].ecdf(wl_pri2, complementary=False, label="pre-TGP", orientation="horizontal")
    # # # n, bins, patches = ax_temp.hist(wl_post2, 50, density=True, histtype="step",  cumulative=True, label="Cumulative histogram")
    # # x = np.linspace(min(wl_post2), max(wl_post2))
    # # y = ((1 / (np.sqrt(2 * np.pi) * 50)) * np.exp(-0.5 * (1 / 50 * (x - 50)) ** 2))
    # # y = y.cumsum()
    # # y /= y[-1]
    # # # Complementary cumulative distributions.
    # # a = plt.ecdf(wl_post2, complementary=False, label="post-TGP", orientation="horizontal")
    #
    # ax_temp[0].set_yticks([37, 43, 49, 55])
    # ax_temp[0].set_yticklabels(['37', '43', '49', '55'], fontname='Arial', fontsize=22)
    # ax_temp[0].set_xticks([0, 0.25, 0.5, 0.75, 1])
    # ax_temp[0].set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontname='Arial', fontsize=22)
    # # ax_temp.set_ylabel('Water level/m', fontname='Arial', fontsize=28, fontweight='bold')
    # # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
    # ax_temp[0].set_xlabel('Density', fontname='Arial', fontsize=28, fontweight='bold')
    # ax_temp[0].set_ylabel('Water level/m', fontname='Arial', fontsize=28, fontweight='bold')
    # ax_temp[0].set_xlim([-0.03, 1.03])
    # ax_temp[0].set_ylim([37, 55])
    # ax_temp[0].legend()
    #
    # x = np.linspace(min(wl_pri3), max(wl_pri3))
    # y = ((1 / (np.sqrt(2 * np.pi) * 50)) * np.exp(-0.5 * (1 / 50 * (x - 50)) ** 2))
    # y = y.cumsum()
    # y /= y[-1]
    #
    # # Complementary cumulative distributions.
    # ax_temp[1].ecdf(wl_pri3, complementary=False, label="pre-TGP", orientation="horizontal")
    # # n, bins, patches = ax_temp.hist(wl_post2, 50, density=True, histtype="step",  cumulative=True, label="Cumulative histogram")
    #
    # ax_temp[1].set_yticks([12, 18, 24, 30])
    # ax_temp[1].set_yticklabels(['12', '18', '24', '30'], fontname='Arial', fontsize=22)
    # ax_temp[1].set_xticks([0, 0.25, 0.5, 0.75, 1])
    # ax_temp[1].set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontname='Arial', fontsize=22)
    # x = np.linspace(min(wl_post3), max(wl_post3))
    # y = ((1 / (np.sqrt(2 * np.pi) * 50)) * np.exp(-0.5 * (1 / 50 * (x - 50)) ** 2))
    # y = y.cumsum()
    # y /= y[-1]
    # # Complementary cumulative distributions.
    # ax_temp[1].ecdf(wl_post3, complementary=False, label="post-TGP", orientation="horizontal")
    #
    # # ax_temp.set_xticks(a)
    # # ax_temp.set_xticklabels(c, fontname='Arial', fontsize=24)
    # # ax_temp.set_ylabel('Water level/m', fontname='Arial', fontsize=28, fontweight='bold')
    # # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
    # ax_temp[1].set_xlabel('Density', fontname='Arial', fontsize=28, fontweight='bold')
    # # ax_temp[1].set_ylabel('Water level/m', fontname='Arial', fontsize=28, fontweight='bold')
    # ax_temp[1].set_xlim([-0.03, 1.03])
    # ax_temp[1].set_ylim([12, 30])
    # ax_temp[1].legend()
    # plt.savefig(f'G:\\B_papers_patents\\RA_Dam operations enhance floodplain vegetation resistance and resilience but compress lateral heterogeneity\\A_fig_nc\\S_NC_FigS3\\{sec}_wl_freq_nc.png', dpi=500)

    # fig_temp, ax_temp = plt.subplots(nrows=1, ncols=1, figsize=(11, 6), constrained_layout=True)
    # wl_dic = {'wl': [], 'status': []}
    # s_ = 36
    # for _ in wl_pri2:
    #     wl_dic['status'].append('Pre-TGP period')
    #     wl_dic['wl'].append(int(np.floor(_)))
    #
    # for _ in wl_post2:
    #     wl_dic['status'].append('Post-TGP period')
    #     wl_dic['wl'].append(int(np.floor(_)))
    #
    # sns.histplot(data=wl_dic, x="wl", hue="status", palette=[(127/256, 163/256, 222/256), (247/256, 247/256, 247/256)], multiple="dodge", shrink=1.45, stat='density', alpha=0.9)
    #
    # # # Manually add dashed lines for category 'C'
    # i = 0
    # for container in ax_temp.containers:
    #     for patch in container.patches:
    #         if np.mod(i, 2) == 0:  # This checks if the patch is for category 'C'
    #             patch.set_hatch('/')  # Set dashed lines
    #             patch.set_facecolor((247/256, 247/256, 247/256))
    #         elif np.mod(i, 2) == 1:
    #             patch.set_hatch('')              # This checks if the patch is for category 'C' # Set dashed lines
    #             patch.set_facecolor((127/256, 163/256, 222/256))
    #     i += 1
    #
    # ax_temp.set_xticks([38, 42, 46, 50, 54])
    # ax_temp.set_xticklabels(['38', '42', '46', '50', '54'], fontname='Arial', fontsize=22)
    # ax_temp.set_yticks([0, 0.05, 0.10, 0.15, 0.2, 0.25])
    # ax_temp.set_yticklabels(['0%', '5%', '10%', '15%', '20%', '25%'], fontname='Arial', fontsize=22)
    # ax_temp.set_ylabel('Density', fontname='Arial', fontsize=26, fontweight='bold')
    # ax_temp.set_xlabel('Water level/m', fontname='Arial', fontsize=26, fontweight='bold')
    # ax_temp.set_ylim([0, 0.25])
    # ax_temp.set_xlim([35.5, 52])
    # ax_temp.get_legend()
    #
    # plt.savefig(f'G:\\B_papers_patents\\RA_Dam operations enhance floodplain vegetation resistance and resilience but compress lateral heterogeneity\\A_fig_nc\\A_NC_Fig1\\{sec}_hist.png', dpi=500)
    # plt.close()
    # pass


def tabS2_func():

    bar_dic = {'name': [], 'pre': [], 'post': []}
    for bar in ['guanzhou',  'tuqizhou', 'nanmenzhou', 'tuanzhou', 'huxianzhou', 'mayangzhou', 'shazhou', 'tianxingzhou', 'huojianzhou', 'nanyangzhou', 'wuguizhou',
                'daijiazhou', 'guniuzhou', 'xinzhou', 'shanjiazhou', 'zhangjiazhou', 'baizhou', 'dongcaozhou', 'fuxingzhou', 'xinyuzhou', 'zhongzhou']:
        try:
            print(bar)
            bar_shp = f'E:\\Z_Phd_Other_stuff\\2024_12_11_MCB_area\\study_area_shapefile\\{bar}.shp'
            ff_pre_name = f'G:\\B_papers_patents\RA_Dam operations enhance floodplain vegetation resistance and resilience but compress lateral heterogeneity\\A_fig_nc\\S_Table2\\ff_{bar}.tif'
            ff_post_name = f'G:\\B_papers_patents\RA_Dam operations enhance floodplain vegetation resistance and resilience but compress lateral heterogeneity\\A_fig_nc\\S_Table2\\ff_post_{bar}.tif'
            ff_all_name = f'G:\\B_papers_patents\RA_Dam operations enhance floodplain vegetation resistance and resilience but compress lateral heterogeneity\\A_fig_nc\\S_Table2\\ff_all_{bar}.tif'
            veg_pre_name = f'G:\\B_papers_patents\RA_Dam operations enhance floodplain vegetation resistance and resilience but compress lateral heterogeneity\\A_fig_nc\\S_Table2\\veg_pre_{bar}.tif'
            veg_post_name = f'G:\\B_papers_patents\RA_Dam operations enhance floodplain vegetation resistance and resilience but compress lateral heterogeneity\\A_fig_nc\\S_Table2\\veg_post_{bar}.tif'
            if not os.path.exists(ff_pre_name):
                gdal.Warp(ff_pre_name, 'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Pre_TGD\\inun_DT_inundation_frequency_pretgd.TIF', cutlineDSName=bar_shp, cropToCutline=True,)
            if not os.path.exists(ff_post_name):
                gdal.Warp(ff_post_name, 'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\inun_DT_inundation_frequency_posttgd.TIF', cutlineDSName=bar_shp, cropToCutline=True,)
            if not os.path.exists(ff_all_name):
                gdal.Warp(ff_all_name, 'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\inun_factor\\DT_inundation_frequency.TIF', cutlineDSName=bar_shp, cropToCutline=True,)
            if not os.path.exists(veg_pre_name):
                gdal.Warp(veg_pre_name, 'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11\\veg_pre_tgd.TIF', cutlineDSName=bar_shp, cropToCutline=True,)
            if not os.path.exists(veg_post_name):
                gdal.Warp(veg_post_name, 'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11\\veg_post_tgd.TIF', cutlineDSName=bar_shp, cropToCutline=True,)

            ff_all_ds = gdal.Open(ff_all_name)
            ff_all_arr = ff_all_ds.GetRasterBand(1).ReadAsArray()
            ff_ds = gdal.Open(ff_pre_name)
            ff_arr = ff_ds.GetRasterBand(1).ReadAsArray()
            ff_post_ds = gdal.Open(ff_post_name)
            ff_post_arr = ff_post_ds.GetRasterBand(1).ReadAsArray()
            veg_pre_ds = gdal.Open(veg_pre_name)
            veg_post_ds = gdal.Open(veg_post_name)
            veg_pre_arr = veg_pre_ds.GetRasterBand(1).ReadAsArray()
            veg_post_arr = veg_post_ds.GetRasterBand(1).ReadAsArray()
            veg_pre_arr[veg_pre_arr == 0] = np.nan
            veg_post_arr[veg_post_arr == 0] = np.nan
            ff_all_arr[ff_all_arr == 0] = np.nan
            veg_post_arr[np.logical_and(np.isnan(veg_post_arr), np.isnan(veg_pre_arr))] = -200
            veg_pre_arr[np.logical_and(veg_post_arr == -200, np.isnan(veg_pre_arr))] = -200

            veg_pre_list, veg_post_list, veg_pre_ff, veg_post_ff, veg_pre_mean, veg_post_mean = [], [], [], [], [], []
            # Generate the cumulative curve
            for _ in [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.20, 0.15, 0.1, 0.051]:
                veg_post_arr2 = copy.deepcopy(veg_post_arr)
                veg_post_arr2[np.logical_or(ff_all_arr > _, ff_all_arr < _ - 0.05)] = -200
                veg_post_arr2 = veg_post_arr2.flatten()
                veg_post_arr2 = veg_post_arr2[veg_post_arr2 != -200]
                veg_post_list.extend(veg_post_arr2.tolist())
                veg_post_ff.extend([_ for __ in range(veg_post_arr2.shape[0])])
                veg_post_mean.append(np.nanmean(veg_post_arr2))

            for _ in [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.20, 0.15, 0.1, 0.051]:
                veg_pre_arr2 = copy.deepcopy(veg_pre_arr)
                veg_pre_arr2[np.logical_or(ff_all_arr > _, ff_all_arr < _ - 0.05)] = -200
                veg_pre_arr2 = veg_pre_arr2.flatten()
                veg_pre_arr2 = veg_pre_arr2[veg_pre_arr2 != -200]
                if _ == 0.5:
                    veg_pre_arr2 = veg_pre_arr2 - 0.025
                veg_pre_list.extend(veg_pre_arr2.tolist())
                veg_pre_ff.extend([_ for __ in range(veg_pre_arr2.shape[0])])
                veg_pre_mean.append(np.nanmean(veg_pre_arr2))
            veg_pre_mean.reverse()
            veg_post_mean.reverse()

            df = pd.DataFrame({'ff': veg_pre_ff, 'pre': veg_pre_list, 'post': veg_post_list})
            ff_list = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.20, 0.15, 0.1, 0.051]
            ff_list.reverse()
            wass_pre_list = [wasserstein_distance(df[df['ff'] == __]['pre'].dropna(), df[df['ff'] == 0.051]['pre'].dropna()) for __ in ff_list]
            wass_post_list = [wasserstein_distance(df[df['ff'] == __]['post'].dropna(), df[df['ff'] == 0.051]['post'].dropna()) for __ in ff_list]
            wass_pre_all, wass_post_all = [], []
            for __ in ff_list:
                for ___ in ff_list:
                    if __ != ___:
                        wass_pre_all.append(wasserstein_distance(df[df['ff'] == __]['pre'].dropna(), df[df['ff'] == ___]['pre'].dropna()))
                        wass_post_all.append(wasserstein_distance(df[df['ff'] == __]['post'].dropna(), df[df['ff'] == ___]['post'].dropna()))
            wass_post_all = np.nanmean(wass_post_all)
            wass_pre_all = np.nanmean(wass_pre_all)
            bar_dic['name'].append(bar)
            bar_dic['pre'].append(wass_pre_all)
            bar_dic['post'].append(wass_post_all)
        except:
            print(traceback.format_exc())
            print(f'bar {bar} failed!')
            bar_dic['name'].append(bar)
            bar_dic['pre'].append(np.nan)
            bar_dic['post'].append(np.nan)

    bar_df = pd.DataFrame(bar_dic)
    bar_df.to_csv('G:\B_papers_patents\RA_Dam operations enhance floodplain vegetation resistance and resilience but compress lateral heterogeneity\A_fig_nc\S_Table2\\bar_w.csv')


def figS6_nc_func():
    data = scipy.io.loadmat('G:\\B_papers_patents\\RA_Dam operations enhance floodplain vegetation resistance and resilience but compress lateral heterogeneity\\A_fig_nc\\S_NC_FigS6\\20240709182211r.mat')
    # cs_ = pd.read_csv('G:\\B_papers_patents\\RA_Dam operations enhance floodplain vegetation resistance and resilience but compress lateral heterogeneity\\A_fig_nc\\S_NC_FigS6\\20240709182211r - Copy.csv')
    # dis = 0
    # dis_all = list(cs_['Cell Size (m)'])
    # for _ in range(cs_.shape[0]):
    #     dep = 0
    #     for _
    a = 1


def tabS1_func():

    inun_pre_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig12\\DT_inundation_frequency_pretgd.TIF')
    inun_post_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig12\\DT_inundation_frequency_posttgd.TIF')
    veg_pre_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig12\\veg_pre_tgd.TIF')
    veg_post_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig12\\veg_post_tgd.TIF')
    ele_pre_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig12\\ele_DT_inundation_frequency_pretgd.TIF')
    ele_post_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig12\\ele_DT_inundation_frequency_posttgd.TIF')
    id_diff_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig12\\id_difftgd.TIF')
    ih_diff_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig12\\ih_difftgd.TIF')
    veg_type_ds = gdal.Open('G:\\B_papers_patents\RA_Dam operations enhance floodplain vegetation resistance and resilience but compress lateral heterogeneity\\A_fig_nc\\A_NC_Fig4\\VEG_type.TIF')

    inun_pre_arr = inun_pre_ds.GetRasterBand(1).ReadAsArray()
    inun_post_arr = inun_post_ds.GetRasterBand(1).ReadAsArray()
    veg_pre_arr = veg_pre_ds.GetRasterBand(1).ReadAsArray()
    veg_post_arr = veg_post_ds.GetRasterBand(1).ReadAsArray()
    ele_pre_arr = ele_pre_ds.GetRasterBand(1).ReadAsArray()
    ele_post_arr = ele_post_ds.GetRasterBand(1).ReadAsArray()
    veg_type_arr = veg_type_ds.GetRasterBand(1).ReadAsArray()

    inun_pre_arr[inun_pre_arr > 0.95] = np.nan
    inun_post_arr[inun_post_arr > 0.95] = np.nan
    inun_pre_arr[inun_pre_arr <= 0.05] = np.nan
    inun_post_arr[inun_post_arr <= 0.05] = np.nan
    veg_pre_arr[np.isnan(veg_pre_arr)] = 0
    veg_post_arr[np.isnan(veg_post_arr)] = 0
    veg_pre_arr[np.isnan(veg_pre_arr)] = 0
    veg_post_arr[np.isnan(veg_post_arr)] = 0
    veg_pre_arr[np.logical_and(veg_pre_arr == 0, veg_post_arr == 0)] = np.nan
    veg_post_arr[np.logical_and(np.isnan(veg_pre_arr), veg_post_arr == 0)] = np.nan

    for sec, cord in zip(['yz', 'jj', 'ch', 'hh', 'all'], [[0, 2950], [2950, 6100], [6100, 10210], [10210, 16537], [0, 16537]]):

        print(f'-----------------------{sec}-area------------------------')
        print('All area: ' + str(np.sum(~np.isnan(veg_type_arr[:, cord[0]: cord[1]])) * 0.03 * 0.03))
        print('Transition from non-vegetated to vegetated: ' + str(np.sum(veg_type_arr[:, cord[0]: cord[1]] == 1) * 0.03 * 0.03))
        print('Transition from vegetated to non-vegetated: ' + str(np.sum(veg_type_arr[:, cord[0]: cord[1]] == 5) * 0.03 * 0.03))
        print('vegetation increment: ' + str(np.sum(veg_type_arr[:, cord[0]: cord[1]] == 3) * 0.03 * 0.03))
        print('Pronounced vegetation increment: ' + str(np.sum(veg_type_arr[:, cord[0]: cord[1]] == 2) * 0.03 * 0.03))
        print('vegetation decrement: ' + str(np.sum(veg_type_arr[:, cord[0]: cord[1]] == 4) * 0.03 * 0.03))

        print(f'-----------------------{sec}-ratio------------------------')
        print('Transition from non-vegetated to vegetated: ' + str(np.sum(veg_type_arr[:, cord[0]: cord[1]] == 1) / np.sum(~np.isnan(veg_type_arr))))
        print('Transition from vegetated to non-vegetated: ' + str(np.sum(veg_type_arr[:, cord[0]: cord[1]] == 5) / np.sum(~np.isnan(veg_type_arr))))
        print('vegetation increment: ' + str(np.sum(veg_type_arr[:, cord[0]: cord[1]] == 3) / np.sum(~np.isnan(veg_type_arr))))
        print('Pronounced vegetation increment: ' + str(np.sum(veg_type_arr[:, cord[0]: cord[1]] == 2) / np.sum(~np.isnan(veg_type_arr))))
        print('vegetation decrement: ' + str(np.sum(veg_type_arr[:, cord[0]: cord[1]] == 4) / np.sum(~np.isnan(veg_type_arr))))

        inun_diff = inun_post_arr - inun_pre_arr
        veg_diff = veg_post_arr - veg_pre_arr
        ele_diff = ele_post_arr - ele_pre_arr
        veg_diff_por = veg_post_arr / veg_pre_arr
        veg_diff_por[np.isinf(veg_diff_por)] = np.nan

        print(f'-----------------------{sec}-veg-diff-----------------------')
        print('veg-diff' + str(np.nanmean(veg_diff_por)))
        print('veg-diff Transition from non-vegetated to vegetated: ' + str(np.nanmean(veg_diff_por[veg_pre_arr == 0])))
        print('veg-diff Transition from vegetated to non-vegetated: ' + str(np.nanmean(veg_diff_por[veg_post_arr == 0])))
        print('veg-diff vegetation increment: ' + str(np.nanmean(veg_diff_por[np.logical_and(veg_diff >= 0, veg_diff < 0.15)])))
        print('veg-diff Pronounced vegetation increment: ' + str(np.nanmean(veg_diff_por[veg_diff >= 0.15])))
        print('veg-diff vegetation decrement: ' + str(np.nanmean(veg_diff_por[veg_diff < 0])))

        print(f'-----------------------{sec}-inundation-frequency-----------------------')
        print('veg-diff' + str(np.nanmean(inun_diff)))
        print('inundation frequncy Transition from non-vegetated to vegetated: ' + str(np.nanmean(inun_diff[veg_pre_arr == 0])))
        print('inundation frequncy Transition from vegetated to non-vegetated: ' + str(np.nanmean(inun_diff[veg_post_arr == 0])))
        print('inundation frequncy vegetation increment: ' + str(np.nanmean(inun_diff[np.logical_and(veg_diff >= 0, veg_diff < 0.15)])))
        print('inundation frequncy Pronounced vegetation increment: ' + str(np.nanmean(inun_diff[veg_diff >= 0.15])))
        print('inundation frequncy vegetation decrement: ' + str(np.nanmean(inun_diff[veg_diff < 0])))

        print(f'-----------------------{sec}-elevation-difference-----------------------')
        print('veg-diff' + str(np.nanmean(ele_diff)))
        print('ele diff Transition from non-vegetated to vegetated: ' + str(np.nanmean(ele_diff[veg_pre_arr == 0])))
        print('ele diff Transition from vegetated to non-vegetated: ' + str(np.nanmean(ele_diff[veg_post_arr == 0])))
        print('ele diff vegetation increment: ' + str(np.nanmean(ele_diff[np.logical_and(veg_diff >= 0, veg_diff < 0.15)])))
        print('ele diff Pronounced vegetation increment: ' + str(np.nanmean(ele_diff[veg_diff >= 0.15])))
        print('ele diff vegetation decrement: ' + str(np.nanmean(ele_diff[veg_diff < 0])))


def figS7_nc_func():

    plt.rcParams['font.family'] = ['Arial', 'SimHei']
    plt.rc('font', size=16)
    plt.rc('axes', linewidth=2)

    up_b, low_b = 300, 1800
    thalweg = 635
    new_arr = np.load(f'G:\\B_papers_patents\\RA_Dam operations enhance floodplain vegetation resistance and resilience but compress lateral heterogeneity\\A_fig_nc\\S_NC_FigS7\\arr_{str(thalweg)}_40.npy')
    new_arr = new_arr[300:900, :]
    num = 5

    start_list, start_list2 = [], []
    for _ in range(new_arr.shape[1]):
        for __ in range(new_arr.shape[0]):
            if ~np.isnan(new_arr[__, _]):
                start_list.append(__)
                break
            if __ == new_arr.shape[0] - 1:
                start_list.append(np.nan)

    for _ in range(new_arr.shape[1]):
        ymin, ymax = max(0 , _ - 20), min(new_arr.shape[1], _ + 20)
        start = start_list[_]

        if start < np.nanmean(start_list[ymin: ymax]) * 2 / 3:
            new_arr[0: int(np.nanmean(start_list[ymin: ymax])), _] = np.nan
            start_list2.append(int(np.nanmean(start_list[ymin: ymax])))
        else:
            start_list2.append(start_list[_])

    end_list, end_list2 = [], []
    inverse_list = [_ for _ in range(new_arr.shape[0])]
    inverse_list.reverse()
    for _ in range(new_arr.shape[1]):
        for __ in inverse_list:
            if ~np.isnan(new_arr[__, _]):
                end_list.append(__)
                break
            if __ == 0:
                end_list.append(np.nan)

    for _ in range(new_arr.shape[1]):
        ymin, ymax = max(0 , _ - 20), min(new_arr.shape[1], _ + 20)
        end = end_list[_]

        if end > np.nanmean(end_list[ymin: ymax]) * 2 / 2:
            new_arr[int(np.nanmean(end_list[ymin: ymax])):, _] = np.nan
            end_list2.append(int(np.nanmean(end_list[ymin: ymax])))
        else:
            end_list2.append(end_list[_])

    new_arr[0: 30, :] = np.nan

    new_arr2 = np.zeros_like(new_arr) * np.nan
    for _ in range(new_arr2.shape[1]):
        for pix_ in range(new_arr2.shape[0]):
            if ~np.isnan(new_arr[pix_, _]):
                new_arr2[pix_ - 1, _] = 1
                break

        for pix_ in range(new_arr2.shape[0]):
            if ~np.isnan(new_arr[-1 - pix_, _]):
                new_arr2[-1 - pix_, _] = 1
                break

    new_arr[new_arr == -200] = np.nan
    channel_arr = np.zeros_like(new_arr) * np.nan
    for q in range(len(start_list2)):
        if ~np.isnan(start_list2[q]) and ~np.isnan(end_list2[q]) and start_list2[q] < end_list2[q]:
            channel_arr[start_list2[q]: end_list2[q], q] = 1

    cmap1 = sns.color_palette("coolwarm", as_cmap=True)
    cmap1 = sns.color_palette("viridis", as_cmap=True)
    # cmap1 = sns.diverging_palette(0, 255, sep=1, n=32, center="light", as_cmap=True)
    cmap1 = sns.diverging_palette(240, 10, n=36, as_cmap=True)
    # cmap1 = sns.diverging_palette(120, 60, s=80, l=55, n=9, as_cmap=True)
    fig, ax = plt.subplots(num, figsize=(20, 13), constrained_layout=True)
    dis_start = 0
    for q in range(num):
        ax[q].set_facecolor((0.98, 0.98, 0.98))
        ax[q].plot(np.linspace(0, int(np.floor(new_arr.shape[1]) / num), 100), np.linspace(thalweg-up_b, thalweg-up_b, 100), color=(0.1, 0.2,0.8), ls ='-.', lw=1., zorder=3)
        new_arr_yz = new_arr[:, int(q * np.floor(new_arr.shape[1]) / num): int((q + 1) * np.floor(new_arr.shape[1]) / num)]
        new_arr_yz2 = new_arr2[:, int(q * np.floor(new_arr.shape[1]) / num): int((q + 1) * np.floor(new_arr.shape[1]) / num)]
        cax = ax[q].imshow(new_arr_yz, vmin=-0.15, vmax=0.25, cmap='RdYlGn', zorder=1)
        cax2 = ax[q].imshow(new_arr_yz2, cmap='gist_gray', zorder=2)
        channel_arr_yz = channel_arr[:, int(q * np.floor(new_arr.shape[1]) / num): int((q + 1) * np.floor(new_arr.shape[1]) / num)]
        ax[q].imshow(channel_arr_yz, cmap='PiYG', vmin=0, vmax=1.5, alpha=0.1)
        ax_xtick_list, ax_xlabel_list = [], []
        ax_ytick_list, ax_ylabel_list = [], []
        for _ in range(int(q * np.floor(new_arr.shape[1]) / num),  int((q + 1) * np.floor(new_arr.shape[1]) / num)):
            if np.mod(_, 750) == 0:
                ax_xtick_list.append(_ - int(q * np.floor(new_arr.shape[1]) / num))
                ax_xlabel_list.append(int(_ / 25))

        ax[q].set_xticks(ax_xtick_list)
        ax[q].set_xticklabels([str(_) for _ in ax_xlabel_list])

        for _ in range(new_arr_yz.shape[0]):
            if np.mod(_ - (thalweg-up_b), 100) == 0:
                ax_ytick_list.append(_)
                ax_ylabel_list.append(int((_ - (thalweg-up_b)) / 100))
        ax[q].set_ylim(535, 35)
        ax[q].set_yticks(ax_ytick_list)
        ax[q].set_yticklabels([str(_) for _ in ax_ylabel_list])
        dis_start = int((q + 1) * np.floor(new_arr.shape[1]) / num)
    # plt.colorbar(cax)
    plt.savefig(f'G:\\B_papers_patents\\RA_Dam operations enhance floodplain vegetation resistance and resilience but compress lateral heterogeneity\\A_fig_nc\\S_NC_FigS7\\fig11_nc_635_40_v5.png', dpi=1600)


def figs8_func():

    plt.rcParams['font.family'] = ['Arial', 'SimHei']
    plt.rc('font', size=24)
    plt.rc('axes', linewidth=3)

    pre_TGD_ds = gdal.Open('G:\A_Landsat_Floodplain_veg\Water_level_python\Pre_TGD\ele_DT_inundation_frequency_pretgd.TIF')
    post_TGD_ds = gdal.Open('G:\A_Landsat_Floodplain_veg\Water_level_python\Post_TGD\ele_DT_inundation_frequency_posttgd.TIF')
    pre_TGD_arr = pre_TGD_ds.GetRasterBand(1).ReadAsArray() - 0.58
    post_TGD_arr = post_TGD_ds.GetRasterBand(1).ReadAsArray() -1.55
    # post_TGD_arr[np.isnan(pre_TGD_arr)] = np.nan
    # pre_TGD_arr[np.isnan(post_TGD_arr)] = np.nan
    diff_arr = post_TGD_arr - pre_TGD_arr

    for sec, cord, domain, up in zip(['yz', 'jj', 'ch', 'hh'], [[0, 3950], [950, 6100], [6100, 10210], [10210, 16537]], [[25,55], [20,50], [15,30], [10,25]], [0.2,0.3,0.3,0.3]):
        pre_tgd_data = pre_TGD_arr[:, cord[0]: cord[1]].flatten()
        post_tgd_data = post_TGD_arr[:, cord[0]: cord[1]].flatten()
        diff_data = diff_arr[:, cord[0]: cord[1]].flatten()
        post_tgd_data = np.sort(np.delete(post_tgd_data, np.argwhere(np.isnan(post_tgd_data))))
        pre_tgd_data = np.sort(np.delete(pre_tgd_data, np.argwhere(np.isnan(pre_tgd_data))))
        diff_data = np.sort(np.delete(diff_data, np.argwhere(np.isnan(diff_data))))
        plt.rc('axes', linewidth=3)
        fig = plt.figure(figsize=(7, 5), layout="constrained")

        ax_temp = fig.subplots(1, 1, sharex=True, sharey=True)
        ax_temp.grid(axis='x', color=(210 / 256, 210 / 256, 210 / 256), zorder=5)
        bins = 0.001
        pre_num = []
        pre_cdf = []
        post_num = []
        post_cdf = []
        for _ in range(domain[0] * 1000 + 1, domain[1] * 1000 + 1):
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
        ax_temp.plot(np.linspace(itr_list.index(max(itr_list))/1000, itr_list.index(max(itr_list))/1000, 100),np.linspace(post_list[itr_list.index(max(itr_list))], pre_list[itr_list.index(max(itr_list))], 100), color=(1, 0, 0), zorder=8)
        print(str(post_list[itr_list.index(max(itr_list))]))
        print(pre_list[itr_list.index(max(itr_list))])
        # ax_temp.fill_between(pre_num, np.linspace(0, 0, 10001), post_cdf, color=(1, 1, 1), edgecolor='none', lw=0.0, zorder=4)
        # ax_temp.fill_between(pre_num, pre_cdf, np.linspace(1, 1, 10001), color=(1, 1, 1), edgecolor='none', lw=0.0, zorder=4)
        ax_temp.fill_betweenx(pre_num, np.linspace(0, 0, (domain[1] - domain[0]) * 1000), np.linspace(post_cdf[-2], post_cdf[-2], (domain[1] - domain[0]) * 1000), color=(0.9, 0.9, 0.9), edgecolor=(0.5, 0.5, 0.5), lw=1.0, zorder=5)
        ax_temp.fill_betweenx(pre_num, np.linspace(0, 0, (domain[1] - domain[0]) * 1000), post_cdf, color=(0.9, 0.9, 0.9), alpha=0.1)
        ax_temp.fill_betweenx(pre_num, post_cdf, pre_cdf, hatch='-', color=(1, 1, 1), edgecolor=(0, 0, 0), lw=2.5,zorder=6)
        ax_temp.plot(pre_cdf, pre_num, lw=3, color=(44/256, 86/256, 200/256), label="Pre-TGD",zorder=7, )
        ax_temp.plot(post_cdf, post_num, lw=3, color=(200/256, 13/256, 18/256), label="Post-TGD",zorder=7)
        ax_temp.set_ylabel('Elevation/m', fontname='Arial', fontsize=25, fontweight='bold', )
        ax_temp.set_xlabel('Exceedance probability', fontname='Arial', fontsize=25, fontweight='bold')
        ax_temp.legend(fontsize=20)
        ax_temp.set_xlim(-0.05, 1.05)
        ax_temp.set_ylim(domain[0], domain[1])
        ax_temp.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax_temp.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontname='Arial', fontsize=24)
        plt.savefig(f'G:\B_papers_patents\RA_Dam operations enhance floodplain vegetation resistance and resilience but compress lateral heterogeneity\A_fig_nc\S_NC_FigS8\\{sec}_ele.png', dpi=500)
        plt.close()

        fig = plt.figure(figsize=(7, 5), layout="constrained")
        ax_temp = fig.subplots(1, 1, sharex=True, sharey=True)
        ax_temp.grid(axis='x', color=(210 / 256, 210 / 256, 210 / 256), zorder=5)
        ax_temp.plot(np.linspace(0,0.6,100), np.linspace(0,0,100), c=(0,0,0), lw=2, zorder=1)
        print(f'mean{str(np.nanmean(diff_data))}')
        print(f'median{str(np.nanmedian(diff_data))}')
        print(f'rmse{str(np.sqrt(np.nanmean((diff_data - np.nanmean(diff_data)) ** 2)))}')
        sns.histplot(y=diff_data, stat='density', color=(0.12, 0.25, 1), fill=False, kde=True, zorder=3)
        ax_temp.set_ylim(-10, 10)
        ax_temp.set_xlim(0, up)
        ax_temp.set_ylabel('Elevation difference/m', fontname='Arial', fontsize=25, fontweight='bold', )
        ax_temp.set_xlabel('Density', fontname='Arial', fontsize=25, fontweight='bold')
        plt.savefig(f'G:\B_papers_patents\RA_Dam operations enhance floodplain vegetation resistance and resilience but compress lateral heterogeneity\A_fig_nc\S_NC_FigS8\\{sec}_ele_diff.png', dpi=500)
        plt.close()

figS3nc_func()
# tabS2_func()
# # tabS1_func()
# figs8_func()