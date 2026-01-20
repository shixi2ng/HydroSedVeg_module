import os.path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import numpy as np
import pandas as pd
import basic_function as bf
from GEDI_toolbox.GEDI_main import GEDI_df
import copy
import xgboost as xgb
import cupy as cp

import time
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import itertools
from scipy.stats import pearsonr, kurtosis, variation, cramervonmises_2samp, wasserstein_distance

shap.initjs()


def Ensemble_bagging_contribution(df: pd.DataFrame, x_feature, y_feature, max_d=None, model_name=None, normalised_feature=True):

    if max_d is None:
        max_d = [5]
    n_est_ = 1000000
    child_weight = [15]
    leaf_indicator = [2]
    gamma_list = [0]
    subsample_list = [0.79]
    lambda_list = [10]
    alpha_list = [4]

    if model_name is None:
        model_name = 'XGboost'
    elif model_name in ['XGboost', 'RFR', 'Ridge']:
        model_name = model_name
    else:
        raise ValueError('Model is not supported!')

    df = df.dropna(ignore_index=True)
    x_ = np.array(df[x_feature])
    y_ = np.array(df[y_feature])
    x_feature2 = x_feature + ['Reach', 'Veg_type']
    x_2 = np.array(df[x_feature2])
    y_2 = np.array(df[y_feature])
    if normalised_feature:
        for _ in range(x_.shape[1]):
            q = np.sort(np.abs(x_[:, _].flatten()))[::-1][0]
            x_[:, _] = x_[:, _] / q
        for _ in range(y_.shape[1]):
            q = np.sort(np.abs(y_[:, _].flatten()))[::-1][0]
            y_[:, _] = y_[:, _] / q
        for _ in range(x_.shape[1]):
            q = np.sort(np.abs(x_2[:, _].flatten()))[::-1][0]
            x_2[:, _] = x_2[:, _] / q
        for _ in range(y_.shape[1]):
            q = np.sort(np.abs(y_2[:, _].flatten()))[::-1][0]
            y_2[:, _] = y_2[:, _] / q
    x_train_, x_test_, y_train_, y_test_ = train_test_split(x_, y_, test_size=0.2, random_state=42)
    x_train2_, x_test2_, y_train2_, y_test2_ = train_test_split(x_2, y_2, test_size=0.2, random_state=42)
    if model_name == 'XGboost':
        x_train_ = cp.array(x_train_)
        y_train_ = cp.array(y_train_)
        x_test_ = cp.array(x_test_)
        y_test_ = cp.array(y_test_)

    elif model_name == 'Ridge' or model_name == 'RFR':
        x_train_ = np.array(x_)
        y_train_ = np.array(y_)
        x_test_ = np.array(x_test_)
        y_test_ = np.array(y_test_)

    else:
        return None, None

    importance_all, indices_all, score_all = None, None, None

    if model_name == 'Ridge':
        model = Ridge(alpha=0.5)
        model.fit(x_train_, y_train_)
        coefficients = model.coef_
        print("Coefficients of features:\n")
        for feature, coef in zip(x_feature, coefficients.flatten().tolist()):
            print(f"{feature}: {str(coef)}")
        explainer = shap.LinearExplainer(model, x_train_)
        shap_values = explainer.shap_values(x_train_)
        shap.summary_plot(shap_values, x_train_, feature_names=x_feature)
        shap_values_explained = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=x_train_, feature_names=x_feature)
        clustering = shap.utils.hclust(x_train_, y_train_)
        shap.plots.bar(shap_values_explained, clustering=clustering, clustering_cutoff=0.5)

        mdi_importances = pd.Series(coefficients.flatten().tolist(), index=x_feature)
        tree_importance_sorted_idx = np.argsort(coefficients.flatten())

        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        mdi_importances.sort_values().plot.barh(ax=ax1)
        ax1.set_xlabel("Gini importance")
        fig.suptitle(
            "Impurity-based vs. permutation importances on multicollinear features (train set)"
        )
        _ = fig.tight_layout()

    else:
        res_dic = {f'Train_{str(y_feature[0])}_MAE': [], f'Train_{str(y_feature[0])}_MSE': [],
                   f'Train_{str(y_feature[0])}_RMSE': [], f'Test_{str(y_feature[0])}_MAE': [],
                   f'Test_{str(y_feature[0])}_MSE': [], f'Test_{str(y_feature[0])}_RMSE': [],
                   f'Train_score': [], f'Test_score': [], 'Optimal_Est': [], 'Max_depth': [], 'Leaf_node': [], 'min_child': [], 'gamma': [], 'subsample': [], 'lambda': [], 'alpha': [], 'feature_importance': []}

        for max_d_, min_child_, leaf_indi, gamma_, subs_, lambda_, alpha_ in itertools.product(max_d, child_weight, leaf_indicator, gamma_list, subsample_list, lambda_list, alpha_list):
            res_dic['Max_depth'].append(max_d_)
            res_dic['min_child'].append(min_child_)
            res_dic['Leaf_node'].append(0)
            res_dic['gamma'].append(gamma_)
            res_dic['subsample'].append(subs_)
            res_dic['lambda'].append(lambda_)
            res_dic['alpha'].append(alpha_)

            start_time = time.time()
            if model_name == 'RFR':
                model = RandomForestRegressor(max_depth=max_d_,
                                              n_estimators=n_est_,
                                              max_features=1,
                                              min_samples_leaf=4,
                                              n_jobs=-1).fit(x_train_, y_train_)
            elif model_name == 'XGboost':
                model = xgb.XGBRegressor(max_depth=max_d_,
                                         n_estimators=n_est_,
                                         max_leaves=0,
                                         learning_rate=0.3,
                                         reg_lambda=lambda_,
                                         reg_alpha=alpha_,
                                         colsample_bylevel=1,
                                         min_child_weight=min_child_,
                                         gamma=gamma_,
                                         subsample=subs_,
                                         colsample_bytree=subs_,
                                         eval_metric=metrics.mean_squared_error,
                                         early_stopping_rounds=50,
                                         device='cuda', ).fit(x_train_, y_train_, eval_set=[(x_train_, y_train_), (x_test_, y_test_)])

            print(f'End time 4 {model_name}: {str(time.time()-start_time)}')
            res_dic['Optimal_Est'].append(model.get_booster().best_iteration)
            y_pred = model.predict(x_train_)
            y_pred2 = model.predict(x_test_)
            score = model.score(x_train_.get(), y_train_.get())
            score2 = model.score(x_test_.get(), y_test_.get())
            print('----------------------------------------------------------------------------------------------')
            print(f'X_feature:{str(x_feature)}, Y_feature:{str(y_feature)}, Max depth:{str(max_d_)}, N_est:{str(n_est_)}')
            print('随机森林模型Train得分: ', score)
            res_dic[f'Train_score'].append(score)
            print('Train MAE:', metrics.mean_absolute_error(y_train_.get(), y_pred))
            res_dic[f'Train_{str(y_feature[0])}_MAE'].append(metrics.mean_absolute_error(y_train_.get(), y_pred))
            print('Train MSE:', metrics.mean_squared_error(y_train_.get(), y_pred))
            res_dic[f'Train_{str(y_feature[0])}_MSE'].append(metrics.mean_squared_error(y_train_.get(), y_pred))
            print('Train RMSE:', np.sqrt(metrics.mean_squared_error(y_train_.get(), y_pred)))
            res_dic[f'Train_{str(y_feature[0])}_RMSE'].append(np.sqrt(metrics.mean_squared_error(y_train_.get(), y_pred)))

            print('随机森林模型Test得分: ', score2)
            res_dic[f'Test_score'].append(score2)
            print('Test MAE:', metrics.mean_absolute_error(y_test_.get(), y_pred2))
            res_dic[f'Test_{str(y_feature[0])}_MAE'].append(metrics.mean_absolute_error(y_test_.get(), y_pred2))
            print('Test MSE:', metrics.mean_squared_error(y_test_.get(), y_pred2))
            res_dic[f'Test_{str(y_feature[0])}_MSE'].append(metrics.mean_squared_error(y_test_.get(), y_pred2))
            print('Test RMSE:', np.sqrt(metrics.mean_squared_error(y_test_.get(), y_pred2)))
            res_dic[f'Test_{str(y_feature[0])}_RMSE'].append(np.sqrt(metrics.mean_squared_error(y_test_.get(), y_pred2)))
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            rank = [x_feature[f] for f in indices]
            res_dic['feature_importance'].append(rank)# [::-1]表示将各指标按权重大小进行排序输出
            if score_all is None:
                score_all = score
                indices_all = indices
                importance_all = importances
            elif score > score_all:
                score_all = score
                indices_all = indices
                importance_all = importances
            for f in range(x_train_.get().shape[1]):
                print("%2d) %-*s %f" % (f + 1, 30, x_feature[indices[f]], importances[indices[f]]))
            print('----------------------------------------------------------------------------------------------')

            # score_list = res_dic[f'Test_score']
            # explainer = shap.Explainer(model, feature_names=x_feature)
            # shap_values = explainer(x_train_)
            # shap.plots.bar(shap_values)

        df_ = pd.DataFrame(res_dic)
        df_.to_csv('G:\A_Landsat_Floodplain_veg\Paper\A_fig_nc\A_NC_Fig4\RF\\v1\\result.csv')
        itr = model.get_booster().best_iteration

        attribute_dic = {'All_feature_rank': [], 'All_feature_importance': [], 'Pearson_r': []}
        x_train_ = cp.array(x_)
        y_train_ = cp.array(y_)
        model = xgb.XGBRegressor(max_depth=max_d_,
                                 n_estimators=itr,
                                 max_leaves=0,
                                 learning_rate=0.3,
                                 reg_lambda=lambda_,
                                 reg_alpha=alpha_,
                                 colsample_bylevel=1,
                                 min_child_weight=min_child_,
                                 gamma=gamma_,
                                 subsample=subs_,
                                 colsample_bytree=subs_,
                                 device='cuda', ).fit(x_train_, y_train_)
        for f in range(x_train_.get().shape[1]):
            print("%2d) %-*s %f" % (f + 1, 30, x_feature[indices[f]], importances[indices[f]]))
            attribute_dic['All_feature_rank'].append(x_feature[indices[f]])
            attribute_dic['All_feature_importance'].append(importances[indices[f]])
            attribute_dic['Pearson_r'].append(pearsonr(x_[:, indices[f]].flatten(), y_.flatten())[0])

        for value in [1.0, 2.0, 3.0, 4.0, 5.0]:
            attribute_dic[f'Type_{str(value)}_feature_rank'] = []
            attribute_dic[f'Type_{str(value)}_feature_importance'] = []
            attribute_dic[f'Type_{str(value)}_Pearson_r'] = []
            x_train2_type = x_train_[np.where(x_train2_[:, 11] == value)[0]]
            y_train2_type = y_train_[np.where(x_train2_[:, 11] == value)[0]]
            model = xgb.XGBRegressor(max_depth=max_d_,
                                     n_estimators=itr,
                                     max_leaves=0,
                                     learning_rate=0.3,
                                     reg_lambda=lambda_,
                                     reg_alpha=alpha_,
                                     colsample_bylevel=1,
                                     min_child_weight=min_child_,
                                     gamma=gamma_,
                                     subsample=subs_,
                                     colsample_bytree=subs_,
                                     device='cuda', ).fit(x_train2_type, y_train2_type)
            print('----------------------Type()----------------------------------------')
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            for f in range(x_train_.get().shape[1]):
                print("%2d) %-*s %f" % (f + 1, 30, x_feature[indices[f]], importances[indices[f]]))
                attribute_dic[f'Type_{str(value)}_feature_rank'].append(x_feature[indices[f]])
                attribute_dic[f'Type_{str(value)}_feature_importance'].append(importances[indices[f]])
                attribute_dic[f'Type_{str(value)}_Pearson_r'].append(pearsonr(x_train2_type[:, indices[f]].flatten().get(), y_train2_type.flatten().get())[0])

        for value in [1.0, 2.0, 3.0, 4.0]:
            attribute_dic[f'Reach_{str(value)}_feature_rank'] = []
            attribute_dic[f'Reach_{str(value)}_feature_importance'] = []
            attribute_dic[f'Reach_{str(value)}_Pearson_r'] = []
            x_train2_type = x_train_[np.where(x_train2_[:, 10] == value)[0]]
            y_train2_type = y_train_[np.where(x_train2_[:, 10] == value)[0]]
            model = xgb.XGBRegressor(max_depth=max_d_,
                                     n_estimators=itr,
                                     max_leaves=0,
                                     learning_rate=0.3,
                                     reg_lambda=lambda_,
                                     reg_alpha=alpha_,
                                     colsample_bylevel=1,
                                     min_child_weight=min_child_,
                                     gamma=gamma_,
                                     subsample=subs_,
                                     colsample_bytree=subs_,
                                     device='cuda', ).fit(x_train2_type, y_train2_type)
            print('----------------------Type()----------------------------------------')
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            for f in range(x_train_.get().shape[1]):
                print("%2d) %-*s %f" % (f + 1, 30, x_feature[indices[f]], importances[indices[f]]))
                attribute_dic[f'Reach_{str(value)}_feature_rank'].append(x_feature[indices[f]])
                attribute_dic[f'Reach_{str(value)}_feature_importance'].append(importances[indices[f]])
                attribute_dic[f'Reach_{str(value)}_Pearson_r'].append(pearsonr(x_train2_type[:, indices[f]].flatten().get(), y_train2_type.flatten().get())[0])

        df_ = pd.DataFrame(attribute_dic)
        df_.to_csv('G:\A_Landsat_Floodplain_veg\Paper\A_fig_nc\A_NC_Fig4\RF\\v1\\importance.csv')

        explainer = shap.Explainer(model, feature_names=x_feature)
        x_train2_type1 = x_train_[np.where(x_train2_[:, 10] == 1.0)[0]]
        shap_values = explainer(x_train2_type1)
        shap.plots.bar(shap_values)

    return indices_all, importance_all


if __name__ == '__main__':
    cur_dic = {}
    cur_dic[f'test3_rmse'] = []
    cur_dic[f'test1_rmse'] = []
    cur_dic[f'test2_rmse'] = []
    cur_dic[f'test_rmse'] = []
    cur_dic[f'train3_rmse'] = []
    cur_dic[f'train1_rmse'] = []
    cur_dic[f'train2_rmse'] = []
    cur_dic[f'train_rmse'] = []
    for max_d in range(5, 50):
        _ = bf.file_filter('G:\A_veg\S2_all\XGB_vhm\GEDI_ds_V3\RFR\\', ['_nes500_', f'_maxd{str(max_d)}_'], and_or_factor='and')
        if len(_) == 0:
            cur_dic[f'test3_rmse'].append(np.nan)
            cur_dic[f'test2_rmse'].append(np.nan)
            cur_dic[f'test1_rmse'].append(np.nan)
            cur_dic[f'test_rmse'].append(np.nan)
            cur_dic[f'train3_rmse'].append(np.nan)
            cur_dic[f'train2_rmse'].append(np.nan)
            cur_dic[f'train1_rmse'].append(np.nan)
            cur_dic[f'train_rmse'].append(np.nan)
        else:
            pd_temp = pd.read_csv(_[0])
            cur_dic[f'test3_rmse'].append(pd_temp.loc[4, 'Test_index_at_RMSE'])
            cur_dic[f'test2_rmse'].append(pd_temp.loc[4, 'Test_index_RMSE'])
            cur_dic[f'test1_rmse'].append(pd_temp.loc[4, 'Test_index_phe_RMSE'])
            cur_dic[f'test_rmse'].append(pd_temp.loc[4, 'Test_index_phe_at_RMSE'])
            cur_dic[f'train3_rmse'].append(pd_temp.loc[4, 'Train_index_at_RMSE'])
            cur_dic[f'train2_rmse'].append(pd_temp.loc[4, 'Train_index_RMSE'])
            cur_dic[f'train1_rmse'].append(pd_temp.loc[4, 'Train_index_phe_RMSE'])
            cur_dic[f'train_rmse'].append(pd_temp.loc[4, 'Train_index_phe_at_RMSE'])

    fig4, ax4 = plt.subplots(figsize=(12, 8), constrained_layout=True)
    ax4.plot(np.linspace(5, 49, 45), cur_dic[f'test3_rmse'], label='Test_index_at_RMSE')
    ax4.plot(np.linspace(5, 49, 45), cur_dic[f'test2_rmse'], label='Test_index_RMSE')
    ax4.plot(np.linspace(5, 49, 45), cur_dic[f'test1_rmse'], label='Test_index_phe_RMSE')
    ax4.plot(np.linspace(5, 49, 45), cur_dic[f'test_rmse'], label='Test_index_phe_at_RMSE')
    ax4.plot(np.linspace(5, 49, 45), cur_dic[f'train_rmse'], label='Train_index_RMSE')
    ax4.plot(np.linspace(5, 49, 45), cur_dic[f'train1_rmse'], label='Train_index_phe_RMSE')
    ax4.plot(np.linspace(5, 49, 45), cur_dic[f'train2_rmse'], label='Train_index_phe_at_RMSE')
    ax4.plot(np.linspace(5, 49, 45), cur_dic[f'train2_rmse'], label='Train_index_at_RMSE')
    plt.legend()
    plt.show()


    for n_est in range(500, 1000):
        for max_d in range(20, 200):
            gedi_ds = GEDI_df('G:\\A_veg\\S2_all\\XGB_vhm\\GEDI_ds_V3\\floodplain_2020_high_quality_merged.csv')
            gedi_df = gedi_ds.GEDI_df
            gedi_df = gedi_df.dropna()
            index_list = ['S2_OSAVI_20m_noninun_linear_interpolation',
                          'S2_NDVI_20m_noninun_linear_interpolation', 'Tandem-X DEM',
                          'S2_B8A_noninun_linear_interpolation',  'S2_B9_noninun_linear_interpolation', 'S2_B8_noninun_linear_interpolation',
                          'S2_B7_noninun_linear_interpolation',  'S2_B6_noninun_linear_interpolation',
                          'S2_B5_noninun_linear_interpolation', ''
                          'S2_B4_noninun_linear_interpolation', 'S2_B3_noninun_linear_interpolation', 'S2_B2_noninun_linear_interpolation',]
            # phe_list = ['S2phemetric_DR', 'S2phemetric_DR2', 'S2phemetric_EOS', 'S2phemetric_GR', 'S2phemetric_peak_doy', 'S2phemetric_peak_vi', 'S2phemetric_SOS', 'S2phemetric_trough_vi']
            phe_list = ['S2phemetric_EOS', 'S2phemetric_peak_doy', 'S2phemetric_peak_vi', 'S2phemetric_SOS', 'S2phemetric_trough_vi', 'S2phemetric_static_TEMP', 'S2phemetric_static_DPAR']
            at_list = ['S2_accumulated_TEMP_relative', 'S2_accumulated_DPAR_relative', 'S2phemetric_peak_TEMP', 'S2phemetric_peak_DPAR', ]
            gedi_df = gedi_df.loc[gedi_df['Canopy Height (rh100)'] < 5]

            res_dic = {}
            for model in range(6):
                if model == 0:
                    temp = index_list
                    name = 'index'
                elif model == 1:
                    temp = phe_list
                    name = 'phe'
                elif model == 2:
                    temp = at_list
                    name = 'at'
                elif model == 3:
                    temp = copy.copy(index_list)
                    temp.extend(phe_list)
                    name = 'index_phe'
                elif model == 4:
                    temp = copy.copy(index_list)
                    temp.extend(at_list)
                    name = 'index_at'
                elif model == 5:
                    temp = copy.copy(index_list)
                    temp.extend(phe_list)
                    temp.extend(at_list)
                    name = 'index_phe_at'

                res_dic[f'Test_{name}_MAE'], res_dic[f'Test_{name}_MSE'], res_dic[f'Test_{name}_RMSE'] = [], [], []
                res_dic[f'Train_{name}_MAE'], res_dic[f'Train_{name}_MSE'], res_dic[f'Train_{name}_RMSE'] = [], [], []

                for year in [2019, 2020, 2021, 2022]:
                    x_train = gedi_df.loc[gedi_df['Date'] // 1000 != year][temp]
                    y_train = gedi_df.loc[gedi_df['Date'] // 1000 != year]['Canopy Height (rh100)']
                    x_test = gedi_df.loc[gedi_df['Date'] // 1000 == year][temp]
                    y_test = gedi_df.loc[gedi_df['Date'] // 1000 == year]['Canopy Height (rh100)']
                    feat_labels = x_train.columns[0:]

                    RFR = RandomForestRegressor(max_depth=max_d,
                                                n_estimators=n_est,
                                                max_features=1,
                                                min_samples_leaf=4,
                                                n_jobs=-1)
                    RFR.fit(x_train, y_train)
                    y_pred = RFR.predict(x_train)
                    y_pred2 = RFR.predict(x_test)
                    score = RFR.score(x_train, y_train)
                    print('----------------------------------------------------------------------------------------------')
                    print(f'Year:{str(year)}, Index:{str(name)}, Max depth:{str(max_d)}, N_est:{str(n_est)}')
                    print('随机森林模型得分： ', score)
                    print('Train MAE:', metrics.mean_absolute_error(y_train, y_pred))
                    res_dic[f'Train_{name}_MAE'].append(metrics.mean_absolute_error(y_train, y_pred))
                    print('Train MSE:', metrics.mean_squared_error(y_train, y_pred))
                    res_dic[f'Train_{name}_MSE'].append(metrics.mean_squared_error(y_train, y_pred))
                    print('Train RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
                    res_dic[f'Train_{name}_RMSE'].append(np.sqrt(metrics.mean_squared_error(y_train, y_pred)))

                    print('Test MAE:', metrics.mean_absolute_error(y_test, y_pred2))
                    res_dic[f'Test_{name}_MAE'].append(metrics.mean_absolute_error(y_test, y_pred2))
                    print('Test MSE:', metrics.mean_squared_error(y_test, y_pred2))
                    res_dic[f'Test_{name}_MSE'].append(metrics.mean_squared_error(y_test, y_pred2))
                    print('Test RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
                    res_dic[f'Test_{name}_RMSE'].append(np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
                    importances = RFR.feature_importances_
                    indices = np.argsort(importances)[::-1]  # [::-1]表示将各指标按权重大小进行排序输出
                    for f in range(x_train.shape[1]):
                        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

                    print('----------------------------------------------------------------------------------------------')

                res_dic[f'Train_{name}_MAE'].append(np.mean(np.array(res_dic[f'Train_{name}_MAE'])))
                res_dic[f'Train_{name}_MSE'].append(np.mean(np.array(res_dic[f'Train_{name}_MSE'])))
                res_dic[f'Train_{name}_RMSE'].append(np.mean(np.array(res_dic[f'Train_{name}_RMSE'])))
                res_dic[f'Test_{name}_MAE'].append(np.mean(np.array(res_dic[f'Test_{name}_MAE'])))
                res_dic[f'Test_{name}_MSE'].append(np.mean(np.array(res_dic[f'Test_{name}_MSE'])))
                res_dic[f'Test_{name}_RMSE'].append(np.mean(np.array(res_dic[f'Test_{name}_RMSE'])))

            res_df = pd.DataFrame(res_dic)
            for _ in range(10000):
                if not os.path.exists(f'G:\\A_veg\\S2_all\\GEDI_v3\\RFR\\RF_res_maxd{str(max_d)}_nes{str(n_est)}_v{str(_)}.csv'):
                    res_df.to_csv(f'G:\\A_veg\\S2_all\\GEDI_v3\\RFR\\RF_res_maxd{str(max_d)}_nes{str(n_est)}_v{str(_)}.csv')
                    break

    cur_dic = {}
    for minleaf in range(2, 5):
        cur_dic[f'train_rmse_{str(minleaf)}'] = []
        cur_dic[f'test_rmse_{str(minleaf)}'] = []
        for max_d in range(10, 501):
            _ = bf.file_filter('G:\\A_veg\\S2_all\\GEDI_v3\\', [f'ml{str(minleaf)}', '_nes500_', f'_maxd{str(max_d)}_'], and_or_factor='and')
            if len(_) == 0:
                cur_dic[f'train_rmse_{str(minleaf)}'].append(np.nan)
                cur_dic[f'test_rmse_{str(minleaf)}'].append(np.nan)
            else:
                pd = pd.read_csv(_[0])
                cur_dic[f'train_rmse_{str(minleaf)}'].append(pd.iloc[4, 'Test_index_RMSE'])
                cur_dic[f'train_rmse_{str(minleaf)}'].append(pd.iloc[4, 'Test_index_phe_at_RMSE'])

    fig4, ax4 = plt.subplots(figsize=(12, 8), constrained_layout=True)
    ax4.plot(np.linspace(10, 501, 492), cur_dic[f'train_rmse_2'])
    ax4.plot(np.linspace(10, 501, 492), cur_dic[f'train_rmse_3'])
    ax4.plot(np.linspace(10, 501, 492), cur_dic[f'train_rmse_4'])
    ax4.plot(np.linspace(10, 501, 492), cur_dic[f'test_rmse_2'])
    ax4.plot(np.linspace(10, 501, 492), cur_dic[f'test_rmse_3'])
    ax4.plot(np.linspace(10, 501, 492), cur_dic[f'test_rmse_4'])

    plt.show()

