import als_preprocess as als
import als_imbl as alsi

import pandas as pd
import constants
import sys
from pathlib import Path
import numpy as np


def load_data_baselines(features):
    train_infile = constants.DATA_FILE_train
    test_infile  = constants.DATA_FILE_test
    test_data = pd.read_csv(test_infile, low_memory=False)
    train_data = pd.read_csv(train_infile, low_memory=False)

    train_data.replace(" ", np.nan, inplace=True)
    test_data.replace(" ", np.nan, inplace=True)
    if 'Evolution' not in train_data.columns:
        train_data['Evolution'] = 'No'
        test_data['Evolution'] = 'No'
    train_data = train_data[features]

    print("Train y INI : ", len(train_data))

    #test_data['Evolution'] = 'No'
    test_data = test_data[features]

    print("Test y INI : ", len(test_data))
    return train_data, test_data


def compute_temporal_table(data, n, features,baseline_temporal):

    data = data[features]
    data_dict = als.df_to_dict(data)

    sps = als.compute_consecutive_snapshots_n(
        data_dict, n, 'Evolution')

    mats, y = als.create_matrix_temporal(data_dict, sps, n)

    mats.fillna(0, inplace=True)

    #baseline_temporal = constants.BASELINE_DIR_T

    Path(baseline_temporal).mkdir(parents=True, exist_ok=True)
    mats['Evolution'] = y
    mats = mats.groupby('Patient_ID').first().reset_index()
    mats.to_csv(baseline_temporal +
                "{}TPS_baseline_temporal.csv".format(n), index=False)
    
def compute_static_table(train_data,test_data, n, features, train_baseline_static, test_baseline_static):
    train_data = train_data[features].copy()
    train_data.dropna(inplace = True, ignore_index = True)
    data_dict = als.df_to_dict(train_data)

    sps = als.compute_consecutive_snapshots_n(
        data_dict, n, 'Evolution')

    train_mats, y_train = als.create_matrix_static(data_dict, sps)
    train_len = len(train_mats)
    test_data = test_data[features].copy()
    test_data.dropna(inplace = True, ignore_index = True)

    data_dict = als.df_to_dict(test_data)

    sps = als.compute_consecutive_snapshots_n(
        data_dict, n, 'Evolution')

    test_mats, y_test = als.create_matrix_static(data_dict, sps)
    mats = pd.concat([train_mats, test_mats], ignore_index=True)
    #y = y_train + y_test
    #if features != ['REF', 'Evolution']:
    encoded_features = []
    for name in features:
        if name in ['Gender', 'UMNvsLMN', 'C9orf72', 'Onset']:
            encoded_features.append(name)
    mats = alsi.label_encoder_als(mats, encoded_features)

    mats.fillna(0, inplace=True)
    
    train_mats = mats.iloc[:train_len, :].copy()
    #print(len(train_mats))
    #print(len(y_train))
    train_mats['Evolution'] = y_train
    
    test_mats = mats.iloc[train_len:, :].copy()
    #print(len(test_mats))
    #print(len(y_test))
    test_mats['Evolution'] = y_test
    #print(len(test_mats))
    #baseline_static = constants.BASELINE_DIR_S

    Path(train_baseline_static).mkdir(parents=True, exist_ok=True)
    Path(test_baseline_static).mkdir(parents=True, exist_ok=True)
    
    train_mats = train_mats.groupby('Patient_ID').first().reset_index()
    test_mats = test_mats.groupby('Patient_ID').first().reset_index()
    train_mats.to_csv(train_baseline_static+"{}TPS_baseline_static.csv".format(n), index=False)
    test_mats.to_csv(test_baseline_static+"{}TPS_baseline_static.csv".format(n), index=False)


n = int(sys.argv[1])
constants.get_config(sys.argv[2])
#if constants.MODEL== 'simple':
features = [constants.REF_FEATURE] + list(constants.TEMPORAL_FEATURES.keys()) + list(constants.STATIC_FEATURES.keys())+ ['Evolution']
#elif constants.MODEL== 'temp_static':
#features = [constants.REF_FEATURE] + list(constants.TEMPORAL_FEATURES.keys()) + list(constants.STATIC_FEATURES.keys()) + ['Evolution']


train_data, test_data = load_data_baselines(features)

t = compute_temporal_table(
    train_data, n, [constants.REF_FEATURE] + list(constants.TEMPORAL_FEATURES.keys()) + ['Evolution'], constants.BASELINE_DIR_T_train)

t = compute_temporal_table(
    test_data, n, [constants.REF_FEATURE] + list(constants.TEMPORAL_FEATURES.keys()) + ['Evolution'], constants.BASELINE_DIR_T_test)

#s = compute_static_table(train_data, n, [constants.REF_FEATURE] +
                        #list(constants.STATIC_FEATURES.keys()) + ['Evolution'], constants.BASELINE_DIR_S_train)
s = compute_static_table(train_data, test_data, n, [constants.REF_FEATURE] +
                        list(constants.STATIC_FEATURES.keys()) + ['Evolution'], constants.BASELINE_DIR_S_train, constants.BASELINE_DIR_S_test)
