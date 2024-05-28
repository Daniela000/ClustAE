import preprocessing.constants as constants
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import xgboost as xg
from sklearn.linear_model import LinearRegression
import torch
import copy
from get_autoencoder_reps import finetune_model, transform_data, get_encoded_reps
import sys
import matplotlib.pyplot as plt

def string_to_list(string):
    # Remove the brackets
    string = string[1:-1]
    # Split the string by spaces and convert each element to an integer
    my_list = [float(x) for x in string.split()]
    return my_list

def regressors(X_train, y_train, X_test, y_test,outputfile):
    models = [RandomForestRegressor(n_estimators=100, max_depth=7500, random_state=0), SVR(), DecisionTreeRegressor(random_state = 0), xg.XGBRegressor(eval_metric='logloss', random_state=42, use_label_encoder=False), LinearRegression()]
    best = models[0]
    best_mse = 100
    for regressor in models:
        print("Regressor: " + type(regressor).__name__)

        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        print('R squared', regressor.score(X_test, y_test))

        if metrics.mean_squared_error(y_test, y_pred) < best_mse:
            best_mse = metrics.mean_squared_error(y_test, y_pred) 
            best = regressor
    print('BEST CLF: ', type(best).__name__)
    best.fit(X_train, y_train)
    y_pred = best.predict(X_test)
    print('***************** BEST REG Mean Squared Error: ', metrics.mean_squared_error(y_test, y_pred))
    x = np.arange(0, len(y_test))
    plt.plot(x, y_pred, label = "Predicted")
    plt.plot(x, y_test, label = "Y")
    plt.legend()
    #plt.show()
    plt.savefig(constants.TRAJECTORY_DIR_train + outputfile)
    plt.clf()


def specialized_prediction(model, train_group, y_train, test_group, y_test,i):
    model = finetune_model(model, train_group, test_group)
    #predict time to NIV    
    model = model.eval()
    X_train = get_encoded_reps(model,train_group, [], 'simple')
    X_test = get_encoded_reps(model,test_group, [], 'simple')
    regressors(X_train, y_train, X_test, y_test, 'reg_spec_prediction{}.pdf'.format(i))
    #regressors(X_train, y_train, X_test, y_test, 'spec_precition{}.csv'.format(i))

if __name__ == "__main__":
    constants.get_config(sys.argv[1])
    n = constants.MIN_APP
    type_model = constants.MODEL

    model = torch.load(type_model + '_turim.pt')
    model =model.eval()

    print('*** Time to NIV prediciton ***')

    df_train = pd.read_csv(constants.LABELS_DIR_train + '/niv_labels.csv')
    df_train.replace(" ", np.nan, inplace=True)
    df_train = df_train.dropna(subset=['niv_years_precise'])
    df_train['niv_years_precise'] = df_train['niv_years_precise'].astype('float')
    df_train = df_train.loc[df_train['niv_years_precise'] >= 0]
    #df_train.drop(df_train.loc[df_train['niv_years_precise'].astype('float')< 0].index, inplace=True)
    df_train = df_train.loc[df_train['NIV'] == 1]    
    y_train = df_train['niv_years_precise'].values

    df_test = pd.read_csv(constants.LABELS_DIR_test + '/time_to_event_labels.csv')
    df_test.replace(" ", np.nan, inplace=True)
    #print(df_test.loc[df_test['niv_years_precise'].astype('float')< 0])
    #df_test.drop(df_test.loc[df_test['niv_years_precise'].astype('float')< 0].index, inplace=True)
    df_test.dropna(subset=['niv_years_precise'], inplace = True)
    df_test['niv_years_precise'] = df_test['niv_years_precise'].astype('float')
    df_test = df_test.loc[df_test['niv_years_precise'] >= 0]
    
    
    df_test = df_test.loc[df_test['NIV'] == 1]
    y_test = df_test['niv_years_precise'].values
    
    print('************ No Stratification *****************')
    print('********* Total **************************')
    regressors(list(map(string_to_list, df_train['Reps'].values)), y_train, list(map(string_to_list, df_test['Reps'].values)), y_test, 'reg_no_strat_overall.pdf')

    print('********* Per group **************************')
    train_clusters = []
    test_clusters = []
    for i in range(constants.N_CLUST):
        print('CLUSTER ' + str(i))
        train_clusters.append(df_train.loc[df_train['Labels'] == i])
        test_clusters.append(df_test.loc[df_test['Labels'] == i])
        regressors(list(map(string_to_list, df_train['Reps'].values)), y_train, list(map(string_to_list,test_clusters[i]['Reps'].values)), test_clusters[i]['niv_years_precise'],'reg_no_strat_pred_{}.pdf'.format(i))

    print('************ Stratification *****************')

    temp_data =  pd.read_csv(constants.BASELINE_DIR_T_train + '{}TPS_baseline_temporal.csv'.format(n))
    static_data =  pd.read_csv(constants.BASELINE_DIR_S_train + '{}TPS_baseline_static.csv'.format(n))

    val_temp_data =  pd.read_csv(constants.BASELINE_DIR_T_test + '{}TPS_baseline_temporal.csv'.format(n))
    val_static_data =  pd.read_csv(constants.BASELINE_DIR_S_test + '{}TPS_baseline_static.csv'.format(n))

    
    for i in range(constants.N_CLUST):
        new_model = copy.deepcopy(model)
        print('CLUSTER ' + str(i))
        
        train_data = pd.merge(temp_data ,train_clusters[i][['REF']].rename(columns={constants.REF_FEATURE: 'Patient_ID'}), on = 'Patient_ID', how='inner')
        print(len(train_data))
        #print(train_data.loc[train_data['niv_years_precise'] < 0])
        train_refs, y_train, dynamic_train_set, static_train_set = transform_data(train_data,static_data)
        
        test_data = pd.merge(val_temp_data,test_clusters[i][['REF']].rename(columns={constants.REF_FEATURE: 'Patient_ID'}), on = 'Patient_ID', how='inner')
        val_refs, y_test, dynamic_val_set, static_val_set = transform_data(test_data,val_static_data)
        print('train:', len(dynamic_train_set))
        print('train:', len(train_clusters[i]['niv_years_precise']))
        print('test:', len(dynamic_val_set))
        print('test:', len(test_clusters[i]['niv_years_precise']))
        specialized_prediction(new_model, dynamic_train_set, train_clusters[i]['niv_years_precise'], dynamic_val_set, test_clusters[i]['niv_years_precise'],i)
