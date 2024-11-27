import preprocessing.constants as constants
import pandas as pd
import numpy as np
import torch
import copy
import sys
from utils import sens, spec,smote, random_undersample,normalize 
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from get_autoencoder_reps import finetune_model, data_to_matrix, get_encoded_reps
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import optuna
import math
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.model_selection import GridSearchCV
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier


def warm_start_classifiers(X_train, y_train, sp_X_train, sp_y_train, X_test, y_test, outputfile):
    table = { "Model": [], "Params" : [], "AUC": [], "CA": [], "Sens": [], "Spec": []}
    #scaler = StandardScaler()    
    
    print("Train\n", pd.Series(y_train).value_counts())
    print("Test\n", pd.Series(y_test).value_counts())
    print("Specialized train \n", pd.Series(sp_y_train).value_counts())


    X_train, y_train = random_undersample(X_train, y_train)
    X_train, y_train= smote(X_train, y_train)

    sp_X_train, sp_y_train = random_undersample(sp_X_train, sp_y_train)
    sp_X_train, sp_y_train= smote(sp_X_train, sp_y_train)
    #resample = SMOTEENN(smote=SMOTE(k_neighbors=2), random_state = 42)
    #X_train, y_train = resample.fit_resample(X_train, y_train)
    #sp_X_train, sp_y_train = resample.fit_resample(sp_X_train, sp_y_train)

    print("Train after resampling\n", pd.Series(y_train).value_counts())
    print("Specialized Train after resampling\n", pd.Series(sp_y_train).value_counts())

    models_and_parameters = {
        'LogisticRegression': (LogisticRegression(solver='liblinear', random_state=42), {
            'C': [0.01, 0.1, 1, 10], 'class_weight':['balanced', None]
        }),
        'RandomForestClassifier': (RandomForestClassifier(random_state=42), {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }),
        'XGBClassifier': (xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42), {
            'n_estimators': [10, 50, 100],
            'learning_rate': [0.01, 0.1, 0.2]
        })
        #'CatBoostClassifier': (CatBoostClassifier(logging_level = 'Silent',random_state=42), {
        #    'depth': [2,5,10],
        #    'learning_rate': [0.01, 0.1, 0.2],
    	#    'bagging_temperature': [0,0.01, 0.05,1]})

    }

    y_train = np.array(list(map(lambda label: 0 if label == 'N' else 1, y_train)))
    sp_y_train = np.array(list(map(lambda label: 0 if label == 'N' else 1, sp_y_train)))
    y_test = np.array(list(map(lambda label: 0 if label == 'N' else 1, y_test)))

    # Custom scoring function for GridSearchCV
    scoring = {'AUC': 'roc_auc', 'CA': make_scorer(accuracy_score)}

    # Grid search over each model
    best_model = None
    best_auc = 0

    for model_name, (model, param_grid) in models_and_parameters.items():
        model.fit(X_train, y_train)
        general_weights = model.get_params() 
        model.set_params(**general_weights)  # Apply warm start weights

        grid_search = GridSearchCV(model, param_grid, scoring=scoring, refit='AUC', cv=5)
        grid_search.fit(sp_X_train, sp_y_train)
        
        best_estimator = grid_search.best_estimator_
        y_pred = best_estimator.predict(X_test)
        y_proba = best_estimator.predict_proba(X_test)[:, 1] if hasattr(best_estimator, "predict_proba") else None

        auc_score = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        ca_score = accuracy_score(y_test, y_pred)
        
        # Sensitivity and specificity calculations (assumes custom functions `sens` and `spec`)
        sensitivity = sens(y_test, y_pred)
        specificity = spec(y_test, y_pred)
        
        # Store results
        table["Model"].append(model_name)
        table["Params"].append(grid_search.best_params_)
        table["AUC"].append(auc_score)
        table["CA"].append(ca_score)
        table["Sens"].append(sensitivity)
        table["Spec"].append(specificity)
        
        # Track best model
        if auc_score and auc_score > best_auc:
            best_auc = auc_score
            best_model = best_estimator

    # Convert results to a DataFrame and save
    results_df = pd.DataFrame(table)
    results_df.to_csv(constants.BASELINE_DIR_T_test + outputfile, index=False)
    
    # Print best model details
    print('***************** BEST CLF:', type(best_model).__name__)
    print('***************** BEST CLF AUC:', best_auc)
    return best_auc

def classifiers(X_train, y_train, X_test, y_test, outputfile):
    table = { "Model": [], "Params" : [], "AUC": [], "CA": [], "Sens": [], "Spec": []}
    #scaler = StandardScaler()    
    
    print("Train\n", pd.Series(y_train).value_counts())
    print("Test\n", pd.Series(y_test).value_counts())

    X_train, y_train = random_undersample(X_train, y_train)
    X_train, y_train= smote(X_train, y_train)
    #resample = SMOTEENN(smote=SMOTE(k_neighbors=2), random_state = 42)
    #X_train, y_train = resample.fit_resample(X_train, y_train)

    print("Train after resampling\n", pd.Series(y_train).value_counts())

    models_and_parameters = {
        'LogisticRegression': (LogisticRegression(solver='liblinear', random_state=42), {
            'C': [0.01, 0.1, 1, 10], 'class_weight':['balanced', None]
        }),
        'KNeighborsClassifier': (KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }),
        'GaussianNB': (GaussianNB(), {
            'var_smoothing': [1e-9, 1e-8, 1e-7]
        }),
        'SVC': (SVC(probability=True, random_state=42), {
            'C': [1, 10, 100],
            'kernel': ['rbf', 'linear']
        }),
        'RandomForestClassifier': (RandomForestClassifier(random_state=42), {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }),
        'XGBClassifier': (xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42), {
            'n_estimators': [10, 50, 100],
            'learning_rate': [0.01, 0.1, 0.2]
        })
        #'CatBoostClassifier': (CatBoostClassifier(logging_level = 'Silent',random_state=42), {
        #    'depth': [2,5,10],
        #    'learning_rate': [0.01, 0.1, 0.2],
    	#    'bagging_temperature': [0,0.01, 0.05,1]})
    }

    y_train = np.array(list(map(lambda label: 0 if label == 'N' else 1, y_train)))
    y_test = np.array(list(map(lambda label: 0 if label == 'N' else 1, y_test)))

    # Custom scoring function for GridSearchCV
    scoring = {'AUC': 'roc_auc', 'CA': make_scorer(accuracy_score)}

    # Grid search over each model
    best_model = None
    best_auc = 0

    for model_name, (model, param_grid) in models_and_parameters.items():
        grid_search = GridSearchCV(model, param_grid, scoring=scoring, refit='AUC', cv=5)
        grid_search.fit(X_train, y_train)
        
        best_estimator = grid_search.best_estimator_
        y_pred = best_estimator.predict(X_test)
        y_proba = best_estimator.predict_proba(X_test)[:, 1] if hasattr(best_estimator, "predict_proba") else None

        auc_score = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        ca_score = accuracy_score(y_test, y_pred)
        
        # Sensitivity and specificity calculations (assumes custom functions `sens` and `spec`)
        sensitivity = sens(y_test, y_pred)
        specificity = spec(y_test, y_pred)
        
        # Store results
        table["Model"].append(model_name)
        table["Params"].append(grid_search.best_params_)
        table["AUC"].append(auc_score)
        table["CA"].append(ca_score)
        table["Sens"].append(sensitivity)
        table["Spec"].append(specificity)
        
        # Track best model
        if auc_score and auc_score > best_auc:
            best_auc = auc_score
            best_model = best_estimator

    # Convert results to a DataFrame and save
    results_df = pd.DataFrame(table)
    results_df.to_csv(constants.BASELINE_DIR_T_test + outputfile, index=False)
    
    # Print best model details
    print('***************** BEST CLF:', type(best_model).__name__)
    print('***************** BEST CLF AUC:', best_auc)
    return best_auc

if __name__ == "__main__":
    constants.get_config(sys.argv[1])
    n = constants.MIN_APP
    type_model = constants.MODEL

    model = torch.load("./src/" + type_model + '_turim_no_nan_val.pt')
    label_class = joblib.load("./src/simple_nearest_centroids_original.joblib")

    features = []
    for i in range(constants.MIN_APP):
            feature_time = [str(i) + item for item in list(constants.TEMPORAL_FEATURES.keys())]
            features = features + feature_time
    static_features = ['Patietn_ID'] + list(constants.STATIC_FEATURES.keys()) + ['Evolution']

    print('*** Prognostic Prediction ***')
    temp_data =  pd.read_csv(constants.BASELINE_DIR_T_train + '{}TPS_baseline_temporal.csv'.format(n))
    #train_labels = pd.read_csv(constants.TOP_FOLDER + '/train/results/labels.csv')
    #train_labels.drop(columns=['Evolution'], inplace=True)
    temp_data['Patient_ID'] = temp_data['Patient_ID'].astype(str)
    #train_labels['Patient_ID'] = train_labels['Patient_ID'].astype(str)
    #temp_data_labels = pd.merge(temp_data, train_labels, on='Patient_ID', how='inner')
    
    val_temp_data =  pd.read_csv(constants.BASELINE_DIR_T_test + '{}TPS_baseline_temporal.csv'.format(n))
    #test_labels = pd.read_csv(constants.TOP_FOLDER + '/test/results/labels.csv')
    #test_labels.drop(columns=['Evolution'], inplace=True)
    val_temp_data['Patient_ID'] = val_temp_data['Patient_ID'].astype(int)
    #test_labels['Patient_ID'] = test_labels['Patient_ID'].astype(int)
    #val_temp_data_labels = pd.merge(val_temp_data, test_labels, on='Patient_ID', how='inner')
    
    print('************ No Stratification *****************')
    print('********* Total **************************')
    #x_train, x_test = normalize(temp_data[features], val_temp_data[features])
    scaler = joblib.load("./src/tclustae_scaler.joblib")
    x_train = scaler.transform(temp_data[features].values)
    x_test = scaler.transform(val_temp_data[features].values)

    x_train = pd.DataFrame(x_train, columns=features)
    x_train, _ = data_to_matrix(x_train, [], type_model)
    x_train = get_encoded_reps(model, x_train, [], type_model)
    train_labels = label_class.predict(x_train)

    x_test = pd.DataFrame(x_test, columns=features)
    x_test, _ = data_to_matrix(x_test, [], type_model)
    x_test = get_encoded_reps(model, x_test, [], type_model)
    test_labels = label_class.predict(x_test)

    classifiers(x_train, temp_data['Evolution'], x_test, val_temp_data['Evolution'],'/no_strat_pred_overall.csv')
    
    print('********* Per group **************************')
    for i in range(constants.N_CLUST):
        print('CLUSTER ' + str(i))
        x_test_clust = x_test[test_labels== i]
        y_test_clust = val_temp_data.loc[test_labels == i]['Evolution']
        classifiers(x_train, temp_data['Evolution'], x_test_clust, y_test_clust, '/no_strat_pred_{}.csv'.format(i))
    print('************ Stratification *****************')
    for i in range(constants.N_CLUST):
        print('CLUSTER ' + str(i))
        test_clusters = val_temp_data.loc[test_labels == i]
        train_clusters = temp_data.loc[train_labels== i]

        sp_x_train, x_test = normalize(train_clusters[features], test_clusters[features])
        warm_start_classifiers(x_train, temp_data['Evolution'], sp_x_train, train_clusters['Evolution'], x_test, test_clusters['Evolution'], '/spec_predition{}.csv'.format(i))
 