import preprocessing.constants as constants
import pandas as pd
import numpy as np
import torch
import copy
import sys
from utils import sens, spec,smote, random_undersample 
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from get_autoencoder_reps import finetune_model, transform_data, get_encoded_reps, transform_data_test
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import optuna
import math
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.model_selection import GridSearchCV

def string_to_list(string):
    # Remove the brackets
    string = string[1:-1]
    # Split the string by spaces and convert each element to an integer
    my_list = [float(x) for x in string.split()]
    return my_list

def esemble(X_train, y_train, X_test, y_test):
    y_train = np.array(list(map(lambda label: 0 if label == 'N' else 1, y_train)))
    y_test = np.array(list(map(lambda label: 0 if label == 'N' else 1, y_test)))

    X_train, y_train = random_undersample(X_train, y_train)
    X_train, y_train= smote(X_train, y_train)
    
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(X_train)     
    X_test = scaler.fit_transform(X_test) 


    #X_train = get_encoded_reps(model,X_train, [], 'simple')
    #X_test = get_encoded_reps(model,X_test, [], 'simple')

    classifiers_list = [('knn', KNeighborsClassifier(n_neighbors = 3)), ('gnb', GaussianNB()), ('svm',SVC(random_state = 0, probability=True)), ('rf', RandomForestClassifier(n_estimators=10, random_state=0)),('xgb', xgb.XGBClassifier(eval_metric='logloss', random_state=42,use_label_encoder=False))]
    eclf1 = VotingClassifier(estimators=classifiers_list, voting = 'soft')
    eclf1 = eclf1.fit(X_train, y_train)
    y_predicted = eclf1.predict(X_test)

    try:
        print("AUC: ", roc_auc_score(y_test, eclf1.predict_proba(X_test)[:, 1]))
    except ValueError:
        pass
    
    print("CA: ", accuracy_score(y_test, y_predicted))
    print("Sensitivity: ", sens(y_test, y_predicted))   
    print("Specificity: ", spec(y_test, y_predicted))


def classifiers(X_train, y_train, X_test, y_test, outputfile):
    table = { "Model": [], "Params" : [], "AUC": [], "CA": [], "Sens": [], "Spec": []}
    #scaler = StandardScaler()    
    nY = pd.Series(y_train).value_counts()['Y']
    nN = pd.Series(y_train).value_counts()['N']
    print(" Train\n", pd.Series(y_train).value_counts())

    nY = pd.Series(y_test).value_counts()['Y']
    nN = pd.Series(y_test).value_counts()['N']
    print(" Test\n", pd.Series(y_test).value_counts())

    X_train, y_train = random_undersample(X_train, y_train)
    X_train, y_train= smote(X_train, y_train)
    

    #X_train = scaler.fit_transform(X_train)     
    #X_test = scaler.fit_transform(X_test) 
    
    #X_train = [x for x in X_train if str(x) != 'nan'] 
    #y_train = [x for x in y_train if str(x) != 'nan'] 


    models_and_parameters = {
        'LogisticRegression': (LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42), {
            'C': [0.01, 0.1, 1, 10]
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
            'max_depth': [None, 10, 20]
        }),
        'XGBClassifier': (xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42), {
            'n_estimators': [10, 50, 100],
            'learning_rate': [0.01, 0.1, 0.2]
        })
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

def specialized_prediction(model, train_group, train_static, y_train, test_group, test_static, y_test,i, type_model):
    #model = finetune_model(model, train_group, test_group)
    #predict endpoint  
    model = model.eval()

    X_train = get_encoded_reps(model,train_group, train_static, type_model)
    X_test = get_encoded_reps(model,test_group, test_static, type_model)
    
    classifiers(X_train, y_train, X_test, y_test,'/spec_predition{}.csv'.format(i))

"""
def objective(trial):
    #dt_cf = trial.suggest_float('dt_cf', {0.15, 0.20, 0.25, 0.30})
    knn = trial.suggest_int('knn', 1, 11,2)
    #svm_comp = trial.suggest_float('svm_comp', {math.e**-1, math.e**-2, math.e**-3,1, math.e**1, math.e**2, math.e**3})
    svm_p_degree = trial.suggest_int('svm_p_degree', 1,3,1)
    svm_g = trial.suggest_float('svm_comp', math.e**-1, math.e**3, step = math.e)
    #nb_kernel = trial.suggest_categorical('nb_kernel', [True, False])
    rf_tree = trial.suggest_int('rf_tree', 5, 20, 5)
    #lr_ridge = trial.suggest_float('svm_comp', {math.e**-9, math.e**-8, math.e**-7,math.e**-5, math.e**-4, math.e**-2, math.e**-3})


    constants.get_config(sys.argv[1])
    n = constants.MIN_APP
    type_model = constants.MODEL

    model = torch.load(type_model + 'turim_niv_180.pt')
    #model =model.eval()
    features = []
    for i in range(constants.MIN_APP):
            feature_time = [str(i) + item for item in list(constants.TEMPORAL_FEATURES.keys())]
            features = features + feature_time

    print('*** Need for NIV 180 days prediciton ***')

    df_train = pd.read_csv(constants.LABELS_DIR_train + '/labels.csv')
    df_train.replace(" ", np.nan, inplace=True)   
    
    df_test = pd.read_csv(constants.LABELS_DIR_test + '/labels.csv')
    df_test.replace(" ", np.nan, inplace=True)
    y_test = df_test['Evolution'].values

    temp_data =  pd.read_csv(constants.BASELINE_DIR_T_train + '{}TPS_baseline_temporal.csv'.format(n))
    static_data =  pd.read_csv(constants.BASELINE_DIR_S_train + '{}TPS_baseline_static.csv'.format(n))

    train_data = pd.merge(temp_data ,df_train, on = ['Patient_ID', 'Evolution'], how='inner') 
    smote_train_data = pd.DataFrame(columns = ['Patient_ID'] + features + ['Evolution'])   
    smote_train_data[features],smote_train_data['Evolution'] = smote(train_data[features], train_data['Evolution'])
    smote_train_data[features],smote_train_data['Evolution'] = random_undersample(smote_train_data[features],smote_train_data['Evolution'])
    smote_train_data.dropna(subset = features, inplace = True)
    train_refs, y_train, dynamic_train_set, static_train_set = transform_data(smote_train_data,static_data)
    val_temp_data =  pd.read_csv(constants.BASELINE_DIR_T_test + '{}TPS_baseline_temporal.csv'.format(n))
    val_temp_data = pd.merge(val_temp_data ,df_test, on = ['Patient_ID', 'Evolution'], how='inner')  
    val_static_data =  pd.read_csv(constants.BASELINE_DIR_S_test + '{}TPS_baseline_static.csv'.format(n))
    #val_refs, y_test, dynamic_val_set, static_val_set = transform_data(val_temp_data[['Patient_ID'] + features + ['Evolution']],val_static_data)

    
    train_clusters = []
    test_clusters = []
    auc = []
    print('************ Stratification *****************')

    for i in range(constants.N_CLUST):
        new_model = copy.deepcopy(model)
        print('CLUSTER ' + str(i))
        train_clusters.append(train_data.loc[train_data['Labels'] == i])
        test_clusters.append(val_temp_data.loc[val_temp_data['Labels'] == i])
        #train_data = pd.merge(temp_data ,train_clusters[i][['Patient_ID']], on = ['Patient_ID', 'Evolution'], how='inner')
        smote_train_data = pd.DataFrame(columns = ['Patient_ID'] + features + ['Evolution'])
        smote_train_data[features],smote_train_data['Evolution'] = smote(train_clusters[i][features], train_clusters[i]['Evolution'])
        smote_train_data[features],smote_train_data['Evolution'] = random_undersample(smote_train_data[features],smote_train_data['Evolution'])
        smote_train_data.dropna(subset = features, inplace = True)
        #train_data[features], train_data['Evolution'] = smote(train_data[features], train_data['Evolution'])

        #static_train_data = pd.merge(static_data ,train_clusters[i][['Patient_ID']], on = ['Patient_ID', 'Evolution'], how='inner')
        train_refs, y_train, dynamic_train_set, static_train_set = transform_data(smote_train_data,static_data)

        #two_dim = [[item for sublist in inner_list for item in sublist] for inner_list in dynamic_train_set]
        #two_dim,y_train = smote(two_dim, y_train)
        
        # Define dimensions of original 3-dimensional list
        #cols = len(dynamic_train_set[0][0])

        # Convert 2-dimensional list back to 3-dimensional list
        #dynamic_train_set = [[two_dim[i][j:j+cols] for j in range(0, len(two_dim[i]), cols)] for i in range(len(two_dim))]
        #print(dynamic_train_set)
        
        #test_data = pd.merge(val_temp_data,test_clusters[i][['Patient_ID']], on = 'Patient_ID', how='inner')
        #static_test_data = pd.merge(val_static_data ,test_clusters[i][['Patient_ID']], on = 'Patient_ID', how='inner')
        val_refs, y_test, dynamic_val_set, static_val_set = transform_data(test_clusters[i][['Patient_ID'] + features + ['Evolution']],val_static_data)
        #dynamic_val_set= list(map(string_to_list, test_clusters[i]['Reps'].values))
        #y_test = test_clusters[i]['Evolution'].values
        
        #print('train:', len(dynamic_train_set))
        #print('test:', len(dynamic_val_set))
        best_auc =specialized_prediction(new_model, dynamic_train_set, y_train, dynamic_val_set, y_test, i, knn, svm_p_degree,svm_g, rf_tree)
        auc.append(best_auc)
        #classifiers(train_clusters[i][features], train_clusters[i]['Evolution'], test_clusters[i][features], test_clusters[i]['Evolution'], 'spec_precition_{}.csv'.format(i))
    return auc

#def hiperparameter_selection():

# Create an Optuna study
study = optuna.create_study(directions=['maximize','maximize','maximize','maximize'], sampler=optuna.samplers.TPESampler(seed=0))

# Run the optimization loop
study.optimize(objective, n_trials=25)

# Print the best hyperparameters found by Optuna
print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

#for trial in study.best_trials:
    #print(f"\tnumber: {trial.number}")
    #print(f"\tparams: {trial.params}")
    #print(f"\tvalues: {trial.values}")

trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[1])
print(f"Trial with highest accuracy: ")
print(f"\tnumber: {trial_with_highest_accuracy.number}")
print(f"\tparams: {trial_with_highest_accuracy.params}")
print(f"\tvalues: {trial_with_highest_accuracy.values}")
print('Best hyperparameters:', study.best_params)

"""   
if __name__ == "__main__":
    constants.get_config(sys.argv[1])
    n = constants.MIN_APP
    type_model = constants.MODEL

    model = torch.load("./src/" + type_model + '_turim_no_nan_val.pt')
    label_class = joblib.load("./src/simple_nearest_centroids_original.joblib")
    #model =model.eval()
    features = []
    for i in range(constants.MIN_APP):
            feature_time = [str(i) + item for item in list(constants.TEMPORAL_FEATURES.keys())]
            features = features + feature_time
    static_features = ['Patietn_ID'] + list(constants.STATIC_FEATURES.keys()) + ['Evolution']
    print('*** Prognostic Prediction ***')

    #df_train = pd.read_csv(constants.LABELS_DIR_train + '/labels.csv')
    #df_train.replace(" ", np.nan, inplace=True)   
    
    #df_test = pd.read_csv(constants.LABELS_DIR_test + '/labels.csv')
    #df_test.replace(" ", np.nan, inplace=True)
    #y_test = df_test['Evolution'].values

    temp_data =  pd.read_csv(constants.BASELINE_DIR_T_train + '{}TPS_baseline_temporal.csv'.format(n))
    static_data =  pd.read_csv(constants.BASELINE_DIR_S_train + '{}TPS_baseline_static.csv'.format(n))
    train_refs, y_train, dynamic_train_set, static_train_set = transform_data_test(temp_data[['Patient_ID'] + features + ['Evolution']],static_data, type_model)
    train_reps = get_encoded_reps(model, dynamic_train_set, static_train_set, type_model)
    train_labels = label_class.predict(train_reps)
    #print(len(train_labels))
    temp_data['Labels'] = train_labels
    static_data['Labels'] = np.empty(len(static_data))
    #df_train = pd.DataFrame({'Patient_ID': temp_data['Patient_ID'].values, 'Labels': train_labels, 'Evolution': temp_data['Evolution'].values })


    #train_data = pd.merge(temp_data ,df_train, on = ['Patient_ID', 'Evolution'], how='inner') 
    #smote_train_data = train_data.copy()
    #smote_train_data = smote_train_data[['Patient_ID'] + features + ['Evolution']]
    #smote_train_data = pd.DataFrame(columns = ['Patient_ID'] + features + ['Evolution'])   
    #smote_train_data[features],smote_train_data['Evolution'] = smote(train_data[features], train_data['Evolution'])
    #smote_train_data[features],smote_train_data['Evolution'] = random_undersample(smote_train_data[features],smote_train_data['Evolution'])
    #smote_train_data.dropna(subset = features, inplace = True)
    #train_refs, y_train, dynamic_train_set, static_train_set = transform_data(smote_train_data,static_data)
    #print(smote_train_data)
    val_temp_data =  pd.read_csv(constants.BASELINE_DIR_T_test + '{}TPS_baseline_temporal.csv'.format(n))
    #val_temp_data = pd.merge(val_temp_data ,df_test, on = ['Patient_ID', 'Evolution'], how='inner')  
    val_static_data =  pd.read_csv(constants.BASELINE_DIR_S_test + '{}TPS_baseline_static.csv'.format(n))
    val_refs, y_test, dynamic_val_set, static_val_set = transform_data_test(val_temp_data[['Patient_ID'] + features + ['Evolution']],val_static_data, type_model)
    test_reps = get_encoded_reps(model, dynamic_val_set, static_val_set, type_model)
    test_labels = label_class.predict(test_reps)
    val_temp_data['Labels'] = test_labels
    val_static_data['Labels'] = np.empty(len(val_static_data))
    #df_test = pd.DataFrame({'Patient_ID': val_temp_data['Patient_ID'].values, 'Labels': test_labels, 'Evolution': val_temp_data['Evolution'].values})
    #dynamic_val_set = get_encoded_reps(model,dynamic_val_set, [], 'simple')

    print('************ No Stratification *****************')
    print('********* Total **************************')
    #classifiers(no_strat_train_x, no_strat_train_y, list(map(string_to_list, df_test['Reps'].values)), y_test, 'no_strat_pred_overall.csv')
    #dynamic_train_set = get_encoded_reps(model,dynamic_train_set, [], 'simple')
    #dynamic_val_set = get_encoded_reps(model,dynamic_val_set, [], 'simple')
    #classifiers(dynamic_train_set, y_train, dynamic_val_set, y_test, 'no_strat_pred_overall.csv')
    #dynamic_train_set = get_encoded_reps(model,dynamic_train_set, [], 'simple')

       
    classifiers(train_reps, y_train, test_reps, y_test,'/no_strat_pred_overall.csv')
    #classifiers(train_data[features].values, train_data['Evolution'].values, val_temp_data[features].values, val_temp_data['Evolution'].values,'no_strat_pred_overall.csv')
    #classifiers(list(map(string_to_list, df_train['Reps'].values)), df_train['Evolution'].values, list(map(string_to_list, df_test['Reps'].values)), y_test, 'no_strat_pred_overall.csv')
    #anomalie_detection(list(map(string_to_list, df_train['Reps'].values)), df_train['Evolution'].values, list(map(string_to_list, df_test['Reps'].values)), y_test)
    print('********* Per group **************************')
    train_clusters = []
    train_clusters_static = []
    test_clusters = []
    test_clusters_static = []
    for i in range(constants.N_CLUST):
        print('CLUSTER ' + str(i))
        #train_clusters.append(df_train.loc[df_train['Labels'] == i])
        #test_clusters.append(df_test.loc[df_test['Labels'] == i])
        train_clusters.append(temp_data.loc[temp_data['Labels'] == i])
        train_clusters_static.append(static_data.loc[static_data['Labels'] == i])
        test_clusters.append(val_temp_data.loc[val_temp_data['Labels'] == i])
        test_clusters_static.append(val_static_data.loc[val_static_data['Labels'] == i])
        val_refs, y_test, dynamic_val_set, static_val_set = transform_data_test(test_clusters[i][['Patient_ID'] + features + ['Evolution']],test_clusters_static[i], type_model)

        #dynamic_train_set = get_encoded_reps(model,dynamic_train_set, [], 'simple')
        #classifiers(train_data[features].values, train_data['Evolution'].values, test_clusters[i][features].values, test_clusters[i]['Evolution'].values,'no_strat_pred_{}.csv'.format(i))
        #dynamic_val_set = get_encoded_reps(model,dynamic_val_set, [], 'simple')
        #dynamic_train_set = get_encoded_reps(model,dynamic_train_set, [], 'simple')
        dynamic_val_set = get_encoded_reps(model,dynamic_val_set, static_val_set, type_model)
        classifiers(train_reps, y_train, dynamic_val_set, y_test, '/no_strat_pred_{}.csv'.format(i))
        #esemble(dynamic_train_set, y_train, dynamic_val_set, y_test)
        #print(test_clusters[i]['Evolution'].values)
        #print(y_train)
        #classifiers(no_strat_train_x, no_strat_train_y, list(map(string_to_list,test_clusters[i]['Reps'].values)), test_clusters[i]['Evolution'].values, 'no_strat_pred_{}.csv'.format(i))
        #classifiers(train_data[features], train_data['Evolution'], test_clusters[i][features], test_clusters[i]['Evolution'], 'no_strat_pred_{}.csv'.format(i))
        #anomalie_detection(list(map(string_to_list, df_train['Reps'].values)), df_train['Evolution'].values, list(map(string_to_list,test_clusters[i]['Reps'].values)), test_clusters[i]['Evolution'].values)
    print('************ Stratification *****************')

    for i in range(constants.N_CLUST):
        new_model = copy.deepcopy(model)
        print('CLUSTER ' + str(i))
        smote_train_data = train_clusters[i].copy()
        smote_train_data = smote_train_data[['Patient_ID'] + features + ['Evolution']]
        #train_data = pd.merge(temp_data ,train_clusters[i][['Patient_ID']], on = ['Patient_ID', 'Evolution'], how='inner')
        #smote_train_data = pd.DataFrame(columns = ['Patient_ID'] + features + ['Evolution'])
        #smote_train_data[features],smote_train_data['Evolution'] = smote(train_clusters[i][features], train_clusters[i]['Evolution'])
        #smote_train_data[features],smote_train_data['Evolution'] = random_undersample(smote_train_data[features],smote_train_data['Evolution'])
        #smote_train_data.dropna(subset = features, inplace = True)
        train_refs, y_train, dynamic_train_set, static_train_set = transform_data_test(smote_train_data,train_clusters_static[i], type_model)

        #train_data[features], train_data['Evolution'] = smote(train_data[features], train_data['Evolution'])

        #static_train_data = pd.merge(static_data ,train_clusters[i][['Patient_ID']], on = ['Patient_ID', 'Evolution'], how='inner')
        #train_refs, y_train, dynamic_train_set, static_train_set = transform_data(smote_train_data,static_data)

        #two_dim = [[item for sublist in inner_list for item in sublist] for inner_list in dynamic_train_set]
        #two_dim,y_train = smote(two_dim, y_train)
        
        # Define dimensions of original 3-dimensional list
        #cols = len(dynamic_train_set[0][0])

        # Convert 2-dimensional list back to 3-dimensional list
        #dynamic_train_set = [[two_dim[i][j:j+cols] for j in range(0, len(two_dim[i]), cols)] for i in range(len(two_dim))]
        #print(dynamic_train_set)
        
        #test_data = pd.merge(val_temp_data,test_clusters[i][['Patient_ID']], on = 'Patient_ID', how='inner')
        #static_test_data = pd.merge(val_static_data ,test_clusters[i][['Patient_ID']], on = 'Patient_ID', how='inner')
        val_refs, y_test, dynamic_val_set, static_val_set = transform_data_test(test_clusters[i][['Patient_ID'] + features + ['Evolution']],test_clusters_static[i], type_model)
        #dynamic_val_set= list(map(string_to_list, test_clusters[i]['Reps'].values))
        #y_test = test_clusters[i]['Evolution'].values
        
        #print('train:', len(dynamic_train_set))
        #print('test:', len(dynamic_val_set))
        specialized_prediction(new_model, dynamic_train_set,static_train_set, y_train, dynamic_val_set,static_val_set, y_test, i, type_model)
        #classifiers(train_clusters[i][features].values, train_clusters[i]['Evolution'].values, test_clusters[i][features].values, test_clusters[i]['Evolution'].values, 'spec_precition_{}.csv'.format(i))
        #esemble(train_clusters[i][features], train_clusters[i]['Evolution'], test_clusters[i][features],test_clusters[i]['Evolution'])