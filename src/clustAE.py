from utils import hierarchical_clustering, tsne, pacmap_func,simple_trajectories,parse_data, classifier,test_classification 
import pandas as pd 
import subprocess
import preprocessing.constants as constants
import sys
from get_autoencoder_reps import train, transform_data, get_encoded_reps, transform_data_test
from sklearn.model_selection import train_test_split
import torch
from pathlib import Path


if __name__ == "__main__":
    
    constants.get_config(sys.argv[1])
    n = constants.MIN_APP
    type_model = constants.MODEL
    pretrain = constants.PRETRAIN

    #create longitudinal tables
    cmd = "python3 src/preprocessing/longitudinal_tables_strat.py {} {}".format(n, sys.argv[1]) 
    print(cmd)
    subprocess.call(cmd, shell=True)

    temp_data =  pd.read_csv(constants.BASELINE_DIR_T_train + '{}TPS_baseline_temporal.csv'.format(n))
    static_data =  pd.read_csv(constants.BASELINE_DIR_S_train + '{}TPS_baseline_static.csv'.format(n))
    if type_model == 'temp_static':
        temp_data = temp_data[temp_data['Patient_ID'].isin(static_data['Patient_ID'])]
        temp_data.to_csv(constants.BASELINE_DIR_T_train + '{}TPS_baseline_temporal.csv'.format(n))
    train_refs, y_train, dynamic_train_set, static_train_set = transform_data(temp_data,static_data)
    

    val_temp_data =  pd.read_csv(constants.BASELINE_DIR_T_test + '{}TPS_baseline_temporal.csv'.format(n))
    val_static_data =  pd.read_csv(constants.BASELINE_DIR_S_test + '{}TPS_baseline_static.csv'.format(n))
    if type_model == 'temp_static':
        val_temp_data = val_temp_data[val_temp_data['Patient_ID'].isin(val_static_data['Patient_ID'])]
        val_temp_data.to_csv(constants.BASELINE_DIR_T_test + '{}TPS_baseline_temporal.csv'.format(n))
    val_refs, y_test, dynamic_val_set, static_val_set = transform_data_test(val_temp_data,val_static_data, type_model)

    #dynamic_train_set, dynamic_val_set, static_train_set, static_val_set, y_train,y_test,train_refs,val_refs = train_test_split(dynamic_data,static_data, y_true,refs, test_size=0.1,random_state = 42,stratify = y_true)
    if not pretrain:
        model = train(static_train_set, static_val_set, dynamic_train_set, dynamic_val_set, type_model)
    else:
        #checkpoint = torch.load('gen_model.pt')
        model = torch.load(type_model + '_turim_no_nan_val.pt')
    model =model.eval()
        #model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #epoch = checkpoint['epoch']
        #loss = checkpoint['loss']
       
    train_reps = get_encoded_reps(model, dynamic_train_set, static_train_set,type_model)
    test_reps = get_encoded_reps(model, dynamic_val_set, static_val_set,type_model)

    #find the clusters in data
    print('*** TRAIN CLUSTERING ***')
    
    patients = parse_data(constants.DATA_FILE_train, constants.TRAJECTORY_DIR_train, constants.VISUALIZATION_DIR_train)
    train_labels = hierarchical_clustering(train_reps, constants.TRAJECTORY_DIR_train)


    #visualize the representations
    #tsne(reps, labels, contants.VISUALIZATION_DIR_train)
    #pacmap_func(reps,labels,contants.VISUALIZATION_DIR_train)

    df = pd.DataFrame() 
    df['Patient_ID'] = train_refs
    df['Labels'] = train_labels
    df['Reps'] = list(train_reps)
    df['Evolution'] = temp_data['Evolution']
    df.to_csv(constants.LABELS_DIR_train + '/labels.csv') 


    patients = pd.merge(patients,df[['Patient_ID', 'Labels']].rename(columns={'Patient_ID':constants.REF_FEATURE}), on = constants.REF_FEATURE)

    patients = patients[patients['Labels'].notna()]
    
    clusters = []
    print(patients['REF'])
    for i in range(constants.N_CLUST):
        clusters.append(patients.loc[patients['Labels'] == i])
    simple_trajectories(clusters, constants.TRAJECTORY_DIR_train) 

    #find the clusters in data
    print('*** TEST CLUSTERING ***')
    
    test_patients = parse_data(constants.DATA_FILE_test, constants.TRAJECTORY_DIR_test, constants.VISUALIZATION_DIR_test)
    #labels = hierarchical_clustering(test_reps)
    test_labels = classifier(train_reps, test_reps,train_labels)
    #labels = test_classification(train_reps, test_reps,train_labels)
    #visualize the representations
    #tsne(reps, labels)
    #pacmap_func(reps,labels)

    df = pd.DataFrame() 
    df['Patient_ID'] = val_refs
    df['Labels'] = test_labels
    df['Reps'] = list(test_reps)
    df['Evolution'] = val_temp_data['Evolution']
    df.to_csv(constants.LABELS_DIR_test + '/labels.csv') 


    test_patients = pd.merge(test_patients,df[['Patient_ID', 'Labels']].rename(columns={'Patient_ID':constants.REF_FEATURE}), on = constants.REF_FEATURE)

    test_patients = test_patients[test_patients['Labels'].notna()]
    
    clusters = []
    for i in range(constants.N_CLUST):
        clusters.append(test_patients.loc[test_patients['Labels'] == i])
    simple_trajectories(clusters, constants.TRAJECTORY_DIR_test)


    print('=========================== GET MERGED TRAJECTORIES =============================================')
    
    clusters = []
    patients = pd.concat([patients, test_patients])
    for i in range(constants.N_CLUST):
        clusters.append(patients.loc[patients['Labels'] == i])
    all_traj_path = constants.TOP_FOLDER + "turim_lisbon_traj/"
    Path(all_traj_path).mkdir(parents=True, exist_ok=True)
    simple_trajectories(clusters, all_traj_path)