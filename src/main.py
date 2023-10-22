from importlib.util import module_from_spec
from re import S
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from LSTM_AE import LSTMAutoencoder
from Temp_Static_AE import STAutoencoder
from FSG_AE import FSGAutoencoder
#from dynamic_AE import DynamicLSTMAutoencoder
import pandas as pd
import numpy as np
import pacmap
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
import os
import optuna
import errno
import lightning as L
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestCentroid
import sys
from sklearn.model_selection import train_test_split
from Predictor_AE import PredAutoencoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
sys.setrecursionlimit(100000)
import time

torch.manual_seed(0)
#random.seed(1)

def random_undersample(X, y):
    nY = pd.Series(y).value_counts()['Y']
    nN = pd.Series(y).value_counts()['N']
    print("Before RU\n", pd.Series(y).value_counts())
    if nY/nN < 0.5:
        rus = RandomUnderSampler(random_state=0, sampling_strategy=0.5)
        X, y = rus.fit_resample(X, y)
    return X, y

def smote(X,y):
    sm = sm = SMOTE(random_state=0)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res



def wss(X, labels):
    #Compute the centroid of each cluster, which represents the mean of the data points in that cluster.
    cluster_centroids = []
    for i in np.unique(labels):
        centroid = np.mean(X[labels == i], axis=0)
        cluster_centroids.append(centroid)
    #Calculate the squared Euclidean distance between each data point and its corresponding cluster centroid.
    squared_distance = 0
    for i in range(X.shape[0]):
        cluster_index = labels[i]
        distance = np.sum((X[i] - cluster_centroids[cluster_index]) ** 2)
        squared_distance += distance
    #Obtain the Within-cluster Sum of Squares by summing up the squared distances.
    return squared_distance

def bss(X, labels):
    #Compute the overall mean of the dataset, which represents the centroid of all data points
    overall_mean = np.mean(X, axis=0)
    #Compute the mean of each cluster, which represents the centroid of the data points in that cluster.
    cluster_means = []
    for i in np.unique(labels):
        cluster_mean = np.mean(X[labels == i], axis=0)
        cluster_means.append(cluster_mean)
    #Calculate the squared Euclidean distance between each cluster mean and the overall mean, weighted by the number of data points in the corresponding cluster.
    n_samples = X.shape[0]
    n_clusters = len(cluster_means)
    #Calculate the squared Euclidean distance between each cluster mean and the overall mean, weighted by the number of data points in the corresponding cluster.
    weights = []
    for i in np.unique(labels):
        n_points = np.sum(labels == i)
        weight = n_points / n_samples
        weights.append(weight)

    squared_distance = 0
    for i in range(n_clusters):
        distance = np.sum((cluster_means[i] - overall_mean) ** 2)
        squared_distance += distance * weights[i]
    return squared_distance

def get_all_appointments(filename):
    dynamic_features = ['ALSFRSb', 'ALSFRSsUL', 'ALSFRSsLL', 'R']
    df = pd.read_csv(filename)
    sequences = df.groupby('REF')
    lenghts = df.groupby('REF').size().to_list()

    dataset = []
    refs = []
    for ref, group in sequences:
        grp = []
        refs.append(ref)
        for row_index, row in group.iterrows(): 
            grp.append(row[dynamic_features].to_list())
        dataset.append(grp)
    return refs,dataset, lenghts

def get_n_appointments(df, dynamic_features, static_features, n=6):
    df = df.copy()

    #df = pd.read_csv(filename)
    df.dropna(subset = dynamic_features, inplace=True)
    
    le = preprocessing.LabelEncoder() 
    for col in static_features:
        if col not in ['DiagnosticDelay']:
            #df[col] = df[col].fillna('_')
            df[col]=le.fit_transform(df[col])
        #df[col]=(df[col]-df[col].min())/(df[col].max()-df[col].min())
    df['MITOS-stage'] = 6 - df['MITOS-stage']
    
    # normalize the feature matrix X
    #scaler = StandardScaler()
    scaler = MinMaxScaler()
    df[dynamic_features] = scaler.fit_transform(df[dynamic_features])
    df[static_features] = scaler.fit_transform(df[static_features])
    #ul = df.loc[]
    sequences = df.groupby('REF')
    dynamic_dataset = []
    static_dataset = []
    refs = []
    for ref, group in sequences:
     
        grp1 = []
        grp2 = []
        
        i = 0 
        if group.shape[0] < n:
            continue
        refs.append(ref)
        for row_index, row in group.iterrows(): 
            if i < 3:
                i+=1
                continue
            grp1.append(row[dynamic_features].to_list())
            #grp2.append(row[static_features].to_list())
            #print(grp)
            i+=1
            if i ==n:
                break
        dynamic_dataset.append(grp1)
        static_dataset.append(grp2)
    return refs, dynamic_dataset, static_dataset

def get_3appointments(df, dynamic_features, static_features, type):
    df = df.copy()

    #df = pd.read_csv(filename)
    df.dropna(subset = dynamic_features, inplace=True)
    
    le = preprocessing.LabelEncoder() 
    df['BMI_at_1st_visit'].replace('obeso', 40, inplace = True)
    for col in static_features:
        if col not in ['Diagnostic Delay', 'BMI_at_1st_visit']:
            #df[col] = df[col].fillna('_')
            df[col]=le.fit_transform(df[col])
        #df[col]=(df[col]-df[col].min())/(df[col].max()-df[col].min())
    if type == 'temp_static':
        df.dropna(subset = static_features, inplace=True)
    df['MITOS-stage'] = 4- df['MITOS-stage']
    
    # normalize the feature matrix X
    scaler = MinMaxScaler()
    #scaler = StandardScaler()
    df[dynamic_features] = scaler.fit_transform(df[dynamic_features])
    df[static_features] = scaler.fit_transform(df[static_features])
    if type == 'predictor':
        df['Evolution'] = df['Evolution'].map(dict(Y=1, N=0))
    #ul = df.loc[]
    if type == 'fsg':
        df['FS'] = 0
        cpy_df = df.copy()
        cpy_df['FS'] = 1
        cpy_df['REF'] =  'dup' + cpy_df['REF'].astype(str)
        df = pd.concat([df,cpy_df])
        
        sequences = df.groupby('REF')
        sequences = [(key, group) for key, group in sequences]
        #print('Before', sequences[0])
        random.shuffle(sequences)
    else:    
        sequences = df.groupby('REF')
    dynamic_dataset = []
    static_dataset = []
    predictor  = []
    fsg = []
    refs = []
    for ref, group in sequences:
        if type == 'fsg':
            fs = df.loc[df['REF'] == ref, 'FS'].values[0]
     
        grp1 = []       
        i = 0 
        if group.shape[0] < 3:
            continue
        refs.append(ref)
        if type == 'fsg':
            fsg.append(fs)
        for row_index, row in group.iterrows(): 
            grp1.append(row[dynamic_features].to_list())
        
            #print(grp)
            i+=1
            if i ==3:
                break
        if type == 'fsg' and fs: # if is a fake sample (fs ==1)
            random.shuffle(grp1)
        dynamic_dataset.append(grp1)
        static_dataset.append(row[static_features].to_list())
        if type == 'predictor':
            predictor.append(row['Evolution'])
    return refs, dynamic_dataset, static_dataset, predictor, fsg

def get_merged_dataset(filename1,filename2, dynamic_features):


    df1 = pd.read_csv(filename1)
    df2 = pd.read_csv(filename2)
    df2.rename(columns = {'ALSFRS_1': 'P1', 'ALSFRS_2': 'P2', 'ALSFRS_3': 'P3', 'ALSFRS_4': 'P4', 'ALSFRS_5': 'P5', 'ALSFRS_6': 'P6', 'ALSFRS_7': 'P7', 'ALSFRS_8': 'P8', 'ALSFRS_9': 'P9', 'ALSFRS_10': 'P10', 'ALSFRS_11': 'P11', 'ALSFRS_12': 'P12'}, inplace = True)
    df = pd.concat([df1,df2])
    df.dropna(subset = dynamic_features, inplace=True)

    #ul = df.loc[]
    sequences = df.groupby('REF')
    sequences = [group for _, group in sequences]
    random.shuffle(sequences)
    df = pd.concat(sequences).reset_index(drop=True)
    #sequences = sequences[0].to_frame()
    df.to_csv('merged_dataset.csv')
    return df

def get_finetuning_data(df, colname, characteristic, dynamic_features, static_features):
   #df = pd.read_csv(filename)
    df = df.copy()
    df.dropna(subset = dynamic_features, inplace=True)
    
    df = df.loc[df[colname].isin(characteristic)]
    le = preprocessing.LabelEncoder() 
    for col in static_features:
        if col not in ['DiagnosticDelay']:
            df[col] = df[col].fillna('_')
            df[col]=le.fit_transform(df[col])
        #df[col]=(df[col]-df[col].min())/(df[col].max()-df[col].min())
    df['MITOS-stage'] = 6 - df['MITOS-stage']
    
    # normalize the feature matrix X
    scaler = MinMaxScaler()
    df[dynamic_features] = scaler.fit_transform(df[dynamic_features])
    df[static_features] = scaler.fit_transform(df[static_features])
    #ul = df.loc[]
    sequences = df.groupby('REF')
    dynamic_dataset = []
    static_dataset = []
    refs = []
    for ref, group in sequences:
     
        grp1 = []
        grp2 = []
        
        i = 0 
        if group.shape[0] < 3:
            continue
        refs.append(ref)
        for row_index, row in group.iterrows(): 
            grp1.append(row[dynamic_features].to_list())
            #grp2.append(row[static_features].to_list())
            #print(grp)
            i+=1
            if i ==3:
                break
        dynamic_dataset.append(grp1)
        static_dataset.append(grp2)
 
    return refs, dynamic_dataset


def tsne(tempData, labels, output_name, n_clust):
    colors = ['#0174DF', 'orange', '#04B45F', '#DF0101','black']
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(tempData)
    new = {}
    #new = tempData.copy()
    new['tsne-2d-one'] = X_2d[:,0]
    new['tsne-2d-two'] = X_2d[:,1]

    fig = plt.figure(figsize=(16,10))
    sns.scatterplot(
        x = "tsne-2d-one", y = "tsne-2d-two",
        hue = labels,
        palette = colors[:n_clust],
        data = new,
        legend = "full"
)
    fig.savefig(output_name + '/tsne.pdf')
    plt.clf()

def pacmap_func(tempData, labels,output_name, n_clust):
    colors = ['#0174DF', 'orange', '#04B45F', '#DF0101','black']
    embedding = pacmap.PaCMAP(n_components=2, random_state=0)
    X_2d = embedding.fit_transform(tempData)
    new = {}
    #new = tempData.copy()
    new['pacmap-2d-one'] = X_2d[:,0]
    new['pacmap-2d-two'] = X_2d[:,1]

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x = "pacmap-2d-one", y = "pacmap-2d-two",
        hue = labels,
        palette = colors[:n_clust],
        data = new,
        legend = "full"
    )

    plt.savefig(output_name + '/pacmap.pdf')
    plt.clf()

def hierarchical_clustering(data,output_name, n_clust):
    
    plt.ylabel('distance')
    clusters = shc.linkage(data, method="ward", metric="euclidean")

    #shc.dendrogram(clusters, p = n_clust, truncate_mode = 'lastp', # show only the last p merged clusters
                #show_leaf_counts = False, no_plot=True)
    #leaves = shc.dendrogram(clusters,no_plot=True)['leaves']
    
    


    Ward_model = AgglomerativeClustering(n_clusters= n_clust, metric='euclidean', linkage='ward')
    cluster_labels = Ward_model.fit_predict(data)

    
    shc.dendrogram(clusters,p= 20, truncate_mode ='lastp',show_leaf_counts=False, leaf_font_size=4)
    leaves = shc.dendrogram(clusters, no_plot = True)['leaves']
    #print(R["leaves"])
    #leaves_sorted = sorted(leaves)
    #print(len(leaves_sorted))
    # Check if any leaves are out of bounds
    if max(leaves) >= len(data):
        # Remove out-of-bounds leaves
        leaves = [i for i in leaves if i < len(data)]
    #print(len(leaves_sorted))
    #print(len(labels))
    # Map the sorted leaves to the labels assigned by sklearn
    #sklearn_labels = labels[leaves_sorted]
    #scipy_labels = [labels[i] for i in leaves] 


    ordered_labels = cluster_labels[leaves]
    for i in range(1, len(ordered_labels)):
        if ordered_labels[i] != ordered_labels[i-1]:
            print(ordered_labels[i] )
         
    print(ordered_labels)


    plt.savefig(output_name + '/dendrogram.pdf')
    plt.clf()

    print('Silhouette Score: ', silhouette_score(data, cluster_labels, metric='euclidean'))
    print('Calinski Harabasz Score: ', calinski_harabasz_score(data, cluster_labels))
    print('Davies Bouldin Score: ', davies_bouldin_score(data, cluster_labels))
    print('WSS: ', wss(data, cluster_labels))
    print('BSS: ', bss(data, cluster_labels))

    return cluster_labels, silhouette_score(data, cluster_labels, metric='euclidean'), calinski_harabasz_score(data, cluster_labels), davies_bouldin_score(data, cluster_labels)

def plot_representations(points, labels, output_name,n_clust):
    new = {}
    #new = tempData.copy()
    new['2d-one'] = points[:,0]
    new['2d-two'] = points[:,1]
    colors = ['#0174DF', 'orange', '#04B45F', '#DF0101']
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x = "2d-one", y = "2d-two",
        hue = labels,
        palette = colors[:n_clust],
        data = new,
        legend = "full"
    )

    plt.savefig(output_name + '/representations.pdf')
    plt.clf()
def classifier(clustering, data, y_train,training):
    print(y_train)
    if training:
        clustering= clustering.fit(data, y_train)
        cluster_labels = clustering.predict(data)
    else:
        cluster_labels = clustering.predict(data)
    
    #print('Silhouette Score: ', silhouette_score(data, cluster_labels, metric='euclidean'))
    #print('Calinski Harabasz Score: ', calinski_harabasz_score(data, cluster_labels))
    #print('Davies Bouldin Score: ', davies_bouldin_score(data, cluster_labels))
    #print('WSS: ', wss(data, cluster_labels))
    #print('BSS: ', bss(data, cluster_labels))
    return clustering, cluster_labels
def nearest_centroid(output_name, X_train, y_train, X_test, y_test, static_train, static_test, type):
    
    train_points = []
    for i in range(0, len(X_train), batch_size):
        if type == 'simple' or type == 'fsg':
            _,proj= model.encoder(torch.tensor(X_train[i:i+batch_size]))
        elif type == 'temp_static' or type =='predictor':
            static, proj,_= model.encoder(torch.tensor(X_train[i:i+batch_size]), torch.tensor(static_train[i:i+batch_size]))
            static = static.squeeze(1)
            proj = torch.cat((proj,static), dim =1)

        for point in proj:
            train_points.append(point.detach().numpy())
    train_points = np.array(train_points)
    test_points = []
    for i in range(0, len(X_test), batch_size):
        if type == 'simple' or type == 'fsg':
            _,proj= model.encoder(torch.tensor(X_test[i:i+batch_size]))
        elif type == 'temp_static' or type == 'predictor':
            static, proj,_= model.encoder(torch.tensor(X_test[i:i+batch_size]), torch.tensor(static_test[i:i+batch_size]))
            static = static.squeeze(1)
            proj = torch.cat((proj,static), dim =1)

        for point in proj:
            test_points.append(point.detach().numpy())
    
    test_points = np.array(test_points)

    X_train=np.asarray(X_train)
    X_train = X_train.reshape(X_train.shape[0], -1)
    print(X_train)
    X_test=np.asarray(X_test)
    X_test = X_test.reshape(X_test.shape[0], -1)


    clf = NearestCentroid()
    clf.fit(train_points, y_train)
    accuracy = clf.score(test_points, y_test)
    cluster_labels = clf.predict(test_points)
    print('Train accuracy: ', clf.score(train_points, y_train))
    print("Test Accuracy:", accuracy)
    #print('Silhouette Score: ', silhouette_score(data, cluster_labels, metric='euclidean'))
    #print('Calinski Harabasz Score: ', calinski_harabasz_score(data, cluster_labels))
    #print('Davies Bouldin Score: ', davies_bouldin_score(data, cluster_labels))
    #print('WSS: ', wss(data, cluster_labels))
    #print('BSS: ', bss(data, cluster_labels))

    #df = pd.DataFrame() 
    #df['Patient_ID'] = refs
    #df['Labels'] = cluster_labels

    # saving the dataframe 
    #df.to_csv(output_name +'/labels.csv') 
def get_visualizations(type, refs, dynamic_dataset, static_dataset, output_name, model, n_clust, batch_size):
#def get_visualizations(clustering, refs, dataset, output_name, model, n_clust, batch_size, training, y_train):
    points = []
    for i in range(0, len(dynamic_dataset), batch_size):
        if type == 'simple'or type == 'fsg':
            _, proj= model.encoder(torch.tensor(dynamic_dataset[i:i+batch_size]))
        elif type == 'temp_static' or type == 'predictor' :
            static, proj,_= model.encoder(torch.tensor(dynamic_dataset[i:i+batch_size]), torch.tensor(static_dataset[i:i+batch_size]))
            static = static.squeeze(1)
            proj = torch.cat((proj,static), dim =1)
        for point in proj:
            points.append(point.detach().numpy())
    points = np.array(points)
    #print(len(points))
    #points = [j for i in points for j in i]
    #   points = np.array(points)
    #print(points)
    #clustering, labels = classifier(clustering, points, y_train,training)
    labels,sc,ch,db = hierarchical_clustering(points, output_name, n_clust)
    df = pd.DataFrame() 
    df['Patient_ID'] = refs
    df['Labels'] = labels

    # saving the dataframe 
    df.to_csv(output_name +'/labels.csv') 
    #plot_representations(points, labels, output_name, 4)
    #pacmap_func(points, labels, output_name, n_clust)
    #tsne(points, labels, output_name, n_clust)
    #return clustering
    return sc,ch,db
"""
def objective(trial):
    dynamic_features = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P8', 'P9', 'P10', 'P11', 'P12']
    type = 'simple'
    #dynamic_features = ['ALSFRSb', 'ALSFRSsUL', 'ALSFRSsLL', 'R', 'ALSFRS-R', 'ALSFRSsT'] #progRate3 , 'MITOS-stage'
    static_features = ['C9orf72', 'Gender', 'ALS_familiar_history', 'UMNvsLMN', 'Onset', 'DiagnosticDelay', 'Weightloss_>10%']

    # Define the hyperparameters to optimize
    bidirectional1 = trial.suggest_categorical('bidirectional1', [True, False])
    bidirectional2 = trial.suggest_categorical('bidirectional2', [True, False])
    #dropout = trial.suggest_float('dropout', 0, 1)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    hidden_size = trial.suggest_int('hidden_size', 2, 8)
    batch_size = trial.suggest_int('batch_size', 2, 16)
    #learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log = True)
    #epochs = trial.suggest_int('num_epochs', 10, 200)
    
    num_head = trial.suggest_categorical('num_head', [1, 2, 4])
    #num_head2 = trial.suggest_categorical('num_head2', [1, 2, 4, 8,16])
    n_clust = trial.suggest_int('n_clust', 3, 6)
    lr = trial.suggest_float('lr', 0, 1)

    data =  pd.read_csv('merged_dataset.csv')
    ratio = 0.7   

    tot_refs, data, static_data, y_true, fsg = get_3appointments(data, dynamic_features, static_features, type)
    total_rows = len(data)
    train_size = int(total_rows*ratio)

    # Split data into test and train
    dynamic_train_set = data[0:train_size]
    static_train_set = static_data[0:train_size]
    train_refs = tot_refs[0:train_size]
    dynamic_val_set = data[train_size:]
    static_val_set = static_data[train_size:]
    val_refs= tot_refs[train_size:]
    #val_refs, dynamic_val_set, static_val_set = get_n_appointments("../../snapshots/new_sc_lisbon_no_nan_snapshots_independent.csv", dynamic_features, static_features)
    

    n_features = len(dynamic_features)
    n_static_features = len(static_features)
    seq_len = 3

    patience = 2
    trigger_times = 0


    # Model Initialization
    model = LSTMAutoencoder(seq_len,n_features, hidden_size, num_layers, bidirectional1,bidirectional2,num_head)

    #model = AttenLSTMAutoencoder(n_features, hidden_size, num_layers, dropout)
    # Validation using MSE Loss function
    loss_function = torch.nn.MSELoss()
    # Using an Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                lr = lr, weight_decay=0.01)
    epochs = 200
    history = dict(train=[], val=[])
    
    for epoch in range(epochs):
        model = model.train()
        train_losses = []
        for i in range(0, len(dynamic_train_set), batch_size):
            dynamic_record = dynamic_train_set[i:i+batch_size]            
            dynamic_record = torch.tensor(dynamic_record)

            # The gradients are set to zero           
            optimizer.zero_grad()
            # Output of Autoencoder
            reconstructed = model(dynamic_record) 
            # Calculating the loss function
            #encoded,_ = model.encoder(dynamic_record)
            #decoded = model.decoder(encoded)
            #layer_losses = [optimizer(layer1(decoded), layer2(dynamic_record)) for layer1,layer2 in zip(decoder_layers,encoder_layers)]
            #loss = sum(layer_losses)
            loss = loss_function(reconstructed,dynamic_record)
            loss.backward()
            # .step() performs parameter update
            optimizer.step()                     
            # Storing the losses in a list for plotting
            train_losses.append(loss.item())
        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for i in range(0, len(dynamic_val_set), batch_size):
                dynamic_record = dynamic_val_set[i:i+batch_size]
                dynamic_record = torch.tensor(dynamic_record)

                reconstructed = model(dynamic_record)
                #encoded,_ = model.encoder(dynamic_record)
                #decoded = model.decoder(encoded)
                #layer_losses = [optimizer(layer1(decoded), layer2(dynamic_record)) for layer1,layer2 in zip(nn.Sequential(model.decoder),model.encoder)]
                #loss = sum(layer_losses)
                
                # Calculating the loss function
                loss = loss_function(reconstructed,dynamic_record)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        #early stop
        if epoch > 1 and history['val'][-1] < val_loss:
            trigger_times += 1
            print('Trigger Times:', trigger_times)
        else:
            trigger_times = 0
        if trigger_times >= patience:
            print('Early stopping!')
            break
            
            
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
    
    trigger_times = 0
    print("FINETUNING!")
    #batch_size = 3
    print(len(finetuning_data))
    print(len(finetuning_test))
    #finetune
    for epoch in range(epochs):
        model = model.train()
        train_losses = []
        for i in range(0, len(finetuning_data), batch_size):
            dynamic_record = finetuning_data[i:i+batch_size]            
            dynamic_record = torch.tensor(dynamic_record)

            # The gradients are set to zero           
            optimizer.zero_grad()
            # Output of Autoencoder
            reconstructed = model(dynamic_record) 
            # Calculating the loss function
            #encoded,_ = model.encoder(dynamic_record)
            #decoded = model.decoder(encoded)
            #layer_losses = [optimizer(layer1(decoded), layer2(dynamic_record)) for layer1,layer2 in zip(decoder_layers,encoder_layers)]
            #loss = sum(layer_losses)
            loss = loss_function(reconstructed,dynamic_record)
            loss.backward()
            # .step() performs parameter update
            optimizer.step()                     
            # Storing the losses in a list for plotting
            train_losses.append(loss.item())
        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for i in range(0, len(finetuning_test), batch_size):
                dynamic_record = finetuning_test[i:i+batch_size]
                dynamic_record = torch.tensor(dynamic_record)

                reconstructed = model(dynamic_record)
                #encoded,_ = model.encoder(dynamic_record)
                #decoded = model.decoder(encoded)
                #layer_losses = [optimizer(layer1(decoded), layer2(dynamic_record)) for layer1,layer2 in zip(nn.Sequential(model.decoder),model.encoder)]
                #loss = sum(layer_losses)
                
                # Calculating the loss function
                loss = loss_function(reconstructed,dynamic_record)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        #early stop
        if epoch > 1 and history['val'][-1] < val_loss: #0.000001:
            trigger_times += 1
            print('Trigger Times:', trigger_times)
        else:
            trigger_times = 0
        if trigger_times >= patience:
            print('Early stopping!')
            break
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
      
    output_folder = 'all_features' 
    #output_folder = 'nearest_centroids'
    if epoch == 199 or trigger_times == patience:
        try: 
            os.mkdir(output_folder+str(n_clust)+'_clust_train')               
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
        try: 
            os.mkdir(output_folder+str(n_clust)+'_clust_test')               
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
         
        #get_visualizations(train_refs,dynamic_train_set, 'all_features_AE_3_clust_train', model,n_clust, batch_size)
    sc_train, ch_train, db_train = get_visualizations(type, train_refs, dynamic_train_set,static_train_set, output_folder+str(n_clust)+'_clust_train', model,n_clust, batch_size)
    sc_test, ch_test, db_test=get_visualizations(type, val_refs,dynamic_val_set,static_val_set, output_folder+str(n_clust)+'_clust_test', model,n_clust,batch_size)
        #sc_train, ch_train, db_train =  get_visualizations(train_refs,dynamic_train_set, 'simple_AE_3_clust_train', model,3, batch_size)
        #sc_test, ch_test, db_test= get_visualizations(val_refs,dynamic_val_set, 'simple_AE_3_clust_test', model,3, batch_size)

    return train_loss, val_loss,sc_train
    #return loss.item()
    #history['train'].append(train_loss)
    #history['val'].append(val_loss)
    #print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

# Create an Optuna study
study = optuna.create_study(directions=['minimize', 'minimize', 'maximize'], sampler=optuna.samplers.TPESampler(seed=0))

# Run the optimization loop
study.optimize(objective, n_trials=25)

# Print the best hyperparameters found by Optuna
print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

for trial in study.best_trials:
    print(f"\tnumber: {trial.number}")
    print(f"\tparams: {trial.params}")
    print(f"\tvalues: {trial.values}")

#trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[1])
#print(f"Trial with highest accuracy: ")
#print(f"\tnumber: {trial_with_highest_accuracy.number}")
#print(f"\tparams: {trial_with_highest_accuracy.params}")
#print(f"\tvalues: {trial_with_highest_accuracy.values}")
#print('Best hyperparameters:', study.best_params)

"""
if __name__ == "__main__":
    type = sys.argv[1]
    #dynamic_features = ['progb', 'progUL', 'progLL', 'progT', 'progR', 'progRate3','ALSFRSb', 'ALSFRSsUL', 'ALSFRSsLL', 'R', 'ALSFRS-R', 'ALSFRSsT']
    #dynamic_features = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P8', 'P9', 'P10', 'P11', 'P12']
    dynamic_features = ['ALSFRSb', 'ALSFRSsUL', 'ALSFRSsLL', 'R', 'ALSFRS-R', 'ALSFRSsT' , 'MITOS-stage'] #progRate3 , 'MITOS-stage'
    static_features = ['Gender', 'BMI_at_1st_visit', 'Diagnostic Delay', 'Age_onset'] #'DiagnosticDelay', 'UMNvsLMN'
    data =  pd.read_csv('../../snapshots/no_nan_aipals_2023_snapshots_independent.csv')
    #data =  pd.read_csv('../../snapshots/comparison_aipals_2023_snapshots_independent.csv')
    #y_1 = pd.read_csv('temp_model6_clust_train/labels.csv')
    #y_2 = pd.read_csv('temp_model6_clust_test/labels.csv')
    #y = pd.concat([y_1, y_2], ignore_index=True)
    #data = get_merged_dataset("../../snapshots/lisbon_data_evolution_C1_90.csv","../../snapshots/turin_data_evolution_C1_90.csv", dynamic_features)
    #data = get_merged_dataset("../../snapshots/new_sc_lisbon_no_nan_snapshots_independent.csv","../../snapshots/new_sc_turin_no_nan_snapshots_independent.csv", dynamic_features)
    # Select ratio
    ratio = 0.7
    sil = []
    ch = []
    db = []
    tot_refs, data, static_data, y_true, fsg = get_3appointments(data, dynamic_features, static_features, type)
    total_rows = len(data)
    train_size = int(total_rows*ratio)
    #print(len(data))
    n_iter = 5
    n_runs = 0
    #train_refs, dynamic_train_set, static_train_set = get_3appointments(data, dynamic_features, static_features)
    #val_refs, dynamic_val_set, _ = get_n_appointments(data, dynamic_features, static_features)
    while n_runs <n_iter:

        
        #Split data into test and train
        #dynamic_val_set = data[train_size:]
        dynamic_train_set, dynamic_val_set, train_refs,val_refs = train_test_split(data,tot_refs, test_size=0.3)
        static_train_set, static_val_set, train_refs,val_refs = train_test_split(static_data,tot_refs, test_size=0.3)
        #dynamic_train_set = data[0:train_size]
        #static_train_set = static_data[0:train_size]
        #train_refs = tot_refs[0:train_size]
        #dynamic_val_set = data[train_size:]
        #static_val_set = static_data[train_size:]
        #val_refs= tot_refs[train_size:]
        if type == 'predictor' : 
            train_y_true =y_true[0:train_size]
            val_y_true = y_true[train_size:]
        if type == 'fsg':
            train_fsg_true =fsg[0:train_size]
            val_fsg_true = fsg[train_size:]
        
        #dynamic_train_set, dynamic_val_set, y_train, y_test = train_test_split(data, y['Labels'], test_size=0.3, random_state=42, stratify = y['Labels'])
        #static_train_set, static_val_set, y_train, y_test = train_test_split(static_data, y['Labels'], testtrain_test_split_size=0.3, random_state=42, stratify = y['Labels'])
        #train_y_set,val_y_set, y_train, y_test = (y_true, y['Labels'], test_size=0.3, random_state=42, stratify = y['Labels'])
        #train_fsg_set,val_fsg_set, y_train, y_test = train_test_split(fsg, y['Labels'], test_size=0.3, random_state=42, stratify = y['Labels'])
        #val_refs, dynamic_val_set, static_val_set = get_3appointments(test_data, dynamic_features, static_features)

        #train_refs, dynamic_train_set, static_train_set = get_3appointments(train_data, dynamic_features, static_features)
        
        #val_refs, dynamic_val_set, static_val_set = get_n_appointments(data, dynamic_features, static_features)
        
        #train_refs, dynamic_train_set, static_train_set = get_3appointments(train_data, dynamic_features, static_features)
        #val_refs, dynamic_val_set, static_val_set = get_3appointments(test_data, dynamic_features, static_features)

        #finetuning_refs, finetuning_data = get_finetuning_data(train_data, 'Onset', ['bulbar', 'generalized'], dynamic_features,static_features)
        #test_finetuning_refs, finetuning_test = get_finetuning_data(test_data, 'Onset', ['bulbar', 'generalized'], dynamic_features,static_features)
        n_features = len(dynamic_features)
        n_static_features = len(static_features)
        hidden_size = 2
        num_layers = 1
        seq_len = 3

        patience = 2
        trigger_times = 0

        bidirectional1 = True 
        bidirectional2 = False
        num_head = 4

        lr = 1e-4
        
        # Model Initialization
        if type == 'simple':
            model = LSTMAutoencoder(seq_len,n_features, hidden_size, num_layers, bidirectional1,bidirectional2,num_head)
            output_folder = 'temp_model_thesis_randm_'+ str(n_runs)
        elif type == 'temp_static':
            model = STAutoencoder(seq_len,n_features, hidden_size, num_layers, bidirectional1,bidirectional2,num_head,n_static_features)
            output_folder = 'temp_static_model_thesis'
            alpha = 0.5
        elif type == 'predictor':
            model = PredAutoencoder(seq_len,n_features, hidden_size, num_layers, bidirectional1,bidirectional2,num_head,n_static_features)
            output_folder = 'predictor_model'
            alpha = 0.5
            outcome_loss = torch.nn.BCELoss()
        elif type == 'fsg':
            output_folder = 'fsg_model_thesis'
            model = FSGAutoencoder(seq_len,n_features, hidden_size, num_layers, bidirectional1,bidirectional2,num_head)
            alpha = 0.5
            #outcome_loss = torch.nn.BCELoss()
            class_loss =torch.nn.BCELoss()

        #model = AttenLSTMAutoencoder(n_features, hidden_size, num_layers, dropout)
        # Validation using MSE Loss function
        loss_function = torch.nn.MSELoss()
        # Using an Adam Optimizer
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr = lr, weight_decay=0.001)
        epochs = 200
        history = dict(train=[], val=[])
        batch_size = 2
        n_clust = 4
        pretrain = False
        #print(train_set)
        start = time.time()

        if not pretrain:
            for epoch in range(epochs):
                model = model.train()
                train_losses = []
                for i in range(0, len(dynamic_train_set), batch_size):
                    dynamic_record = dynamic_train_set[i:i+batch_size]            
                    dynamic_record = torch.tensor(dynamic_record)

                    if type == 'temp_static' or type == 'predictor':
                        static_record = static_train_set[i:i+batch_size]
                        static_record = torch.tensor(static_record)                       

                    # The gradients are set to zero           
                    optimizer.zero_grad()
                    # Output of Autoencoder
                    if type == 'simple':
                        reconstructed = model(dynamic_record) 
                        loss = loss_function(reconstructed,dynamic_record)
                    elif type == 'temp_static':
                        dynamic_reconstructed, static_reconstructed  = model(dynamic_record,static_record)
                        dynamic_loss = loss_function(dynamic_reconstructed,dynamic_record)
                        static_loss = loss_function(static_reconstructed,static_record)
                        loss = alpha*dynamic_loss + (1-alpha)*static_loss
                    elif type == 'predictor':
                        y_true = torch.tensor(train_y_true[i:i+batch_size], dtype = torch.float32)
                        dynamic_reconstructed, static_reconstructed, outcome  = model(dynamic_record,static_record)
                        dynamic_loss = loss_function(dynamic_reconstructed,dynamic_record)
                        static_loss = loss_function(static_reconstructed,static_record)
                        loss = 0.05*(alpha*dynamic_loss + (1-alpha)*static_loss) +  0.95*outcome_loss(outcome, y_true)
                    elif type == 'fsg':
                        #y_true = torch.tensor(train_y_true[i:i+batch_size], dtype = torch.float32)
                        fsg_true = torch.tensor(train_fsg_true[i:i+batch_size], dtype = torch.float32)
                        dynamic_reconstructed, fsg  = model(dynamic_record)
                        dynamic_loss = loss_function(dynamic_reconstructed,dynamic_record)
                        #static_loss = loss_function(static_reconstructed,static_record)
                        #loss = (alpha*dynamic_loss + (1-alpha)*static_loss) +  class_loss(fsg,fsg_true)
                        loss = 0.95*dynamic_loss + 0.05*class_loss(fsg,fsg_true)
                    
                    loss.backward()
                    # .step() performs parameter update
                    optimizer.step()                     
                    # Storing the losses in a list for plotting
                    train_losses.append(loss.item())
                val_losses = []
                model = model.eval()
                with torch.no_grad():
                    for i in range(0, len(dynamic_val_set), batch_size):
                        dynamic_record = dynamic_val_set[i:i+batch_size]
                        dynamic_record = torch.tensor(dynamic_record)

                        if type == 'temp_static' or type =='predictor':
                            static_record = static_val_set[i:i+batch_size]
                            static_record = torch.tensor(static_record)

                        # Output of Autoencoder
                        if type == 'simple':
                            reconstructed = model(dynamic_record) 
                            loss = loss_function(reconstructed,dynamic_record)
                        elif type == 'temp_static':
                            dynamic_reconstructed, static_reconstructed  = model(dynamic_record,static_record)
                            dynamic_loss = loss_function(dynamic_reconstructed,dynamic_record)
                            static_loss = loss_function(static_reconstructed,static_record)
                            loss = alpha*dynamic_loss + (1-alpha)*static_loss
                        elif type == 'predictor':
                            y_true = torch.tensor(val_y_true[i:i+batch_size], dtype = torch.float32)
                            dynamic_reconstructed, static_reconstructed, outcome  = model(dynamic_record,static_record)
                            dynamic_loss = loss_function(dynamic_reconstructed,dynamic_record)
                            static_loss = loss_function(static_reconstructed,static_record)
                            
                            loss = 0.05*(alpha *dynamic_loss + (1-alpha)*static_loss) +  0.95*outcome_loss(outcome, y_true)
                        elif type == 'fsg':
                            #y_true = torch.tensor(train_y_true[i:i+batch_size], dtype = torch.float32)
                            fsg_true = torch.tensor(train_fsg_true[i:i+batch_size], dtype = torch.float32)
                            dynamic_reconstructed, fsg  = model(dynamic_record)
                            dynamic_loss = loss_function(dynamic_reconstructed,dynamic_record)
                            #static_loss = loss_function(static_reconstructed,static_record)
                            #loss = (alpha*dynamic_loss + (1-alpha)*static_loss) +  class_loss(fsg,fsg_true)
                            loss = 0.95*dynamic_loss + 0.05*class_loss(fsg,fsg_true)
                        val_losses.append(loss.item())

                train_loss = np.mean(train_losses)
                val_loss = np.mean(val_losses)

                #early stop
                if epoch > 1 and history['val'][-1] < val_loss:
                    trigger_times += 1
                    print('Trigger Times:', trigger_times)
                else:
                    trigger_times = 0
                if trigger_times >= patience:
                    print('Early stopping!')
                    break
                    
                    
                history['train'].append(train_loss)
                history['val'].append(val_loss)
                print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

            EPOCH = epoch
            PATH = "model.pt"
            LOSS = train_loss

            torch.save({
                        'epoch': EPOCH,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': LOSS,
                        }, PATH)


        else:
            checkpoint = torch.load('model.pt')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            trigger_times = patience

        """

        trigger_times = 0
        print("FINETUNING!")
        #batch_size = 3
        print(len(finetuning_data))
        print(len(finetuning_test))
        #finetune

        for epoch in range(epochs):
            model = model.train()
            train_losses = []
            for i in range(0, len(finetuning_data), batch_size):
                dynamic_record = finetuning_data[i:i+batch_size]            
                dynamic_record = torch.tensor(dynamic_record)

                # The gradients are set to zero           
                optimizer.zero_grad()
                # Output of Autoencoder
                reconstructed = model(dynamic_record) 
                # Calculating the loss function
                #encoded,_ = model.encoder(dynamic_record)
                #decoded = model.decoder(encoded)
                #layer_losses = [optimizer(layer1(decoded), layer2(dynamic_record)) for layer1,layer2 in zip(decoder_layers,encoder_layers)]
                #loss = sum(layer_losses)
                loss = loss_function(reconstructed,dynamic_record)
                loss.backward()
                # .step() performs parameter update
                optimizer.step()                     
                # Storing the losses in a list for plotting
                train_losses.append(loss.item())
            val_losses = []
            model = model.eval()
            with torch.no_grad():
                for i in range(0, len(finetuning_test), batch_size):
                    dynamic_record = finetuning_test[i:i+batch_size]
                    dynamic_record = torch.tensor(dynamic_record)

                    reconstructed = model(dynamic_record)
                    #encoded,_ = model.encoder(dynamic_record)
                    #decoded = model.decoder(encoded)
                    #layer_losses = [optimizer(layer1(decoded), layer2(dynamic_record)) for layer1,layer2 in zip(nn.Sequential(model.decoder),model.encoder)]
                    #loss = sum(layer_losses)
                    
                    # Calculating the loss function
                    loss = loss_function(reconstructed,dynamic_record)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            #early stop
            if epoch > 1 and abs(history['val'][-1] - val_loss) < 0.001 : #0.000001:
                trigger_times += 1
                print('Trigger Times:', trigger_times)
            else:
                trigger_times = 0
            if trigger_times >= patience:
                print('Early stopping!')
                break
            history['train'].append(train_loss)
            history['val'].append(val_loss)
            print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
        """

        #output_folder = '6app_experiment'
        #output_folder = 'all_features' 
        #output_folder = 'nearest_centroids'
        if epoch == 199 or trigger_times == patience:
            try: 
                os.mkdir(output_folder+str(n_clust)+'_clust_train')               
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
            try: 
                os.mkdir(output_folder+str(n_clust)+'_clust_test')               
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        
            #get_visualizations(train_refs,dynamic_train_set, 'all_features_AE_3_clust_train', model,n_clust, batch_size)
            #clustering = GaussianMixture(n_components=n_clust)
        #clustering = NearestCentroid()
        #y_train = pd.read_csv('6app_experiment4_clust_train/labels.csv')
        #clustering = get_visualizations(clustering, train_refs,dynamic_train_set, output_folder+str(n_clust)+'_clust_train', model,n_clust, batch_size, True, y_train['Labels'])
        #get_visualizations(clustering, val_refs,dynamic_val_set, output_folder+str(n_clust)+'_clust_test', model,n_clust,batch_size,False,y_train['Labels'])
        #y_test =pd.read_csv('all_features4_clust_test/labels.csv')
        #get_visualizations(finetuning_refs, finetuning_data, 'prog'+str(n_clust)+'_clust_train', model,n_clust, batch_size)
        #get_visualizations(test_finetuning_refs, finetuning_test, 'prog'+str(n_clust)+'_clust_test', model,n_clust,batch_size)

        ss,ch_s,db_s = get_visualizations(type, train_refs, dynamic_train_set,static_train_set, output_folder+str(n_clust)+'_clust_train', model,n_clust, batch_size)
        get_visualizations(type, val_refs,dynamic_val_set, static_val_set,output_folder+str(n_clust)+'_clust_test', model,n_clust,batch_size)
        end = time.time()
        print('TIME: ', end-start)
        if ss == 0.0:
            continue
        n_runs +=1
        sil.append(ss)
        ch.append(ch_s)
        db.append(db_s)
        print(n_runs)
        #nearest_centroid(output_folder+str(n_clust)+'_clust_test', dynamic_train_set, y_train, dynamic_val_set, y_test,static_train_set, static_val_set, type)

    ax = plt.figure().gca()
    ax.plot(history['train'])
    ax.plot(history['val'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train', 'test'])
    plt.savefig(output_folder+str(n_clust)+'_clust_train/loss.pdf')

        #plt.show()

    print('Sihlouette: %f +\- %f', np.mean(sil), np.std(sil))
    print('CH: {}%f +\- {}%f', np.mean(ch), np.std(ch))
    print('DB: {}%f +\- {}%f', np.mean(db), np.std(db))


    

