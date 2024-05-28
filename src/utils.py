import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import errno
import pacmap
import preprocessing.constants as constants
from pathlib import Path
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
import csv
from sklearn.metrics import confusion_matrix
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
from s_dbw import S_Dbw

def parse_data(data_file, trajectory_dir, visualization_dir):
    try:
        n = constants.MIN_APP
        snapshots_name = data_file # snapshots file

        Path(trajectory_dir).mkdir(parents=True, exist_ok=True) 
        Path(visualization_dir).mkdir(parents=True, exist_ok=True) 
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

    patients = pd.read_csv(snapshots_name, low_memory=False)
    patients.replace(" ", np.nan, inplace=True)
    #evolution = patients['Evolution']
    patients[list(constants.TEMPORAL_FEATURES.keys())] =patients[list(constants.TEMPORAL_FEATURES.keys())].astype(float)
    # remove patietns with less than MIN_APP appointments
    #counts = patients[constants.REF_FEATURE].value_counts()
    #mask = counts >= constants.MIN_APP
    #filtered_patients = patients[patients[constants.REF_FEATURE].isin(counts[mask].index)]
    #filtered_patients = patients[patients[constants.REF_FEATURE].isin(train_refs)]
    #filtered_patients = filtered_patients.groupby(constants.REF_FEATURE).first().reset_index()

    return patients

def smote(X,y):
    print('BEFORE SMOTE: ', len(X))
    counter = Counter(y)
    print(counter)

    sm = sm = SMOTE(sampling_strategy = 0.8, k_neighbors = 4, random_state=0)
    X_res, y_res = sm.fit_resample(X, y)

    print('After SMOTE: ', len(X_res))
    counter = Counter(y_res)
    print(counter)

    return X_res, y_res

def random_undersample(X, y):
    nY = pd.Series(y).value_counts()['Y']
    nN = pd.Series(y).value_counts()['N']
    print("Before RU\n", pd.Series(y).value_counts())
    #if nY/nN < 0.6:
    rus = RandomUnderSampler(sampling_strategy = 0.6, random_state=0)
    X, y = rus.fit_resample(X, y)
    return X, y

def get_color_list():
    colormap = plt.cm.get_cmap('rainbow')
    colors = [colormap(i/constants.N_CLUST ) for i in range(constants.N_CLUST )]
    return colors

def tsne(tempData, labels, dir):
    """
    Computes the tsne dimensionality reduction 

    Parameters
    ----------
    tempData: data to plot
    labels: labels of each patient in data
    output_name: name of the output folder
    """
    colors = get_color_list()
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(tempData.values)

    new = tempData.copy()
    new['tsne-2d-one'] = X_2d[:,0]
    new['tsne-2d-two'] = X_2d[:,1]
    fig = plt.figure(figsize=(16,10))
    sns.scatterplot(
        x = "tsne-2d-one", y = "tsne-2d-two",
        hue = labels,
        palette = colors,
        data = new,
        legend = "full"
)
    fig.savefig(dir + 'tsne.pdf')

def pacmap_func(tempData, labels, dir):
    """
    Computes the pacmap dimensionality reduction 

    Parameters
    ----------
    tempData: data to plot
    labels: labels of each patient in data
    output_name: name of the output folder
    """
    colors = get_color_list()
    embedding = pacmap.PaCMAP(n_components=2, random_state=0)
    X_2d = embedding.fit_transform(tempData.values)

    new = tempData.copy()
    new['pacmap-2d-one'] = X_2d[:,0]
    new['pacmap-2d-two'] = X_2d[:,1]

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x = "pacmap-2d-one", y = "pacmap-2d-two",
        hue = labels,
        palette = colors,
        data = new,
        legend = "full"
    )

    plt.savefig(dir + 'pacmap.pdf')

def hierarchical_clustering(data, dir):
    """
    Computes the agglomerative clustering 

    Parameters
    ----------
    data: data to cluster
    """
    plt.ylabel('distance')
    clusters = shc.linkage(data, method="ward", metric="euclidean")
    print(clusters)
    with open(constants.TOP_FOLDER +"dendrogram_distances.csv", "w") as f:
        wr = csv.writer(f)
        wr.writerows(clusters)

    shc.dendrogram(clusters, p = 20, truncate_mode = 'lastp', # show only the last p merged clusters
                show_leaf_counts = False) 
    plt.gcf()
    plt.savefig(dir + '/dendrogram.pdf')

    Ward_model = AgglomerativeClustering(n_clusters= constants.N_CLUST, metric='euclidean', linkage='ward')
    Ward_model.fit(data)

    cluster_labels = Ward_model.fit_predict(data)

    
    shc.dendrogram(clusters,p= 20, truncate_mode ='lastp',show_leaf_counts=False, leaf_font_size=4)
    leaves = shc.dendrogram(clusters, no_plot = True)['leaves']

    if max(leaves) >= len(data):
        # Remove out-of-bounds leaves
        leaves = [i for i in leaves if i < len(data)]

    ordered_labels = cluster_labels[leaves]
    for i in range(1, len(ordered_labels)):
        if ordered_labels[i] != ordered_labels[i-1]:
            print(ordered_labels[i] )
         
    print(ordered_labels)

    print('Silhouette Score: ', silhouette_score(data, Ward_model.labels_, metric='euclidean'))
    print('Calinski Harabasz Score: ', calinski_harabasz_score(data, Ward_model.labels_))
    print('Davies Bouldin Score: ', davies_bouldin_score(data, Ward_model.labels_))
    print('IDC: ', clusters[(constants.N_CLUST - 1)*(-1), 2] - clusters[constants.N_CLUST*(-1), 2])
    print('sdbw: ', S_Dbw(data, Ward_model.labels_))

    return Ward_model.labels_


def slope(x1, y1, x2, y2):
    m = (y2-y1)/(x2-x1)
    return m

def format_mogp_axs(ax, max_x=8, x_step=1.0, y_label=[0,24,48], y_minmax=(-3, 53)):
    ax.set_xlim([0, max_x])
    ax.set_xticks(np.arange(0, max_x + 1, x_step))
    ax.set_yticks(y_label)
    ax.set_ylim(y_minmax)
    return ax


def simple_trajectories(clusters, trajectory_dir):
    """
    Computes the trajectories of each clustering in the temporal features

    Parameters
    ----------
    clusters: list of n_clust lists with each list comprising the snapshots of the patients in the corresponding cluster
    """
    features = ['ALSFRSb', 'ALSFRSsUL', 'ALSFRSsLL', 'R', 'ALSFRS-R', 'ALSFRSsT' , 'MITOS-stage']
    colors = get_color_list()
    #colors_1 = [colors[1], '#A52A2A', colors[3], '#E759AC']
    #print(clusters)
    #for feature in list(constants.TEMPORAL_FEATURES.keys()):
    for feature in features:
        plt.clf()
        fig, ax = plt.subplots()

        max_val =0
        for j in range(constants.N_CLUST):
            prg = clusters[j].groupby(constants.REF_FEATURE)[feature]           

            lst = []
            for _, group in prg:
                lst.append(group.values)
            
            #transposed_bad = [list(filter(None,i)) for i in zip_longest(*lst)]
            #max_length = max(len(inner) for inner in lst)
            max_len = max(len(i) for i in lst)
            transposed = [[i[o] for i in lst if len(i) > o] for o in range(max_len)]
            #transposed = [[inner[i] if i < len(inner) else None for inner in lst] for i in range(max_length)]
            n_samples = []
  
            if feature == 'ALSFRS-R':
                means = np.array(48)
                ci = np.array(0)
                max_val = 48
            elif feature == 'ALSFRSsUL' or feature == 'ALSFRSsLL' or feature == 'ALSFRSsT':
                #print(feature)
                means = np.array(8)
                ci = np.array(0)
                max_val = 8
            elif feature == 'ALSFRSb' or feature == 'R':
                means = np.array(12)
                ci = np.array(0)
                max_val = 12                
            else:
                means = np.array(0)
                ci = np.array(0)
                
            for values in transposed:

                #if j == 1 and feature == 'MITOS-stage':
                    #print(n_samples)
                    #print(values)
                if values != []:
                    if max_val < np.nanmax(values):
                        max_val = np.nanmax(values)
                    n_samples.append(len(values))
                    ci = np.append(ci, 1.96 * np.nanstd(values)/np.sqrt(len(values)))
                    if feature != 'MITOS-stage':
                       
                        #print(values)
                        means = np.append(means,    np.nanmean(values))
                    else:

                        #print(len(values))
                        means = np.append(means,    np.nanmedian(values))
            app = np.arange(0,len(means))
            #num_pat = 'n = {}'.format(len(clusters[j].groupby('REF')))
            
            ax.plot(app, means, marker = '.', color = colors[j], label = 'Cluster ' + str(j+1))
            ax.fill_between(app, (means-ci), (means+ci), color=colors[j], alpha=.1)

        
            slope_value=[]
            for i in range(1,6):
                v=slope(i-1,means[i-1], i, means[i])
                slope_value.append(v)
                plt.text( i-0.5 , (means[i] + means[i-1])/2, str(round(v,2)), fontsize=5, color = colors[j])
                
                plt.text(i, means[i] + 0.001, str(n_samples[i-1]), fontsize=8, color = colors[j], fontweight= 'bold')

        format_mogp_axs(ax, 5, 1, y_label=[0,max_val/2,max_val], y_minmax = (0, max_val+1))

        plt.xlabel("Appointments")
        plt.ylabel(str(feature))
        plt.legend()
        fig.savefig(trajectory_dir + str(feature) + '.pdf')
        plt.close()

def box_plot(data):
    first_app = data.groupby(constants.REF_FEATURE).first().reset_index()
    third_app = data.groupby(constants.REF_FEATURE).nth(2).reset_index()
    columns = ['ALSFRS-R', 'ALSFRSsUL', 'ALSFRSsLL', 'ALSFRSsT', 'ALSFRSb', 'R']

    plt.clf()
    ax = plt.axes()
    first_app.boxplot(column = columns)
    ax.set_title("Lisbon Boxplot 1st App")
    plt.savefig(constants.VISUALIZATION_DIR + '1st_app_distribution_proact.pdf')
    plt.clf()
    ax = plt.axes()
    third_app.boxplot(column = columns)
    ax.set_title("Lisbon Boxplot 3rd App")
    plt.savefig(constants.VISUALIZATION_DIR + '3rd_app_distribution_proact.pdf')
        

def classifier(x_train,x_test, y_train):
    #def nearest_centroid(X_train, y_train, X_test):
    #clf = RandomForestClassifier(random_state=0)
    clf = NearestCentroid()
    clf.fit(x_train, y_train)
    labels = clf.predict(x_test)
    joblib.dump(clf, "./simple_nearest_centroids_original.joblib")
    print('Accuracy: ', clf.score(x_train, y_train))
    
    print('Silhouette Score: ', silhouette_score(x_test, labels, metric='euclidean'))
    print('Calinski Harabasz Score: ', calinski_harabasz_score(x_test, labels))
    print('Davies Bouldin Score: ', davies_bouldin_score(x_test, labels))
    print('sdbw: ', S_Dbw(x_test, labels))
    return labels


def test_classification(x_train,x_test, y_train):
    labels = []

    for point in x_test:
        # Calculate distances using NumPy's broadcasting
        distances = np.sqrt(np.sum((x_train - point)**2, axis=1))
    
        # Get the index of the smallest distance
        nearest_index = np.argmin(distances)
        labels.append(y_train[nearest_index])
    #print('Accuracy: ', clf.score(x_train, y_train))
    
    print('Silhouette Score: ', silhouette_score(x_test, labels, metric='euclidean'))
    print('Calinski Harabasz Score: ', calinski_harabasz_score(x_test, labels))
    print('Davies Bouldin Score: ', davies_bouldin_score(x_test, labels))
    return labels

def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]


def sens(y_true, y_pred): return tp(y_true, y_pred) / \
    (fn(y_true, y_pred) + tp(y_true, y_pred))


def spec(y_true, y_pred): return tn(y_true, y_pred) / \
    (fp(y_true, y_pred) + tn(y_true, y_pred))
