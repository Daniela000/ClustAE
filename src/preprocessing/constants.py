from pyexpat import model
import yaml as yy
from yaml.loader import Loader


def get_config(config_file):
    s = open(config_file, 'r')
    cfs = yy.load(s, Loader=Loader)
    globals().update(cfs)

    global BASELINE_DIR_S_train
    global BASELINE_DIR_T_train
    global TRAJECTORY_DIR_train
    global VISUALIZATION_DIR_train
    global LABELS_DIR_train
    global BASELINE_DIR_S_test
    global BASELINE_DIR_T_test
    global TRAJECTORY_DIR_test
    global VISUALIZATION_DIR_test
    global LABELS_DIR_test
    global TOP_FOLDER
    global DATA_FILE
    global MIN_APP
    global SNAPSHOTS_FILE
    global N_CLUST
    global REF_FEATURE
    global MODEL
    global PRETRAIN

    BASELINE_DIR_S_train = TOP_FOLDER + "train/baselines/static/"
    BASELINE_DIR_T_train = TOP_FOLDER + "train/baselines/temporal/"

    BASELINE_DIR_S_test = TOP_FOLDER + "test/baselines/static/"
    BASELINE_DIR_T_test = TOP_FOLDER + "test/baselines/temporal/"

    LABELS_DIR_train = TOP_FOLDER + "train/results/"
    TRAJECTORY_DIR_train = TOP_FOLDER + "train/results/trajectories/"
    VISUALIZATION_DIR_train = TOP_FOLDER + "train/results/visualization/"

    LABELS_DIR_test = TOP_FOLDER + "test/results/"
    TRAJECTORY_DIR_test = TOP_FOLDER + "test/results/trajectories/"
    VISUALIZATION_DIR_test = TOP_FOLDER + "test/results/visualization/"
    