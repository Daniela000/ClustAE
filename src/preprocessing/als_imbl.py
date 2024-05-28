from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC
from sklearn import preprocessing
import pandas as pd
import numpy as np
import constants


def random_undersample(X, y):
    nY = pd.Series(y).value_counts()['Y']
    nN = pd.Series(y).value_counts()['N']
    print("Before RU\n", pd.Series(y).value_counts())
    if nY/nN < 0.5:
        rus = RandomUnderSampler(random_state=0, sampling_strategy=0.5)
        X, y = rus.fit_resample(X, y)
    return X, y


def smote(X, y, n, disc=False, bic=False):
    print("Before SMOTE\n", pd.Series(y).value_counts())

    if not disc and not bic:
        f_categorical_ix = list(map(lambda f: list(constants.TEMPORAL_FEATURES.keys()).index(f[0]), filter(
            lambda x: x[1] == 'categorical', constants.TEMPORAL_FEATURES.items())))
        f_categorical_ix_ini = list(map(lambda y: y + 1, f_categorical_ix))
        f_categorical_ix = []
        for ix in range(0, n):
            f_categorical_ix += list(map(lambda y: y +
                                     len(constants.TEMPORAL_FEATURES)*ix, f_categorical_ix_ini))
    if bic:
        f_categorical_ix = list(map(lambda f: list(constants.STATIC_FEATURES.keys()).index(f[0]), filter(
            lambda x: x[1] == 'categorical', constants.STATIC_FEATURES.items())))
        f_categorical_ix = list(map(lambda y: y + 1, f_categorical_ix))
    f_categorical_ix = [0] + f_categorical_ix

    sm = SMOTENC(random_state=2,
                 categorical_features=f_categorical_ix, k_neighbors=3)

    try:
        X_res, y_res = sm.fit_resample(X, y)
    except:
        X_res, y_res = X, y

    return X_res, y_res


def label_encoder(wri):
    le = preprocessing.LabelEncoder()
    le.fit(wri["0El Escorial reviewed criteria"])
    wri["0El Escorial reviewed criteria"] = list(
        map(lambda a: a+1, le.transform(wri["0El Escorial reviewed criteria"])))
    # print("EERC", le.inverse_transform([3,5,6]))

    le = preprocessing.LabelEncoder()
    le.fit(wri["0UMN vs LMN"])
    wri["0UMN vs LMN"] = list(
        map(lambda a: a+1, le.transform(wri["0UMN vs LMN"])))
    # print("UMNLM",le.inverse_transform([1]))

    le = preprocessing.LabelEncoder()
    le.fit(wri["0Onset form"])
    wri["0Onset form"] = list(
        map(lambda a: a+1, le.transform(wri["0Onset form"])))
    # print("Onset",le.inverse_transform([1,4]))

    le = preprocessing.LabelEncoder()
    le.fit(wri["0C9orf72"])
    wri["0C9orf72"] = list(map(lambda a: a+1, le.transform(wri["0C9orf72"])))
    # print("C9oerf", le.inverse_transform([1,2]))
    return wri


def label_encoder_als(wri, features):
    le = preprocessing.LabelEncoder()
    for f in features:
        le.fit(wri[f])
        wri[f] = list(map(lambda a: a+1, le.transform(wri[f])))
    return wri


def one_hot_encode(wri):
    for feature in list(constants.STATIC_FEATURES.keys()):
        column0 = '0' + feature
        ohe = pd.get_dummies(wri[column0], prefix=feature, dummy_na=False)
        wri = pd.concat([wri, ohe], axis=1)
        wri.drop([column0], axis=1, inplace=True)

    return wri
