#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from SPS_classify import *



# --------------------------
#
# Purpose:
# sending the Trainingsdata to the ML Algorithm
# --------------------------

# row_end is set on the beginning if default to 1, but bevor its used its going to be set on the right value
def import_data(row_start=0, row_end=0, test_size=0.3, filter_features=[]):
    data_spaltennamen = ['Time',  # 0
                         'Volumen_Tank_1',  # 1
                         'Temperatur_Tank_1',  # 2
                         'Druck_nach_Pumpe_1',  # 3
                         'Volumenstrom_nach_Pumpe_1',  # 4
                         'Druck_nach_Pumpe_2',  # 5
                         'Volumenstrom_nach_Pumpe_2',  # 6
                         'Volumenstrom_nach_Rohr_1',  # 7
                         'Druck_nach_Rohr_1',  # 8
                         'Druck_nach_Rohr_2',  # 9
                         'Volumenstrom_nach_Rohr_2',  # 10
                         'Volumen_Tank_2',  # 11
                         'Temperatur_Tank_2',  # 12
                         'Wasserlevel_Tank_1',  # 13
                         'Wasserlevel_Tank_2',  # 14
                         'Druck_vor_Pumpe_1',  # 15
                         'Druck_vor_Pumpe_2',  # 16
                         'Timedelta',
                         'Anomaly',
                         'SPS_druck',
                         'SPS_fluss',
                         'SPS_vol',
                         'SPS_temp',
                         'SPS_Anomaly']

    data = pd.read_csv("Werte_Fluidsimulation_4000s.csv", names=data_spaltennamen, header=None) # read in the training data

    if (row_end == 0):
        row_end = data.shape[0]

    # if specific features are asked, only return them, otherwise give the whole Dataframe back
    data = data.iloc[row_start:row_end]
    data = SPS_classify(data)
    # Add the timedelta
    data['Timedelta'] = data['Time'].diff()
    #assign a true label of 0 to all the training data
    data['Anomaly'] = 0
    # Drop the phase of transient oscillation
    data = data.iloc[100:]

    features = []
    #filter the features if required
    if (len(filter_features) != 0):
        for filter_features_int in filter_features:
            features.append(data_spaltennamen[filter_features_int])
        data = data[data['Anomaly'] == 0].loc[:, features]  # finds all features
    else:
        features = []
        data = data[data['Anomaly'] == 0]

    df = data

    X, X_test, Y, y_test = train_test_split(df, df['Anomaly'], test_size=0.9, random_state=0)
    X = X.drop(['Anomaly'], axis=1)

    f_length = len(data_spaltennamen) - 1
    features = data_spaltennamen

    return df, X, Y, f_length, features


df, X, Y, f_length, features = import_data(0, 0, 0, [])