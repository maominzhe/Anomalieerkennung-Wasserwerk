import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#https://scikit-learn.org/stable/modules/mixture.html
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
#%matplotlib inline
from matplotlib.patches import Ellipse
from sklearn.metrics import confusion_matrix
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import math
from SPS_classify import *
# !/usr/bin/env python
# coding: utf-8

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

    dataToSend = pd.read_csv("SPS_Daten.csv", names=data_spaltennamen, index_col=False)
    dataToSend['Timedelta'] = dataToSend['Time'].diff()
    dataToSend = SPS_classify(dataToSend)
    row_count = len(dataToSend)
    dataToSend.loc[0:3459, 'Anomaly'] = 0
    dataToSend.loc[3459:5126, 'Anomaly'] = 1
    dataToSend.loc[5126:6959, 'Anomaly'] = 0
    dataToSend.loc[6959:9593, 'Anomaly'] = 1
    dataToSend.loc[9593:10127, 'Anomaly'] = 0
    dataToSend.loc[10127:11837, 'Anomaly'] = 1
    dataToSend.loc[11837:row_count, 'Anomaly'] = 0
    dataToSend = dataToSend.iloc[100:]
    dataToSend = dataToSend.reset_index(drop=True)

    df = dataToSend

    df_0 = df[df.Anomaly == 0]  # Dataset with non-fraudulent only
    df_1 = df[df.Anomaly == 1]  # Dataset with fraudulent only
    df_0 = df_0[data_spaltennamen]  # Select two most correlated features for now
    df_1 = df_1[data_spaltennamen]

    # Split non-anomaly data in 90% for training GMM and 10% for cross-validation and testing
    X_0, X_test_0, y_train_0, y_test_0 = train_test_split(df_0.drop(['Anomaly'], axis=1),
                                                          df_0['Anomaly'], test_size=0.1, random_state=0)

    X_train_0, X_val, y_train, y_val = train_test_split(X_0, y_train_0, test_size=0.3, random_state=0)

    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(df_1.drop(['Anomaly'], axis=1),
                                                                df_1['Anomaly'], test_size=0.5, random_state=0)

    X_test_all = np.vstack([X_test_0, X_test_1])

    y_test_all = np.hstack([y_test_0, y_test_1])

    # y_test_all = df['Anomaly']

    f_length = len(data_spaltennamen) - 1

    features = data_spaltennamen

    return df, X_train_0, X_val, X_test_all, y_train, y_test_all, f_length, features


# def import_data(row_start=0, row_end=0, filter_features=[], test_size=0.3):
df, X_train_0, X_val, X_test_all, y_train, y_test_all, f_length, features = import_data(0, 0, 0, [])

##############################################
#training the model
##############################################

standard_scalar = StandardScaler()
X_train_0 = standard_scalar.fit_transform(X_train_0)
pca = PCA() #n_components=0.99

transformed_data = pca.fit_transform(X_train_0)
x_train_principal = pd.DataFrame(data=transformed_data)
d = transformed_data

##############################################
#GMM process
##############################################

GMM = GaussianMixture(n_components=1) #n_components=1
GMM.fit(x_train_principal.values)
features = features[0:16]
X_test_all = standard_scalar.transform(X_test_all)
newDataPrincipalComponents = pca.transform(X_test_all)
principalDf = pd.DataFrame(data=newDataPrincipalComponents)
data_score= GMM.score_samples(principalDf)

##############################################
#setting different T-werte and save the predicted labels
##############################################
T_1 = np.percentile(data_score, 1, axis=0, keepdims=True)
T_2 = np.percentile(data_score, 60, axis=0, keepdims=True)
T_3 = np.percentile(data_score, 70, axis=0, keepdims=True)
T_4 = np.percentile(data_score, 75, axis=0, keepdims=True)
T_5 = np.percentile(data_score, 80, axis=0, keepdims=True)
T_6 = np.percentile(data_score, 90, axis=0, keepdims=True)
T_7 = np.percentile(data_score, 100, axis=0, keepdims=True)
labels_pred_1 = data_score< T_1
labels_pred_2 = data_score< T_2
labels_pred_3 = data_score< T_3
labels_pred_4 = data_score< T_4
labels_pred_5 = data_score< T_5
labels_pred_6 = data_score< T_6
labels_pred_7 = data_score< T_7



#anomaly_index = np.argwhere(labels_pred == True)


a1 = confusion_matrix(y_test_all, labels_pred_1)
a2 = confusion_matrix(y_test_all, labels_pred_2)
a3 = confusion_matrix(y_test_all, labels_pred_3)
a4 = confusion_matrix(y_test_all, labels_pred_4)
a5 = confusion_matrix(y_test_all, labels_pred_5)
a6 = confusion_matrix(y_test_all, labels_pred_6)
a7 = confusion_matrix(y_test_all, labels_pred_7)

tpr_1 = a1[1,1]/(a1[1,0]+a1[1,1])
tpr_2 = a2[1,1]/(a2[1,0]+a2[1,1])
tpr_3 = a3[1,1]/(a3[1,0]+a3[1,1])
tpr_4 = a4[1,1]/(a4[1,0]+a4[1,1])
tpr_5 = a5[1,1]/(a5[1,0]+a5[1,1])
tpr_6 = a6[1,1]/(a6[1,0]+a6[1,1])
tpr_7 = a7[1,1]/(a7[1,0]+a7[1,1])

fpr_1 = a1[0,1]/(a1[0,0]+a1[0,1])
fpr_2 = a2[0,1]/(a2[0,0]+a2[0,1])
fpr_3 = a3[0,1]/(a3[0,0]+a3[0,1])
fpr_4 = a4[0,1]/(a4[0,0]+a4[0,1])
fpr_5 = a5[0,1]/(a5[0,0]+a5[0,1])
fpr_6 = a6[0,1]/(a6[0,0]+a6[0,1])
fpr_7 = a7[0,1]/(a7[0,0]+a7[0,1])

tpr = np.array([tpr_7,tpr_6,tpr_5,tpr_4,tpr_3,tpr_2,tpr_1])
fpr = np.array([fpr_7,fpr_6,fpr_5,fpr_4,fpr_3,fpr_2,fpr_1])

plt.figure()
lw = 2
roc_auc = auc(fpr,tpr) #auc berechnen
plt.plot(fpr, tpr, color='darkorange', lw=lw, marker = 'o', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

#######################
#f1_score
#######################

from sklearn.metrics import f1_score

f1_1 = f1_score(y_test_all, labels_pred_1)
f1_2 = f1_score(y_test_all, labels_pred_2)
f1_3 = f1_score(y_test_all, labels_pred_3)
f1_4 = f1_score(y_test_all, labels_pred_4)
f1_5 = f1_score(y_test_all, labels_pred_5)
f1_6 = f1_score(y_test_all, labels_pred_6)
f1_7 = f1_score(y_test_all, labels_pred_7)