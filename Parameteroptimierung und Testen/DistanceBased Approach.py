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
from sklearn.metrics import confusion_matrix
import math

# !/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import train_test_split
from SPS_classify import *
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

def classifier_1(major_components, minor_components):
    major = major_components > c1_1
    minor = minor_components > c2_1
    #minor = 0
    return np.logical_or(major,minor)
def classifier_2(major_components, minor_components):
    major = major_components > c1_2
    minor = minor_components > c2_2
    #minor = 0
    return np.logical_or(major,minor)
def classifier_3(major_components, minor_components):
    major = major_components > c1_3
    minor = minor_components > c2_3
    #minor = 0
    return np.logical_or(major,minor)
def classifier_4(major_components, minor_components):
    major = major_components > c1_4
    minor = minor_components > c2_4
    #minor = 0
    return np.logical_or(major,minor)
def classifier_5(major_components, minor_components):
    major = major_components > c1_5
    minor = minor_components > c2_5
    #minor = 0
    return np.logical_or(major,minor)
def classifier_6(major_components, minor_components):
    major = major_components > c1_6
    minor = minor_components > c2_6
    #minor = 0
    return np.logical_or(major,minor)
def classifier_7(major_components, minor_components):
    major = major_components > c1_7
    minor = minor_components > c2_7
    #minor = 0
    return np.logical_or(major,minor)
def distance_cal(np):
    lambdas = pca.singular_values_
    lambdas[lambdas==0]=1e-40
    M = np * np / lambdas
    return M

# define classifier
def classifier(major_components, minor_components):
    major = major_components > c1_1
    minor = minor_components > c2_1
   # majoc = 0
    return np.logical_or(major,minor)

#PCA process

standard_scalar = StandardScaler()
# centered_training_data = standard_scalar.fit_transform(x_train)
X_train = standard_scalar.fit_transform(X_val)
X_test_all_fit = standard_scalar.transform(X_test_all)
pca = PCA()
#pca.fit(centered_training_data)
pca.fit(X_train)
#x_train = standard_scalar.fit_transform(centered_training_data)
transformed_data = pca.fit_transform(X_train)
x_train_principal = pd.DataFrame(data = transformed_data)
d = transformed_data

#######################
# distance calculation
#######################

M = distance_cal(d) # M is the Mahalanobis distance
# we take the first q components for calculation
q = 1
# calculate the threshold c1
major_components = M[:,range(q)]
major_components_distance = np.sum(major_components, axis=1)
s = sum(pca.explained_variance_ratio_[:q])
print("Explained variance by first " + str(q) +" terms:" + str(s))

#looking for the least components
r = list(pca.explained_variance_ratio_ < 0.2)  # determine minor components, who account for less than 50 % of the information
minor_components = M[:, r]
minor_components_distance = np.sum(minor_components, axis=1)  # calculate the distance of the minor components
#define the two thresholds.
components = pd.DataFrame({'major_components': major_components_distance,
                               'minor_components': minor_components_distance})
c1_1 = components.quantile(0.999)['major_components']
c2_1 = components.quantile(0.999)['minor_components']

c1_2 = components.quantile(0.99)['major_components']
c2_2 = components.quantile(0.99)['minor_components']

c1_3 = components.quantile(0.95)['major_components']
c2_3 = components.quantile(0.95)['minor_components']

c1_4 = components.quantile(0.9)['major_components']
c2_4 = components.quantile(0.9)['minor_components']

c1_5 = components.quantile(0.7)['major_components']
c2_5 = components.quantile(0.7)['minor_components']

c1_6 = components.quantile(0.5)['major_components']
c2_6 = components.quantile(0.5)['minor_components']

c1_7 = components.quantile(0)['major_components']
c2_7 = components.quantile(0)['minor_components']

print(c1_1)
print(c2_1)
########################
#apply the testing data
########################

transformed_data_test = pca.transform(X_test_all_fit)
x_test_principalDf = pd.DataFrame(data = transformed_data_test)
d_test = transformed_data_test
M_test = distance_cal(d_test)
major_components_test = M_test[:,range(q)]
major_components_test_distance = np.sum(major_components_test, axis=1)
minor_components_test = M_test[:, r]
minor_components_test_distance = np.sum(minor_components_test, axis=1)  # calculate the distance of the minor components
labels_pred = classifier(major_components=major_components_test_distance
                        ,minor_components=minor_components_test_distance)

###########################################
#calculate tpr,fpr with different quantile
############################################

y_pred_1 = classifier_1(major_components=major_components_test_distance
                        ,minor_components=minor_components_test_distance)
y_pred_2 = classifier_2(major_components=major_components_test_distance
                        ,minor_components=minor_components_test_distance)
y_pred_3 = classifier_3(major_components=major_components_test_distance
                        ,minor_components=minor_components_test_distance)
y_pred_4 = classifier_4(major_components=major_components_test_distance
                        ,minor_components=minor_components_test_distance)
y_pred_5 = classifier_5(major_components=major_components_test_distance
                        ,minor_components=minor_components_test_distance)
y_pred_6 = classifier_6(major_components=major_components_test_distance
                        ,minor_components=minor_components_test_distance)
y_pred_7 = classifier_7(major_components=major_components_test_distance
                        ,minor_components=minor_components_test_distance)

a1 = confusion_matrix(y_test_all, y_pred_1)
a2 = confusion_matrix(y_test_all, y_pred_2)
a3 = confusion_matrix(y_test_all, y_pred_3)
a4 = confusion_matrix(y_test_all, y_pred_4)
a5 = confusion_matrix(y_test_all, y_pred_5)
a6 = confusion_matrix(y_test_all, y_pred_6)
a7 = confusion_matrix(y_test_all, y_pred_7)


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

###########################################
#plot the final graph
###########################################



roc_auc = auc(fpr,tpr) #auc berechnen
lw=2
plt.plot(fpr, tpr, color='darkorange', lw=lw, marker = 'o', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.plot([0,1],[0.8,0.8],linestyle=':',linewidth='1',color = 'black')
plt.plot([0,1],[0.6,0.6],linestyle=':',linewidth='1',color = 'black')
plt.plot([0,1],[0.4,0.4],linestyle=':',linewidth='1',color = 'black')
plt.plot([0,1],[0.2,0.2],linestyle=':',linewidth='1',color = 'black')
#for a,b in zip(fpr,tpr):
#    plt.text(a, b, (a,b),ha='center', va='bottom', fontsize=10)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()


#f1_score

from sklearn.metrics import f1_score
f1_1 = f1_score(y_test_all, y_pred_1)
f1_2 = f1_score(y_test_all, y_pred_2)
f1_3 = f1_score(y_test_all, y_pred_3)
f1_4 = f1_score(y_test_all, y_pred_4)
f1_5 = f1_score(y_test_all, y_pred_5)
f1_6 = f1_score(y_test_all, y_pred_6)
f1_7 = f1_score(y_test_all, y_pred_7)

