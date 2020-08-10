#!/usr/bin/env python
# coding: utf-8

import socket
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Process, Queue
import time
from random import randrange
import socket
import pickle
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import matplotlib.lines as mlines
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from matplotlib.patches import Ellipse
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from import_data import *
from confusion_matrix import *
import time
import datetime

# CONFIG SECTION
UDP_IP = "127.0.0.1"
UDP_PORT = 4005
WINDOW_SIZE = 5 # in seconds
#----------------------
# Algorithm parameters:
c1_quantile = 0.95
c2_quantile = 0.95
q = 1  # number of major components, which preserve 50 % of the information in total
r_explained = 0.2  # all principal components with a variance less then r_explained are used as minor components
#---------------------


# ASYNCHRONOUS Task: collects the data, until the buffer is full
#                    when it is full, it is sent to the main function
def f(q):
    while 1:
        # Make UDP Connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))

        # Initialize a Buffer
        buffer = []
        lastSend = time.monotonic()
        plt.scatter(0, 0)

        while True:
            data, addr = sock.recvfrom(2048)  # buffer size is 2048 bytes
            data = pickle.loads(data)

            buffer.append(data)

            if lastSend <= time.monotonic()-WINDOW_SIZE:
                lastSend = time.monotonic();
                q.put(buffer)
                buffer = []


if __name__ == '__main__':
    # Import the Training data from a file
    df, X, Y, f_length, features = import_data(0, 0, 0, [])
    Trainings_time = X['Time']  # saves the timestamps of all the training data to ensure, that they are excluded from the testing process
    standard_scalar = StandardScaler()  # Normalize the data
    X = standard_scalar.fit_transform(X)

    # initialize PCA with n components, such that 99 % of the information are preserved
    pca = PCA() #n_components=0.99, svd_solver='full'
    transformed_data = pca.fit_transform(X)
    X_principal = pd.DataFrame(data=transformed_data)

    ts = time.time()


    # function to calculate the distance
    def distance_cal(np):
        lambdas = pca.singular_values_
        lambdas[lambdas == 0] = 1e-40
        M = np * np / lambdas
        return M

    # define classifier
    def classifier(major_components, minor_components):
        major = major_components > c1
        minor = minor_components > c2
        # major = 0
        return np.logical_or(major, minor)

    M = distance_cal(transformed_data)  # calculate the distance matrix

    # number of major components, which preserve 50 % of the information in total
    s = sum(pca.explained_variance_ratio_[:q])
    print("Explained variance by first " + str(q) + " principal component(s):" + str(s))
    major_components = M[:, range(q)]  # determine major_components
    major_components_distance = np.sum(major_components, axis=1)  # calculate the distance of the major components

    r = list(pca.explained_variance_ratio_ < r_explained)  # determine minor components, who account for less than 50 % of the information
    minor_components = M[:, r]
    minor_components_distance = np.sum(minor_components, axis=1)# calculate the distance of the minor components

    # define the two thresholds.
    components = pd.DataFrame({'major_components': major_components_distance,
                               'minor_components': minor_components_distance})
    c1 = components.quantile(c1_quantile)['major_components']
    c2 = components.quantile(c2_quantile)['minor_components']

    # Initialize the data plots
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(5, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Druck')
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title('Wasserlevel')
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_title('Volumenstrom')
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.set_title('Temperatur')
    ax5 = fig.add_subplot(gs[4, 0])
    ax5.set_title('Anomalie Level')
    ax = fig.add_subplot(gs[:, 1])
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('PCA', fontsize=20)
    colors = ['r', 'g']
    legend = ["Non-Anomaly", "Anomaly"]
    custom_lines = [Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='g', markersize=5),
                     Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='r', markersize=5)]
    plt.legend(custom_lines, legend)
    ax.scatter(0, 0, c='w', s=12)
    # plot the two principal components with the highest variance
    ax.scatter(X_principal[0], X_principal[1], s=12)

    # Initialize the plot for the Confusion Matrix
    fig_CM, ax_CM = plt.subplots()
    im_CM = ax_CM.imshow(np.empty([2,2]), interpolation='nearest', cmap=plt.cm.Blues)
    colorbar_CM=ax_CM.figure.colorbar(im_CM, ax=ax_CM)
    txt_CM=[0,0,0,0]
    classes = ['Non-Anomaly', 'Anomaly']
    ax_CM.set(xticks=np.arange(2),
           yticks=np.arange(2),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax_CM.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Initialize a handle for the text object in the confusion matrix
    for i in range(2):
        for j in range(2):
            txt_CM[i + 2 * j] = ax_CM.text(j, i, "0",
                                     ha="center", va="center",
                                     color="black")
    plt.pause(0.00001)

    # Setup the asynchronous UDP Task and start the process to receive the Data
    qq = Queue()
    p = Process(target=f, args=(qq,))
    p.start()

    forExport = np.empty([0])

    labels_pred = np.empty([0])
    labels_time = np.empty([0])
    labels_true = np.empty([0])
    timediff = np.empty([0])

    # Start a while loop to process the data from qq if the queue is full and send to qq
    # The While will run until the User stops it
    newData = qq.get
    run = 1

    # Prepare the Plots
    ax1.plot([],[], 'r-', label='Druck nach Pumpe 1')
    ax1.plot([],[], 'g-', label='Druck nach Pumpe 2')
    ax1.plot([],[], 'y-', label='Druck vor Pumpe 1')
    ax1.plot([],[], 'b-', label='Druck vor Pumpe 2')
    ax1.plot([],[], 'm-', label='P nach Rohr 1')
    ax1.plot([],[], 'c-', label='P nach Rohr 2')

    ax2.plot([], [], 'r-', label='Wasserlevel Tank 1')
    ax2.plot([], [], 'g-', label='Wasserlevel Tank 2')
    # ax2.plot([], [], 'y-', label='Volumen Tank 1')
    # ax2.plot([], [], 'b-', label='Volumen Tank 2')

    ax3.plot([], [], 'r-', label='Volumenstrom nach Pumpe 1')
    ax3.plot([], [], 'g-', label='Volumenstrom nach Pumpe 2')
    ax3.plot([], [], 'b-', label='Volumenstrom nach Rohr 1')
    ax3.plot([], [], 'y-', label='Volumenstrom nach Rohr 2')
    ax4.plot([], [], 'b-', label='Temperatur Tank 1')
    ax4.plot([], [], 'g-', label='Temperatur Tank 2')
    ax5.plot([], [], linestyle='--', c='b', linewidth=2, label='Real Anomaly')

    # Make Legends
    ax1.legend(loc='upper right', fontsize=7, ncol=1)
    ax2.legend(loc='upper left', fontsize=7, ncol=1)
    ax3.legend(loc='upper left', fontsize=7, ncol=1)
    ax4.legend(loc='upper left', fontsize=7, ncol=1)
    ax5.legend(loc='upper left', fontsize=7, ncol=1)



    while run:
        # Get the Data from the queue
        try:
            newData = qq.get(True, 10)  # waiting for data sent via udp; timeout after 10 seconds
            start = datetime.datetime.now()
        except:
            print("Received no data for 10 seconds.")
            break

        # Create a dataframe for the Data
        newData = pd.DataFrame(data=newData, columns=features)
        # drops the data, which was used for training
        training_data = newData[np.logical_and(newData['Time'].isin(Trainings_time), newData['Anomaly'] == 0)].index
        newData = newData.drop(training_data, axis=0)
        if newData.empty:
            continue
        labels_true=np.append(labels_true, newData['Anomaly'])
        time=newData['Time']
        # Plot the newData in the Liveplot
        ax1.plot(time, newData[features[3]], 'r-')
        ax1.plot(time, newData[features[5]], 'g-')
        ax1.plot(time, newData[features[15]], 'y-')
        ax1.plot(time, newData[features[16]], 'b-')
        ax2.plot(time, newData[features[13]], 'r-')
        ax2.plot(time, newData[features[14]], 'g-')
        ax3.plot(time, newData[features[4]], 'r-')
        ax3.plot(time, newData[features[6]], 'g-')
        ax3.plot(time, newData[features[7]], 'b-')
        ax3.plot(time, newData[features[10]], 'y-')
        ax4.plot(time, newData[features[2]], 'b-')
        ax4.plot(time, newData[features[12]], 'g-')
        ax5.plot(time, newData[features[18]], linestyle='--', c='b', linewidth=2)

        newData = newData.drop(['Anomaly'], axis=1)
        newData = standard_scalar.transform(newData)  # normalize the testing data
        transformed_data_test = pca.transform(newData)  # find the values for the principal components
        principalDf = pd.DataFrame(data=transformed_data_test)

        # calculate the distances of the major and minor component
        M_test = distance_cal(transformed_data_test)
        major_components_test = M_test[:, range(q)]
        major_components_test_distance = np.sum(major_components_test, axis=1)
        minor_components_test=M_test[:,r]
        minor_components_test_distance = np.sum(minor_components_test, axis=1)

        # predict the labels using the classifier function
        labels_pred=np.append(labels_pred,classifier(major_components=major_components_test_distance
                                 , minor_components=minor_components_test_distance))
        labels_time = np.append(labels_time, time)
        # after the data processing has finished stop the timer
        end = datetime.datetime.now()
        timediff = np.append(timediff, (end - start) / len(newData))
        # Initialize empty Array for storing the Values from each Window, that can be used in the Plot
        x = []
        y = []
        c = []

        # Iterate through every Data sample and draw the scatter plot
        for i in range(newData.shape[0]):
            x.append(principalDf.loc[i, 0])
            y.append(principalDf.loc[i, 1])
            if (labels_pred[labels_pred.size+i-newData.shape[0]] == 1): # check the current window for anomalies
                c.append(colors[0])
                ax.scatter(principalDf.loc[i, 0], principalDf.loc[i, 1],
                           c='r', s=20)
            else:
                # If no anomaly
                c.append(colors[1])
        p0 = ax.scatter(x, y, c=c, s=20)
        # Make Legends
        ax1.legend(loc='upper right', fontsize=7, ncol=1)
        ax2.legend(loc='upper left', fontsize=7, ncol=1)
        ax3.legend(loc='upper left', fontsize=7, ncol=1)
        ax4.legend(loc='upper left', fontsize=7, ncol=1)
        ax5.legend(loc='upper left', fontsize=7, ncol=1)

        ax5.plot(time, labels_pred[labels_pred.size-newData.shape[0]:labels_pred.size], linestyle='-', c='r', linewidth=1)
        # plot the confusion matrix
        plot_confusion_matrix(fig_CM,ax_CM,txt_CM,colorbar_CM, labels_true[0:labels_pred.size], labels_pred)
        plt.pause(0.00001)
    #After the timeout, draw the ROC AUC and print the F1 score
    try:
        print("The F1 Score is: ", str(f1_score(labels_true, labels_pred)))
        print("The data processing time was on average: ", str(np.mean(timediff)))
    except ValueError:
        print("The F1 Score and the ROC AUC could not be determined.")
plt.show()

