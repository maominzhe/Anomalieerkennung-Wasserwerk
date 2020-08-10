#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
import socket
import pickle
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score
from import_data import *
from confusion_matrix import *
from roc_auc import *
import time
import datetime

# UDP Connection
# CONFIG SECTION
UDP_IP = "127.0.0.1"
UDP_PORT = 4005
WINDOW_SIZE = 3  # how many data samples are received before further processing

#----------------------
# Algorithm parameters:
T = 67.1  # T value used as a threshold for evaluating the gmm score of the data
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
    Trainings_time=X['Time'] # saves the timestamps of all the training data to ensure, that they are excluded from the testing process
    standard_scalar = StandardScaler()  # Normalize the data
    X = standard_scalar.fit_transform(X)

    # initialize PCA with n components, such that 99 % of the information are preserved
    pca = PCA()  # n_components=0.99, svd_solver='full')
    transformed_data = pca.fit_transform(X)
    X_principal = pd.DataFrame(data=transformed_data)

    ts = time.time()

    # Fit the PCA into GMM
    GMM = GaussianMixture(n_components=1)
    GMM.fit(X_principal.values)

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
    im_CM = ax_CM.imshow(np.empty([2, 2]), interpolation='nearest', cmap=plt.cm.Blues)
    colorbar_CM = ax_CM.figure.colorbar(im_CM, ax=ax_CM)
    txt_CM = [0, 0, 0, 0]
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

    labels_pred = np.empty([0])
    labels_time = np.empty([0])
    labels_true = np.empty([0])
    timediff=np.empty([0])
    # Start a while loop to process the data from qq if the queue is full and send to qq
    # The While will run until the User stops it
    newData = qq.get
    while 1:
        # Get the Data from the queue
        try:
            newData = qq.get(True, 10)  # waiting for data sent via udp; timeout after 10 seconds
            start = datetime.datetime.now()
        except:
            print("Received no data for 10 seconds.")
            break
       # Create a dataframe for the Data
        newData = pd.DataFrame(data=newData, columns=features)
        #drops the data, which was used for training
        training_data=newData[np.logical_and(newData['Time'].isin(Trainings_time), newData['Anomaly']==0)].index
        newData=newData.drop(training_data,axis=0)
        if newData.empty:
            continue
        labels_true = np.append(labels_true, newData['Anomaly'])
        time = newData['Time']
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

        data_score = GMM.score_samples(principalDf)  # Calculate the GMM score
        labels_pred = np.append(labels_pred, data_score<T)  # classify the data points by using the T value
        labels_time = np.append(labels_time, time)
        #after the data processing has finished stop the timer
        end = datetime.datetime.now()
        timediff=np.append(timediff, (end-start)/len(newData))
        # Initialize empty Array for storing the Values from each Window, that can be used in the Plot
        x = []
        y = []
        c = []

        # Iterate through every Data sample and draw the scatter plot
        for i in range(newData.shape[0]):
            x.append(principalDf.loc[i, 0])
            y.append(principalDf.loc[i, 1])
            if (labels_pred[labels_pred.size + i - newData.shape[0]] == 1):  # check the current window for anomalies
                c.append(colors[0])
                ax.scatter(principalDf.loc[i, 0], principalDf.loc[i, 1],
                           c='r', s=20)
            else:
                # If no anomaly
                c.append(colors[1])
        p0 = ax.scatter(x, y, c=c, s=20)

        ax5.plot(time, labels_pred[labels_pred.size - newData.shape[0]:labels_pred.size], linestyle='-', c='r', linewidth=1)
        # plot the confusion matrix
        plot_confusion_matrix(fig_CM, ax_CM, txt_CM, colorbar_CM, labels_true[0:labels_pred.size], labels_pred)
        plt.pause(0.00001)

        labels_time_anomalyTable = pd.DataFrame(data=labels_time, columns=["time"])
        labels_pred_anomalyTable = pd.DataFrame(data=labels_pred, columns=["anomaly"])
        labels_time_anomalyTable["anomaly"] = labels_pred_anomalyTable
        anomalyTable = labels_time_anomalyTable
    # After the timeout, draw the ROC AUC and print the F1 score
    try:
        print("The F1 Score is: ", str(f1_score(labels_true, labels_pred)))
        print("The data processing time was on average: ", str(np.mean(timediff)))
    except ValueError:
        print("The F1 Score and the ROC AUC could not be determined.")
plt.show()

