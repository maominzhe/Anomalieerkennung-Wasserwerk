# Anomaly-detection-waterworks

The aim of the project is to compare two technical approaches to increase the safety of communication technology in safety-critical infrastructures using the example of a waterworks. 

Content:
In the following, the individual files/folders are listed and the function is briefly summarized.

-------------------------------
Werte_Fluidsimulation_4000s.csv
-------------------------------
This file contains all normal data and is used for training the application programs.
If the waterworks is configured, this file must be replaced by a new data sample.

-------------
SPS_Daten.csv
-------------
Contains the data generated by the simulation (including anomalies) and in addition a column indicating whether or not an anomaly is present.
This file is used to imitate the UDP connection for the user programs and for parameter optimization

--------------
import_data.py
--------------
The values come from the file "Values_Fluidsimulation_Fluidsimulation_4000s.csv".
This file contains no anomalies.
The following values are read in:

['Time', 'Volumen_Tank_1', 'Temperatur_Tank_1', 'Druck_nach_Pumpe_1',
 'Volumenstrom_nach_Pumpe_1', 
'Druck_nach_Pumpe_2', 'Volumenstrom_nach_Pumpe_2', 'Volumenstrom_nach_Rohr_1', 'P_nach_Rohr_1', 'P_nach_Rohr_2', 'Volumenstrom_nach_Rohr_2',
'Volumen_Tank_2', 'Temperatur_Tank_2',  'Wasserlevel_Tank_1', 'Wasserlevel_Tank_2', 'Druck_vor_Pumpe_1',
'Druck_vor_Pumpe_2',
  'Anomaly',
 'Timedelta']

This function is called to read in the training data of the user programs.
63 % of all data is used for training, 27 % for validation and 10 % for testing.

-----------------
UDP_send_data.py
-----------------
This program imitates the UDP transmission and sends the data from "SPS_daten.csv" sequentially.
It is started simultaneously with a selected user program.

-----------------
SPS_classify.py
-----------------
This program is used to simulate the classification of the data using rule-based monitoring.
The decision of the PLC is used as an additional input dimension in the ML algorithms during the training and test phase.

------------------------
DistanceBased Approach.py
------------------------
Here you can find the application program that implements the distance-based approach.
The training data is read in with import_data and the data from the PLC is received via UDP.

---------------------------------
GaussianMixtureModel Approach.py
---------------------------------
Here you can find the application program which implements the GMM approach.
The training data is read in with import_data and the data from the PLC is received via UDP.

-------------------
confusion_matrix.py
-------------------
Input parameters are the actual test data and predicted classes. 
A confusion matrix is plotted from the input data by the confusion_matrix function from the ScikitLearn library.

-------------------------------
Parameter optimization and testing
-------------------------------
The two user programs can also be found in this folder in a modified form.
They were used to manually perform parameter optimization using ROC_AUC Curve and F1 Score and to test the results.
