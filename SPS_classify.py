#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# --------------------------
#
# Purpose:
# classifying the data using the SPS
# --------------------------

#----------------------
# Grenzwerte:
grenzemax_druck=1.4
grenzemax_fluss=0.002
grenzemin_vol=1
grenzemin_temp=273.15
grenzemax_temp=373.15
#----------------------
def SPS_classify(Process_data):
    # data is compared to the respective thresholds and the outcome is assigned to the columns as an integer
    Process_data['SPS_druck']=np.multiply(np.any([Process_data['Druck_nach_Pumpe_1']>grenzemax_druck, Process_data['Druck_nach_Pumpe_2']>grenzemax_druck,
                                      Process_data['Druck_vor_Pumpe_1']>grenzemax_druck, Process_data['Druck_vor_Pumpe_2']>grenzemax_druck,
                                      Process_data['Druck_nach_Rohr_1']>grenzemax_druck, Process_data['Druck_nach_Rohr_2']>grenzemax_druck], axis=0),1)
    Process_data['SPS_fluss'] = np.multiply(np.any([Process_data['Volumenstrom_nach_Pumpe_1'] > grenzemax_fluss, Process_data['Volumenstrom_nach_Pumpe_2'] > grenzemax_fluss,
                                         Process_data['Volumenstrom_nach_Rohr_1'] > grenzemax_fluss, Process_data['Volumenstrom_nach_Rohr_2'] > grenzemax_fluss], axis=0),1)
    Process_data['SPS_vol'] = np.multiply(np.any([Process_data['Wasserlevel_Tank_1'] < grenzemin_vol, Process_data['Wasserlevel_Tank_2'] < grenzemin_vol], axis=0),1)
    Process_data['SPS_temp'] = np.multiply(np.any([Process_data['Temperatur_Tank_1'] < grenzemin_temp, Process_data['Temperatur_Tank_2'] < grenzemin_temp,
                                       Process_data['Temperatur_Tank_1'] > grenzemax_temp, Process_data['Temperatur_Tank_2'] > grenzemax_temp], axis=0),1)
    Process_data['SPS_Anomaly']=np.multiply(np.any([Process_data['SPS_druck'],Process_data['SPS_fluss'],Process_data['SPS_vol'],Process_data['SPS_temp']], axis=0),1)
    return Process_data