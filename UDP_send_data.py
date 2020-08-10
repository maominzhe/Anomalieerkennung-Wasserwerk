import socket
from random import randrange
import pickle
import pandas as pd
import time
from datetime import datetime
from SPS_classify import *

# CONFIG SECTION
UDP_IP = "127.0.0.1"
UDP_PORT = 4005
SPS_HACKED = 1 # 0: messages will be shown. 1: SPS was hacked and no warnings will be reported


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
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
dataToSend = SPS_classify(dataToSend)
dataToSend['Timedelta'] = dataToSend['Time'].diff()
row_count = len(dataToSend)
dataToSend.loc[0:3359,'Anomaly']=0
dataToSend.loc[3359:5026,'Anomaly']=1
dataToSend.loc[5026:6859,'Anomaly']=0
dataToSend.loc[6859:9493,'Anomaly']=1
dataToSend.loc[9493:10027,'Anomaly']=0
dataToSend.loc[10027:11737,'Anomaly']=1
dataToSend.loc[11737:row_count,'Anomaly']=0
dataToSend = dataToSend.iloc[11000:] # delete after usage
dataToSend = dataToSend.reset_index(drop=True)
row_count = len(dataToSend)

if(SPS_HACKED):
    dataToSend.loc['SPS_druck'] = 0
    dataToSend.loc['SPS_fluss'] = 0
    dataToSend.loc['SPS_vol'] = 0
    dataToSend.loc['SPS_temp'] = 0
    dataToSend.loc['SPS_Anomaly'] = 0

#send over udp until all data has been sent
for i in range(0, row_count):
    now = datetime.now()
    timestamp = datetime.timestamp(now)

    rows = dataToSend.iloc[i]
    MESSAGE = [rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], rows[6], rows[7], rows[8], rows[9], rows[10],
               rows[11], rows[12], rows[13], rows[14], rows[15], rows[16], rows[17],rows[18],rows[19],rows[20],rows[21],rows[22],rows[23]]
    print(MESSAGE)
    data = pickle.dumps(MESSAGE)
    time.sleep(dataToSend.loc[i,'Timedelta']/2000)  # delays execution by timedelta/1 = Realtime
    sock.sendto(data, (UDP_IP, UDP_PORT))
