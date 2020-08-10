# Anomalieerkennung-Wasserwerk
The aim of the project is to compare two technical approaches to increase the safety of communication technology in safety-critical infrastructures using the example of a waterworks. 

Inhalt:
Nachfolgend werden die einzelnen Dateien/Ordner aufgezählt und es wird kurz die Funktion zusammengefasst.

-------------------------------
Werte_Fluidsimulation_4000s.csv
-------------------------------
Diese Datei enthält alle normalen Daten und wird zum Training der Anwendungsprogramme genutzt.
Bei einer Konfiguration des Wasserwerks ist diese Datei durch ein neues Datensample zu ersetzen.

-------------
SPS_Daten.csv
-------------
Beinhaltet die von der Simulation erzeugten Daten (inkl. Anomalien) und zusätzlich dazu eine Spalte, die anzeigt, ob es sich um eine Anomalie handelt oder nicht.
Diese Datei wird zur Nachahmung der UDP Verbindung für die Anwenderprogramme verwendet sowie zur Parameteroptimierung

--------------
import_data.py
--------------
Die Werte stammen aus der Datei "Werte_Fluidsimulation_Fluidsimulation_4000s.csv".
Diese beinhaltet keine Anomalien.
Es werden die folgenden Werte eingelesen:

['Time', 'Volumen_Tank_1', 'Temperatur_Tank_1', 'Druck_nach_Pumpe_1',
 'Volumenstrom_nach_Pumpe_1', 
'Druck_nach_Pumpe_2', 'Volumenstrom_nach_Pumpe_2', 'Volumenstrom_nach_Rohr_1', 'P_nach_Rohr_1', 'P_nach_Rohr_2', 'Volumenstrom_nach_Rohr_2',
'Volumen_Tank_2', 'Temperatur_Tank_2',  'Wasserlevel_Tank_1', 'Wasserlevel_Tank_2', 'Druck_vor_Pumpe_1',
'Druck_vor_Pumpe_2',
  'Anomaly',
 'Timedelta']

Diese Funktion wird zum Einlesen der Trainingsdaten der Anwenderprogramme aufgerufen.
Es werden 63 % aller Daten zum Training genutzt, 27 % zur Validierung und 10 % zum Testen.

-----------------
UDP_send_data.py
-----------------
Dieses Programm ahmt die UDP Übertragung nach und verschickt die Daten aus "SPS_daten.csv" sequentiell.
Es wird nebenläufig neben einem gewählten Anwenderprogramm gestartet.

-----------------
SPS_classify.py
-----------------
Dieses Programm wird genutzt, die Klassifizierung der Daten mittels der Regelbasierten Überwachung zu simulieren.
Die Entscheidung der SPS wird in den ML Algorithmen während der Training- und Testphase als zusätzliche Eingangsdimension verwendet.

------------------------
DistanceBased Approach.py
------------------------
Hier ist das Anwenderprogramm zu finden, welches den Distanzbasierten Ansatz implementiert.
Die Trainingsdaten werden mit import_data eingelesen und die Daten der SPS werden per UDP entgegengenommen.

---------------------------------
GaussianMixtureModel Approach.py
---------------------------------
Hier ist das Anwenderprogramm zu finden, welches den GMM Ansatz implementiert.
Die Trainingsdaten werden mit import_data eingelesen und die Daten der SPS werden per UDP entgegengenommen.

-------------------
confusion_matrix.py
-------------------
Eingangsparameter sind  die tatsächlichen Testdaten und vorhergesagten Klassen. 
Aus den eingegebenen Daten wird eine Confusion-Matrix durch die Funktion confusion_matrix aus der ScikitLearn-Bibliothek geplottet.

-------------------------------
Parameteroptimierung und Testen
-------------------------------
Die beiden Anwenderprogramme sind auch in diesem Ordner in veränderter Form zu finden.
Sie wurden genutzt, um manuell eine Parameteroptimierung mittels ROC_AUC Curve und F1 Score vorzunehmen und die Ergebnisse zu testen.
