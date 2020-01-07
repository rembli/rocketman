######################################################
# ROCKETMAN
#######################################################

MOTIVATION:

Kompletten Liefcycle einer ML-App am Beispiel durchspielen.
Idee für das Beispiel: Computer soll ein abfotografiertes Notenblatt spielen.

AKTUELLE LIMITATIONEN:

- Bilder muessen ein bestimmtest Format haben (PNG, nicht zu große Auflösung)
- Bisher wird nur immer eine Note gleichzeitig erkannt und gespielt
- Noten nur von der untersten bis zur obersten Notenlinie und kein höheren oder tieferen Töne
- Dauer der Note wird nicht berücksichtigt
- # oder Bass-Schlüssel wird nicht berücksichtigt

DATEN:

+ /pieces: die eingescannte Musik-Stücke
+ /lines: die per pre-processing extrahierten Notenlinien
+ /notes: die per pre-processing extrahierten einzelnen Noten bzw. Akkorde
+ /cluster: Noten bzw. Akkorder mit grosser Ähnlichkeit soll geclustert werden, um das Labeling zu erleichtern
+ /labels: Gleiche Noten werden die gleichen Katgeorien zugewiesen

SCRIPTS:

Skripts: C:\Data\Dev\PyCharmProjects\rocketman\src
1. preprocessing: Aus einem Notenblatt einzelne Noten extrathieren und clustern
    - peakdetect: Hilfs-Klasse, um die Darstellung einer Note in einer Notenlinie zu identifzieren
2. mechanicalturk: PyGame-App um die Noten manuell zu lablen
    - soundshift: Hilfs-Klasse, um aus einer WAV-Datei unterschiedliche Noten zu erzeugen
3. training: Trainieren des ML Models auf den Labels

--> scan2sound: die Umsetzung der Idee in einer ersten Version

EINGESETZT TECHNIKEN:

- CV2 für Image Manipulationen
- DBSCAN für Clustering
- KERAS für Notenerkennung
- PYGAME für UI

REQUIREMENTS erzeugt mit
pipreqs /path/to/project