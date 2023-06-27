import os
import pandas as pd

def prune_accelerometer_files(root_folder, offset_start_sec, offset_end_sec):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file == "Accelerometer.csv":
                csv_file = os.path.join(root, file)
                df = pd.read_csv(csv_file)

                # Anfang und Ende der Zeitachse entsprechend des Offsets festlegen
                start_time = offset_start_sec
                end_time = df.iloc[-1, 0] - offset_end_sec

                # Pruning der Daten
                pruned_df = df[(df.iloc[:, 0] >= start_time) & (df.iloc[:, 0] <= end_time)]

                # Speichern der bearbeiteten Datei als "accelerometer_pruned.csv" im gleichen Ordner
                pruned_file = os.path.join(os.path.dirname(csv_file), "Accelerometer_pruned.csv")
                pruned_df.to_csv(pruned_file, index=False)

                print(f"Pruned file created: {pruned_file}")


root_folder = "???"
offset_start_sec = 10  # 10 Sekunden vom Anfang der Daten entfernen
offset_end_sec = 10 # 10 Sekunden vom Ende der Daten entfernen

prune_accelerometer_files(root_folder, offset_start_sec, offset_end_sec)

