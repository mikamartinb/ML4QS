import os
import pandas as pd
import numpy as np
import lowpassfilter

def create_movement_dataframe(root_folder):
    # DataFrame "Bewegung" initialisieren
    bewegung_df = pd.DataFrame(columns=["Mean X", "Mean Y", "Mean Z", "Standardabweichung X",
                                        "Standardabweichung Y", "Standardabweichung Z",
                                        "Min X", "Min Y", "Min Z", "Max X", "Max Y", "Max Z", "Label"])

    # Durch den übergeordneten Ordner iterieren
    for root, dirs, files in os.walk(root_folder):
        # Durch die Unterverzeichnisse iterieren
        for dir in dirs:
            subdir_path = os.path.join(root, dir)

            # Überprüfen, ob eine "accelerometer.csv" in diesem Unterverzeichnis existiert
            csv_file = os.path.join(subdir_path, "accelerometer.csv")
            if os.path.isfile(csv_file):
                # DataFrame aus der "accelerometer.csv" erstellen
                df = pd.read_csv(csv_file)

                # Berechnung der statistischen Werte
                mean_x = df.iloc[:, 1].mean()
                mean_y = df.iloc[:, 2].mean()
                mean_z = df.iloc[:, 3].mean()

                std_x = df.iloc[:, 1].std()
                std_y = df.iloc[:, 2].std()
                std_z = df.iloc[:, 3].std()

                min_x = df.iloc[:, 1].min()
                min_y = df.iloc[:, 2].min()
                min_z = df.iloc[:, 3].min()

                max_x = df.iloc[:, 1].max()
                max_y = df.iloc[:, 2].max()
                max_z = df.iloc[:, 3].max()

                # Neue Zeile zum DataFrame "Bewegung" hinzufügen
                row = {
                    "Mean X": mean_x,
                    "Mean Y": mean_y,
                    "Mean Z": mean_z,
                    "Standardabweichung X": std_x,
                    "Standardabweichung Y": std_y,
                    "Standardabweichung Z": std_z,
                    "Min X": min_x,
                    "Min Y": min_y,
                    "Min Z": min_z,
                    "Max X": max_x,
                    "Max Y": max_y,
                    "Max Z": max_z,
                    "Label": "Gehen"
                }
                bewegung_df = bewegung_df.append(row, ignore_index=True)

    return bewegung_df


df = create_movement_dataframe("ALLE GEHEN")

