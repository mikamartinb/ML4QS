import os
import pandas as pd

def merge_csv_files(project_folder, filenames, output_file):
    dataframes = []

    # Durch alle Dateinamen iterieren
    for filename in filenames:
        file_path = os.path.join(project_folder, f"{filename}.csv")

        # CSV-Datei lesen und zum Dataframe hinzufügen
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            dataframes.append(df)

    # Dataframes zusammenführen
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Zusammengeführtes Dataframe als CSV-Datei speichern
    output_path = os.path.join(project_folder, output_file)
    merged_df.to_csv(output_path, index=False)
    print(f"Data merged and saved to {output_path}")


# Beispielaufruf der Funktion
project_folder = "/Users/mika/.conda/envs/ML4QS"  # Pfad zu Ihrem Python-Projektordner
filenames = ["ALLE AUTO", "ALLE RAD", "ALLE FAHRSTUHL", "ALLE TREPPEHOCH",
             "ALLE TREPPERUNTER", "ALLE STEHEN", "ALLE GEHEN", "ALLE LAUFEN"]  # Liste der Dateinamen
output_file = "DATEN.csv"  # Name der Ausgabedatei
merge_csv_files(project_folder, filenames, output_file)
