import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

def create_movement_dataframe(root_folder):
    bewegung_df = pd.DataFrame(columns=["Mean X", "Mean Y", "Mean Z", "Standardabweichung X",
                                        "Standardabweichung Y", "Standardabweichung Z",
                                        "Min X", "Max X", "Min Y", "Max Y", "Min Z", "Max Z",
                                        "Lokale Peaks X", "Lokale Peaks Y", "Lokale Peaks Z",
                                        "Label"])

    for root, dirs, files in os.walk(root_folder):
        for dir in dirs:
            subdir_path = os.path.join(root, dir)

            csv_file = os.path.join(subdir_path, "accelerometer_pruned.csv")
            if os.path.isfile(csv_file):
                df = pd.read_csv(csv_file)

                mean_x = df.iloc[:, 1].mean()
                mean_y = df.iloc[:, 2].mean()
                mean_z = df.iloc[:, 3].mean()

                std_x = df.iloc[:, 1].std()
                std_y = df.iloc[:, 2].std()
                std_z = df.iloc[:, 3].std()

                min_x = df.iloc[:, 1].min()
                max_x = df.iloc[:, 1].max()
                min_y = df.iloc[:, 2].min()
                max_y = df.iloc[:, 2].max()
                min_z = df.iloc[:, 3].min()
                max_z = df.iloc[:, 3].max()

                local_peaks_x = calculate_local_peaks(df.iloc[:, 1])
                local_peaks_y = calculate_local_peaks(df.iloc[:, 2])
                local_peaks_z = calculate_local_peaks(df.iloc[:, 3])

                row = {
                    "Mean X": mean_x,
                    "Mean Y": mean_y,
                    "Mean Z": mean_z,
                    "Standardabweichung X": std_x,
                    "Standardabweichung Y": std_y,
                    "Standardabweichung Z": std_z,
                    "Min X": min_x,
                    "Max X": max_x,
                    "Min Y": min_y,
                    "Max Y": max_y,
                    "Min Z": min_z,
                    "Max Z": max_z,
                    "Lokale Peaks X": local_peaks_x,
                    "Lokale Peaks Y": local_peaks_y,
                    "Lokale Peaks Z": local_peaks_z,
                    "Label": "TreppeRunter"
                }
                bewegung_df = bewegung_df.append(row, ignore_index=True)

    return bewegung_df


def calculate_local_peaks(accel_data):
    fs = 200  # Abtastfrequenz

    # Butterworth-Tiefpassfilter anwenden
    b, a = butter(4, 0.1, fs=fs, btype='low')
    padlen = min(len(accel_data), 15)  # Anpassung der padlen-Werte
    filtered_data = filtfilt(b, a, accel_data, padlen=padlen)

    # Finden der lokalen Peaks
    peaks, _ = find_peaks(filtered_data)

    return len(peaks)


def prune_accelerometer_files(root_folder, offset_start_sec, offset_end_sec):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file == "Accelerometer.csv":
                csv_file = os.path.join(root, file)
                df = pd.read_csv(csv_file)

                start_time = offset_start_sec
                end_time = df.iloc[-1, 0] - offset_end_sec

                pruned_df = df[(df.iloc[:, 0] >= start_time) & (df.iloc[:, 0] <= end_time)]

                pruned_file = os.path.join(os.path.dirname(csv_file), "accelerometer_pruned.csv")
                pruned_df.to_csv(pruned_file, index=False)

                print(f"Pruned file created: {pruned_file}")


def main(root_folder):
    offset_start_sec = 3
    offset_end_sec = 3
    prune_accelerometer_files(root_folder, offset_start_sec, offset_end_sec)

    feature_data = create_movement_dataframe(root_folder)
    feature_data.to_csv(f"{root_folder}.csv", index=False)
    print(f"Feature dataset created: {root_folder}.csv")


if __name__ == "__main__":
    root_folder = input("Bitte geben Sie den Namen des Root-Ordners ein:")
    main(root_folder)
