import os
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

def apply_lowpass_filter(data, lowcut, highcut, fs):
    b, a = butter(4, [lowcut, highcut], fs=fs, btype='band')
    return filtfilt(b, a, data)

def calculate_local_peaks(accel_data, fs, lowcut, highcut):
    b, a = butter(4, lowcut, fs=fs, btype='low')
    padlen = min(len(accel_data), 15)
    filtered_data = filtfilt(b, a, accel_data, padlen=padlen)

    peaks, _ = find_peaks(filtered_data)

    return len(peaks)

def prune_and_filter_data(input_file, output_file, offset_start_sec, offset_end_sec, lowcut, highcut):
    df = pd.read_csv(input_file)

    fs = 200  # Abtastfrequenz

    start_time = offset_start_sec
    end_time = df.iloc[-1, 0] - offset_end_sec

    pruned_df = df[(df.iloc[:, 0] >= start_time) & (df.iloc[:, 0] <= end_time)]

    for col in pruned_df.columns[1:]:
        pruned_df[col] = apply_lowpass_filter(pruned_df[col], lowcut, highcut, fs)

    pruned_df.to_csv(output_file, index=False)

def create_movement_dataframe(pruned_file, lowcut, highcut, person_name, movement_name):
    df = pd.read_csv(pruned_file)

    fs = 200  # Abtastfrequenz

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

    local_peaks_x = calculate_local_peaks(df.iloc[:, 1], fs, lowcut, highcut)
    local_peaks_y = calculate_local_peaks(df.iloc[:, 2], fs, lowcut, highcut)
    local_peaks_z = calculate_local_peaks(df.iloc[:, 3], fs, lowcut, highcut)

    # Restlicher Code zur Extraktion der weiteren Features bleibt unverändert

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
        "Label": movement_name,
        "Person_Name": person_name
    }

    return row

def main(root_folder):
    offset_start_sec = 6
    offset_end_sec = 6
    lowcut = 1  # Setze deine gewünschte Cut-off-Frequenz hier ein
    highcut = 4  # Setze deine gewünschte Cut-off-Frequenz hier ein

    movement_name = input("Bitte geben Sie den Bewegungsnamen ein:")

    feature_data = []

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file == "Accelerometer.csv":
                input_file = os.path.join(root, file)
                output_file = os.path.join(root, f"{movement_name}.csv")

                prune_and_filter_data(input_file, output_file, offset_start_sec, offset_end_sec, lowcut, highcut)

                person_name = os.path.basename(root).split('_')[0]  # Extrahiere den Namen aus dem Verzeichnisnamen
                feature_row = create_movement_dataframe(output_file, lowcut, highcut, person_name, movement_name)
                feature_data.append(feature_row)

    feature_df = pd.DataFrame(feature_data)
    output_csv = os.path.join(root_folder, f"{root_folder}_features.csv")
    feature_df.to_csv(output_csv, index=False)
    print(f"Combined feature dataset created: {output_csv}")

if __name__ == "__main__":
    root_folder = input("Bitte geben Sie den Namen des Root-Ordners ein:")
    main(root_folder)
