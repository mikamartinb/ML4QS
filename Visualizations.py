# Visualisierung Offset-Funktion
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import butter, filtfilt

def visualize_data(csv_file_path):
    # CSV-Datei einlesen
    df = pd.read_csv(csv_file_path)

    # Visualisierung der x, y und z Werte in einem Subplot
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))

    # x-Werte plotten
    axes[0].plot(df['Time (s)'], df['Acceleration x (m/s^2)'], color='darkred')
    axes[0].set_title('X')

    # y-Werte plotten
    axes[1].plot(df['Time (s)'], df['Acceleration y (m/s^2)'], color='darkred')
    axes[1].set_title('Y')

    # z-Werte plotten
    axes[2].plot(df['Time (s)'], df['Acceleration z (m/s^2)'], color='darkred')
    axes[2].set_title('Z')

    plt.tight_layout()
    plt.show()

def print_individual_std(data_path):
    data = pd.read_csv(data_path)

    std_columns = ['Standardabweichung X', 'Standardabweichung Y', 'Standardabweichung Z']

    for index, row in data.iterrows():
        print(f"Datenpunkt {index + 1}:")
        for column in std_columns:
            std_value = row[column]
            print(f"  {column}: {std_value:.6f}")
        print("-" * 30)

def plot_csv_data(csv_filename):
    # CSV-Datei einlesen
    data = pd.read_csv(csv_filename)

    # Daten aus der CSV-Datei extrahieren
    time = data.iloc[:, 0]  # Index 0 entspricht "Time (s)"
    x = data.iloc[:, 1]     # Index 1 entspricht "X (m/s^2)"
    y = data.iloc[:, 2]     # Index 2 entspricht "Y (m/s^2)"
    z = data.iloc[:, 3]     # Index 3 entspricht "Z (m/s^2)"

    # Erstelle Subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    # Plot für X-Beschleunigung
    axs[0].plot(time, x, linestyle='-')
    axs[0].set_title('X-Beschleunigung')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('X (m/s^2)')

    # Plot für Y-Beschleunigung
    axs[1].plot(time, y, color='orange', linestyle='-')
    axs[1].set_title('Y-Beschleunigung')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Y (m/s^2)')

    # Plot für Z-Beschleunigung
    axs[2].plot(time, z, color='green', linestyle='-')
    axs[2].set_title('Z-Beschleunigung')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Z (m/s^2)')

    # Layout anpassen
    plt.tight_layout()

    # Anzeigen der Plots
    plt.show()

def apply_lowpass_filter(data, lowcut, highcut, fs):
    b, a = butter(4, [lowcut, highcut], fs=fs, btype='band')
    return filtfilt(b, a, data)

def plot_data_with_filter(csv_path, lowcut, highcut, fs):
    df = pd.read_csv(csv_path)
    time = df.iloc[:, 0]
    data = df.iloc[:, 1]

    offset = 9  # Sekunden Offset

    # Kürzen der Daten mit Offset
    start_time = offset
    end_time = time.iloc[-1] - offset
    df_offset = df[(df.iloc[:, 0] >= start_time) & (df.iloc[:, 0] <= end_time)]
    time_offset = df_offset.iloc[:, 0]
    data_offset = df_offset.iloc[:, 1]

    # Anwenden des Lowpass-Filters auf die Daten mit Offset
    data_filtered = apply_lowpass_filter(data_offset, lowcut, highcut, fs)

    # Erstellen der Subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].plot(time_offset, data_offset, color='blue')
    axs[0].set_title("Daten ohne Filter")
    axs[0].set_xlabel("Zeit (s)")
    axs[0].set_ylabel("Daten")

    axs[1].plot(time_offset, data_filtered, color='red')
    axs[1].set_title("Daten mit Lowpass-Filter")
    axs[1].set_xlabel("Zeit (s)")
    axs[1].set_ylabel("Daten (gefiltert)")

    plt.tight_layout()
    plt.show()

# Aufruf der Funktion mit dem Pfad zu deiner CSV-Datei und den gewünschten Filtereinstellungen
#csv_path = "/Users/mika/.conda/envs/ML4QS/ALLE GEHEN/mika_gehen16/Accelerometer.csv"
#lowcut = 0.5  # Untere Grenzfrequenz des Filters
#highcut = 1.8  # Obere Grenzfrequenz des Filters
#fs = 200.0  # Abtastfrequenz in Hz

#plot_data_with_filter(csv_path, lowcut, highcut, fs)


# Beispielaufruf der Funktion mit dem Dateinamen deiner CSV-Datei
#csv_file_path = '/Users/mika/.conda/envs/ML4QS/ALLE AUTO/Jonte_auto8/accelerometer.csv'
#plot_csv_data(csv_file_path)

# visualize_data(csv_file_path)
