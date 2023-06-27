import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


def apply_lowpass_filter(csv_file):
    df = pd.read_csv(csv_file)

    # Indizes der zu filternden Spalten
    accel_cols = [1, 2, 3]  # 2. bis 4. Spalte (0-basiert)

    # Anwenden des Lowpass-Filters auf die angegebenen Spalten
    for col in accel_cols:
        column_name = df.columns[col]
        df[column_name] = butter_lowpass_filter(df.iloc[:, col].values)

    # Extrahieren der Zeit und der Beschleunigungsdaten
    time = df.iloc[:, 0].values
    accel_data = df.iloc[:, accel_cols].values

    # Plot der unverarbeiteten Daten
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, accel_data[:, 0], label='Acceleration x')
    plt.plot(time, accel_data[:, 1], label='Acceleration y')
    plt.plot(time, accel_data[:, 2], label='Acceleration z')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.title('Unfiltered Data')
    plt.legend()

    # Anwenden des Lowpass-Filters auf die angegebenen Spalten
    accel_data_filtered = butter_lowpass_filter(accel_data)

    # Plot der gefilterten Daten
    plt.subplot(2, 1, 2)
    plt.plot(time, accel_data_filtered[:, 0], label='Filtered Acceleration x')
    plt.plot(time, accel_data_filtered[:, 1], label='Filtered Acceleration y')
    plt.plot(time, accel_data_filtered[:, 2], label='Filtered Acceleration z')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.title('Filtered Data')
    plt.legend()

    # Anpassen der Subplot-Abstände
    plt.tight_layout()

    # Anzeigen des Plots
    plt.show()


def butter_lowpass_filter(data, cutoff=3, fs=100, order=5):
    nyquist_freq = 0.5 * fs
    normalized_cutoff = cutoff / nyquist_freq
    b, a = butter(order, normalized_cutoff, btype='low', analog=False, output='ba')
    y = filtfilt(b, a, data, axis=0)
    return y



apply_lowpass_filter("/Users/mika/.conda/envs/ML4QS/ALLE TREPPEHOCH/Mika_treppehoch8/Accelerometer.csv")