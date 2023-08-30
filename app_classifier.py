import pandas as pd
from scipy.signal import butter, lfilter, find_peaks
import joblib

def trim_csv_with_offset_return(data, offset_seconds=6):
    fs = 1 / (data.iloc[1, 0] - data.iloc[0, 0])
    offset_samples = int(offset_seconds * fs)
    trimmed_data = data.iloc[offset_samples:-offset_samples, :]
    return trimmed_data

def apply_lowpass_filter_to_dataframe(data, low_cut=0.5, high_cut=5):
    fs = 1 / (data.iloc[1, 0] - data.iloc[0, 0])

    def butter_lowpass(cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    filtered_data = data.copy()
    for col_index in range(1, 4):
        filtered_data.iloc[:, col_index] = butter_lowpass_filter(data.iloc[:, col_index], high_cut, fs)

    return filtered_data

def calculate_local_peaks(data):
    peaks_x, _ = find_peaks(data.iloc[:, 1], distance=50)
    peaks_y, _ = find_peaks(data.iloc[:, 2], distance=50)
    peaks_z, _ = find_peaks(data.iloc[:, 3], distance=50)
    return len(peaks_x), len(peaks_y), len(peaks_z)

# Lade die trainierten Modellgewichte
clf = joblib.load("trained_model.pkl")

# Lade das Testdatenset (Rohdaten)
test_data = pd.read_csv("/Users/mika/.conda/envs/ML4QS/ALLE RAD/Alex_fahrrad7/Accelerometer.csv")

# Anwende Vorverarbeitung (Offset, Lowpass-Filter)
trimmed_data = trim_csv_with_offset_return(test_data)
print(trimmed_data)
filtered_data = apply_lowpass_filter_to_dataframe(trimmed_data, high_cut=0.15)
print(filtered_data)
# Berechne die Features
local_peaks_x, local_peaks_y, local_peaks_z = calculate_local_peaks(filtered_data)

# Erstelle die Eingabe-Features für das Modell (ohne "mean")
input_features = [
    local_peaks_x, local_peaks_y, local_peaks_z,
    filtered_data.iloc[:, 1].std(), filtered_data.iloc[:, 2].std(), filtered_data.iloc[:, 3].std(),
    filtered_data.iloc[:, 1].min(), filtered_data.iloc[:, 1].max(),
    filtered_data.iloc[:, 2].min(), filtered_data.iloc[:, 2].max(),
    filtered_data.iloc[:, 3].min(), filtered_data.iloc[:, 3].max()
]

# Mache Vorhersage mit dem trainierten Modell
prediction = clf.predict([input_features])

print("Vorhersage:", prediction)
