import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq

# Annahme: Dein Acceleration-Datensatz ist in einem Pandas DataFrame namens 'data' gespeichert
# Annahme: Die Beschleunigungswerte sind in den Spalten 'x', 'y' und 'z' enthalten

data = pd.read_csv("/Users/mika/.conda/envs/ML4QS/ALLE STEHEN/mika_stehen9/Accelerometer.csv")
# Schritt 1: Daten vorbereiten
# Extrahiere die Beschleunigungswerte in den drei Achsen
x = data['X (m/s^2)'].values
y = data['Y (m/s^2)'].values
z = data['Z (m/s^2)'].values

# Schritt 2: Fourier-Transformation durchführen
# Verwende beispielsweise die x-Achse für die Transformation
sampling_rate = 200  # Abtastrate von 200 Hz

# Führe die Fourier-Transformation durch
x_fft = fft(x)
freq = fftfreq(len(x), 1/sampling_rate)

# Schritt 3: Analyse der Frequenzkomponenten
# Die FFT gibt komplexe Werte zurück, daher nehmen wir den Betrag (Amplitude)
x_fft_abs = np.abs(x_fft)

# Schritt 4: Visualisierung der Ergebnisse (optional)
import matplotlib.pyplot as plt

plt.plot(freq, x_fft_abs)
plt.xlabel('Frequenz (Hz)')
plt.ylabel('Amplitude')
plt.show()
