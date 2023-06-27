import pandas as pd
import matplotlib.pyplot as plt

#dataframes
df = pd.read_csv("/Users/mika/.conda/envs/ML4QS/Mika_trepperunter1""/Accelerometer.csv")
df.drop('Time (s)', inplace=True, axis=1)

x = df['X (m/s^2)']
y = df['Y (m/s^2)']
z = df['Z (m/s^2)']

#visuals
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

x.plot(x='x', y='x_values', ax=axs[0], color='red')
axs[0].set_title('X Plot')

y.plot(x='x', y='y_values', ax=axs[1], color='blue')
axs[1].set_title('Y Plot')

z.plot(x='x', y='z_values', ax=axs[2], color='green')
axs[2].set_title('Z Plot')

plt.show()
