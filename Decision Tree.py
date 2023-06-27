from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Lade den Datensatz
data = pd.read_csv('DATEN.csv')

# Aufteilen der Features und Labels
features = data.drop('Label', axis=1)
labels = data['Label']

# Aufteilen in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialisiere den Decision Tree Klassifikator
dt = DecisionTreeClassifier()

# Listen zur Aufzeichnung der Genauigkeiten
train_accuracies = []
test_accuracies = []

# Trainiere das Modell und speichere die Genauigkeiten
for i in range(1, 100):
    dt.fit(X_train, y_train)
    train_predictions = dt.predict(X_train)
    test_predictions = dt.predict(X_test)
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Plot der Genauigkeiten
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, 100), train_accuracies, label='Trainingsgenauigkeit')
plt.plot(np.arange(1, 100), test_accuracies, label='Testgenauigkeit')
plt.xlabel('Anzahl der Trainingsschritte')
plt.ylabel('Genauigkeit')
plt.title('Genauigkeiten von Trainings- und Testdaten')
plt.legend()
plt.show()

