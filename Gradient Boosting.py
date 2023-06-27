from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Lade den Datensatz
data = pd.read_csv('DATEN.csv')

# Aufteilen der Features und Labels
features = data.drop('Label', axis=1)
labels = data['Label']

# Aufteilen in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialisiere den Gradient Boosting Klassifikator
gb = GradientBoostingClassifier(n_estimators=100)

# Trainiere das Modell
gb.fit(X_train, y_train)

# Vorhersagen für Trainings- und Testdaten
train_predictions = gb.predict(X_train)
test_predictions = gb.predict(X_test)

# Ausgabe der Genauigkeiten
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)
print("Trainingsgenauigkeit: {:.2f}%".format(train_accuracy * 100))
print("Testgenauigkeit: {:.2f}%".format(test_accuracy * 100))

# Klassifikationsbericht und Konfusionsmatrix für Trainingsdaten
print("Klassifikationsbericht für Trainingsdaten:")
print(classification_report(y_train, train_predictions))
print("Konfusionsmatrix für Trainingsdaten:")
print(confusion_matrix(y_train, train_predictions))

# Klassifikationsbericht und Konfusionsmatrix für Testdaten
print("Klassifikationsbericht für Testdaten:")
print(classification_report(y_test, test_predictions))
print("Konfusionsmatrix für Testdaten:")
print(confusion_matrix(y_test, test_predictions))
