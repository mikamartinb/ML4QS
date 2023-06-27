import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Lade den Datensatz
data = pd.read_csv('DATEN.csv')

# Aufteilen der Features und Labels
features = data.drop('Label', axis=1)
labels = data['Label']

# Aufteilen in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialisiere den SVM-Klassifikator
svm = SVC()

# Trainiere das Modell
svm.fit(X_train, y_train)

# Vorhersagen für das Testset
predictions = svm.predict(X_test)

# Evaluierung der Genauigkeit
accuracy = accuracy_score(y_test, predictions)
print("Genauigkeit: {:.2f}%".format(accuracy * 100))

# Klassifikationsbericht
print("Klassifikationsbericht:")
print(classification_report(y_test, predictions))

# Konfusionsmatrix
print("Konfusionsmatrix:")
print(confusion_matrix(y_test, predictions))
