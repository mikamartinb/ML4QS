from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Lade den Datensatz
data = pd.read_csv('DATEN.csv')

# Aufteilen der Features und Labels
features = data.drop('Label', axis=1)
labels = data['Label']

# Aufteilen in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Definiere eine Liste von K-Werten, die du ausprobieren möchtest
k_values = [3, 5, 7, 9, 11]

# Durchlaufe verschiedene Werte von K und validiere das Modell
for k in k_values:
    # Initialisiere den KNN-Klassifikator mit dem aktuellen K-Wert
    knn = KNeighborsClassifier(n_neighbors=k)

    # K-Fold Cross Validation
    scores = cross_val_score(knn, X_train, y_train, cv=5)

    # Trainiere das Modell
    knn.fit(X_train, y_train)

    # Vorhersagen für das Testset
    predictions = knn.predict(X_test)

    # Evaluierung der Genauigkeit
    accuracy = accuracy_score(y_test, predictions)

    # Ausgabe der Ergebnisse
    print("K = {}: Durchschnittliche Genauigkeit über 5-Fold Cross Validation: {:.2f}%".format(k, scores.mean() * 100))
    print("K = {}: Genauigkeit auf dem Testset: {:.2f}%".format(k, accuracy * 100))

    # Klassifikationsbericht
    print("Klassifikationsbericht:")
    print(classification_report(y_test, predictions))

    # Konfusionsmatrix
    print("Konfusionsmatrix:")
    print(confusion_matrix(y_test, predictions))

    print("\n")
