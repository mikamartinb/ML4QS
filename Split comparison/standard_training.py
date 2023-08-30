import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Lade die Daten aus der CSV-Datei
data = pd.read_csv("../dataset.csv")

# Entferne die Spalte "Person_Name" aus den Features
X = data.drop(columns=["Label", "Person_Name", "Mean X", "Mean Y", "Mean Z"])  # "Label" und "Person_Name" entfernen

# Verwende die "Label"-Spalte als das Ziel (y)
y = data["Label"]

# Teile die Daten in Trainings- und Testsets auf
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisiere den Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Trainiere den Classifier auf den Trainingsdaten
clf.fit(X_train, y_train)

# Mache Vorhersagen auf den Testdaten
y_pred = clf.predict(X_test)

# Bewertung der Vorhersagen
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_rep)
