import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Lade die Daten aus der CSV-Datei
data = pd.read_csv("dataset.csv")

# Entferne die Spalte "Person_Name" aus den Features
X = data.drop(columns=["Label", "Person_Name"])  # "Label" und "Person_Name" entfernen

# Verwende die "Label"-Spalte als das Ziel (y)
y = data["Label"]

# Liste der Featuregruppen
feature_groups = [
    ["Mean X", "Mean Y", "Mean Z"],
    ["Standardabweichung X", "Standardabweichung Y", "Standardabweichung Z"],
    ["Min X", "Min Y", "Min Z"],
    ["Max X", "Max Y", "Max Z"],
    ["Lokale Peaks X", "Lokale Peaks Y", "Lokale Peaks Z"]
]

# Initialisiere leere Liste, um die Ergebnisse zu speichern
results = []

# Trainiere und evaluieren des Random Forests mit allen Features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
results.append(("Alle Features", accuracy, classification_rep))

# Iteriere über jede Featuregruppe und trainiere/evaluiere Random Forests ohne diese Gruppe
for group in feature_groups:
    # Wähle nur die Features der aktuellen Gruppe aus
    selected_features = [col for col in X.columns if col not in group]
    X_group = X[selected_features]

    # Teile die Daten in Trainings- und Testsets auf
    X_train_group, X_test_group, y_train_group, y_test_group = train_test_split(X_group, y, test_size=0.2,
                                                                                random_state=42)

    # Initialisiere den Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Trainiere den Classifier auf den Trainingsdaten
    clf.fit(X_train_group, y_train_group)

    # Mache Vorhersagen auf den Testdaten
    y_pred_group = clf.predict(X_test_group)

    # Bewertung der Vorhersagen
    accuracy_group = accuracy_score(y_test_group, y_pred_group)
    classification_rep_group = classification_report(y_test_group, y_pred_group)

    results.append(("Ohne " + ", ".join(group), accuracy_group, classification_rep_group))

# Zeige die Ergebnisse für jeden Random Forest
for name, accuracy, classification_rep in results:
    print(f"Random Forest {name}")
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_rep)
    print("=" * 50)
