import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Lade die Daten aus der CSV-Datei
data = pd.read_csv("../dataset.csv")

# Entferne die Spalte "Person_Name" und die mean-Features aus den Features
X = data.drop(columns=["Label", "Person_Name", "Mean X", "Mean Y", "Mean Z"])

# Verwende die "Label"-Spalte als das Ziel (y)
y = data["Label"]

# Wandele die Personennamen in Kleinbuchstaben um
data["Person_Name"] = data["Person_Name"].str.lower()

# Finde alle einzigartigen Personen (Namen)
unique_persons = data["Person_Name"].unique()

# Initialisiere leere Listen, um die Ergebnisse zu speichern
accuracies = []
classification_reports = []

# Iteriere über jede Person und wende train_test_split an
for person in unique_persons:
    person_mask = data["Person_Name"] == person
    X_person = X[person_mask]
    y_person = y[person_mask]

    X_train, X_test, y_train, y_test = train_test_split(X_person, y_person, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    accuracies.append(accuracy)
    classification_reports.append(classification_rep)

# Zeige die durchschnittliche Genauigkeit
average_accuracy = sum(accuracies) / len(accuracies)
print("Durchschnittliche Genauigkeit:", average_accuracy)

# Zeige die Genauigkeiten für jede einzelne Person
for person, accuracy, classification_rep in zip(unique_persons, accuracies, classification_reports):
    print(f"\nGenauigkeit für {person}: {accuracy}")
    print(f"Classification Report für {person}:\n", classification_rep)
