import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Annahme: Ihre Daten sind in einer Datei namens 'dataset.csv'
data = pd.read_csv('../dataset.csv')

# Spalten, die nicht verwendet werden sollen
columns_to_exclude = ["Label", "Person_Name", "Mean X", "Mean Y", "Mean Z"]

# Personenliste erstellen (unabhängig von Groß-/Kleinschreibung)
persons = data['Person_Name'].str.lower().unique()

# Hyperparameter-Raster für die zufällige Suche festlegen
param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

# Speichert die besten Parameter für jede Person
best_params_per_person = {}

# Schleife über alle Personen
for person in persons:
    # Daten für die aktuelle Person auswählen
    person_data = data[data['Person_Name'].str.lower() == person]

    # Alle Spalten außer den ausgeschlossenen auswählen
    X = person_data.drop(columns=columns_to_exclude)
    y = person_data['Label']

    # Aufteilung in Trainings- und Testdaten
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialisierung des Random Forest Classifiers mit den besten gemeinsamen Parametern
    best_common_params = {
        'n_estimators': 150,
        'max_depth': 20,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'auto'
    }
    model = RandomForestClassifier(random_state=42, **best_common_params)

    # Modell trainieren
    model.fit(X_train, y_train)

    # Vorhersagen für Testdaten
    y_pred = model.predict(X_test)

    # Genauigkeit berechnen und speichern
    accuracy = accuracy_score(y_test, y_pred)
    best_params_per_person[person] = accuracy

# Durchschnittliche Genauigkeit über alle Personen berechnen
average_accuracy = sum(best_params_per_person.values()) / len(best_params_per_person)

# Ausgabe der besten Parameter und der durchschnittlichen Genauigkeit
print("Beste Parameter für alle Personen: ", best_common_params)
print("Durchschnittliche Genauigkeit über alle Personen: {:.2f}".format(average_accuracy))
