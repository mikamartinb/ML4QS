import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Annahme: Ihre Daten sind in einer Datei namens 'dataset.csv'
data = pd.read_csv('../dataset.csv')

# Spalten, die nicht verwendet werden sollen
columns_to_exclude = ["Label", "Person_Name", "Mean X", "Mean Y", "Mean Z"]

# Personenliste erstellen (unabhängig von Groß-/Kleinschreibung)
persons = data['Person_Name'].str.lower().unique()

# Hyperparameter-Raster für die Gittersuche festlegen
param_grid = {
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

    # Aufteilung in Trainings- und Testdaten
    X_train, X_test, y_train, y_test = train_test_split(person_data.drop(columns=columns_to_exclude), person_data['Label'], test_size=0.2, random_state=42)

    # Initialisierung der Gittersuche
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        scoring='accuracy',
        cv=3  # Anzahl der Kreuzvalidierungsiterationen
    )

    # Modell trainieren mit Gittersuche
    grid_search.fit(X_train, y_train)

    # Beste Parameter für diese Person speichern
    best_params_per_person[person] = grid_search.best_params_

# Ermittle die am häufigsten vorkommenden Parameter über alle Personen
common_best_params = {}
for param in param_grid.keys():
    param_values = [params[param] for params in best_params_per_person.values()]
    most_common_param = max(set(param_values), key=param_values.count)
    common_best_params[param] = most_common_param

# Initialisierung des Random Forest Classifiers mit den besten gemeinsamen Parametern
model = RandomForestClassifier(random_state=42, **common_best_params)

# Genauigkeiten mit den besten gemeinsamen Parametern speichern
accuracies = []

# Schleife über alle Personen
for person in persons:
    # Daten für die aktuelle Person auswählen
    person_data = data[data['Person_Name'].str.lower() == person]

    # Aufteilung in Trainings- und Testdaten
    X_train, X_test, y_train, y_test = train_test_split(person_data.drop(columns=columns_to_exclude), person_data['Label'], test_size=0.2, random_state=42)

    # Modell trainieren
    model.fit(X_train, y_train)

    # Vorhersagen für Testdaten
    y_pred = model.predict(X_test)

    # Genauigkeit berechnen und speichern
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Durchschnittliche Genauigkeit über alle Personen berechnen
average_accuracy = sum(accuracies) / len(accuracies)

# Ausgabe der besten Parameter und der durchschnittlichen Genauigkeit
print("Gemeinsame beste Parameter für alle Personen: ", common_best_params)
print("Durchschnittliche Genauigkeit über alle Personen: {:.2f}".format(average_accuracy))
