import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV


# Lade die Daten aus der CSV-Datei
data = pd.read_csv("../dataset.csv")

# Entferne die Spalte "Person_Name" und die mean-Features aus den Features
X = data.drop(columns=["Label", "Person_Name", "Mean X", "Mean Y", "Mean Z"])

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

# Definiere die Parameter, die du optimieren möchtest
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialisiere den Random Forest Classifier
clf = RandomForestClassifier(random_state=42)

# Initialisiere die Grid Search mit Kreuzvalidierung
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')

# Führe die Grid Search durch
grid_search.fit(X_train, y_train)

# Zeige die besten gefundenen Parameter
print("Beste Parameter:", grid_search.best_params_)

# Verwende das Modell mit den besten Parametern
best_clf = grid_search.best_estimator_

# Evaluierung auf den Testdaten
y_pred = best_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Beste Genauigkeit:", accuracy)
print("\nClassification Report:\n", classification_rep)
