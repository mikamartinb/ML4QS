import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Lade die Daten aus der CSV-Datei
data = pd.read_csv("../dataset.csv")

# Entferne die Spalte "Person_Name" und die mean-Features aus den Features
X = data.drop(columns=["Label", "Person_Name", "Mean X", "Mean Y", "Mean Z"])

# Verwende die "Label"-Spalte als das Ziel (y)
y = data["Label"]

# Teile die Daten in Trainings- und Testsets auf
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definiere die Parameter, die du optimieren möchtest
param_dist = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialisiere den Random Forest Classifier
clf = RandomForestClassifier(random_state=42)

# Initialisiere die Random Search mit Kreuzvalidierung
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)

# Führe die Random Search durch
random_search.fit(X_train, y_train)

# Zeige die besten gefundenen Parameter
print("Beste Parameter:", random_search.best_params_)

# Verwende das Modell mit den besten Parametern
best_clf = random_search.best_estimator_

# Evaluierung auf den Testdaten
y_pred = best_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Beste Genauigkeit:", accuracy)
print("\nClassification Report:\n", classification_rep)

# Beste Parameter: {'n_estimators': 150, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 10}
# Beste Genauigkeit: 0.9425287356321839