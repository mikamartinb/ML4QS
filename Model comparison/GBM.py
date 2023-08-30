import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Lade die Daten aus der CSV-Datei
data = pd.read_csv("dataset.csv")

# Entferne die Spalte "Person_Name"
data = data.drop(columns=["Person_Name"])

# Konvertiere Label in numerische Werte
label_encoder = LabelEncoder()
data["Label"] = label_encoder.fit_transform(data["Label"])

# Teile die Daten in Features (X) und Ziel (y) auf
X = data.drop(columns=["Label"])
y = data["Label"]

# Teile die Daten in Trainings- und Testsets auf
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisiere den Gradient Boosting Classifier
gbm = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)  # Passe die Hyperparameter nach Bedarf an
gbm.fit(X_train, y_train)

# Mache Vorhersagen auf den Testdaten
y_pred = gbm.predict(X_test)

# Bewertung der Vorhersagen
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_rep)
