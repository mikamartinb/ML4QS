import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Lade die Daten aus der CSV-Datei
data = pd.read_csv("dataset.csv")

# Entferne die Spalte "Person_Name" und die mean-Features aus den Features
X = data.drop(columns=["Label", "Person_Name", "Mean X", "Mean Y", "Mean Z"])

# Verwende die "Label"-Spalte als das Ziel (y)
y = data["Label"]

# Teile die Daten in Trainings- und Testsets auf
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisiere den Random Forest Classifier
clf = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=10, max_features='sqrt', min_samples_leaf=1, min_samples_split=5) # optimale Parameter

# Trainiere den Classifier auf den Trainingsdaten
clf.fit(X_train, y_train)

# Speichere das trainierte Modell mit joblib
joblib.dump(clf, "trained_model.pkl")

# Mache Vorhersagen auf den Trainings- und Testdaten
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Erstelle die Konfusionsmatrix für Trainings- und Testdaten
conf_matrix_train = confusion_matrix(y_train, y_train_pred)
conf_matrix_test = confusion_matrix(y_test, y_test_pred)

# Erstelle die Heatmaps mit Farbthema Rot
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Reds', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel('Vorhersage')
plt.ylabel('Wahre Klasse')
plt.title('Konfusionsmatrix - Trainingsdaten')

plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Reds', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel('Vorhersage')
plt.ylabel('Wahre Klasse')
plt.title('Konfusionsmatrix - Testdaten')

plt.tight_layout()
plt.show()

# Bewertung der Vorhersagen auf Testdaten
accuracy = accuracy_score(y_test, y_test_pred)
classification_rep = classification_report(y_test, y_test_pred)

print("Accuracy auf Testdaten:", accuracy)
print("\nClassification Report auf Testdaten:\n", classification_rep)
