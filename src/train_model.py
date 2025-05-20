import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import pickle
from utils import prepare_features

# Dynamischer Pfad
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_path, 'data', 'ai4i2020.csv')
model_dir = os.path.join(base_path, 'models')

# Daten laden
print("Lese CSV:", data_path)
df = pd.read_csv(data_path)
y = df['Machine failure']
features = ['Torque [Nm]', 'Tool wear [min]', 'Rotational speed [rpm]', 'Process temperature [K]', 'Type']

# Feature Columns initialisieren (z. B. aus vollständigen Dummy-Daten)
dummy_reference = pd.get_dummies(df[features], drop_first=True)
X = prepare_features(df[features], dummy_reference.columns.tolist())

# Modelle trainieren
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
iso.fit(X)

# Modelle speichern
os.makedirs(model_dir, exist_ok=True)

with open(os.path.join(model_dir, 'model_rf.pkl'), 'wb') as f:
    pickle.dump((rf, X.columns.tolist()), f)

with open(os.path.join(model_dir, 'model_iso.pkl'), 'wb') as f:
    pickle.dump((iso, X.columns.tolist()), f)

print("✅ Modelle gespeichert in:", model_dir)
