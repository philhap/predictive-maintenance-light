# 📁 src/predict_from_file.py
import pandas as pd
import pickle
import os

# Basisverzeichnis ermitteln (relativ zu dieser Datei)
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_path, 'data', 'prediction-data.csv')
rf_model_path = os.path.join(base_path, 'models', 'model_rf.pkl')
iso_model_path = os.path.join(base_path, 'models', 'model_iso.pkl')
output_path = os.path.join(base_path, 'data', 'predictions.csv')

# Eingabedaten laden
print("Lade Eingabedaten von:", data_path)
input_df = pd.read_csv(data_path)

# Modelle laden
with open(rf_model_path, 'rb') as f:
    rf, rf_cols = pickle.load(f)

with open(iso_model_path, 'rb') as f:
    iso, iso_cols = pickle.load(f)

# Eingabedaten vorbereiten
X = pd.get_dummies(input_df, drop_first=True)
for col in rf_cols:
    if col not in X.columns:
        X[col] = 0
X = X[rf_cols]

# Vorhersagen berechnen
rf_proba = rf.predict_proba(X)[:, 1]
anomaly_flag = (iso.predict(X) == -1).astype(int)
tool_wear_scaled = (input_df['Tool wear [min]'] - input_df['Tool wear [min]'].min()) / (input_df['Tool wear [min]'].max() - input_df['Tool wear [min]'].min())

# Risiko-Score berechnen
risk_score = 0.5 * rf_proba + 0.3 * tool_wear_scaled + 0.2 * anomaly_flag

def classify_risk(score):
    if score < 0.3:
        return 'Unkritisch'
    elif score < 0.6:
        return 'Verdächtig'
    else:
        return 'Hochrisiko'

risk_label = [classify_risk(s) for s in risk_score]

# Ergebnisse speichern
input_df['rf_proba'] = rf_proba
input_df['anomaly'] = anomaly_flag
input_df['tool_wear_scaled'] = tool_wear_scaled
input_df['risk_score'] = risk_score
input_df['risk_label'] = risk_label

input_df.to_csv(output_path, index=False)
print("✅ Vorhersagen gespeichert in:", output_path)


