import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Daten laden
df = pd.read_csv('data/ai4i2020.csv')
y = df['Machine failure']
features = ['Torque [Nm]', 'Tool wear [min]', 'Rotational speed [rpm]', 'Process temperature [K]', 'Type']
X = pd.get_dummies(df[features], drop_first=True)

# Modell trainieren
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Modell speichern
with open('models/model_rf.pkl', 'wb') as f:
    pickle.dump((model, X.columns.tolist()), f)  