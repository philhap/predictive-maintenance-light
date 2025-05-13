import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest

# ----------------------------
# 1. Daten vorbereiten
# ----------------------------
df = pd.read_csv('data/ai4i2020.csv')

# Ziel und Features
features = ['Torque [Nm]', 'Tool wear [min]', 'Rotational speed [rpm]', 'Process temperature [K]', 'Type']
X = pd.get_dummies(df[features], drop_first=True)

# Random Forest für Wahrscheinlichkeiten
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, df['Machine failure'])
df['rf_proba'] = rf.predict_proba(X)[:, 1]

# Isolation Forest
iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
df['anomaly_flag'] = (iso.fit_predict(X) == -1).astype(int)

# Tool wear normieren
df['tool_wear_scaled'] = (df['Tool wear [min]'] - df['Tool wear [min]'].min()) / (df['Tool wear [min]'].max() - df['Tool wear [min]'].min())

# Kombinierter Score
df['risk_score'] = (
    0.5 * df['rf_proba'] +
    0.3 * df['tool_wear_scaled'] +
    0.2 * df['anomaly_flag']
)

# Risikostufe zuweisen
def classify_risk(score):
    if score < 0.3:
        return 'Unkritisch'
    elif score < 0.6:
        return 'Verdächtig'
    else:
        return 'Hochrisiko'

df['risk_label'] = df['risk_score'].apply(classify_risk)

# ----------------------------
# 2. Streamlit-Dashboard
# ----------------------------
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

st.title("🔧 Predictive Maintenance – Fehlerindikator Dashboard")

# Filterbereich
st.sidebar.header("🔍 Filter")
risk_filter = st.sidebar.multiselect(
    "Risikostufe auswählen",
    options=df['risk_label'].unique(),
    default=df['risk_label'].unique()
)

df_filtered = df[df['risk_label'].isin(risk_filter)]

# Tabelle
st.subheader("🧾 Übersicht gemessener Maschinenzustände")
st.dataframe(df_filtered[['Torque [Nm]', 'Tool wear [min]', 'Process temperature [K]', 'rf_proba', 'anomaly_flag', 'risk_score', 'risk_label']].round(3))

# Verteilung
st.subheader("📊 Verteilung der Risikostufen")
fig, ax = plt.subplots()
sns.countplot(x='risk_label', data=df_filtered, order=['Unkritisch', 'Verdächtig', 'Hochrisiko'], palette="Set2", ax=ax)
st.pyplot(fig)

# Export
st.download_button(
    label="📤 Exportiere gefilterte Daten als CSV",
    data=df_filtered.to_csv(index=False).encode('utf-8'),
    file_name='fehlerindikator_export.csv',
    mime='text/csv'
)
