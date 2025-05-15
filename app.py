# 📁 app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.ensemble import IsolationForest

# ----------------------------
# 1. Daten & Modelle laden
# ----------------------------
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'data', 'ai4i2020.csv')
rf_model_path = os.path.join(base_path, 'models', 'model_rf.pkl')
iso_model_path = os.path.join(base_path, 'models', 'model_iso.pkl')

# Daten laden
df = pd.read_csv(data_path)

# Modelle laden
with open(rf_model_path, 'rb') as f:
    rf, rf_cols = pickle.load(f)

with open(iso_model_path, 'rb') as f:
    iso, _ = pickle.load(f)

# Features vorbereiten
features = ['Torque [Nm]', 'Tool wear [min]', 'Rotational speed [rpm]', 'Process temperature [K]', 'Type']
X = pd.get_dummies(df[features], drop_first=True)
for col in rf_cols:
    if col not in X.columns:
        X[col] = 0
X = X[rf_cols]

# Berechnungen
df['rf_proba'] = rf.predict_proba(X)[:, 1]
df['anomaly_flag'] = (iso.predict(X) == -1).astype(int)
df['tool_wear_scaled'] = (df['Tool wear [min]'] - df['Tool wear [min]'].min()) / (df['Tool wear [min]'].max() - df['Tool wear [min]'].min())
df['risk_score'] = 0.5 * df['rf_proba'] + 0.3 * df['tool_wear_scaled'] + 0.2 * df['anomaly_flag']

def classify_risk(score):
    if score < 0.3:
        return 'Unkritisch'
    elif score < 0.6:
        return 'Verdächtig'
    else:
        return 'Hochrisiko'

df['risk_label'] = df['risk_score'].apply(classify_risk)

# ----------------------------
# 2. Streamlit UI
# ----------------------------
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("🔧 Predictive Maintenance – Fehlerindikator Dashboard")

# Filteroption
st.sidebar.header("🔍 Filter")
risk_filter = st.sidebar.multiselect(
    "Risikostufe auswählen",
    options=df['risk_label'].unique(),
    default=df['risk_label'].unique()
)

df_filtered = df[df['risk_label'].isin(risk_filter)]

# Tabelle
st.subheader("🧾 Übersicht der Maschinenzustände")
st.dataframe(df_filtered[['Torque [Nm]', 'Tool wear [min]', 'rf_proba', 'anomaly_flag', 'risk_score', 'risk_label']].round(3))

# Balkendiagramm
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

# ----------------------------
# 3. Manuelle Eingabe
# ----------------------------
st.markdown("---")
st.subheader("🛠️ Manuelle Eingabe: Risikobewertung simulieren")

with st.form("manual_input"):
    torque = st.number_input("🔧 Drehmoment [Nm]", min_value=0.0, max_value=100.0, value=40.0)
    tool_wear = st.number_input("🛠️ Tool Wear [min]", min_value=0.0, max_value=250.0, value=150.0)
    rpm = st.number_input("🔄 Drehzahl [rpm]", min_value=0.0, max_value=3000.0, value=1500.0)
    temp = st.number_input("🌡️ Prozesstemperatur [K]", min_value=250.0, max_value=400.0, value=310.0)
    machine_type = st.selectbox("🏭 Maschinentyp", ["L", "M", "H"])

    submitted = st.form_submit_button("✅ Risiko berechnen")

    if submitted:
        input_dict = {
            'Torque [Nm]': torque,
            'Tool wear [min]': tool_wear,
            'Rotational speed [rpm]': rpm,
            'Process temperature [K]': temp,
            'Type': machine_type
        }
        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df, drop_first=True)

        for col in rf_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[rf_cols]

        rf_score = rf.predict_proba(input_encoded)[0][1]
        tool_wear_scaled = (tool_wear - df['Tool wear [min]'].min()) / (df['Tool wear [min]'].max() - df['Tool wear [min]'].min())
        anomaly = 1 if iso.predict(input_encoded)[0] == -1 else 0
        risk_score = 0.5 * rf_score + 0.3 * tool_wear_scaled + 0.2 * anomaly

        if risk_score < 0.3:
            label = "🟢 Unkritisch"
            color = "green"
        elif risk_score < 0.6:
            label = "🟠 Verdächtig"
            color = "orange"
        else:
            label = "🔴 Hochrisiko"
            color = "red"

        st.markdown(f"### 🔍 Ergebnis:")
        st.markdown(f"**Risikostufe:** <span style='color:{color}; font-size:24px'>{label}</span>", unsafe_allow_html=True)
        st.markdown(f"**Risikowert:** `{risk_score:.3f}`")
        st.markdown(f"**RF-Wahrscheinlichkeit:** `{rf_score:.3f}`  |  **Tool wear (skaliert):** `{tool_wear_scaled:.3f}`  |  **Anomalie erkannt:** `{bool(anomaly)}`")

