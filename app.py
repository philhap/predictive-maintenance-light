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

st.markdown("---")
st.subheader("🛠️ Manuelle Eingabe: Risikobewertung simulieren")

with st.form("manual_input"):
    st.markdown("### 📥 Sensorwerte eingeben")
    
    torque = st.number_input("🔧 Drehmoment [Nm]", min_value=0.0, max_value=100.0, value=40.0)
    tool_wear = st.number_input("🛠️ Tool Wear [min]", min_value=0.0, max_value=250.0, value=150.0)
    rpm = st.number_input("🔄 Drehzahl [rpm]", min_value=0.0, max_value=3000.0, value=1500.0)
    temp = st.number_input("🌡️ Prozesstemperatur [K]", min_value=250.0, max_value=400.0, value=310.0)
    machine_type = st.selectbox("🏭 Maschinentyp", ["M", "H"])  

    submitted = st.form_submit_button("✅ Risiko berechnen")

    if submitted:
        
       # Eingabedaten vorbereiten
        input_dict = {
            'Torque [Nm]': torque,
            'Tool wear [min]': tool_wear,
            'Rotational speed [rpm]': rpm,
            'Process temperature [K]': temp,
            'Type': machine_type
        }

        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df, drop_first=True)

        # Sicherstellen, dass alle Modellspalten enthalten sind
        for col in X.columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0  # fehlende Spalte → auffüllen

        # Spaltenreihenfolge anpassen
        input_encoded = input_encoded[X.columns]


        # Modellvorhersagen
        rf_score = rf.predict_proba(input_encoded)[0][1]
        tool_wear_scaled = (tool_wear - df['Tool wear [min]'].min()) / (df['Tool wear [min]'].max() - df['Tool wear [min]'].min())
        anomaly_flag = iso.predict(input_encoded)[0]
        anomaly = 1 if anomaly_flag == -1 else 0

        # Kombinierter Risiko-Score
        risk_score = 0.5 * rf_score + 0.3 * tool_wear_scaled + 0.2 * anomaly

        # Klassifikation
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
