# 📁 app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from src.utils import prepare_features

# ----------------------------
# 1. Daten & Modelle laden
# ----------------------------
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'data', 'ai4i2020.csv')
rf_model_path = os.path.join(base_path, 'models', 'model_rf.pkl')
iso_model_path = os.path.join(base_path, 'models', 'model_iso.pkl')

df = pd.read_csv(data_path)

with open(rf_model_path, 'rb') as f:
    rf, rf_cols = pickle.load(f)

with open(iso_model_path, 'rb') as f:
    iso, _ = pickle.load(f)

features = ['Torque [Nm]', 'Tool wear [min]', 'Rotational speed [rpm]', 'Process temperature [K]', 'Type']
X = prepare_features(df[features], rf_cols)

df['rf_proba'] = rf.predict_proba(X)[:, 1]
df['anomaly_flag'] = (iso.predict(X) == -1).astype(int)
df['tool_wear_scaled'] = (df['Tool wear [min]'] - df['Tool wear [min]'].min()) / (df['Tool wear [min]'].max() - df['Tool wear [min]'].min())
df['risk_score'] = 0.5 * df['rf_proba'] + 0.3 * df['tool_wear_scaled'] + 0.2 * df['anomaly_flag']

RISK_ORDER = ['Unkritisch', 'Verdächtig', 'Hochrisiko']
RISK_COLORS = {
    'Unkritisch': '#10B981',
    'Verdächtig': '#F59E0B',
    'Hochrisiko': '#EF4444',
}


def classify_risk(score):
    if score < 0.3:
        return 'Unkritisch'
    elif score < 0.6:
        return 'Verdächtig'
    else:
        return 'Hochrisiko'


df['risk_label'] = df['risk_score'].apply(classify_risk)

# ----------------------------
# 2. Streamlit UI – Setup
# ----------------------------
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2.5rem; padding-bottom: 3rem; }
        h1, h2, h3 { letter-spacing: -0.01em; }
        .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
        .stTabs [data-baseweb="tab"] {
            padding: 0.6rem 1.1rem;
            border-radius: 10px;
            background: rgba(255,255,255,0.03);
        }
        .stTabs [aria-selected="true"] {
            background: rgba(99,102,241,0.18) !important;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.03);
            padding: 14px 16px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.06);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🔧 Predictive Maintenance")
    st.caption(
        "Demo-Dashboard zur Risikoeinschätzung von Fräsmaschinen "
        "auf Basis des AI4I-2020-Datensatzes."
    )
    st.markdown("**Modelle**")
    st.markdown(
        "- Random Forest (Klassifikation)\n"
        "- Isolation Forest (Anomalie)\n"
        "- Tool-Wear-Skalierung"
    )
    st.divider()
    st.caption(f"📦 {len(df):,} Datensätze geladen")

st.title("🔧 Predictive Maintenance – Fehlerindikator")
st.caption("Eigene Sensordaten simulieren oder die Datenbasis explorieren.")

tab_sim, tab_data = st.tabs(["🛠️  Simulation", "📊  Datenbasis"])

# ----------------------------
# 3. Tab: Simulation
# ----------------------------
with tab_sim:
    st.subheader("Risikobewertung simulieren")
    st.write(
        "Gib Sensorwerte einer Maschine ein und erhalte eine Risiko-Einschätzung "
        "basierend auf dem kombinierten Fehlerindikator."
    )

    with st.form("manual_input"):
        c1, c2 = st.columns(2)
        with c1:
            torque = st.number_input("🔧 Drehmoment [Nm]", min_value=0.0, max_value=100.0, value=40.0)
            tool_wear = st.number_input("🛠️ Tool Wear [min]", min_value=0.0, max_value=250.0, value=150.0)
            rpm = st.number_input("🔄 Drehzahl [rpm]", min_value=0.0, max_value=3000.0, value=1500.0)
        with c2:
            temp = st.number_input("🌡️ Prozesstemperatur [K]", min_value=250.0, max_value=400.0, value=310.0)
            machine_type = st.selectbox("🏭 Maschinentyp", ["L", "M", "H"])

        submitted = st.form_submit_button("✅ Risiko berechnen", use_container_width=True)

    if submitted:
        input_dict = {
            'Torque [Nm]': torque,
            'Tool wear [min]': tool_wear,
            'Rotational speed [rpm]': rpm,
            'Process temperature [K]': temp,
            'Type': machine_type,
        }
        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df, drop_first=True)

        for col in rf_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[rf_cols]

        rf_score = rf.predict_proba(input_encoded)[0][1]
        tool_wear_scaled = (tool_wear - df['Tool wear [min]'].min()) / (df['Tool wear [min]'].max() - df['Tool wear [min]'].min())
        anomaly = int(iso.predict(input_encoded)[0] == -1)
        risk_score = 0.5 * rf_score + 0.3 * tool_wear_scaled + 0.2 * anomaly
        label = classify_risk(risk_score)
        color = RISK_COLORS[label]

        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {color}26 0%, {color}0D 100%);
                border-left: 4px solid {color};
                padding: 18px 22px;
                border-radius: 12px;
                margin: 18px 0 8px 0;
            ">
                <div style="font-size: 13px; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.08em;">
                    Risikostufe
                </div>
                <div style="font-size: 30px; font-weight: 700; color: {color}; margin-top: 4px;">
                    {label}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Risiko-Score", f"{risk_score:.3f}")
        m2.metric("RF-Wahrscheinlichkeit", f"{rf_score:.3f}")
        m3.metric("Tool wear (skaliert)", f"{tool_wear_scaled:.3f}")
        m4.metric("Anomalie erkannt", "Ja" if anomaly else "Nein")

# ----------------------------
# 4. Tab: Datenbasis (EDA)
# ----------------------------
with tab_data:
    st.subheader("Übersicht & Verteilung")
    st.write("Filtere den Datensatz nach Risikostufe und sieh dir Verteilung sowie Detailwerte an.")

    risk_filter = st.multiselect(
        "🔍 Risikostufen filtern",
        options=RISK_ORDER,
        default=RISK_ORDER,
    )
    df_filtered = df[df['risk_label'].isin(risk_filter)]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Datensätze", f"{len(df_filtered):,}")
    m2.metric("Unkritisch", f"{(df_filtered['risk_label'] == 'Unkritisch').sum():,}")
    m3.metric("Verdächtig", f"{(df_filtered['risk_label'] == 'Verdächtig').sum():,}")
    m4.metric("Hochrisiko", f"{(df_filtered['risk_label'] == 'Hochrisiko').sum():,}")

    col_chart, col_table = st.columns([1, 1.4])

    with col_chart:
        st.markdown("**Verteilung der Risikostufen**")
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        sns.countplot(
            x='risk_label',
            hue='risk_label',
            data=df_filtered,
            order=RISK_ORDER,
            palette=[RISK_COLORS[r] for r in RISK_ORDER],
            legend=False,
            ax=ax,
        )
        ax.set_xlabel('')
        ax.set_ylabel('Anzahl', color='#CBD5E1')
        ax.tick_params(colors='#CBD5E1')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis='y', alpha=0.15)
        st.pyplot(fig)

    with col_table:
        st.markdown("**Maschinenzustände**")
        st.dataframe(
            df_filtered[['Torque [Nm]', 'Tool wear [min]', 'rf_proba', 'anomaly_flag', 'risk_score', 'risk_label']].round(3),
            use_container_width=True,
            height=380,
        )

    st.download_button(
        label="📤 Gefilterte Daten als CSV exportieren",
        data=df_filtered.to_csv(index=False).encode('utf-8'),
        file_name='fehlerindikator_export.csv',
        mime='text/csv',
        use_container_width=True,
    )
