# 🔧 Predictive Maintenance Dashboard (Streamlit)

Ein interaktives Data-Science-Projekt zur Fehlererkennung und Risikobewertung bei Fräsmaschinen mit Hilfe von Machine Learning, basierend auf dem AI4I 2020 Datensatz.

🔗 [Live-Demo auf Streamlit Cloud](https://philhap-predictive-maintenance-light-app-dufrs2.streamlit.app)

## 📌 Projektziel

Dieses Dashboard kombiniert klassische Klassifikation (Random Forest), Anomalie-Erkennung (Isolation Forest) und Werkzeugverschleiß (Tool Wear) zu einem kombinierten **Risikowert**. Ziel ist es, Fehlerzustände frühzeitig zu erkennen, zu simulieren und eine fundierte Grundlage für Predictive-Maintenance-Maßnahmen zu schaffen.

## 🧰 Technologien & Tools

- Python, Pandas, Scikit-Learn
- Streamlit
- Pickle (Modellspeicherung)
- Seaborn & Matplotlib (Visualisierung)

## 📁 Projektstruktur

```
Predictive-Maintenance light/
├── data/
│   ├── ai4i2020.csv
│   ├── detected_anomalies.csv
│   ├── prediction-data.csv
│   └── predictions.csv
├── models/
│   ├── model_rf.pkl
│   └── model_iso.pkl
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_isolation_forest_featurevergleich.ipynb
│   ├── 03_3d_visualisierung.ipynb
│   ├── 04_classification_machine_failure.ipynb
│   ├── 05_classification_expanded.ipynb
│   ├── 06_multilabel_failuretypes.ipynb
│   └── 07_fehlerindikator.ipynb
├── src/
│   ├── train_model.py
│   └── predict_from_file.py
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

## 🚀 Start der App

```bash
pip install -r requirements.txt
streamlit run app.py
```

### 📦 Requirements.txt

```
pandas==2.2.2
scikit-learn==1.4.2
streamlit==1.34.0
matplotlib==3.8.4
seaborn==0.13.2
```

## 🔍 Features der App

- Analyse historischer Maschinendaten inkl. Filter & Visualisierung
- Kombinierter Fehlerindikator: RandomForest + Tool Wear + Isolation Forest
- Interaktive Simulation: Benutzer kann Sensordaten eingeben und Risikostufe berechnen lassen
- CSV-Export der aktuellen Analyse

## ⚙️ Training der Modelle

```bash
python src/train_model.py
```

Speichert die Modelle in `models/model_rf.pkl` und `models/model_iso.pkl`.

## 📊 Batch-Vorhersage via CSV

```bash
python src/predict_from_file.py
```

Liest `data/prediction-data.csv` und schreibt `data/predictions.csv` mit Risiko-Scores & Labels.

## ☁️ Perspektifisches Azure-Deployment (nach realer Datengrundlage)
Ein produktives Deployment der Anwendung über Microsoft Azure ist perspektivisch geplant – insbesondere in Kombination mit echten Maschinendaten. Die aktuelle Version basiert auf einem öffentlich verfügbaren Datensatz (AI4I2020) und dient als Proof of Concept.

Ein späterer Einsatz auf Azure App Service mit Anbindung an Azure IoT Hub ist vorgesehen, sobald ein modellbasiertes Training auf realen Betriebsdaten erfolgt ist.

```bash
#!/bin/bash
streamlit run app.py --server.port=$PORT --server.enableCORS=false
```

## 🧠 Autor & Motivation

Dieses Projekt wurde nach der Weiterbildung zum zertifizierten Data Scientist umgesetzt, um ein praxisnahes End-to-End-Projekt für Bewerbungen und Portfolio aufzubauen. Es dient gleichzeitig als Demonstration eines Fehlerindikators, der prinzipiell auch in eine Echtzeit-Fehleranalyse und IoT-Anbindung via Azure übertragbar ist – auch wenn der zugrunde liegende Datensatz simuliert ist und keine echten Live-Daten umfasst.

---



