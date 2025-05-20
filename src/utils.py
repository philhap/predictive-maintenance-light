import pandas as pd

def prepare_features(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    Einheitliche Feature-Vorbereitung für Training und Inferenz.
    Wandelt kategorische Variablen in Dummies um, füllt fehlende Spalten auf
    und sortiert die Spalten entsprechend dem Training.

    Args:
        df (pd.DataFrame): Eingabedaten (z. B. Originaldaten oder neue Eingaben)
        feature_columns (list): Spaltennamen, die das Modell erwartet (vom Training)

    Returns:
        pd.DataFrame: vorbereitete Feature-Matrix
    """
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Fehlende Spalten ergänzen
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Reihenfolge anpassen
    df_encoded = df_encoded[feature_columns]

    return df_encoded
