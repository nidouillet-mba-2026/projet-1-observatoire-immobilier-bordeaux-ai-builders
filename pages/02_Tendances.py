import streamlit as st
import pandas as pd

st.title("📈 Tendances & Corrélation")

# Fonction R² "From Scratch" (sans sklearn/numpy)
def calculate_r2_manual(df, col_x, col_y):
    data = df[[col_x, col_y]].dropna()
    x = data[col_x].tolist()
    y = data[col_y].tolist()
    n = len(x)
    if n < 2: return 0.0

    sum_x = sum(x)
    sum_y = sum(y)
    sum_x_sq = sum([i**2 for i in x])
    sum_y_sq = sum([i**2 for i in y])
    sum_xy = sum([x[i] * y[i] for i in range(n)])

    numerator = (n * sum_xy) - (sum_x * sum_y)
    denominator = ((n * sum_x_sq - sum_x**2) * (n * sum_y_sq - sum_y**2))**0.5
    
    if denominator == 0: return 0.0
    r = numerator / denominator
    return r**2

try:
    df_dvf = pd.read_csv("donnees/processed/dvf_clean.csv")
    df_scr = pd.read_csv("donnees/processed/annonces_propres.csv")

    st.subheader("Corrélation Prix vs Surface (DVF)")
    r2_val = calculate_r2_manual(df_dvf, 'surface_reelle_bati', 'valeur_fonciere')
    st.metric("Coefficient de détermination (R²)", f"{r2_val:.4f}")

    # Courbe prix moyen par quartier (via le Scraper)
    st.subheader("Prix moyen au m² par quartier (Offre actuelle)")
    if 'quartier' in df_scr.columns:
        trend = df_scr.groupby('quartier')['prix_m2'].mean().sort_values()
        st.line_chart(trend)

except Exception as e:
    st.error(f"Erreur d'analyse : {e}")