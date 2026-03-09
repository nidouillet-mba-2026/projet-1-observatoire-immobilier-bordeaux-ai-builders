import streamlit as st
import pandas as pd

st.title("🔍 Recherche d'Annonces (Scraper)")

@st.cache_data
def load_scrap():
    return pd.read_csv("donnees/processed/annonces_propres.csv")

try:
    df_scr = load_scrap()

    # --- Sidebar Filtres ---
    st.sidebar.header("Paramètres de recherche")
    budget = st.sidebar.slider("Budget (€)", 0, 1000000, (0, 500000))
    surface_min = st.sidebar.number_input("Surface min (m²)", 0, 500, 20)
    
    quartier_list = df_scr['quartier'].unique()
    selected_q = st.sidebar.multiselect("Quartier(s)", quartier_list)
    
    type_list = df_scr['type'].unique()
    selected_t = st.sidebar.multiselect("Type de bien", type_list)

    # --- Filtrage ---
    mask = (df_scr['prix'] >= budget[0]) & (df_scr['prix'] <= budget[1]) & (df_scr['surface'] >= surface_min)
    if selected_q:
        mask &= df_scr['quartier'].isin(selected_q)
    if selected_t:
        mask &= df_scr['type'].isin(selected_t)

    df_filtered = df_scr[mask]

    st.subheader(f"Résultats : {len(df_filtered)} annonces trouvées")
    st.dataframe(df_filtered)

except Exception as e:
    st.error(f"Erreur : Vérifiez le fichier 'donnes/processed/annonces_propres.csv'. Détails : {e}")