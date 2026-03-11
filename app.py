import streamlit as st
import pandas as pd

st.set_page_config(page_title="État du Marché - Toulon", layout="wide")

@st.cache_data
def load_dvf():
    df = pd.read_csv("donnees/processed/dvf_clean.csv")
    # Conversion de la date
    df['date_mutation'] = pd.to_datetime(df['date_mutation'], dayfirst=True)
    return df

st.title("🏠 État du Marché Immobilier (DVF)")
st.markdown("---")

try:
    df_dvf = load_dvf()

    # --- TRAITEMENT DES DONNÉES POUR LES KPI ---
    # On ne garde qu'une ligne par mutation pour ne pas compter les prix plusieurs fois
    df_unique_mutations = df_dvf.drop_duplicates(subset=['id_mutation'])
    
    total_ventes = len(df_unique_mutations)
    prix_moyen_m2 = df_dvf['prix_au_m2'].mean()
    valeur_moyenne = df_unique_mutations['valeur_fonciere'].mean()

    # --- SECTION KPI ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Prix moyen au m²", f"{prix_moyen_m2:,.0f} €".replace(',', ' '))
    col2.metric("Total des transactions", f"{total_ventes}")
    col3.metric("Budget moyen par achat", f"{valeur_moyenne:,.0f} €".replace(',', ' '))

    # --- SECTION GRAPHIQUE ---
    st.subheader("📊 Distribution des prix au m²")
    # On limite entre 1000 et 10000€/m2 pour éviter que les erreurs de saisie n'écrasent le graph à droite
    prix_filtres = df_dvf[(df_dvf['prix_au_m2'] > 1000) & (df_dvf['prix_au_m2'] < 10000)]['prix_au_m2']
    
    # Création d'un histogramme plus propre
    bins = list(range(1000, 10500, 500))
    counts = pd.cut(prix_filtres, bins=bins).value_counts().sort_index()
    # On transforme l'index en string pour l'affichage
    counts.index = [f"{int(b.left)}-{int(b.right)}" for b in counts.index.categories]
    st.bar_chart(counts)

    # --- SECTION TABLEAU ---
    st.subheader("📑 Historique des dernières ventes (DVF)")
    # On ne montre que les colonnes essentielles pour le COMEX
    colonnes_utiles = [
        'date_mutation', 'nature_mutation', 'valeur_fonciere', 
        'type_local', 'surface_reelle_bati', 'prix_au_m2'
    ]
    df_display = df_unique_mutations[colonnes_utiles].sort_values(by='date_mutation', ascending=False)
    
    st.dataframe(
        df_display.head(50), 
        column_config={
            "date_mutation": "Date",
            "valeur_fonciere": st.column_config.NumberColumn("Prix de vente", format="%.0f €"),
            "surface_reelle_bati": "Surface (m²)",
            "prix_au_m2": st.column_config.NumberColumn("Prix m²", format="%.0f €"),
            "type_local": "Type"
        },
        hide_index=True,
        use_container_width=True
    )

except Exception as e:
    st.error(f"Erreur d'analyse : {e}")