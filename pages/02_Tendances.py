import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION DE LA PAGE (Doit être la 1ère commande Streamlit) ---
st.set_page_config(
    page_title="Observatoire Nid Douillet",
    page_icon="🏡",
    layout="wide", # Utilise toute la largeur de l'écran
    initial_sidebar_state="expanded"
)

# --- FONCTIONS FROM SCRATCH (À importer depuis analysis/regression.py et stats.py en prod) ---
def calculate_r2_from_scratch(x, y):
    n = len(x)
    if n < 2: return 0.0
    sum_x, sum_y = sum(x), sum(y)
    sum_x_sq = sum([i**2 for i in x])
    sum_y_sq = sum([i**2 for i in y])
    sum_xy = sum([x[i] * y[i] for i in range(n)])
    numerator = (n * sum_xy) - (sum_x * sum_y)
    denominator = ((n * sum_x_sq - sum_x**2) * (n * sum_y_sq - sum_y**2))**0.5
    if denominator == 0: return 0.0
    return (numerator / denominator)**2

def calculate_mean_from_scratch(data_list):
    if not data_list: return 0.0
    return sum(data_list) / len(data_list)

# --- STYLE CSS PERSONNALISÉ (Optionnel, pour peaufiner) ---
st.markdown("""
    <style>
    /* Adoucir les couleurs des métriques */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR : FILTRES GLOBAUX ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/602/602182.png", width=80) # Logo placeholder
    st.title("🏡 Nid Douillet")
    st.markdown("### Filtres de recherche")
    
    # Filtres typiques pour les primo-accédants
    type_bien = st.selectbox("Type de bien", ["Tous", "Appartement", "Maison"])
    budget_max = st.slider("Budget Maximum (€)", min_value=100000, max_value=450000, value=450000, step=10000)
    
    st.divider()
    st.info("💡 Objectif : Aider les jeunes couples primo-accédants à trouver le meilleur rapport qualité/prix à Toulon.")

# --- CORPS DE L'APPLICATION ---
st.title("📊 Observatoire du Marché Immobilier Toulonnais")
st.markdown("Visualisez les données réelles et les tendances pour conseiller au mieux vos clients.")

# Utilisation des TABS pour organiser l'information sans surcharger la page
tab1, tab2, tab3 = st.tabs(["📈 Tendances & Corrélation", "📍 État du marché", "🎯 Opportunités IA"])

with tab1:
    st.header("Dynamique du marché")
    try:
        # Simulation de chargement de données (à remplacer par tes vrais CSV)
        # df_dvf = pd.read_csv("donnees/processed/dvf_clean.csv").dropna(...)
        # POUR L'EXEMPLE : fausses données respectant le format attendu
        surfaces = [30, 45, 50, 65, 80, 95, 110]
        prix = [120000, 150000, 160000, 210000, 260000, 310000, 340000]
        
        col_metrics, col_chart = st.columns([1, 2])
        
        with col_metrics:
            r2_val = calculate_r2_from_scratch(surfaces, prix)
            st.metric(label="Coefficient $R^{2}$ (Surface vs Prix)", value=f"{r2_val:.4f}", delta="Fiabilité Haute" if r2_val > 0.7 else "Fiabilité Moyenne")
            
            with st.expander("Comment interpréter cette donnée ?"):
                st.write("Le $R^{2}$ (Coefficient de détermination) indique à quel point la surface explique le prix du bien. Plus on est proche de 1, plus la corrélation est forte.")
        
        with col_chart:
            fig, ax = plt.subplots(figsize=(7, 4))
            # Rendre le fond transparent pour s'intégrer au thème Streamlit (clair ou sombre)
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)
            ax.scatter(surfaces, prix, alpha=0.7, color="#ff4b4b", edgecolors="white", linewidth=1)
            ax.set_xlabel("Surface ($m^{2}$)")
            ax.set_ylabel("Prix (€)")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', linestyle='--', alpha=0.5) # Ajout d'une grille discrète
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")

with tab2:
    st.header("Prix par quartier")
    # Simulation des données "Scraper" groupées from scratch
    moyennes_quartiers = {"Mourillon": 4500, "Faron": 4100, "Pont-du-Las": 2800, "Saint-Jean": 3200}
    quartiers_tries = sorted(moyennes_quartiers.items(), key=lambda item: item[1])
    
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    fig2.patch.set_alpha(0)
    ax2.patch.set_alpha(0)
    ax2.barh([q[0] for q in quartiers_tries], [q[1] for q in quartiers_tries], color="#3498db", edgecolor="white")
    ax2.set_xlabel("Prix moyen / $m^{2}$ (€)")
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    st.pyplot(fig2)

with tab3:
    st.header("Biens Sous-évalués")
    st.success("Bientôt disponible : Identification des opportunités en dessous des prix du marché grâce à notre algorithme d'évaluation !")
    # C'est ici que tu brancheras ton scoring.py plus tard !