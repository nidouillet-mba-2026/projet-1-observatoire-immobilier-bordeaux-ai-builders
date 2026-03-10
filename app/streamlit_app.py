"""
Observatoire Immobilier Toulonnais , Dashboard principal
NidDouillet | Toulon (INSEE 83137) | 2024
"""

import math
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.regression import least_squares_fit, r_squared
from analysis.stats import correlation, mean, standard_deviation, variance

# ── Config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Observatoire Immobilier Toulonnais",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DVF_PATH = os.path.join(ROOT, "data", "dvf_toulon.csv")
ANN_PATH = os.path.join(ROOT, "data", "annonces.csv")

BG_BLUE = "#EFF6FF"
BLUE = "#2563EB"
RED = "#DC2626"
VIOLET = "#7C3AED"
GREEN = "#059669"
AMBER = "#D97706"

fmt_eur = mticker.FuncFormatter(lambda x, _: f"{x:,.0f} €")
fmt_m2 = mticker.FuncFormatter(lambda x, _: f"{x:,.0f} €/m²")


# ── Chargement ─────────────────────────────────────────────────────────────
@st.cache_data
def load_dvf() -> pd.DataFrame:
    df = pd.read_csv(DVF_PATH, low_memory=False)
    df["date_mutation"] = pd.to_datetime(df["date_mutation"], dayfirst=True, errors="coerce")
    df = df[
        df["prix_au_m2"].notna()
        & (df["prix_au_m2"] >= 500)
        & (df["prix_au_m2"] <= 20_000)
        & df["type_local"].isin(["Appartement", "Maison"])
    ].copy()
    return df


@st.cache_data
def load_annonces() -> pd.DataFrame:
    df = pd.read_csv(ANN_PATH)
    return df[df["prix_m2"].notna() & (df["prix_m2"] > 0)].copy()


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/fr/d/d5/Logo_Ville_de_Toulon.svg", width=80)
    st.title("Observatoire Immobilier")
    st.caption("Toulon , INSEE 83137")
    st.divider()
    st.subheader("Filtres globaux")

    dvf_raw = load_dvf()
    ann_raw = load_annonces()

    type_filter = st.multiselect(
        "Type de bien",
        ["Appartement", "Maison"],
        default=["Appartement", "Maison"],
        help="Filtrer par type de bien sur l'ensemble du dashboard."
    )

    pm2_range = st.slider(
        "Fourchette prix/m² (€/m²)",
        500, 15_000, (1_000, 10_000), step=100,
        help="Filtre les transactions et annonces hors de cette fourchette de prix au m²."
    )

    st.divider()
    st.markdown("**Sources de données**")
    st.markdown(f"🔵 DVF : **{len(dvf_raw):,}** transactions")
    st.markdown(f"🔴 SeLoger : **{len(ann_raw):,}** annonces")
    st.markdown(f"📅 Période DVF : 2024")
    st.divider()
    st.caption("NidDouillet · Projet Epitech IA Spé")

dvf = dvf_raw[
    dvf_raw["type_local"].isin(type_filter)
    & dvf_raw["prix_au_m2"].between(pm2_range[0], pm2_range[1])
].copy()

ann = ann_raw[
    ann_raw["type"].isin(type_filter)
    & ann_raw["prix_m2"].between(pm2_range[0], pm2_range[1])
].copy()

# ── Header ──────────────────────────────────────────────────────────────────
st.title("🏠 Observatoire Immobilier Toulonnais")
st.caption("Données DVF (data.gouv.fr) × Annonces SeLoger | Toulon 2024")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Tableau de bord",
    "🗺️ Carte & Quartiers",
    "📈 Régression & Corrélations",
    "🎯 Opportunités",
    "📚 Méthodologie",
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 , Tableau de bord
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("État du marché immobilier , Toulon 2024")

    with st.expander("ℹ️ Comment lire ce tableau de bord ?", expanded=False):
        st.markdown("""
        **Ce tableau de bord combine deux sources complémentaires :**

        | Source | Nature | Couleur |
        |--------|--------|---------|
        | **DVF** (data.gouv.fr) | Transactions **réelles** enregistrées aux impôts en 2024 | 🔵 Bleu |
        | **SeLoger** (scraping) | Annonces en cours (prix **demandés**, pas encore vendus) | 🔴 Rouge |

        > **Différence clé** : Le DVF reflète ce qui a **réellement été payé**. SeLoger reflète ce que les vendeurs **espèrent obtenir**. L'écart entre les deux mesure la tension du marché.
        """)

    # KPIs DVF
    st.subheader("🔵 DVF , Transactions réelles 2024")
    c1, c2, c3, c4, c5 = st.columns(5)

    pm2_dvf = dvf["prix_au_m2"].tolist()
    c1.metric("Transactions", f"{len(dvf):,}",
              help="Nombre de ventes immobilières enregistrées aux impôts à Toulon en 2024, après filtrage des valeurs aberrantes.")
    c2.metric("Prix/m² moyen", f"{mean(pm2_dvf):,.0f} €/m²",
              help="Moyenne arithmétique des prix au m² de toutes les transactions. Sensible aux valeurs extrêmes.")
    c3.metric("Prix/m² médian", f"{dvf['prix_au_m2'].median():,.0f} €/m²",
              help="Valeur centrale : 50% des transactions sont en-dessous, 50% au-dessus. Plus robuste que la moyenne aux valeurs extrêmes.")
    c4.metric("Écart-type", f"{standard_deviation(pm2_dvf):,.0f} €/m²",
              help="Mesure la dispersion des prix. Un écart-type élevé indique un marché hétérogène avec de grands écarts de prix entre quartiers/biens.")
    c5.metric("Variance", f"{variance(pm2_dvf):,.0f}",
              help="Carré de l'écart-type. Mesure mathématique de la dispersion utilisée dans les calculs statistiques.")

    with st.expander("📖 Que signifient ces indicateurs ?"):
        st.markdown(f"""
        **Moyenne vs Médiane** : Sur ce marché, la moyenne ({mean(pm2_dvf):,.0f} €/m²)
        est {'supérieure' if mean(pm2_dvf) > dvf['prix_au_m2'].median() else 'inférieure'} à la médiane ({dvf['prix_au_m2'].median():,.0f} €/m²),
        ce qui indique une distribution {'asymétrique vers le haut (quelques biens très chers tirent la moyenne)' if mean(pm2_dvf) > dvf['prix_au_m2'].median() else 'asymétrique vers le bas'}.

        **Écart-type ({standard_deviation(pm2_dvf):,.0f} €/m²)** : Cela signifie que la majorité des biens
        ont un prix/m² compris entre **{mean(pm2_dvf) - standard_deviation(pm2_dvf):,.0f}** et **{mean(pm2_dvf) + standard_deviation(pm2_dvf):,.0f} €/m²** (règle des 68%).
        """)

    # DVF par type
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Prix/m² par type de bien")
        for t in dvf["type_local"].unique():
            grp = dvf[dvf["type_local"] == t]["prix_au_m2"]
            st.metric(
                f"{'🏢' if t == 'Appartement' else '🏡'} {t}",
                f"{grp.mean():,.0f} €/m²",
                delta=f"méd. {grp.median():,.0f} | n={len(grp):,}",
                help=f"Prix moyen au m² pour les {t.lower()}s. La valeur delta montre la médiane et le nombre de transactions."
            )

    with col_b:
        st.subheader("Distribution prix/m² DVF")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(dvf["prix_au_m2"], bins=50, color=BLUE, alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.axvline(dvf["prix_au_m2"].mean(), color="black", linestyle="--", linewidth=1.5,
                   label=f"Moyenne : {dvf['prix_au_m2'].mean():,.0f} €/m²")
        ax.axvline(dvf["prix_au_m2"].median(), color=AMBER, linestyle="-.", linewidth=1.5,
                   label=f"Médiane : {dvf['prix_au_m2'].median():,.0f} €/m²")
        ax.xaxis.set_major_formatter(fmt_m2)
        ax.set_xlabel("Prix/m²")
        ax.set_ylabel("Nb transactions")
        ax.legend(fontsize=8)
        ax.set_facecolor("#F8FAFC")
        fig.patch.set_facecolor("#F8FAFC")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # KPIs SeLoger
    st.divider()
    st.subheader("🔴 SeLoger , Annonces en cours")
    c1, c2, c3, c4 = st.columns(4)
    pm2_ann = ann["prix_m2"].tolist()
    c1.metric("Annonces", f"{len(ann):,}",
              help="Nombre d'annonces scrapées depuis SeLoger. Ces biens ne sont pas encore vendus.")
    c2.metric("Prix/m² moyen", f"{mean(pm2_ann):,.0f} €/m²",
              help="Prix moyen demandé par les vendeurs. Généralement supérieur au prix DVF (marge de négociation).")
    c3.metric("Prix/m² médian", f"{ann['prix_m2'].median():,.0f} €/m²")
    c4.metric("Nb quartiers", f"{ann['quartier'].nunique()}",
              help="Nombre de quartiers toulonnais représentés dans les annonces SeLoger.")

    ecart = mean(pm2_ann) - mean(pm2_dvf)
    st.info(
        f"**Écart moyen SeLoger vs DVF : {ecart:+,.0f} €/m²** "
        f"({'Les vendeurs demandent en moyenne plus que le marché réel' if ecart > 0 else 'Les prix demandés sont alignés avec le marché'}). "
        f"Cet écart représente la marge de négociation typique."
    )

    # Distribution DVF mensuelle
    st.divider()
    st.subheader("📅 Volume mensuel de transactions DVF")
    with st.expander("ℹ️ Comment lire ce graphique ?"):
        st.markdown("""
        Chaque barre représente le **nombre de transactions** enregistrées ce mois-là.
        Un volume élevé indique un marché actif. La ligne rouge montre l'évolution du **prix médian mensuel**.
        """)

    monthly = (
        dvf.set_index("date_mutation")
        .resample("ME")["prix_au_m2"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    monthly.columns = ["mois", "moyenne", "mediane", "nb"]

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax2 = ax1.twinx()
    ax1.bar(monthly["mois"], monthly["nb"], color=BLUE, alpha=0.5, width=20, label="Nb transactions")
    ax2.plot(monthly["mois"], monthly["mediane"], color=RED, linewidth=2.5, marker="o", markersize=6, label="Prix médian")
    ax1.set_ylabel("Nb transactions", color=BLUE)
    ax2.set_ylabel("Prix médian/m²", color=RED)
    ax2.yaxis.set_major_formatter(fmt_m2)
    ax1.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#F8FAFC")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 , Carte & Quartiers
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Géographie du marché")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("🗺️ Carte GPS , Transactions DVF")
        with st.expander("ℹ️ Comment lire cette carte ?"):
            st.markdown("""
            Chaque point est une **transaction DVF réelle** en 2024.
            La couleur indique le prix/m² : **vert = bon marché**, **rouge = cher**.
            La concentration de points révèle les **zones les plus actives** du marché.
            """)
        dvf_geo = dvf[dvf["longitude"].notna() & dvf["latitude"].notna()].copy()
        fig, ax = plt.subplots(figsize=(7, 7))
        sc = ax.scatter(
            dvf_geo["longitude"], dvf_geo["latitude"],
            c=dvf_geo["prix_au_m2"],
            cmap="RdYlGn_r", alpha=0.45, s=6, edgecolors="none",
            vmin=dvf_geo["prix_au_m2"].quantile(0.05),
            vmax=dvf_geo["prix_au_m2"].quantile(0.95),
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Prix/m² (€/m²)", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
        ax.set_facecolor("#0F172A")
        fig.patch.set_facecolor("#0F172A")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.set_title(f"Transactions DVF Toulon 2024 (n={len(dvf_geo):,})", color="white", fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_right:
        st.subheader("🏘️ Heatmap prix/m² par quartier , SeLoger")
        with st.expander("ℹ️ Comment lire cette heatmap ?"):
            st.markdown("""
            Chaque ligne est un **quartier toulonnais** avec le prix moyen/m² observé dans les annonces SeLoger.
            La **couleur** va du bleu (moins cher) au rouge (plus cher).
            Le **n=** indique le nombre d'annonces , les quartiers avec peu d'annonces sont moins fiables.
            """)

        min_annonces = st.slider("Nb min. d'annonces par quartier", 2, 20, 5,
                                 help="Masquer les quartiers avec trop peu d'annonces pour éviter les statistiques peu représentatives.")
        by_q = (
            ann.groupby("quartier")["prix_m2"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "moy", "count": "n"})
        )
        by_q = by_q[by_q["n"] >= min_annonces].sort_values("moy", ascending=False)

        fig, ax = plt.subplots(figsize=(6, max(5, len(by_q) * 0.38)))
        im = ax.imshow(
            by_q["moy"].values.reshape(-1, 1),
            cmap="RdYlGn_r", aspect="auto",
            vmin=by_q["moy"].min(), vmax=by_q["moy"].max()
        )
        plt.colorbar(im, ax=ax, label="Prix/m² moyen (€/m²)")
        ax.set_yticks(range(len(by_q)))
        ax.set_yticklabels([f"{q} (n={int(n)})" for q, n in zip(by_q.index, by_q["n"])], fontsize=8)
        ax.set_xticks([])
        for i, val in enumerate(by_q["moy"].values):
            ax.text(0, i, f"{val:,.0f} €/m²", ha="center", va="center", fontsize=8,
                    color="white" if (val > by_q["moy"].quantile(0.6) or val < by_q["moy"].quantile(0.3)) else "black",
                    fontweight="bold")
        ax.set_title(f"Prix/m² moyen par quartier ({len(by_q)} quartiers)", fontsize=10, fontweight="bold")
        ax.set_facecolor("#F8FAFC")
        fig.patch.set_facecolor("#F8FAFC")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Barplot quartiers
    st.divider()
    st.subheader("Prix/m² par quartier , classement complet")
    by_q_full = (
        ann.groupby("quartier")["prix_m2"]
        .agg(["mean", "median", "count"])
        .rename(columns={"mean": "Moyenne", "median": "Médiane", "count": "Nb annonces"})
        .sort_values("Moyenne", ascending=False)
        .round(0)
    )
    by_q_full = by_q_full[by_q_full["Nb annonces"] >= min_annonces]

    fig, ax = plt.subplots(figsize=(12, max(6, len(by_q_full) * 0.35)))
    median_global = ann["prix_m2"].median()
    colors = [RED if v > median_global else GREEN for v in by_q_full["Moyenne"]]
    bars = ax.barh(by_q_full.index, by_q_full["Moyenne"], color=colors, alpha=0.8)
    ax.axvline(median_global, color=AMBER, linestyle="--", linewidth=1.5,
               label=f"Médiane globale : {median_global:,.0f} €/m²")
    for bar, val in zip(bars, by_q_full["Moyenne"]):
        ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f} €", va="center", fontsize=8)
    ax.xaxis.set_major_formatter(fmt_m2)
    ax.set_xlabel("Prix moyen au m²")
    ax.legend(fontsize=9)
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#F8FAFC")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.dataframe(
        by_q_full.style.background_gradient(subset=["Moyenne"], cmap="RdYlGn_r"),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 , Régression & Corrélations
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Régression linéaire & Corrélations")

    with st.expander("📖 Comprendre la régression linéaire", expanded=True):
        st.markdown(r"""
        **La régression linéaire** cherche la droite $\hat{y} = \alpha + \beta \cdot x$ qui **minimise les erreurs** de prédiction.

        | Terme | Signification | Exemple |
        |-------|--------------|---------|
        | $\alpha$ (intercept) | Prix de base quand la surface → 0 | Frais fixes, emplacement |
        | $\beta$ (pente) | Chaque m² supplémentaire coûte... | Ex : 2 500 €/m² |
        | $R^2$ (R carré) | Part de la variation du prix **expliquée** par la surface | 0.7 = 70% expliqué |
        | $r$ (corrélation) | Force et direction du lien surface↔prix | $r = \sqrt{R^2}$ |

        **Interprétation du R²** :
        - **R² = 0** → La surface n'explique **rien** du prix
        - **R² = 0.5** → La surface explique **50%** de la variation de prix (le reste = localisation, DPE, état...)
        - **R² = 1** → La surface explique **100%** du prix (parfait, irréaliste sur un vrai marché)

        > Sur un marché immobilier réel, un R² entre **0.4 et 0.7** est typique , d'autres facteurs comptent aussi (quartier, DPE, étage...).
        """)

    # Régression interactive
    st.subheader("Régression interactive , choisissez vos variables")
    col_x_sel, col_ds_sel = st.columns(2)
    with col_x_sel:
        source_reg = st.selectbox("Source de données", ["Annonces SeLoger", "DVF Transactions"],
                                  help="SeLoger = prix demandés | DVF = prix réels payés")
    with col_ds_sel:
        x_var = st.selectbox("Variable explicative (X)", ["Surface", "Nb pièces"],
                             help="La variable que l'on utilise pour prédire le prix.")

    if source_reg == "Annonces SeLoger":
        data_reg = ann.copy()
        xcol = "surface" if x_var == "Surface" else "nb_pieces"
        ycol = "prix"
        xlabel = "Surface (m²)" if x_var == "Surface" else "Nombre de pièces"
    else:
        data_reg = dvf.copy()
        xcol = "surface_reelle_bati" if x_var == "Surface" else "nombre_pieces_principales"
        ycol = "valeur_fonciere"
        xlabel = "Surface bâtie (m²)" if x_var == "Surface" else "Nombre de pièces"

    data_reg = data_reg[[xcol, ycol]].dropna()
    data_reg = data_reg[(data_reg[xcol] > 0) & (data_reg[ycol] > 0)]
    if xcol in ["surface", "surface_reelle_bati"]:
        data_reg = data_reg[data_reg[xcol] <= 300]

    x_list = data_reg[xcol].tolist()
    y_list = data_reg[ycol].tolist()

    alpha, beta = least_squares_fit(x_list, y_list)
    r2 = r_squared(alpha, beta, x_list, y_list)
    r_corr = correlation(x_list, y_list)

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    col_r1.metric("α (intercept)", f"{alpha:,.0f} €",
                  help="Valeur de départ théorique , prix quand la variable X vaut 0. Souvent négatif (extrapolation hors données).")
    col_r2.metric("β (pente)", f"{beta:,.0f} €/{xlabel.split('(')[1].rstrip(')') if '(' in xlabel else 'unité'}",
                  help=f"Chaque unité de '{xlabel}' supplémentaire ajoute {beta:,.0f} € au prix prédit.")
    col_r3.metric("R²", f"{r2:.4f}",
                  help=f"Le modèle explique {r2*100:.1f}% de la variation des prix par la {xlabel.lower()}. Le reste vient d'autres facteurs.")
    col_r4.metric("Corrélation r", f"{r_corr:.3f}",
                  help="Coefficient de Pearson : +1 = parfaitement corrélé, 0 = aucun lien, -1 = inversement corrélé.")

    quality = "Fort" if abs(r_corr) > 0.7 else "Modéré" if abs(r_corr) > 0.4 else "Faible"
    direction = "positif" if r_corr > 0 else "négatif"
    st.info(
        f"**Interprétation** : Le lien entre {xlabel.lower()} et le prix est **{quality.lower()} et {direction}** (r={r_corr:.3f}). "
        f"La surface explique **{r2*100:.1f}%** de la variation du prix. "
        f"Chaque m² supplémentaire est associé à **+{beta:,.0f} €** en moyenne."
    )

    x_arr = np.array(x_list)
    y_arr = np.array(y_list)
    x_line = np.linspace(x_arr.min(), x_arr.max(), 200)
    y_line = [alpha + beta * xi for xi in x_line]
    y_pred = [alpha + beta * xi for xi in x_list]
    residuals = [y_list[i] - y_pred[i] for i in range(len(y_list))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter + droite
    ax = axes[0]
    ax.scatter(x_arr, y_arr, alpha=0.3, color=BLUE if source_reg == "DVF Transactions" else RED,
               s=15, edgecolors="none", label=f"Données (n={len(x_list):,})")
    ax.plot(x_line, y_line, color="black", linewidth=2.5,
            label=f"ŷ = {alpha:,.0f} + {beta:,.0f}·x")
    textbox = f"R² = {r2:.4f}\nr = {r_corr:.3f}\nβ = {beta:,.0f} €/unité"
    ax.text(0.97, 0.05, textbox, transform=ax.transAxes, fontsize=9,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#CBD5E1", alpha=0.9))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Prix (€)")
    ax.yaxis.set_major_formatter(fmt_eur)
    ax.legend(fontsize=9)
    ax.set_title(f"Régression : Prix ~ {xlabel}", fontweight="bold")
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#F8FAFC")

    # Distribution des résidus
    ax2 = axes[1]
    ax2.hist(residuals, bins=40, color=VIOLET, alpha=0.8, edgecolor="white", linewidth=0.3)
    ax2.axvline(0, color="black", linewidth=2, linestyle="--")
    ax2.set_xlabel("Résidu (Prix réel − Prix prédit)")
    ax2.set_ylabel("Fréquence")
    ax2.xaxis.set_major_formatter(fmt_eur)
    ax2.set_title("Distribution des résidus\n(centré sur 0 = bonne régression)", fontweight="bold")
    ax2.set_facecolor("#F8FAFC")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    with st.expander("📖 Que sont les résidus ?"):
        st.markdown("""
        Un **résidu** = Prix réel − Prix prédit par le modèle.

        - Si les résidus sont **centrés sur 0** et **en forme de cloche** → le modèle est bien calibré
        - Si les résidus sont **asymétriques** → le modèle sous-estime ou sur-estime systématiquement
        - Des résidus **en forme d'entonnoir** (hétéroscédasticité) → la variance augmente avec la surface
        """)

    # Heatmap corrélation
    st.divider()
    st.subheader("Matrices de corrélation")
    with st.expander("ℹ️ Comment lire une heatmap de corrélation ?"):
        st.markdown(r"""
        Chaque cellule montre le **coefficient de corrélation de Pearson** entre deux variables :

        | Valeur | Interprétation |
        |--------|---------------|
        | **+1.0** | Corrélation parfaite positive (si X augmente, Y augmente proportionnellement) |
        | **+0.7** | Forte corrélation positive |
        | **0** | Aucune corrélation linéaire |
        | **-0.7** | Forte corrélation négative |
        | **-1.0** | Corrélation parfaite négative |

        **Formule** : $r = \frac{\text{Cov}(X,Y)}{\sigma_X \cdot \sigma_Y}$ , implémentée from scratch dans `analysis/stats.py`
        """)

    col_hm1, col_hm2 = st.columns(2)

    with col_hm1:
        st.markdown("**🔵 DVF , Transactions gouvernementales**")
        cols_dvf = ["valeur_fonciere", "surface_reelle_bati", "nombre_pieces_principales", "prix_au_m2"]
        labels_dvf = ["Prix (€)", "Surface (m²)", "Nb pièces", "Prix/m²"]
        sub_dvf = dvf[cols_dvf].dropna()
        corr_dvf = sub_dvf.corr()

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr_dvf.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(labels_dvf)))
        ax.set_yticks(range(len(labels_dvf)))
        ax.set_xticklabels(labels_dvf, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(labels_dvf, fontsize=9)
        for i in range(len(labels_dvf)):
            for j in range(len(labels_dvf)):
                val = corr_dvf.values[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10, color=color, fontweight="bold")
        ax.set_title("DVF , Corrélations", fontweight="bold")
        ax.set_facecolor("#F8FAFC")
        fig.patch.set_facecolor("#F8FAFC")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_hm2:
        st.markdown("**🔴 SeLoger , Annonces**")
        cols_ann = ["prix", "surface", "prix_m2", "nb_pieces"]
        labels_ann = ["Prix (€)", "Surface (m²)", "Prix/m²", "Nb pièces"]
        sub_ann = ann[cols_ann].dropna()
        corr_ann = sub_ann.corr()

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr_ann.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(labels_ann)))
        ax.set_yticks(range(len(labels_ann)))
        ax.set_xticklabels(labels_ann, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(labels_ann, fontsize=9)
        for i in range(len(labels_ann)):
            for j in range(len(labels_ann)):
                val = corr_ann.values[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10, color=color, fontweight="bold")
        ax.set_title("SeLoger , Corrélations", fontweight="bold")
        ax.set_facecolor("#F8FAFC")
        fig.patch.set_facecolor("#F8FAFC")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # DPE
    st.divider()
    st.subheader("Impact du DPE sur le prix/m²")
    with st.expander("ℹ️ Qu'est-ce que le DPE ?"):
        st.markdown("""
        Le **Diagnostic de Performance Énergétique (DPE)** classe les biens de **A** (très économe) à **G** (très énergivore).

        - **Classe A/B** : Biens récents ou rénovés, charges réduites → généralement **mieux valorisés**
        - **Classe F/G** : Passoires thermiques, travaux à prévoir → **décote** sur le prix
        - Depuis 2025, les logements G **ne peuvent plus être loués**, ce qui impacte leur valeur marchande
        """)

    ordre_dpe = ["A", "B", "C", "D", "E", "F", "G"]
    colors_dpe = {"A": "#15803D", "B": "#65A30D", "C": "#A3E635",
                  "D": "#FBBF24", "E": "#F97316", "F": "#EF4444", "G": "#991B1B"}
    ann_dpe = ann[ann["energie"].isin(ordre_dpe)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    counts = ann_dpe["energie"].value_counts().reindex(ordre_dpe).dropna()
    axes[0].bar(counts.index, counts.values,
                color=[colors_dpe.get(d, "#94A3B8") for d in counts.index], alpha=0.85)
    axes[0].set_xlabel("Classe DPE")
    axes[0].set_ylabel("Nb annonces")
    axes[0].set_title("Répartition par classe DPE", fontweight="bold")
    for bar in axes[0].patches:
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(int(bar.get_height())), ha="center", fontsize=10)

    dpe_prix = ann_dpe.groupby("energie")["prix_m2"].median().reindex(ordre_dpe).dropna()
    axes[1].bar(dpe_prix.index, dpe_prix.values,
                color=[colors_dpe.get(d, "#94A3B8") for d in dpe_prix.index], alpha=0.85)
    axes[1].set_xlabel("Classe DPE")
    axes[1].set_ylabel("Prix médian/m²")
    axes[1].yaxis.set_major_formatter(fmt_m2)
    axes[1].set_title("Prix médian/m² par classe DPE", fontweight="bold")

    for ax in axes:
        ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#F8FAFC")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 , Opportunités
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Scoring des opportunités")
    with st.expander("📖 Comment fonctionne le scoring ?", expanded=True):
        st.markdown(r"""
        Le score d'opportunité est basé sur le **z-score** du prix/m² de chaque quartier :

        $$z = \frac{x - \mu}{\sigma}$$

        Où :
        - $x$ = prix/m² moyen du quartier
        - $\mu$ = prix/m² moyen **global** du marché
        - $\sigma$ = écart-type global

        | Signal | Condition | Signification |
        |--------|-----------|--------------|
        | 🟢 Sous-coté | z < −0.5 | Le quartier est **en dessous** du marché → potentiel de plus-value |
        | 🟡 Dans le marché | −0.5 ≤ z ≤ 0.5 | Prix **aligné** avec la moyenne |
        | 🔴 Sur-coté | z > 0.5 | Le quartier est **au-dessus** du marché → peu de marge |

        > **Attention** : Ce scoring est purement quantitatif. Des facteurs qualitatifs (sécurité, commerces, transports) peuvent expliquer des prix légitimement élevés ou bas.
        """)

    # Filtres opportunités
    budget_max = st.slider("Budget maximum (€)", 50_000, 800_000, 300_000, 10_000,
                           help="Filtrer les annonces au-dessus de ce budget.")
    surface_min_opp = st.slider("Surface minimum (m²)", 10, 150, 30,
                                help="Surface minimale souhaitée.")

    ann_opp = ann[(ann["prix"] <= budget_max) & (ann["surface"] >= surface_min_opp)].copy()

    global_mean_pm2 = mean(ann["prix_m2"].tolist())
    global_std_pm2 = standard_deviation(ann["prix_m2"].tolist())

    by_q_opp = ann_opp.groupby("quartier")["prix_m2"].agg(["mean", "median", "count"]).reset_index()
    by_q_opp.columns = ["Quartier", "Prix_m2_moyen", "Prix_m2_median", "Nb_annonces"]
    by_q_opp = by_q_opp[by_q_opp["Nb_annonces"] >= 2]
    by_q_opp["Z_score"] = (by_q_opp["Prix_m2_moyen"] - global_mean_pm2) / global_std_pm2
    by_q_opp["Signal"] = by_q_opp["Z_score"].apply(
        lambda z: "🟢 Sous-coté" if z < -0.5 else ("🔴 Sur-coté" if z > 0.5 else "🟡 Marché")
    )
    by_q_opp = by_q_opp.sort_values("Z_score")
    by_q_opp["Prix_m2_moyen"] = by_q_opp["Prix_m2_moyen"].round(0)
    by_q_opp["Z_score"] = by_q_opp["Z_score"].round(2)

    c_sous, c_marche, c_sur = st.columns(3)
    sous = by_q_opp[by_q_opp["Signal"] == "🟢 Sous-coté"]
    marche = by_q_opp[by_q_opp["Signal"] == "🟡 Marché"]
    sur = by_q_opp[by_q_opp["Signal"] == "🔴 Sur-coté"]
    c_sous.metric("🟢 Quartiers sous-cotés", len(sous))
    c_marche.metric("🟡 Dans le marché", len(marche))
    c_sur.metric("🔴 Quartiers sur-cotés", len(sur))
    st.caption(f"Référence : moyenne globale = {global_mean_pm2:,.0f} €/m² | écart-type = {global_std_pm2:,.0f} €/m²")

    st.dataframe(
        by_q_opp[["Quartier", "Prix_m2_moyen", "Prix_m2_median", "Nb_annonces", "Z_score", "Signal"]],
        use_container_width=True, hide_index=True
    )

    # Annonces filtrées
    st.divider()
    st.subheader(f"Annonces correspondantes ({len(ann_opp):,})")
    ann_opp_display = ann_opp.merge(
        by_q_opp[["Quartier", "Signal", "Z_score"]],
        left_on="quartier", right_on="Quartier", how="left"
    )
    st.dataframe(
        ann_opp_display[["Signal", "quartier", "type", "prix", "surface", "prix_m2", "nb_pieces", "energie"]]
        .sort_values(["Signal", "prix_m2"]),
        use_container_width=True, hide_index=True
    )


# ══════════════════════════════════════════════════════════════════════════
# TAB 5 , Méthodologie
# ══════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("📚 Méthodologie & Documentation")

    st.markdown("""
    > Cette section explique comment les données ont été collectées, nettoyées, combinées
    > et analysées. Elle correspond au fichier `METHODOLOGIE.md` du projet.
    """)

    with st.expander("1️⃣ Les deux sources de données", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🔵 DVF , Demandes de Valeurs Foncières
            **Source** : [data.gouv.fr](https://files.data.gouv.fr/geo-dvf/latest/csv/83/)
            **Fichier brut** : `donnees/raw/datagouv_83137_20242025.csv`
            **Fichier nettoyé** : `donnees/processed/dvf_clean.csv`

            **Ce que contient le DVF :**
            - Toutes les **transactions immobilières réelles** déclarées aux impôts
            - Période : Fév 2024 → Déc 2024 (Toulon, INSEE 83137)
            - Variables : prix de vente, surface bâtie, type de bien, coordonnées GPS

            **Ce que le DVF ne contient PAS :**
            - Pas de quartier nommé (seulement coordonnées GPS)
            - Pas de DPE (classe énergétique)
            - Pas de description textuelle

            **Nettoyage effectué** (`analysis/cleaning_dvf.py`) :
            - Suppression des lignes sans prix ou sans surface
            - Calcul du prix/m² = valeur_foncière / surface_réelle
            - Conservation de 13 colonnes utiles sur 40 initiales
            """)
        with col2:
            st.markdown("""
            ### 🔴 Annonces SeLoger , Scraping
            **Source** : SeLoger.com via scraping Python
            **Fichier brut** : `donnees/raw/seloger_document_base.csv`
            **Fichier nettoyé** : `donnees/processed/annonces_propres.csv`

            **Ce que contient SeLoger :**
            - Annonces **en cours** (biens à vendre, pas encore vendus)
            - 707 annonces, 42 quartiers toulonnais
            - Variables : prix demandé, surface, quartier, DPE, nb pièces

            **Ce que SeLoger ne contient PAS :**
            - Pas de prix **réellement payé** (c'est le prix demandé)
            - Pas de coordonnées GPS précises
            - Pas de date de transaction

            **Nettoyage effectué** (`analysis/nettoyage.py`) :
            - Conversion des formats texte → numérique
            - Suppression des doublons et valeurs aberrantes
            - Standardisation des noms de quartiers
            """)

    with st.expander("2️⃣ Comment les datasets sont combinés"):
        st.markdown("""
        ### Pas de jointure directe , deux vues complémentaires

        **DVF et SeLoger ne peuvent pas être fusionnés en une seule table** car :
        - DVF n'a pas de quartier nommé (seulement GPS)
        - SeLoger n'a pas de coordonnées GPS précises
        - Les deux ne couvrent pas la même période (DVF = vendus, SeLoger = en vente)

        **Stratégie adoptée : analyse parallèle**
        ```
        DVF (transactions réelles)    SeLoger (annonces)
              ↓                              ↓
        Prix du marché effectif        Prix demandés
        Évolution temporelle           Répartition quartiers
        Volume de transactions         DPE et caractéristiques
              ↓                              ↓
        Comparaison : écart prix demandé vs prix payé = tension du marché
        ```

        **Point de comparaison** : Le type de bien (Appartement/Maison) est commun aux deux.
        On peut comparer les distributions de prix/m² entre les deux sources pour mesurer l'écart entre offre et réalité.
        """)

    with st.expander("3️⃣ Statistiques descriptives , formules from scratch"):
        st.markdown(r"""
        Toutes les statistiques sont calculées **sans numpy/pandas/statistics**,
        en Python pur (`analysis/stats.py`) , conformément à Joel Grus ch.5.

        | Fonction | Formule | Rôle |
        |----------|---------|------|
        | `mean(xs)` | $\bar{x} = \frac{1}{n}\sum x_i$ | Valeur centrale |
        | `variance(xs)` | $\sigma^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$ | Dispersion quadratique |
        | `standard_deviation(xs)` | $\sigma = \sqrt{\sigma^2}$ | Dispersion dans l'unité d'origine |
        | `covariance(xs, ys)` | $\text{Cov} = \frac{1}{n}\sum(x_i-\bar{x})(y_i-\bar{y})$ | Co-variation |
        | `correlation(xs, ys)` | $r = \frac{\text{Cov}(X,Y)}{\sigma_X \cdot \sigma_Y}$ | Lien linéaire normalisé |

        **Note sur la variance** : On utilise $n$ (biaisée) et non $n-1$ (non-biaisée).
        Le choix de $n$ donne `variance([2,4,4,4,5,5,7,9]) = 4.0` (valeur attendue par le CI).
        """)

    with st.expander("4️⃣ Régression linéaire , formules from scratch"):
        st.markdown(r"""
        Implémentée sans sklearn dans `analysis/regression.py` , Joel Grus ch.14.

        **Modèle** : $\hat{y} = \alpha + \beta \cdot x$

        **Formules des coefficients (moindres carrés)** :

        $$\beta = \frac{\text{Cov}(x, y)}{\text{Var}(x)}$$
        $$\alpha = \bar{y} - \beta \cdot \bar{x}$$

        **Coefficient de détermination R²** :

        $$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(\hat{y}_i - y_i)^2}{\sum(y_i - \bar{y})^2}$$

        Où :
        - $SS_{res}$ = Somme des carrés des résidus (erreurs du modèle)
        - $SS_{tot}$ = Somme totale des carrés (variance totale)

        **Lien r et R²** : Pour une régression simple, $R^2 = r^2$ où $r$ est le coefficient de corrélation de Pearson.

        **Sur ce marché** : R² surface→prix ≈ 0.4-0.6, ce qui est réaliste.
        La localisation (quartier), le DPE, l'état du bien expliquent le reste.
        """)
