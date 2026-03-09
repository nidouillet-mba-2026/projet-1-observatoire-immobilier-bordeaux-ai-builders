"""
Observatoire Immobilier Toulonnais — Dashboard principal.
NidDouillet — Marche immobilier Toulon (INSEE 83137)
"""

import os
import sys

import pandas as pd
import streamlit as st

# Permet d'importer analysis depuis n'importe quel CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.regression import least_squares_fit, r_squared
from analysis.stats import correlation, mean, standard_deviation

st.set_page_config(
    page_title="Observatoire Immobilier Toulonnais",
    page_icon="🏠",
    layout="wide",
)

st.title("Observatoire Immobilier Toulonnais")
st.caption("NidDouillet — Donnees DVF + SeLoger | Toulon (INSEE 83137)")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@st.cache_data
def load_dvf() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "dvf_toulon.csv")
    df = pd.read_csv(path, sep=";", low_memory=False)
    df["valeur_fonciere"] = pd.to_numeric(df["valeur_fonciere"], errors="coerce")
    df["surface_reelle_bati"] = pd.to_numeric(df["surface_reelle_bati"], errors="coerce")
    df["nombre_pieces_principales"] = pd.to_numeric(df["nombre_pieces_principales"], errors="coerce")
    mask = (
        df["valeur_fonciere"].notna()
        & df["surface_reelle_bati"].notna()
        & (df["surface_reelle_bati"] > 0)
        & (df["valeur_fonciere"] > 0)
    )
    df = df[mask].copy()
    df["prix_m2"] = df["valeur_fonciere"] / df["surface_reelle_bati"]
    df = df[(df["prix_m2"] >= 500) & (df["prix_m2"] <= 20000)]
    return df


@st.cache_data
def load_annonces() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "annonces.csv")
    df = pd.read_csv(path)
    df = df[df["prix_m2"].notna() & (df["prix_m2"] > 0)]
    return df


tab1, tab2, tab3, tab4 = st.tabs([
    "Etat du marche",
    "Filtres annonces",
    "Tendances & regression",
    "Opportunites",
])

# ── Onglet 1 : Etat du marche ──────────────────────────────────────────────
with tab1:
    st.header("Etat du marche DVF — Toulon")
    try:
        dvf = load_dvf()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Transactions DVF", f"{len(dvf):,}")
        col2.metric("Prix/m² moyen", f"{mean(dvf['prix_m2'].tolist()):.0f} €")
        col3.metric("Prix/m² median", f"{dvf['prix_m2'].median():.0f} €")
        col4.metric("Ecart-type prix/m²", f"{standard_deviation(dvf['prix_m2'].tolist()):.0f} €")

        st.subheader("Prix/m² par type de bien")
        if "type_local" in dvf.columns:
            by_type = (
                dvf.groupby("type_local")["prix_m2"]
                .agg(["mean", "median", "count"])
                .rename(columns={"mean": "Moyenne", "median": "Mediane", "count": "Nb transactions"})
                .round(0)
            )
            st.dataframe(by_type)

        st.subheader("Dernieres transactions")
        cols_show = [c for c in ["date_mutation", "type_local", "valeur_fonciere", "surface_reelle_bati", "prix_m2", "adresse_nom_voie"] if c in dvf.columns]
        st.dataframe(dvf[cols_show].head(50), use_container_width=True)
    except Exception as e:
        st.error(f"Donnees DVF non chargees : {e}")

# ── Onglet 2 : Filtres annonces ────────────────────────────────────────────
with tab2:
    st.header("Annonces SeLoger — Toulon")
    try:
        ann = load_annonces()
        col1, col2 = st.columns(2)
        with col1:
            budget = st.slider("Budget max (€)", 50_000, 800_000, 350_000, step=10_000)
            types = st.multiselect("Type de bien", ann["type"].unique().tolist(), default=ann["type"].unique().tolist())
        with col2:
            quartiers = st.multiselect("Quartier", sorted(ann["quartier"].unique().tolist()), default=[])
            surface_min = st.slider("Surface min (m²)", 10, 200, 20)

        filtered = ann[ann["prix"] <= budget]
        if types:
            filtered = filtered[filtered["type"].isin(types)]
        if quartiers:
            filtered = filtered[filtered["quartier"].isin(quartiers)]
        filtered = filtered[filtered["surface"] >= surface_min]

        st.metric("Annonces correspondantes", len(filtered))
        st.dataframe(filtered.sort_values("prix_m2"), use_container_width=True)
    except Exception as e:
        st.error(f"Donnees annonces non chargees : {e}")

# ── Onglet 3 : Tendances & regression ─────────────────────────────────────
with tab3:
    st.header("Tendances prix/m² — Regression lineaire from scratch")
    try:
        ann = load_annonces()
        st.subheader("Prix/m² moyen par quartier")
        by_q = ann.groupby("quartier")["prix_m2"].mean().sort_values(ascending=False).round(0)
        st.bar_chart(by_q)

        st.subheader("Regression prix ~ surface (annonces SeLoger)")
        data = ann[ann["prix_m2"].notna() & ann["surface"].notna()].copy()
        x = data["surface"].tolist()
        y = data["prix"].tolist()
        alpha, beta = least_squares_fit(x, y)
        r2 = r_squared(alpha, beta, x, y)
        corr = correlation(x, y)

        col1, col2, col3 = st.columns(3)
        col1.metric("Alpha (intercept)", f"{alpha:,.0f} €")
        col2.metric("Beta (pente)", f"{beta:,.1f} €/m²")
        col3.metric("R²", f"{r2:.3f}")
        st.caption(f"Correlation surface/prix : {corr:.3f}")
        st.info("Regression implementee from scratch (Joel Grus ch.14) — sans sklearn.")
    except Exception as e:
        st.error(f"Erreur regression : {e}")

# ── Onglet 4 : Opportunites ────────────────────────────────────────────────
with tab4:
    st.header("Opportunites — Scoring par quartier")
    try:
        ann = load_annonces()
        global_mean = mean(ann["prix_m2"].tolist())
        global_std = standard_deviation(ann["prix_m2"].tolist())

        by_q = ann.groupby("quartier")["prix_m2"].mean().reset_index()
        by_q.columns = ["Quartier", "Prix_m2_moyen"]
        by_q["Z-score"] = (by_q["Prix_m2_moyen"] - global_mean) / global_std
        by_q["Signal"] = by_q["Z-score"].apply(
            lambda z: "🟢 Sous-cote" if z < -0.5 else ("🔴 Sur-cote" if z > 0.5 else "🟡 Marche")
        )
        by_q = by_q.sort_values("Prix_m2_moyen")
        by_q["Prix_m2_moyen"] = by_q["Prix_m2_moyen"].round(0)
        by_q["Z-score"] = by_q["Z-score"].round(2)

        st.dataframe(by_q, use_container_width=True)
        st.caption(f"Moyenne globale : {global_mean:.0f} €/m² | Ecart-type : {global_std:.0f} €/m²")
        st.info("Scoring base sur le z-score par rapport a la moyenne du marche. Seuil : ±0.5 sigma.")
    except Exception as e:
        st.error(f"Erreur scoring : {e}")
