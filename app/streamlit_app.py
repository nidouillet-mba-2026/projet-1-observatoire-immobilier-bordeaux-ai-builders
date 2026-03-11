"""
NidDouillet — Observatoire Immobilier Toulonnais
Vue consommateur : aide à la décision d'achat
"""
 
import math
import os
import sys
 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
 
matplotlib.use("Agg")
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.regression import least_squares_fit, r_squared
from analysis.stats import correlation, mean, standard_deviation, variance
 
# ── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NidDouillet — Trouver mon bien à Toulon",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DVF_PATH = os.path.join(ROOT, "data", "dvf_toulon.csv")
ANN_PATH = os.path.join(ROOT, "data", "annonces.csv")
 
BLUE   = "#2563EB"
RED    = "#DC2626"
GREEN  = "#059669"
AMBER  = "#D97706"
VIOLET = "#7C3AED"
 
fmt_eur = mticker.FuncFormatter(lambda x, _: f"{x:,.0f} €")
fmt_m2  = mticker.FuncFormatter(lambda x, _: f"{x:,.0f} €/m²")
 
 
# ── Chargement ───────────────────────────────────────────────────────────────
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
 
 
dvf_raw = load_dvf()
ann_raw = load_annonces()
 
# ── Helpers ──────────────────────────────────────────────────────────────────
def _pm2_to_rgb(val: float, vmin: float, vmax: float) -> list[int]:
    """Mappe un prix/m² sur un dégradé vert→jaune→rouge [R, G, B]."""
    t = max(0.0, min(1.0, (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5))
    if t < 0.5:
        r = int(255 * t * 2)
        g = 200
        b = 80
    else:
        r = 220
        g = int(200 * (1 - (t - 0.5) * 2))
        b = 40
    return [r, g, b, 180]
 
 
def make_dvf_map(df_geo: pd.DataFrame, zoom: float = 13, pitch: float = 0) -> pdk.Deck:
    """Crée un pydeck ScatterplotLayer sur les transactions DVF géolocalisées."""
    vmin = df_geo["prix_au_m2"].quantile(0.05)
    vmax = df_geo["prix_au_m2"].quantile(0.95)
    df_map = df_geo[["latitude", "longitude", "prix_au_m2", "type_local",
                      "surface_reelle_bati", "valeur_fonciere"]].dropna().copy()
    df_map["color"] = df_map["prix_au_m2"].apply(lambda v: _pm2_to_rgb(v, vmin, vmax))
    df_map["tooltip_text"] = df_map.apply(
        lambda r: f"{r['type_local']} · {r['surface_reelle_bati']:.0f} m² · "
                  f"{r['valeur_fonciere']:,.0f} € · {r['prix_au_m2']:,.0f} €/m²",
        axis=1,
    )
    center_lat = df_map["latitude"].mean()
    center_lon = df_map["longitude"].mean()
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position=["longitude", "latitude"],
        get_fill_color="color",
        get_radius=40,
        radius_min_pixels=3,
        radius_max_pixels=12,
        pickable=True,
        auto_highlight=True,
    )
    view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom, pitch=pitch)
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        tooltip={"text": "{tooltip_text}"},
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    )
 
 
def verdict_ecart(ecart_pct: float) -> tuple[str, str]:
    """Retourne (emoji+label, couleur CSS) selon l'écart au prix du marché."""
    if ecart_pct < -10:
        return "🟢 Très bonne affaire", GREEN
    elif ecart_pct < -3:
        return "🟩 En dessous du marché", GREEN
    elif ecart_pct <= 3:
        return "🟡 Prix dans la moyenne", AMBER
    elif ecart_pct <= 12:
        return "🟠 Légèrement au-dessus", AMBER
    else:
        return "🔴 Prix élevé", RED
 
 
def phrase_verdict(bien: pd.Series, prix_predit: float, pm2_quartier: float | None) -> str:
    """Génère une phrase de conclusion en langage naturel pour un bien."""
    ecart = bien["prix"] - prix_predit
    ecart_pct = ecart / prix_predit * 100
 
    surface = bien.get("surface", None)
    quartier = bien.get("quartier", "ce quartier")
    dpe = bien.get("energie", None)
    nb_pieces = bien.get("nb_pieces", None)
 
    lignes = []
 
    # Verdict prix
    if ecart_pct < -10:
        lignes.append(
            f"**Ce bien est affiché {abs(ecart_pct):.0f}% sous le prix attendu** pour sa surface "
            f"({bien['prix']:,.0f} € vs {prix_predit:,.0f} € prédit) — c'est une opportunité rare à Toulon."
        )
    elif ecart_pct < -3:
        lignes.append(
            f"**Ce bien est {abs(ecart_pct):.0f}% moins cher que la moyenne** pour sa surface. "
            f"Il y a de la marge de négociation."
        )
    elif ecart_pct <= 3:
        lignes.append(
            f"**Ce bien est au juste prix du marché** (écart de {ecart_pct:+.0f}%). "
            f"Le prix demandé ({bien['prix']:,.0f} €) correspond à ce qu'on attend pour {surface:.0f} m²."
            if surface else
            f"**Ce bien est au juste prix du marché** (écart de {ecart_pct:+.0f}%). "
            f"Le prix demandé ({bien['prix']:,.0f} €) correspond au modèle."
        )
    elif ecart_pct <= 12:
        lignes.append(
            f"**Ce bien est {ecart_pct:.0f}% au-dessus du prix attendu.** "
            f"Il faudra négocier ou justifier ce surcoût (vue, rénovation, emplacement précis)."
        )
    else:
        lignes.append(
            f"**Attention : ce bien est affiché {ecart_pct:.0f}% au-dessus du marché.** "
            f"Le prix demandé ({bien['prix']:,.0f} €) dépasse nettement ce que les données justifient ({prix_predit:,.0f} €)."
        )
 
    # Contexte quartier
    if pm2_quartier is not None:
        lignes.append(
            f"Dans le quartier **{quartier}**, le prix médian est de **{pm2_quartier:,.0f} €/m²**."
        )
 
    # Contexte DPE
    if dpe in ["F", "G"]:
        lignes.append(
            f"⚠️ DPE **{dpe}** : prévoir des travaux de rénovation énergétique (estimer ~10 000–30 000 €). "
            f"Les passoires thermiques ne pourront plus être louées à partir de 2028."
        )
    elif dpe in ["A", "B"]:
        lignes.append(f"✅ DPE **{dpe}** : excellent bilan énergétique, faibles charges.")
 
    return "  \n".join(lignes)
 
 
# ── Sidebar — Filtres acheteur ────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/fr/d/d5/Logo_Ville_de_Toulon.svg", width=70)
    st.title("Mon projet immobilier")
    st.caption("Toulon · 2024")
    st.divider()
 
    budget_max = st.number_input(
        "💰 Budget maximum (€)",
        min_value=50_000, max_value=1_500_000,
        value=300_000, step=10_000,
        help="Le prix total affiché dans l'annonce."
    )
 
    type_bien = st.multiselect(
        "🏠 Type de bien",
        ["Appartement", "Maison"],
        default=["Appartement"],
    )
 
    nb_pieces_min = st.selectbox(
        "🛏️ Nombre de pièces minimum",
        [1, 2, 3, 4, 5],
        index=2,
        format_func=lambda x: f"{x} pièces (T{x})",
    )
 
    surface_min = st.slider("📐 Surface minimale (m²)", 10, 150, 40)
 
    quartiers_dispo = sorted(ann_raw["quartier"].dropna().unique().tolist())
    quartier_sel = st.multiselect(
        "📍 Quartier(s) souhaité(s)",
        quartiers_dispo,
        default=[],
        placeholder="Tous les quartiers",
    )
 
    st.divider()
    st.caption(f"📊 {len(dvf_raw):,} ventes réelles · {len(ann_raw):,} annonces en cours")
    st.caption("NidDouillet · Epitech IA Spé 2024")
 
# ── Filtrage annonces ─────────────────────────────────────────────────────────
ann_f = ann_raw[
    ann_raw["prix"].le(budget_max)
    & ann_raw["type"].isin(type_bien)
    & ann_raw["surface"].ge(surface_min)
    & ann_raw["nb_pieces"].ge(nb_pieces_min)
].copy()
 
if quartier_sel:
    ann_f = ann_f[ann_f["quartier"].isin(quartier_sel)]
 
# ── Filtrage DVF pour les cartes (type + budget cohérent via prix/m²) ─────────
# On filtre par type et on déduit une fourchette de prix/m² depuis le budget :
# budget_max / surface_min donne le prix/m² max que l'acheteur peut se permettre
_pm2_max_acheteur = budget_max / max(surface_min, 20)
dvf_f = dvf_raw[
    dvf_raw["type_local"].isin(type_bien)
    & dvf_raw["prix_au_m2"].le(_pm2_max_acheteur)
    & dvf_raw["latitude"].notna()
    & dvf_raw["longitude"].notna()
].copy()
 
# Modèle de régression global sur les annonces
_ann_reg = ann_raw[["surface", "prix"]].dropna()
_ann_reg = _ann_reg[(_ann_reg["surface"] > 0) & (_ann_reg["prix"] > 0) & (_ann_reg["surface"] <= 300)]
_x_reg = _ann_reg["surface"].tolist()
_y_reg = _ann_reg["prix"].tolist()
alpha_global, beta_global = least_squares_fit(_x_reg, _y_reg)
r2_global = r_squared(alpha_global, beta_global, _x_reg, _y_reg)
 
# Prix/m² médian par quartier (annonces)
_pm2_par_quartier = ann_raw.groupby("quartier")["prix_m2"].median().to_dict()
 
# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏠 NidDouillet — Observatoire Immobilier Toulonnais")
st.caption("Données DVF (transactions réelles 2024) × Annonces SeLoger · Toulon")
 
if not ann_f.empty:
    n = len(ann_f)
    pm2_moy = ann_f["prix_m2"].mean()
    st.success(
        f"**{n} bien{'s' if n > 1 else ''} correspond{'ent' if n > 1 else ''} à votre recherche** "
        f"— prix/m² moyen dans cette sélection : **{pm2_moy:,.0f} €/m²**"
    )
else:
    st.warning("Aucun bien ne correspond à vos critères. Essayez d'élargir le budget ou les filtres.")
 
st.divider()
 
# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Biens disponibles",
    "📍 Mon quartier",
    "⚖️ Ce bien est-il une bonne affaire ?",
    "📊 Le marché en bref",
])
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Biens disponibles
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header(f"Biens disponibles · {len(ann_f):,} résultat{'s' if len(ann_f) != 1 else ''}")
 
    if ann_f.empty:
        st.info("Modifiez vos critères dans la barre latérale pour voir des biens.")
    else:
        # KPIs synthétiques
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Biens trouvés", f"{len(ann_f)}")
        col2.metric("Budget moyen", f"{ann_f['prix'].mean():,.0f} €")
        col3.metric("Surface moyenne", f"{ann_f['surface'].mean():.0f} m²")
        col4.metric("Prix/m² médian", f"{ann_f['prix_m2'].median():,.0f} €/m²")
 
        st.divider()
 
        # Calcul du verdict pour chaque bien
        ann_f = ann_f.copy()
        ann_f["prix_predit"] = ann_f["surface"].apply(
            lambda s: alpha_global + beta_global * s if pd.notna(s) else None
        )
        ann_f["ecart_pct"] = ((ann_f["prix"] - ann_f["prix_predit"]) / ann_f["prix_predit"] * 100).round(1)
        ann_f["verdict_label"] = ann_f["ecart_pct"].apply(
            lambda e: verdict_ecart(e)[0] if pd.notna(e) else "—"
        )
 
        # Tri par opportunité (plus sous-évalués en premier)
        ann_sorted = ann_f.sort_values("ecart_pct")
 
        # Affichage carte-style pour les N premiers
        st.subheader("Les meilleures opportunités en premier")
 
        for _, row in ann_sorted.head(10).iterrows():
            with st.container(border=True):
                c1, c2, c3 = st.columns([2, 2, 3])
                with c1:
                    st.markdown(f"**{row.get('type', '—')} · {row.get('quartier', '—')}**")
                    st.markdown(f"📐 {row.get('surface', '?'):.0f} m² · 🛏️ {int(row.get('nb_pieces', 0))} pièces")
                    dpe = row.get("energie", None)
                    if pd.notna(dpe) and dpe != "":
                        dpe_colors = {"A": "🟢", "B": "🟢", "C": "🟡", "D": "🟡", "E": "🟠", "F": "🔴", "G": "🔴"}
                        st.markdown(f"DPE : {dpe_colors.get(dpe, '⚪')} **{dpe}**")
                with c2:
                    st.metric("Prix affiché", f"{row['prix']:,.0f} €")
                    st.metric("Prix/m²", f"{row.get('prix_m2', 0):,.0f} €/m²")
                with c3:
                    verdict, color = verdict_ecart(row["ecart_pct"]) if pd.notna(row["ecart_pct"]) else ("—", AMBER)
                    st.markdown(f"### {verdict}")
                    pm2_q = _pm2_par_quartier.get(row.get("quartier"), None)
                    if pd.notna(row.get("prix_predit")):
                        phrase = phrase_verdict(row, row["prix_predit"], pm2_q)
                        st.markdown(phrase)
 
        st.divider()
        st.subheader("Tous les biens · tableau complet")
 
        display_cols = {
            "verdict_label": "Verdict",
            "quartier": "Quartier",
            "type": "Type",
            "prix": "Prix (€)",
            "surface": "Surface (m²)",
            "prix_m2": "€/m²",
            "nb_pieces": "Pièces",
            "energie": "DPE",
            "ecart_pct": "Écart au modèle (%)",
        }
        df_display = ann_sorted[[c for c in display_cols if c in ann_sorted.columns]].rename(columns=display_cols)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
 
        # Carte DVF — contexte de marché, séparée visuellement des annonces
        st.divider()
        st.subheader("🗺️ Contexte de marché · Ventes réelles DVF 2024")
        st.markdown(
            f"> ℹ️ **Ces {len(dvf_f):,} points ne sont pas vos {len(ann_f)} annonces** — "
            f"ce sont les transactions immobilières réellement enregistrées aux impôts en 2024 "
            f"pour des {' et '.join(type_bien).lower()}s avec un prix/m² cohérent avec votre budget. "
            f"Ils vous montrent **où les gens ont acheté** et à quel prix."
        )
        if not dvf_f.empty:
            st.pydeck_chart(make_dvf_map(dvf_f, zoom=13), use_container_width=True)
        else:
            st.info("Aucune transaction DVF ne correspond à ces critères.")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Mon quartier
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Comprendre le marché de mon quartier")
 
    quartiers_avec_data = sorted(ann_raw.groupby("quartier").filter(lambda x: len(x) >= 3)["quartier"].unique())
    quartier_analyse = st.selectbox(
        "Choisissez un quartier à analyser",
        quartiers_avec_data,
        index=quartiers_avec_data.index("Le Mourillon") if "Le Mourillon" in quartiers_avec_data else 0,
    )
 
    ann_q = ann_raw[ann_raw["quartier"] == quartier_analyse]
    pm2_global_med = ann_raw["prix_m2"].median()
    pm2_q_med = ann_q["prix_m2"].median()
    pm2_q_moy = ann_q["prix_m2"].mean()
    ecart_vs_global = (pm2_q_med - pm2_global_med) / pm2_global_med * 100
 
    # Verdict quartier
    if ecart_vs_global < -10:
        signal_q = "🟢 Quartier accessible"
        expl_q = f"Les prix dans ce quartier sont **{abs(ecart_vs_global):.0f}% inférieurs** à la médiane toulonnaise. Idéal pour maximiser la surface dans le budget."
    elif ecart_vs_global < -3:
        signal_q = "🟩 Légèrement en dessous du marché"
        expl_q = f"Ce quartier est **{abs(ecart_vs_global):.0f}% moins cher** que la moyenne toulonnaise."
    elif ecart_vs_global <= 5:
        signal_q = "🟡 Prix dans la moyenne toulonnaise"
        expl_q = "Ce quartier est représentatif du marché toulonnais — ni premium, ni décoté."
    elif ecart_vs_global <= 15:
        signal_q = "🟠 Quartier recherché"
        expl_q = f"Les prix ici sont **{ecart_vs_global:.0f}% au-dessus** de la moyenne — ce quartier est prisé."
    else:
        signal_q = "🔴 Quartier premium"
        expl_q = f"Les prix ici sont **{ecart_vs_global:.0f}% au-dessus** de la moyenne — budget à prévoir."
 
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Prix/m² médian", f"{pm2_q_med:,.0f} €/m²",
                 delta=f"{ecart_vs_global:+.0f}% vs moyenne Toulon")
    col_b.metric("Annonces disponibles", f"{len(ann_q)}")
    col_c.metric("Médiane Toulon", f"{pm2_global_med:,.0f} €/m²")
 
    st.info(f"**{signal_q}** — {expl_q}")
 
    st.divider()
 
    # Carte zoomée sur le quartier — transactions DVF réelles
    st.subheader(f"🗺️ Ventes réelles autour de {quartier_analyse}")
    st.caption("Chaque point = une transaction DVF 2024 enregistrée aux impôts. 🟢 abordable → 🔴 cher")
 
    # dvf_f est déjà filtré par type + budget — on l'utilise directement
    dvf_geo_q = dvf_f.copy()
 
    # Centroïdes approximatifs des quartiers connus de Toulon (lat, lon)
    QUARTIER_COORDS: dict[str, tuple[float, float]] = {
        "Le Mourillon":        (43.115,  5.945),
        "Centre-ville":        (43.124,  5.928),
        "Haute-Ville":         (43.127,  5.930),
        "Le Cap Brun":         (43.108,  5.963),
        "Saint-Jean-du-Var":   (43.130,  5.960),
        "La Rode":             (43.135,  5.920),
        "Sainte-Anne":         (43.140,  5.915),
        "Le Jonquet":          (43.148,  5.908),
        "La Beaucaire":        (43.142,  5.940),
        "Pont-du-Las":         (43.145,  5.935),
        "Siblas":              (43.150,  5.945),
        "Saint-Roch":          (43.132,  5.942),
        "La Loubière":         (43.138,  5.930),
        "Pré-Joli":            (43.155,  5.935),
        "Font-Pré":            (43.157,  5.940),
        "Dardennes":           (43.162,  5.922),
        "La Valette":          (43.138,  5.979),
        "La Garde":            (43.124,  5.980),
        "Six-Fours":           (43.105,  5.820),
        "La Seyne-sur-Mer":    (43.105,  5.882),
        "Ollioules":           (43.136,  5.849),
    }
 
    center = QUARTIER_COORDS.get(quartier_analyse, (43.124, 5.928))
    # Filtrer les DVF dans un rayon ~1.5 km autour du centroïde
    lat_c, lon_c = center
    delta_lat = 0.013  # ~1.5 km
    delta_lon = 0.018
    dvf_zone = dvf_geo_q[
        dvf_geo_q["latitude"].between(lat_c - delta_lat, lat_c + delta_lat)
        & dvf_geo_q["longitude"].between(lon_c - delta_lon, lon_c + delta_lon)
    ]
 
    if not dvf_zone.empty:
        st.pydeck_chart(make_dvf_map(dvf_zone, zoom=14), use_container_width=True)
        st.caption(
            f"{len(dvf_zone):,} transactions dans la zone · "
            f"{', '.join(type_bien)} · budget ≤ {budget_max:,.0f} € · "
            f"cliquez sur un point pour le détail"
        )
    elif not dvf_geo_q.empty:
        st.pydeck_chart(make_dvf_map(dvf_geo_q, zoom=13), use_container_width=True)
        st.caption("Aucune transaction dans le rayon immédiat — vue globale Toulon (filtrée) affichée.")
    else:
        st.info("Aucune transaction DVF ne correspond à ces critères.")
 
    st.divider()
 
    col_left, col_right = st.columns(2)
 
    with col_left:
        st.subheader("Distribution des prix dans ce quartier")
        pm2_q_list = ann_q["prix_m2"].tolist()
        pm2_all_list = ann_raw["prix_m2"].tolist()
 
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(pm2_all_list, bins=40, color="#CBD5E1", alpha=0.6, label="Toulon global", density=True)
        ax.hist(pm2_q_list, bins=20, color=BLUE, alpha=0.8, label=f"{quartier_analyse}", density=True)
        ax.axvline(pm2_q_med, color=BLUE, linewidth=2, linestyle="--",
                   label=f"Médiane {quartier_analyse} : {pm2_q_med:,.0f} €/m²")
        ax.axvline(pm2_global_med, color="#94A3B8", linewidth=1.5, linestyle=":",
                   label=f"Médiane Toulon : {pm2_global_med:,.0f} €/m²")
        ax.xaxis.set_major_formatter(fmt_m2)
        ax.set_xlabel("Prix/m²")
        ax.set_ylabel("Densité")
        ax.legend(fontsize=8)
        ax.set_facecolor("#F8FAFC")
        fig.patch.set_facecolor("#F8FAFC")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
 
    with col_right:
        st.subheader("Comparatif des quartiers toulonnais")
        by_q = (
            ann_raw.groupby("quartier")["prix_m2"]
            .agg(["median", "count"])
            .rename(columns={"median": "moy", "count": "n"})
        )
        by_q = by_q[by_q["n"] >= 3].sort_values("moy", ascending=True)
 
        colors_bar = [BLUE if q == quartier_analyse else ("#E2E8F0" if v <= pm2_global_med else "#FCA5A5")
                      for q, v in zip(by_q.index, by_q["moy"])]
 
        fig, ax = plt.subplots(figsize=(6, max(5, len(by_q) * 0.32)))
        ax.barh(by_q.index, by_q["moy"], color=colors_bar, alpha=0.9)
        ax.axvline(pm2_global_med, color=AMBER, linestyle="--", linewidth=1.5,
                   label=f"Médiane Toulon : {pm2_global_med:,.0f} €/m²")
        for i, (q, row_q) in enumerate(by_q.iterrows()):
            ax.text(row_q["moy"] + 30, i, f"{row_q['moy']:,.0f} €", va="center", fontsize=7)
        ax.xaxis.set_major_formatter(fmt_m2)
        ax.set_xlabel("Prix/m² médian")
        ax.legend(fontsize=8)
        ax.set_facecolor("#F8FAFC")
        fig.patch.set_facecolor("#F8FAFC")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
 
    # DPE dans ce quartier
    ann_q_dpe = ann_q[ann_q["energie"].isin(["A", "B", "C", "D", "E", "F", "G"])]
    if not ann_q_dpe.empty:
        st.divider()
        st.subheader("Bilan énergétique des biens dans ce quartier")
        ordre_dpe = ["A", "B", "C", "D", "E", "F", "G"]
        colors_dpe = {"A": "#15803D", "B": "#65A30D", "C": "#A3E635",
                      "D": "#FBBF24", "E": "#F97316", "F": "#EF4444", "G": "#991B1B"}
        counts_dpe = ann_q_dpe["energie"].value_counts().reindex(ordre_dpe).dropna()
        pct_mauvais = ann_q_dpe["energie"].isin(["F", "G"]).mean() * 100
 
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar(counts_dpe.index, counts_dpe.values,
               color=[colors_dpe[d] for d in counts_dpe.index], alpha=0.9)
        ax.set_xlabel("Classe DPE")
        ax.set_ylabel("Nb annonces")
        ax.set_title(f"DPE · {quartier_analyse}", fontweight="bold")
        for bar in ax.patches:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(int(bar.get_height())), ha="center", fontsize=9)
        ax.set_facecolor("#F8FAFC")
        fig.patch.set_facecolor("#F8FAFC")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
 
        if pct_mauvais > 30:
            st.warning(
                f"⚠️ **{pct_mauvais:.0f}% des biens annoncés à {quartier_analyse} sont en DPE F ou G.** "
                f"Anticipez des travaux de rénovation et une décote sur la revente."
            )
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Ce bien est-il une bonne affaire ?
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Analyser un bien précis")
    st.markdown(
        "Renseignez les caractéristiques d'un bien qui vous intéresse. "
        "Notre modèle vous dira si le prix demandé est justifié."
    )
 
    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    with col_i1:
        surface_input = st.number_input("Surface (m²)", min_value=10, max_value=500, value=65)
    with col_i2:
        prix_input = st.number_input("Prix demandé (€)", min_value=10_000, max_value=2_000_000, value=220_000, step=5_000)
    with col_i3:
        quartier_input = st.selectbox("Quartier", ["(non précisé)"] + quartiers_dispo)
    with col_i4:
        dpe_input = st.selectbox("DPE", ["(non précisé)", "A", "B", "C", "D", "E", "F", "G"])
 
    st.divider()
 
    prix_predit = alpha_global + beta_global * surface_input
    ecart = prix_input - prix_predit
    ecart_pct = ecart / prix_predit * 100
    verdict_label, verdict_color = verdict_ecart(ecart_pct)
    pm2_input = prix_input / surface_input
 
    # Métriques principales
    col_v1, col_v2, col_v3 = st.columns(3)
    col_v1.metric("Prix demandé", f"{prix_input:,.0f} €")
    col_v2.metric("Prix estimé par le marché", f"{prix_predit:,.0f} €",
                  delta=f"{ecart:+,.0f} € ({ecart_pct:+.1f}%)",
                  delta_color="inverse")
    col_v3.metric("Prix/m²", f"{pm2_input:,.0f} €/m²")
 
    # Verdict principal
    st.markdown(f"## {verdict_label}")
 
    # Phrase de conclusion
    pm2_q_input = _pm2_par_quartier.get(quartier_input, None) if quartier_input != "(non précisé)" else None
    bien_sim = pd.Series({
        "prix": prix_input,
        "surface": surface_input,
        "quartier": quartier_input if quartier_input != "(non précisé)" else None,
        "energie": dpe_input if dpe_input != "(non précisé)" else None,
    })
    st.info(phrase_verdict(bien_sim, prix_predit, pm2_q_input))
 
    st.divider()
 
    # Graphique : positionnement du bien sur la droite de régression
    col_g1 = st.columns(1)[0]
 
 
    with col_g1:
        st.subheader("Ce que ça veut dire")
 
        st.markdown(f"""
        Pour **{surface_input} m²**, le marché paye en moyenne :
 
        > ### {prix_predit:,.0f} €
 
        Vous regardez un bien à **{prix_input:,.0f} €**, soit **{ecart_pct:+.1f}%**
        {'de moins' if ecart < 0 else 'de plus'} que la valeur de marché.
 
        ---
        """)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Le marché en bref
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Le marché immobilier toulonnais · Vue d'ensemble")
 
    # KPIs marché global
    pm2_dvf_list = dvf_raw["prix_au_m2"].tolist()
    pm2_ann_list = ann_raw["prix_m2"].tolist()
    ecart_global = mean(pm2_ann_list) - mean(pm2_dvf_list)
 
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Prix/m² médian (ventes réelles)", f"{dvf_raw['prix_au_m2'].median():,.0f} €/m²",
                help="Médiane des transactions DVF 2024 enregistrées aux impôts.")
    col2.metric("Prix/m² médian (annonces)", f"{ann_raw['prix_m2'].median():,.0f} €/m²",
                help="Médiane des annonces SeLoger actuellement en ligne.")
    col3.metric("Écart annonces vs ventes réelles", f"{ecart_global:+,.0f} €/m²",
                delta_color="inverse",
                help="Différence entre ce que les vendeurs demandent et ce qui est réellement payé.")
    col4.metric("Transactions enregistrées (2024)", f"{len(dvf_raw):,}")
 
    if ecart_global > 0:
        st.info(
            f"📌 **Les vendeurs demandent en moyenne {ecart_global:,.0f} €/m² de plus** que ce qui est réellement payé. "
            f"Cet écart ({ecart_global / mean(pm2_dvf_list) * 100:.1f}%) représente la marge de négociation typique à Toulon."
        )
    else:
        st.info("Les prix annoncés sont alignés avec les transactions réelles.")
 
    st.divider()
 
    # Carte heatmap globale Toulon
    st.subheader("🗺️ Carte des prix immobiliers · Toulon 2024")
    st.caption(
        f"**{len(dvf_f):,} transactions** correspondant à vos critères "
        f"({', '.join(type_bien)} · budget ≤ {budget_max:,.0f} €) sur {len(dvf_raw):,} au total · "
        "🟢 = abordable → 🔴 = cher"
    )
 
    tab4_map_col, tab4_legend_col = st.columns([4, 1])
    with tab4_map_col:
        if not dvf_f.empty:
            st.pydeck_chart(make_dvf_map(dvf_f, zoom=12, pitch=30), use_container_width=True)
        else:
            st.info("Aucune transaction DVF ne correspond à ces critères.")
 
    with tab4_legend_col:
        p5  = dvf_f["prix_au_m2"].quantile(0.05) if not dvf_f.empty else 0
        p50 = dvf_f["prix_au_m2"].quantile(0.50) if not dvf_f.empty else 0
        p95 = dvf_f["prix_au_m2"].quantile(0.95) if not dvf_f.empty else 0
        st.markdown("**Légende**")
        st.markdown(f"🟢 **Accessible**  \n< {p5:,.0f} €/m²")
        st.markdown(f"🟡 **Médiane**  \n{p50:,.0f} €/m²")
        st.markdown(f"🔴 **Premium**  \n> {p95:,.0f} €/m²")
        st.divider()
        st.caption("Source : DVF data.gouv.fr")
 
    st.divider()
 
    col_left, col_right = st.columns(2)
 
    with col_left:
        st.subheader("📅 Évolution mensuelle des ventes (DVF 2024)")
        monthly = (
            dvf_raw.set_index("date_mutation")
            .resample("ME")["prix_au_m2"]
            .agg(["median", "count"])
            .reset_index()
        )
        monthly.columns = ["mois", "mediane", "nb"]
 
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax2 = ax1.twinx()
        ax1.bar(monthly["mois"], monthly["nb"], color=BLUE, alpha=0.4, width=20, label="Nb ventes")
        ax2.plot(monthly["mois"], monthly["mediane"], color=RED, linewidth=2.5,
                 marker="o", markersize=5, label="Prix médian/m²")
        ax1.set_ylabel("Nb ventes", color=BLUE)
        ax2.set_ylabel("Prix médian/m²", color=RED)
        ax2.yaxis.set_major_formatter(fmt_m2)
        ax1.set_facecolor("#F8FAFC")
        fig.patch.set_facecolor("#F8FAFC")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
 
        # Tendance en langage naturel
        if len(monthly) >= 3:
            debut = monthly["mediane"].iloc[:3].mean()
            fin = monthly["mediane"].iloc[-3:].mean()
            tendance_pct = (fin - debut) / debut * 100
            if tendance_pct > 3:
                st.success(f"📈 Les prix ont **augmenté de {tendance_pct:.1f}%** sur l'année — marché en hausse.")
            elif tendance_pct < -3:
                st.warning(f"📉 Les prix ont **baissé de {abs(tendance_pct):.1f}%** sur l'année — opportunité d'achat.")
            else:
                st.info(f"➡️ Les prix sont **stables** (variation de {tendance_pct:+.1f}%) — marché équilibré.")
 
    with col_right:
        st.subheader("📍 Classement des quartiers par prix/m²")
        by_q_rank = (
            ann_raw.groupby("quartier")["prix_m2"]
            .agg(["median", "count"])
            .rename(columns={"median": "moy", "count": "n"})
        )
        by_q_rank = by_q_rank[by_q_rank["n"] >= 3].sort_values("moy", ascending=True)
        med_glob = ann_raw["prix_m2"].median()
 
        fig, ax = plt.subplots(figsize=(7, max(5, len(by_q_rank) * 0.30)))
        colors_rank = [GREEN if v <= med_glob else RED for v in by_q_rank["moy"]]
        ax.barh(by_q_rank.index, by_q_rank["moy"], color=colors_rank, alpha=0.75)
        ax.axvline(med_glob, color=AMBER, linestyle="--", linewidth=1.5,
                   label=f"Médiane : {med_glob:,.0f} €/m²")
        for i, (q, row_q) in enumerate(by_q_rank.iterrows()):
            ax.text(row_q["moy"] + 20, i, f"{row_q['moy']:,.0f} €", va="center", fontsize=7)
        ax.xaxis.set_major_formatter(fmt_m2)
        ax.legend(fontsize=8)
        ax.set_facecolor("#F8FAFC")
        fig.patch.set_facecolor("#F8FAFC")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
 
        st.caption("🟢 Vert = sous la médiane (plus accessible) · 🔴 Rouge = au-dessus (plus cher)")
 
