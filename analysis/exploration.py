"""
Analyse exploratoire complète — Observatoire Immobilier Toulonnais
==================================================================
Génère tous les graphiques dans analysis/figures/
Deux sources : DVF (data.gouv.fr) + Annonces SeLoger (scraping)

Usage :
    python3 analysis/exploration.py
"""

import math
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # mode non-interactif

# ── Chemins ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DVF_PATH = os.path.join(ROOT, "donnees", "processed", "dvf_clean.csv")
ANN_PATH = os.path.join(ROOT, "donnees", "processed", "annonces_propres.csv")
FIG_DIR = os.path.join(ROOT, "analysis", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Style global ───────────────────────────────────────────────────────────
PALETTE_DVF = "#2563EB"       # bleu — données gouvernementales
PALETTE_ANN = "#DC2626"       # rouge — données SeLoger
PALETTE_BOTH = "#7C3AED"      # violet — croisement
BG = "#F8FAFC"
GRID_COLOR = "#E2E8F0"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

fmt_eur = mticker.FuncFormatter(lambda x, _: f"{x:,.0f} €")
fmt_m2 = mticker.FuncFormatter(lambda x, _: f"{x:,.0f} €/m²")


# ── Chargement ─────────────────────────────────────────────────────────────
def load_data():
    dvf = pd.read_csv(DVF_PATH, low_memory=False)
    dvf["date_mutation"] = pd.to_datetime(dvf["date_mutation"], dayfirst=True, errors="coerce")
    dvf = dvf[
        dvf["prix_au_m2"].notna()
        & (dvf["prix_au_m2"] >= 500)
        & (dvf["prix_au_m2"] <= 20_000)
        & dvf["type_local"].isin(["Appartement", "Maison"])
    ].copy()

    ann = pd.read_csv(ANN_PATH)
    ann = ann[ann["prix_m2"].notna() & (ann["prix_m2"] > 0)].copy()
    return dvf, ann


# ── Statistiques descriptives textuelles ──────────────────────────────────
def print_stats(dvf: pd.DataFrame, ann: pd.DataFrame):
    sep = "=" * 65
    print(f"\n{sep}")
    print("STATISTIQUES DESCRIPTIVES — DVF Toulon (données gouvernementales)")
    print(sep)
    print(f"Période couverte     : {dvf['date_mutation'].min().date()} → {dvf['date_mutation'].max().date()}")
    print(f"Nb transactions      : {len(dvf):,}")
    print(f"Types de biens       : {dvf['type_local'].value_counts().to_dict()}")
    print(f"\nPrix/m² (€/m²)")
    col = dvf["prix_au_m2"]
    print(f"  Moyenne            : {col.mean():,.0f}")
    print(f"  Médiane            : {col.median():,.0f}")
    print(f"  Écart-type         : {col.std():,.0f}")
    print(f"  Min                : {col.min():,.0f}")
    print(f"  Max                : {col.max():,.0f}")
    print(f"  Q1 (25%)           : {col.quantile(0.25):,.0f}")
    print(f"  Q3 (75%)           : {col.quantile(0.75):,.0f}")
    print(f"  IQR                : {col.quantile(0.75) - col.quantile(0.25):,.0f}")

    print(f"\nPrix transaction (€)")
    col2 = dvf["valeur_fonciere"]
    print(f"  Moyenne            : {col2.mean():,.0f}")
    print(f"  Médiane            : {col2.median():,.0f}")
    print(f"  Écart-type         : {col2.std():,.0f}")

    print(f"\nSurface bâtie (m²)")
    col3 = dvf["surface_reelle_bati"]
    print(f"  Moyenne            : {col3.mean():.1f}")
    print(f"  Médiane            : {col3.median():.1f}")
    print(f"  Écart-type         : {col3.std():.1f}")

    print(f"\nNb pièces moyen     : {dvf['nombre_pieces_principales'].mean():.2f}")

    print(f"\nPrix/m² par type de bien :")
    for t, grp in dvf.groupby("type_local"):
        print(f"  {t:15s} : moy {grp['prix_au_m2'].mean():,.0f}  |  méd {grp['prix_au_m2'].median():,.0f}  |  n={len(grp):,}")

    print(f"\n{sep}")
    print("STATISTIQUES DESCRIPTIVES — Annonces SeLoger (scraping)")
    print(sep)
    print(f"Nb annonces          : {len(ann):,}")
    print(f"Nb quartiers         : {ann['quartier'].nunique()}")
    print(f"Types de biens       : {ann['type'].value_counts().to_dict()}")

    print(f"\nPrix/m² (€/m²)")
    col = ann["prix_m2"]
    print(f"  Moyenne            : {col.mean():,.0f}")
    print(f"  Médiane            : {col.median():,.0f}")
    print(f"  Écart-type         : {col.std():,.0f}")
    print(f"  Min                : {col.min():,.0f}")
    print(f"  Max                : {col.max():,.0f}")
    print(f"  Q1 (25%)           : {col.quantile(0.25):,.0f}")
    print(f"  Q3 (75%)           : {col.quantile(0.75):,.0f}")

    print(f"\nDPE (performance énergétique) :")
    print(f"  {ann['energie'].value_counts().to_dict()}")

    print(f"\nPrix/m² par quartier (top 10 + bottom 5) :")
    by_q = ann.groupby("quartier")["prix_m2"].agg(["mean", "median", "count"]).sort_values("mean", ascending=False)
    print(by_q.head(10).to_string())
    print("  ...")
    print(by_q.tail(5).to_string())


# ── Graphiques ─────────────────────────────────────────────────────────────

def save(name: str):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegardé : analysis/figures/{name}")


# 01 — Distribution prix/m² DVF vs SeLoger
def fig_01_distribution_prix_m2(dvf, ann):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("01 — Distribution des prix au m²\nDVF (transactions réelles) vs Annonces SeLoger", fontsize=14, fontweight="bold")

    for ax, data, color, label, n in [
        (axes[0], dvf["prix_au_m2"], PALETTE_DVF, "DVF — Transactions", len(dvf)),
        (axes[1], ann["prix_m2"], PALETTE_ANN, "SeLoger — Annonces", len(ann)),
    ]:
        ax.hist(data, bins=50, color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.axvline(data.mean(), color="black", linestyle="--", linewidth=1.5, label=f"Moyenne : {data.mean():,.0f} €/m²")
        ax.axvline(data.median(), color="#F59E0B", linestyle="-.", linewidth=1.5, label=f"Médiane : {data.median():,.0f} €/m²")
        ax.set_title(f"{label} (n={n:,})", fontweight="bold")
        ax.set_xlabel("Prix au m² (€/m²)")
        ax.set_ylabel("Nombre de biens")
        ax.xaxis.set_major_formatter(fmt_m2)
        ax.legend(fontsize=9)

    plt.tight_layout()
    save("01_distribution_prix_m2.png")


# 02 — Boxplot prix/m² par type de bien
def fig_02_boxplot_type(dvf, ann):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("02 — Prix au m² par type de bien", fontsize=14, fontweight="bold")

    for ax, data, col_pm2, col_type, color, label in [
        (axes[0], dvf, "prix_au_m2", "type_local", PALETTE_DVF, "DVF"),
        (axes[1], ann, "prix_m2", "type", PALETTE_ANN, "SeLoger"),
    ]:
        groups = [grp[col_pm2].values for _, grp in data.groupby(col_type)]
        labels = list(data[col_type].unique())
        bp = ax.boxplot(groups, patch_artist=True, notch=True, tick_labels=labels,
                        medianprops=dict(color="#F59E0B", linewidth=2))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(f"{label}", fontweight="bold")
        ax.set_ylabel("Prix au m² (€/m²)")
        ax.yaxis.set_major_formatter(fmt_m2)

    plt.tight_layout()
    save("02_boxplot_type_de_bien.png")


# 03 — Évolution temporelle DVF (mensuelle)
def fig_03_evolution_temporelle(dvf):
    monthly = (
        dvf.set_index("date_mutation")
        .resample("ME")["prix_au_m2"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    monthly.columns = ["mois", "moyenne", "mediane", "nb_transactions"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("03 — Évolution temporelle DVF\nPrix/m² et volume de transactions (2024)", fontsize=14, fontweight="bold")

    axes[0].plot(monthly["mois"], monthly["moyenne"], color=PALETTE_DVF, linewidth=2, marker="o", markersize=5, label="Moyenne")
    axes[0].plot(monthly["mois"], monthly["mediane"], color=PALETTE_BOTH, linewidth=2, linestyle="--", marker="s", markersize=5, label="Médiane")
    axes[0].set_ylabel("Prix/m² (€/m²)")
    axes[0].yaxis.set_major_formatter(fmt_m2)
    axes[0].legend()
    axes[0].set_title("Prix au m² moyen et médian par mois", fontweight="bold")

    axes[1].bar(monthly["mois"], monthly["nb_transactions"], color=PALETTE_DVF, alpha=0.7, width=20)
    axes[1].set_ylabel("Nb transactions")
    axes[1].set_xlabel("Mois")
    axes[1].set_title("Volume mensuel de transactions", fontweight="bold")

    plt.tight_layout()
    save("03_evolution_temporelle_dvf.png")


# 04 — Prix/m² par quartier SeLoger (barplot horizontal)
def fig_04_prix_par_quartier(ann):
    by_q = (
        ann.groupby("quartier")["prix_m2"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "moy", "count": "n"})
        .sort_values("moy")
    )
    by_q = by_q[by_q["n"] >= 3]  # quartiers avec au moins 3 annonces

    fig, ax = plt.subplots(figsize=(12, max(7, len(by_q) * 0.35)))
    colors = [PALETTE_ANN if v > by_q["moy"].median() else "#10B981" for v in by_q["moy"]]
    bars = ax.barh(by_q.index, by_q["moy"], color=colors, alpha=0.85)
    ax.axvline(by_q["moy"].median(), color="#F59E0B", linestyle="--", linewidth=1.5,
               label=f"Médiane : {by_q['moy'].median():,.0f} €/m²")
    for bar, (_, row) in zip(bars, by_q.iterrows()):
        ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
                f"{row['moy']:,.0f} € (n={int(row['n'])})", va="center", fontsize=8)
    ax.set_xlabel("Prix moyen au m² (€/m²)")
    ax.set_title("04 — Prix/m² moyen par quartier — Annonces SeLoger\n[vert] Sous médiane  |  [rouge] Au-dessus médiane",
                 fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(fmt_m2)
    ax.legend()
    plt.tight_layout()
    save("04_prix_par_quartier_seloger.png")


# 05 — Scatter prix ~ surface (DVF + annonces)
def fig_05_scatter_prix_surface(dvf, ann):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("05 — Relation Prix ~ Surface\nDVF (transactions) vs SeLoger (annonces)", fontsize=14, fontweight="bold")

    for ax, data, xcol, ycol, color, label in [
        (axes[0], dvf.sample(min(1500, len(dvf)), random_state=42), "surface_reelle_bati", "valeur_fonciere", PALETTE_DVF, "DVF"),
        (axes[1], ann, "surface", "prix", PALETTE_ANN, "SeLoger"),
    ]:
        x = data[xcol].values
        y = data[ycol].values
        ax.scatter(x, y, alpha=0.3, color=color, s=15, edgecolors="none")

        # Droite de régression
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() > 2:
            coeffs = np.polyfit(x[mask], y[mask], 1)
            x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
            y_line = np.polyval(coeffs, x_line)
            ax.plot(x_line, y_line, color="black", linewidth=2, label=f"Régression : y = {coeffs[0]:,.0f}x + {coeffs[1]:,.0f}")
            ss_res = np.sum((y[mask] - np.polyval(coeffs, x[mask])) ** 2)
            ss_tot = np.sum((y[mask] - y[mask].mean()) ** 2)
            r2 = 1 - ss_res / ss_tot
            ax.legend(title=f"R² = {r2:.3f}", fontsize=9)

        ax.set_xlabel(f"Surface (m²)")
        ax.set_ylabel(f"Prix (€)")
        ax.yaxis.set_major_formatter(fmt_eur)
        ax.set_title(f"{label} (n={len(data):,})", fontweight="bold")

    plt.tight_layout()
    save("05_scatter_prix_surface.png")


# 06 — Heatmap de corrélation (DVF)
def fig_06_heatmap_correlation_dvf(dvf):
    cols = ["valeur_fonciere", "surface_reelle_bati", "nombre_pieces_principales", "prix_au_m2"]
    labels = ["Prix total (€)", "Surface (m²)", "Nb pièces", "Prix/m²"]
    subset = dvf[cols].dropna()
    corr = subset.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Coefficient de corrélation")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=11, color=color, fontweight="bold")
    ax.set_title("06 — Heatmap de corrélation — DVF Toulon\n(Transactions gouvernementales 2024)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save("06_heatmap_correlation_dvf.png")


# 07 — Heatmap de corrélation (SeLoger)
def fig_07_heatmap_correlation_seloger(ann):
    cols = ["prix", "surface", "prix_m2", "nb_pieces"]
    labels = ["Prix (€)", "Surface (m²)", "Prix/m²", "Nb pièces"]
    subset = ann[cols].dropna()
    corr = subset.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Coefficient de corrélation")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=11, color=color, fontweight="bold")
    ax.set_title("07 — Heatmap de corrélation — Annonces SeLoger\n(707 annonces scrapées)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save("07_heatmap_correlation_seloger.png")


# 08 — Distribution DPE et prix/m² par DPE
def fig_08_dpe_analyse(ann):
    ordre_dpe = ["A", "B", "C", "D", "E", "F", "G"]
    colors_dpe = {"A": "#15803D", "B": "#65A30D", "C": "#A3E635",
                  "D": "#FBBF24", "E": "#F97316", "F": "#EF4444", "G": "#991B1B"}
    ann_dpe = ann[ann["energie"].isin(ordre_dpe)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("08 — Diagnostic de Performance Énergétique (DPE)\nRépartition et impact sur le prix/m²",
                 fontsize=14, fontweight="bold")

    counts = ann_dpe["energie"].value_counts().reindex(ordre_dpe).dropna()
    bars = axes[0].bar(counts.index, counts.values,
                       color=[colors_dpe.get(d, "#94A3B8") for d in counts.index], alpha=0.85)
    axes[0].set_xlabel("Classe DPE")
    axes[0].set_ylabel("Nombre d'annonces")
    axes[0].set_title("Répartition des annonces par classe DPE", fontweight="bold")
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     str(int(bar.get_height())), ha="center", va="bottom", fontsize=10)

    dpe_prix = ann_dpe.groupby("energie")["prix_m2"].median().reindex(ordre_dpe).dropna()
    axes[1].bar(dpe_prix.index, dpe_prix.values,
                color=[colors_dpe.get(d, "#94A3B8") for d in dpe_prix.index], alpha=0.85)
    axes[1].set_xlabel("Classe DPE")
    axes[1].set_ylabel("Prix médian au m² (€/m²)")
    axes[1].set_title("Prix/m² médian par classe DPE", fontweight="bold")
    axes[1].yaxis.set_major_formatter(fmt_m2)

    plt.tight_layout()
    save("08_dpe_analyse.png")


# 09 — Comparaison DVF vs SeLoger : prix/m² côte à côte
def fig_09_comparaison_sources(dvf, ann):
    fig, ax = plt.subplots(figsize=(12, 5))

    dvf_appt = dvf[dvf["type_local"] == "Appartement"]["prix_au_m2"]
    ann_appt = ann[ann["type"] == "Appartement"]["prix_m2"]
    dvf_mais = dvf[dvf["type_local"] == "Maison"]["prix_au_m2"]
    ann_mais = ann[ann["type"] == "Maison"]["prix_m2"]

    data = [dvf_appt, ann_appt, dvf_mais, ann_mais]
    labels = ["DVF\nAppartement", "SeLoger\nAppartement", "DVF\nMaison", "SeLoger\nMaison"]
    colors = [PALETTE_DVF, PALETTE_ANN, PALETTE_DVF, PALETTE_ANN]

    bp = ax.boxplot(data, patch_artist=True, tick_labels=labels, notch=True,
                    medianprops=dict(color="#F59E0B", linewidth=2.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_ylabel("Prix au m² (€/m²)")
    ax.yaxis.set_major_formatter(fmt_m2)
    ax.set_title("09 — Comparaison DVF vs SeLoger par type de bien\n"
                 "Bleu = données gouvernementales (transactions) | Rouge = annonces (prix demandés)",
                 fontsize=13, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE_DVF, alpha=0.5, label="DVF — transactions réelles"),
        Patch(facecolor=PALETTE_ANN, alpha=0.5, label="SeLoger — prix demandés"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    plt.tight_layout()
    save("09_comparaison_dvf_vs_seloger.png")


# 10 — Carte GPS des transactions DVF (scatter géographique)
def fig_10_carte_gps_dvf(dvf):
    dvf_geo = dvf[dvf["longitude"].notna() & dvf["latitude"].notna()].copy()

    fig, ax = plt.subplots(figsize=(10, 9))
    sc = ax.scatter(
        dvf_geo["longitude"], dvf_geo["latitude"],
        c=dvf_geo["prix_au_m2"],
        cmap="RdYlGn_r",
        alpha=0.5, s=8, edgecolors="none",
        vmin=dvf_geo["prix_au_m2"].quantile(0.05),
        vmax=dvf_geo["prix_au_m2"].quantile(0.95),
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Prix au m² (€/m²)", fontsize=10)
    cbar.formatter = fmt_m2
    cbar.update_ticks()
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("10 — Carte géographique des transactions DVF — Toulon 2024\n"
                 "Vert = bas prix | Rouge = prix élevés",
                 fontsize=13, fontweight="bold")
    ax.set_facecolor("#1E293B")
    fig.patch.set_facecolor("#1E293B")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    plt.tight_layout()
    save("10_carte_gps_dvf.png")


# 11 — Régression linéaire with R² (SeLoger)
def fig_11_regression_r2(ann):
    data = ann[ann["prix_m2"].notna() & ann["surface"].notna() & (ann["surface"] < 300)].copy()
    x = data["surface"].values
    y = data["prix"].values

    coeffs = np.polyfit(x, y, 1)
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = np.polyval(coeffs, x_line)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(x, y, alpha=0.35, color=PALETTE_ANN, s=20, edgecolors="none", label="Annonces SeLoger")
    ax.plot(x_line, y_line, color="black", linewidth=2.5,
            label=f"Régression : y = {coeffs[0]:,.0f}x + {coeffs[1]:,.0f}")

    textbox = (
        f"R² = {r2:.4f}\n"
        f"β (pente) = {coeffs[0]:,.0f} €/m²\n"
        f"α (intercept) = {coeffs[1]:,.0f} €\n"
        f"Corrélation r = {math.sqrt(r2):.3f}"
    )
    ax.text(0.97, 0.05, textbox, transform=ax.transAxes, fontsize=10,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#CBD5E1", alpha=0.9))

    ax.set_xlabel("Surface (m²)")
    ax.set_ylabel("Prix (€)")
    ax.yaxis.set_major_formatter(fmt_eur)
    ax.set_title("11 — Régression linéaire Prix ~ Surface — Annonces SeLoger\n"
                 "From scratch (Joel Grus ch.14) | sans sklearn",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    save("11_regression_lineaire_r2.png")


# 12 — Distribution des surfaces par nb de pièces
def fig_12_surface_par_pieces(dvf):
    dvf_f = dvf[(dvf["nombre_pieces_principales"] >= 1) & (dvf["nombre_pieces_principales"] <= 6)].copy()
    pieces = sorted(dvf_f["nombre_pieces_principales"].unique())
    groups = [dvf_f[dvf_f["nombre_pieces_principales"] == p]["surface_reelle_bati"].values for p in pieces]
    labels = [f"T{int(p)}" for p in pieces]

    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot(groups, patch_artist=True, tick_labels=labels,
                    medianprops=dict(color="#F59E0B", linewidth=2))
    cmap = plt.cm.Blues
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(0.3 + 0.12 * i))
    ax.set_xlabel("Nombre de pièces principales (T1 → T6)")
    ax.set_ylabel("Surface bâtie (m²)")
    ax.set_title("12 — Distribution des surfaces par typologie — DVF Toulon\n"
                 "Les boîtes montrent Q1, médiane, Q3 | moustaches = min/max sans outliers",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save("12_surface_par_pieces_dvf.png")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("Chargement des données...")
    dvf, ann = load_data()
    print(f"  DVF : {len(dvf):,} transactions | Annonces : {len(ann):,} annonces")

    print_stats(dvf, ann)

    print("\nGénération des graphiques...")
    fig_01_distribution_prix_m2(dvf, ann)
    fig_02_boxplot_type(dvf, ann)
    fig_03_evolution_temporelle(dvf)
    fig_04_prix_par_quartier(ann)
    fig_05_scatter_prix_surface(dvf, ann)
    fig_06_heatmap_correlation_dvf(dvf)
    fig_07_heatmap_correlation_seloger(ann)
    fig_08_dpe_analyse(ann)
    fig_09_comparaison_sources(dvf, ann)
    fig_10_carte_gps_dvf(dvf)
    fig_11_regression_r2(ann)
    fig_12_surface_par_pieces(dvf)

    print(f"\nTous les graphiques sont dans : analysis/figures/")
    print("Fichiers générés :")
    for f in sorted(os.listdir(FIG_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
