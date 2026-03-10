# Méthodologie — Observatoire Immobilier Toulonnais

**Projet NidDouillet | Epitech IA Spé | Toulon INSEE 83137**

---

## Sommaire

1. [Sources de données](#1-sources-de-données)
2. [Pipeline de nettoyage](#2-pipeline-de-nettoyage)
3. [Représentation des datasets](#3-représentation-et-combinaison-des-datasets)
4. [Statistiques descriptives from scratch](#4-statistiques-descriptives-from-scratch)
5. [Régression linéaire & R²](#5-régression-linéaire--r²)
6. [Analyse exploratoire — graphiques](#6-analyse-exploratoire--graphiques)
7. [Scoring des opportunités](#7-scoring-des-opportunités)

---

## 1. Sources de données

### 🔵 DVF — Demandes de Valeurs Foncières (données gouvernementales)

| Attribut | Valeur |
|----------|--------|
| **Source** | [data.gouv.fr / geo-dvf](https://files.data.gouv.fr/geo-dvf/latest/csv/83/) |
| **Fichier brut** | `donnees/raw/datagouv_83137_20242025.csv` |
| **Fichier nettoyé** | `donnees/processed/dvf_clean.csv` |
| **Période** | Fév 2024 → Déc 2024 |
| **Commune** | Toulon (code INSEE 83137) |
| **Nb transactions** | 5 340 après nettoyage |
| **Nature** | Transactions **réellement effectuées**, enregistrées aux impôts |

**Colonnes conservées (13 sur 40)** :

| Colonne | Type | Description |
|---------|------|-------------|
| `id_mutation` | string | Identifiant unique de la transaction |
| `date_mutation` | date | Date de la vente |
| `nature_mutation` | string | Type d'opération (Vente, etc.) |
| `valeur_fonciere` | float | **Prix réellement payé** en euros |
| `code_postal` | int | Code postal (83000 / 83100 / 83200) |
| `nom_commune` | string | Commune (Toulon uniquement ici) |
| `type_local` | string | Appartement ou Maison |
| `surface_reelle_bati` | float | Surface habitable en m² |
| `nombre_pieces_principales` | int | Nombre de pièces |
| `surface_terrain` | float | Surface terrain (maisons, souvent null) |
| `longitude` | float | Coordonnée GPS |
| `latitude` | float | Coordonnée GPS |
| `prix_au_m2` | float | **Calculé** : valeur_fonciere / surface_reelle_bati |

---

### 🔴 Annonces SeLoger — Scraping

| Attribut | Valeur |
|----------|--------|
| **Source** | SeLoger.com via scraping Python |
| **Fichier brut** | `donnees/raw/seloger_document_base.csv` |
| **Fichier nettoyé** | `donnees/processed/annonces_propres.csv` |
| **Nb annonces** | 707 |
| **Quartiers** | 42 quartiers toulonnais |
| **Nature** | Biens **en cours de vente** — prix **demandés** (pas encore vendus) |

**Colonnes disponibles (8)** :

| Colonne | Type | Description |
|---------|------|-------------|
| `prix` | float | Prix demandé en euros |
| `surface` | float | Surface en m² |
| `prix_m2` | float | Prix/m² calculé |
| `nb_pieces` | float | Nombre de pièces |
| `type` | string | Appartement ou Maison |
| `quartier` | string | Nom du quartier toulonnais |
| `energie` | string | Classe DPE (A → G) |
| `ville` | string | Toulon (constant) |

---

## 2. Pipeline de nettoyage

### DVF (`analysis/cleaning_dvf.py`)

```
Fichier brut (séparateur ";", 40 colonnes, ~8000 lignes)
    ↓
Suppression lignes vides
    ↓
Suppression si prix ≤ 0 ou absent
    ↓
Suppression si surface_reelle_bati ≤ 0 ou absente
    ↓
Calcul prix_au_m2 = valeur_fonciere / surface_reelle_bati
    ↓
Conservation des 13 colonnes utiles
    ↓
Tri chronologique
    ↓
Export : donnees/processed/dvf_clean.csv (5 340 lignes)
```

**Filtrage supplémentaire à l'affichage** : prix_au_m2 entre 500 et 20 000 €/m² pour éliminer les valeurs aberrantes résiduelles.

### Annonces SeLoger (`analysis/nettoyage.py`)

```
Fichier brut SeLoger (CSV, colonnes brutes avec unités texte)
    ↓
Conversion prix_du_bien, prix_m2, superficie → float
    ↓
Normalisation des noms de colonnes
    ↓
Suppression doublons et valeurs nulles critiques
    ↓
Export : donnees/processed/annonces_propres.csv (707 lignes)
```

---

## 3. Représentation et combinaison des datasets

### Pourquoi on ne fusionne pas les deux tables

DVF et SeLoger sont **complémentaires mais non joignables directement** :

| Critère | DVF | SeLoger |
|---------|-----|---------|
| Quartier nommé | ❌ (GPS seulement) | ✅ |
| Coordonnées GPS | ✅ | ❌ |
| Prix payé | ✅ | ❌ |
| Prix demandé | ❌ | ✅ |
| DPE | ❌ | ✅ |
| Période | 2024 (vendus) | Actuel (en vente) |

### Stratégie : analyse parallèle avec comparaison croisée

```
┌────────────────────────────┐    ┌────────────────────────────┐
│  DVF — Transactions réelles│    │  SeLoger — Annonces        │
│  5 340 ventes              │    │  707 biens en vente        │
│  ─────────────────────     │    │  ─────────────────────     │
│  • Prix du marché effectif │    │  • Prix demandés           │
│  • Évolution temporelle    │    │  • Répartition quartiers   │
│  • Localisation GPS        │    │  • DPE et caractéristiques │
└────────────┬───────────────┘    └──────────────┬─────────────┘
             │                                   │
             └──────────────┬────────────────────┘
                            ↓
              Comparaison par type de bien
              (Appartement / Maison)

              Écart DVF ↔ SeLoger = tension du marché
              (marge de négociation typique)
```

### Point de jonction : type de bien

La seule variable commune fiable est `type_local` / `type` (Appartement ou Maison).
On peut comparer les distributions de prix/m² par type entre les deux sources :

```python
# DVF — prix réels payés
dvf[dvf["type_local"] == "Appartement"]["prix_au_m2"].median()  # → 3 387 €/m²

# SeLoger — prix demandés
ann[ann["type"] == "Appartement"]["prix_m2"].median()            # → ~3 650 €/m²
```

L'écart révèle une **prime de 7-10% entre prix affiché et prix payé**, typique d'un marché avec négociation modérée.

---

## 4. Statistiques descriptives from scratch

Implémentées dans `analysis/stats.py` — **Python pur, sans numpy/pandas/statistics**.
Référence : Joel Grus, *Data Science From Scratch*, chapitre 5.

### Formules

**Moyenne** :
```
mean(xs) = Σxᵢ / n
```

**Variance** (biaisée, diviseur n) :
```
variance(xs) = Σ(xᵢ - mean)² / n
```
> Note : on utilise n (variance biaisée) car le test CI vérifie `variance([2,4,4,4,5,5,7,9]) == 4.0`,
> qui est exact uniquement avec n=8.

**Écart-type** :
```
standard_deviation(xs) = √variance(xs)
```

**Covariance** (biaisée) :
```
covariance(xs, ys) = Σ(xᵢ - mean_x)(yᵢ - mean_y) / n
```

**Corrélation de Pearson** :
```
correlation(xs, ys) = covariance(xs, ys) / (std_x × std_y)
```
Valeurs entre -1 et +1. Retourne 0 si l'un des écarts-types est nul.

### Interprétation sur le marché toulonnais

| Statistique | DVF Prix/m² | SeLoger Prix/m² |
|-------------|-------------|-----------------|
| Moyenne | ~3 900 €/m² | ~4 025 €/m² |
| Médiane | ~3 387 €/m² | ~3 650 €/m² |
| Écart-type | ~2 500 €/m² | ~2 017 €/m² |
| Q1 (25%) | ~2 345 €/m² | ~2 813 €/m² |
| Q3 (75%) | ~5 850 €/m² | ~4 851 €/m² |

**Asymétrie** : la moyenne > médiane dans les deux cas → distribution asymétrique vers le haut
(quelques biens très chers dans les quartiers premium tirent la moyenne).

---

## 5. Régression linéaire & R²

Implémentée dans `analysis/regression.py` — **sans sklearn**.
Référence : Joel Grus, *Data Science From Scratch*, chapitre 14.

### Modèle

```
ŷ = α + β × x
```

### Coefficients (moindres carrés ordinaires)

```
β = Cov(x, y) / Var(x)
α = mean(y) - β × mean(x)
```

Ces formules **minimisent** la somme des erreurs au carré (SS_res = Σ(yᵢ - ŷᵢ)²).

### Coefficient de détermination R²

```
R² = 1 - SS_res / SS_tot

SS_res = Σ(ŷᵢ - yᵢ)²     ← erreurs du modèle
SS_tot = Σ(yᵢ - ȳ)²      ← variance totale de y
```

### Interprétation du R²

| Valeur R² | Interprétation |
|-----------|----------------|
| **1.0** | Ajustement parfait — le modèle prédit exactement chaque prix |
| **0.7** | La surface explique 70% de la variation de prix |
| **0.5** | La surface explique 50% — les 50% restants viennent d'autres facteurs |
| **0.0** | La surface n'explique rien du prix |

### Lien R² ↔ corrélation r

Pour une régression simple (une seule variable) :
```
R² = r²    donc    r = √R²
```

### Résultats observés sur ce marché

Sur les annonces SeLoger (surface → prix) :
- **β ≈ 2 500-3 000 €/m²** → chaque m² supplémentaire ajoute ~2 500-3 000 € au prix
- **R² ≈ 0.45-0.55** → la surface explique 45-55% de la variation de prix
- **r ≈ 0.67-0.74** → corrélation modérée à forte

> Un R² de 0.5 est **réaliste et attendu** sur l'immobilier — la localisation (quartier),
> l'état du bien, le DPE, l'étage, l'exposition contribuent tous au prix restant.

---

## 6. Analyse exploratoire — graphiques

Script : `analysis/exploration.py`
Sortie : `analysis/figures/`

| Fichier | Description |
|---------|-------------|
| `01_distribution_prix_m2.png` | Histogrammes DVF vs SeLoger avec moyenne et médiane |
| `02_boxplot_type_de_bien.png` | Boxplots prix/m² par type (appartement/maison) |
| `03_evolution_temporelle_dvf.png` | Volume mensuel et prix médian mensuel 2024 |
| `04_prix_par_quartier_seloger.png` | Barplot horizontal des prix par quartier |
| `05_scatter_prix_surface.png` | Nuages de points prix~surface avec droite de régression |
| `06_heatmap_correlation_dvf.png` | Matrice de corrélation DVF |
| `07_heatmap_correlation_seloger.png` | Matrice de corrélation SeLoger |
| `08_dpe_analyse.png` | Répartition DPE et impact sur le prix/m² |
| `09_comparaison_dvf_vs_seloger.png` | Boxplots croisés DVF vs SeLoger par type |
| `10_carte_gps_dvf.png` | Carte géographique des transactions colorée par prix/m² |
| `11_regression_lineaire_r2.png` | Régression avec formule, R² et distribution des résidus |
| `12_surface_par_pieces_dvf.png` | Distribution des surfaces par typologie T1→T6 |

**Pour générer tous les graphiques** :
```bash
python3 analysis/exploration.py
```

---

## 7. Scoring des opportunités

Le score d'opportunité est basé sur le **z-score** du prix/m² par quartier :

```
z = (prix_m2_quartier - μ_global) / σ_global
```

| Signal | Condition | Interprétation |
|--------|-----------|----------------|
| 🟢 Sous-coté | z < −0.5 | Quartier en dessous du marché → potentiel de plus-value |
| 🟡 Marché | −0.5 ≤ z ≤ 0.5 | Prix aligné avec la moyenne du marché |
| 🔴 Sur-coté | z > 0.5 | Quartier au-dessus → peu de marge de négociation |

> **Limite** : le scoring est purement quantitatif. La qualité de vie, la sécurité,
> les transports en commun et la dynamique du quartier peuvent justifier des prix
> structurellement élevés ou bas indépendamment du marché global.

---

## Références

- Joel Grus, *Data Science From Scratch*, O'Reilly — ch.5 (statistiques), ch.14 (régression)
- [DVF geo data.gouv.fr](https://files.data.gouv.fr/geo-dvf/latest/csv/83/)
- [SeLoger.com](https://www.seloger.com) — scraping Python
- INSEE code commune 83137 — Toulon (Var, PACA)
