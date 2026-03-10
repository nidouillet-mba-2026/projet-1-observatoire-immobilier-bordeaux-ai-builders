# Observatoire du Marché Immobilier Toulonnais

Projet NidDouillet — Epitech IA Spé
Analyse du marché immobilier de Toulon (INSEE 83137) combinant transactions DVF et annonces SeLoger, avec algorithmes statistiques et régression linéaire implémentés from scratch.

**Application déployée :** https://observatoire-immo-toulon.streamlit.app

---

## Architecture du projet

```
.
├── analysis/
│   ├── stats.py              Statistiques from scratch (moyenne, variance, corrélation)
│   ├── regression.py         Régression linéaire from scratch (moindres carrés, R²)
│   ├── cleaning_dvf.py       Pipeline de nettoyage des données DVF
│   ├── nettoyage.py          Pipeline de nettoyage des annonces SeLoger
│   ├── exploration.py        Génération des graphiques d'analyse exploratoire
│   └── figures/              12 graphiques PNG générés
├── app/
│   └── streamlit_app.py      Dashboard principal (5 onglets)
├── data/
│   ├── dvf_toulon.csv        Données DVF nettoyées (>= 500 transactions)
│   └── annonces.csv          Annonces SeLoger nettoyées
├── donnees/
│   ├── raw/
│   │   ├── datagouv_83137_20242025.csv    Fichier DVF brut (data.gouv.fr)
│   │   └── seloger_document_base.csv      Fichier SeLoger brut (scraping)
│   └── processed/
│       ├── dvf_clean.csv                  DVF nettoyé (5 340 transactions)
│       └── annonces_propres.csv           Annonces nettoyées (707 lignes)
├── tests/
│   ├── test_stats.py         Tests unitaires stats.py
│   ├── test_regression.py    Tests unitaires regression.py
│   └── test_auto_eval.py     Tests d'évaluation CI (ne pas modifier)
├── requirements.txt
├── METHODOLOGIE.md
└── README.md
```

---

## Sources de données

### DVF — Demandes de Valeurs Foncières

Source officielle : [data.gouv.fr](https://files.data.gouv.fr/geo-dvf/latest/csv/83/)

Contient toutes les transactions immobilières réellement enregistrées aux impôts. Ce sont des prix **effectivement payés**, pas des prix demandés.

| Attribut | Valeur |
|----------|--------|
| Période | Fév 2024 → Déc 2024 |
| Commune | Toulon (code INSEE 83137) |
| Transactions après nettoyage | 5 340 |
| Format brut | CSV, séparateur `;`, 40 colonnes |

Colonnes conservées après nettoyage (13 sur 40) :

| Colonne | Type | Description |
|---------|------|-------------|
| `id_mutation` | string | Identifiant unique de la transaction |
| `date_mutation` | date | Date de la vente |
| `nature_mutation` | string | Type d'opération |
| `valeur_fonciere` | float | Prix réellement payé en euros |
| `code_postal` | int | Code postal (83000 / 83100 / 83200) |
| `nom_commune` | string | Commune |
| `type_local` | string | Appartement ou Maison |
| `surface_reelle_bati` | float | Surface habitable en m² |
| `nombre_pieces_principales` | int | Nombre de pièces |
| `surface_terrain` | float | Surface terrain |
| `longitude` | float | Coordonnée GPS |
| `latitude` | float | Coordonnée GPS |
| `prix_au_m2` | float | Calculé : valeur_fonciere / surface_reelle_bati |

### Annonces SeLoger

Source : SeLoger.com via scraping Python

Contient des biens **en cours de vente** — prix demandés, pas encore vendus.

| Attribut | Valeur |
|----------|--------|
| Annonces | 707 |
| Quartiers | 42 quartiers toulonnais |

Colonnes disponibles (8) :

| Colonne | Type | Description |
|---------|------|-------------|
| `prix` | float | Prix demandé en euros |
| `surface` | float | Surface en m² |
| `prix_m2` | float | Prix/m² calculé |
| `nb_pieces` | float | Nombre de pièces |
| `type` | string | Appartement ou Maison |
| `quartier` | string | Nom du quartier |
| `energie` | string | Classe DPE (A à G) |
| `ville` | string | Toulon |

### Pourquoi les deux datasets ne sont pas fusionnés

DVF et SeLoger sont complémentaires mais non joignables directement :

| Critère | DVF | SeLoger |
|---------|-----|---------|
| Quartier nommé | Non (GPS seulement) | Oui |
| Coordonnées GPS | Oui | Non |
| Prix payé | Oui | Non |
| Prix demandé | Non | Oui |
| DPE | Non | Oui |

La stratégie adoptée est une **analyse parallèle** : chaque source est analysée indépendamment, et l'écart entre prix DVF et prix SeLoger mesure la tension du marché (marge de négociation typique : 7-10%).

---

## Pipelines de nettoyage

### DVF — `analysis/cleaning_dvf.py`

```
Fichier brut (~8 000 lignes, séparateur ";")
    → Suppression lignes vides
    → Suppression si prix <= 0 ou absent
    → Suppression si surface_reelle_bati <= 0 ou absente
    → Calcul prix_au_m2 = valeur_fonciere / surface_reelle_bati
    → Conservation des 13 colonnes utiles
    → Tri chronologique
    → Export : donnees/processed/dvf_clean.csv (5 340 lignes)
```

Filtrage supplémentaire à l'affichage dans le dashboard : prix_au_m2 entre 500 et 20 000 €/m².

### Annonces SeLoger — `analysis/nettoyage.py`

```
Fichier brut (colonnes avec unités texte : "350 000 €", "65 m²")
    → Conversion prix, surface, prix_m2 en float (regex)
    → Normalisation des noms de quartiers
    → Suppression doublons et lignes sans prix/surface/DPE/quartier
    → Export : donnees/processed/annonces_propres.csv (707 lignes)
```

---

## Algorithmes from scratch

### Statistiques — `analysis/stats.py`

Implémentées en Python pur, sans numpy/pandas/statistics.
Référence : Joel Grus, *Data Science From Scratch*, chapitre 5.

| Fonction | Formule |
|----------|---------|
| `mean(xs)` | `sum(xs) / n` |
| `median(xs)` | valeur centrale après tri |
| `variance(xs)` | `sum((x - mean)^2 for x in xs) / n` |
| `standard_deviation(xs)` | `sqrt(variance(xs))` |
| `covariance(xs, ys)` | `sum((xi - mx)(yi - my) for i) / n` |
| `correlation(xs, ys)` | `covariance(xs, ys) / (std_x * std_y)` |

La variance utilise le diviseur `n` (biaisée). Ce choix est requis par le CI : `variance([2,4,4,4,5,5,7,9])` doit retourner `4.0`.

### Régression linéaire — `analysis/regression.py`

Implémentée sans sklearn.
Référence : Joel Grus, *Data Science From Scratch*, chapitre 14.

Modèle : `y_hat = alpha + beta * x`

| Fonction | Description |
|----------|-------------|
| `least_squares_fit(x, y)` | Calcule alpha et beta par moindres carrés |
| `predict(alpha, beta, x_i)` | Prédit y pour une valeur x |
| `error(alpha, beta, x_i, y_i)` | Erreur de prédiction pour un point |
| `sum_of_sqerrors(alpha, beta, x, y)` | Somme des erreurs au carré |
| `r_squared(alpha, beta, x, y)` | Coefficient de détermination R² |

Formules des coefficients (moindres carrés ordinaires) :

```
beta  = Cov(x, y) / Var(x)
alpha = mean(y) - beta * mean(x)

R² = 1 - SS_res / SS_tot
   = 1 - sum((y_hat_i - y_i)^2) / sum((y_i - y_mean)^2)
```

Résultats observés sur ce marché (surface → prix) :

| Indicateur | Valeur |
|------------|--------|
| beta | ~2 500 à 3 000 €/m² |
| R² | ~0.45 à 0.55 |
| r (Pearson) | ~0.67 à 0.74 |

---

## Dashboard Streamlit — `app/streamlit_app.py`

Lancé avec `streamlit run app/streamlit_app.py`, le dashboard comporte 5 onglets.

**Sidebar** : filtres globaux (type de bien, fourchette prix/m²) appliqués à l'ensemble des onglets.

### Onglet 1 — Tableau de bord

- KPIs DVF : nombre de transactions, prix/m² moyen, médian, écart-type, variance (calculés via `stats.py`)
- KPIs SeLoger : nombre d'annonces, prix/m² moyen et médian, nombre de quartiers
- Histogramme de distribution des prix/m² DVF
- Écart moyen SeLoger vs DVF (indicateur de tension du marché)
- Volume mensuel de transactions DVF avec évolution du prix médian (double axe)

### Onglet 2 — Carte et Quartiers

- Carte GPS des transactions DVF colorée par prix/m² (vert = bon marché, rouge = cher)
- Heatmap des prix/m² par quartier SeLoger avec filtre sur le nombre minimum d'annonces
- Classement complet des quartiers par prix moyen avec tableau interactif

### Onglet 3 — Régression et Corrélations

- Régression interactive : choix de la source (DVF ou SeLoger) et de la variable explicative (surface ou nombre de pièces)
- Métriques affichées : alpha, beta, R², corrélation r (via `regression.py` et `stats.py`)
- Graphique scatter avec droite de régression
- Distribution des résidus (prix réel − prix prédit)
- Matrices de corrélation DVF et SeLoger
- Analyse de l'impact du DPE sur le prix/m²

### Onglet 4 — Opportunités

- Scoring par z-score : `z = (prix_m2_quartier - mu_global) / sigma_global`
- Classification des quartiers : sous-coté (z < −0.5), dans le marché, sur-coté (z > 0.5)
- Filtres sur le budget maximum et la surface minimale
- Tableau des annonces correspondantes avec leur signal d'opportunité

### Onglet 5 — Méthodologie

- Documentation inline : sources, pipelines, formules statistiques, formules de régression

---

## Analyse exploratoire — `analysis/exploration.py`

Génère 12 graphiques dans `analysis/figures/` :

| Fichier | Contenu |
|---------|---------|
| `01_distribution_prix_m2.png` | Histogrammes DVF vs SeLoger |
| `02_boxplot_type_de_bien.png` | Prix/m² par type de bien |
| `03_evolution_temporelle_dvf.png` | Volume et prix médian mensuel 2024 |
| `04_prix_par_quartier_seloger.png` | Classement des quartiers |
| `05_scatter_prix_surface.png` | Nuages de points avec droite de régression |
| `06_heatmap_correlation_dvf.png` | Matrice de corrélation DVF |
| `07_heatmap_correlation_seloger.png` | Matrice de corrélation SeLoger |
| `08_dpe_analyse.png` | Répartition DPE et impact sur le prix |
| `09_comparaison_dvf_vs_seloger.png` | Boxplots croisés DVF vs SeLoger |
| `10_carte_gps_dvf.png` | Carte géographique des transactions |
| `11_regression_lineaire_r2.png` | Régression avec R² et résidus |
| `12_surface_par_pieces_dvf.png` | Distribution des surfaces par typologie |

```bash
python3 analysis/exploration.py
```

---

## Tests

```
tests/
├── test_stats.py         12 tests : mean, median, variance, std, covariance, correlation
├── test_regression.py     7 tests : predict, error, least_squares_fit, r_squared, sum_of_sqerrors
└── test_auto_eval.py     Évaluation automatique CI (ne pas modifier)
```

Le CI vérifie également que `stats.py` n'importe pas numpy et que `regression.py` n'importe pas sklearn.

```bash
pytest tests/
```

---

## Installation et lancement

```bash
git clone <url-du-repo>
cd <repo>
pip install -r requirements.txt
```

Lancement du dashboard :

```bash
streamlit run app/streamlit_app.py
```

Génération des graphiques d'analyse :

```bash
python3 analysis/exploration.py
```

Nettoyage des données brutes :

```bash
python3 analysis/cleaning_dvf.py   # DVF
python3 analysis/nettoyage.py      # SeLoger
```

---

## Évaluation CI

A chaque `git push`, le CI exécute `tests/test_auto_eval.py` et publie le score dans l'onglet **Actions > Job Summary**.

| Critère | Points |
|---------|--------|
| Statistiques from scratch (`stats.py`) | 15 |
| Régression from scratch (`regression.py`) | 10 |
| Données réelles (DVF >= 500 lignes, annonces présentes) | 15 |
| Application Streamlit déployée avec URL dans le README | 10 |
| Tests unitaires étudiants (>= 3 tests) | 5 |
| **Total automatisé** | **55** |
| Soutenance | 45 |
| **Total** | **100** |

---

## Équipe

| Membre | Rôle |
|--------|------|
| Maxime Ribes | Data Engineer, Frontend |
| Matthieu Barric | Data Engineer, AI Engineer |
| Lena Pedelahore | Data Scientist |
| Alexis De Malet | AI Engineer |
| Lilou Legenvre | Frontend, DevOps |

---

## Références

- Joel Grus, *Data Science From Scratch*, O'Reilly — ch.5 (statistiques), ch.14 (régression)
- DVF : https://files.data.gouv.fr/geo-dvf/latest/csv/83/
- INSEE code commune 83137 — Toulon (Var, PACA)
