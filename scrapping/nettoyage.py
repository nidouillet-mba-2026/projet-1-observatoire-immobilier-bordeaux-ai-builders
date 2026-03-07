import pandas as pd
import re
import os

def clean_text(text):
    if pd.isna(text): return ""
    # Nettoyage des espaces insécables et bizarres de SeLoger
    t = str(text).replace('\u202f', '').replace('\xa0', '').replace(' ', '')
    return t.lower()

def extraire_nombre(texte, type_donnee="prix"):
    t = clean_text(texte)
    chiffres = re.findall(r'\d+', t)
    if not chiffres: return None
    
    if type_donnee == "prix":
        # On prend le nombre le plus long pour éviter les faux positifs
        n = int(max(chiffres, key=len))
        return n if n > 5000 else None
    else:
        # Pour la surface ou pièces, on prend souvent le premier chiffre trouvé
        return int(chiffres[0])

def analyser_ligne(row):
    # Initialisation des variables
    prix, surface, pieces, type_bien, quartier, prix_m2 = [None] * 6
    
    for cellule in row:
        val_str = str(cellule).lower()
        
        # 1. TYPE DE BIEN
        if 'appartement' in val_str: type_bien = 'Appartement'
        elif 'maison' in val_str or 'villa' in val_str: type_bien = 'Maison'

        # 2. PRIX TOTAL
        if '€' in val_str and '/m²' not in val_str and 'm2' not in val_str:
            p = extraire_nombre(val_str, "prix")
            if p: prix = p
            
        # 3. PRIX AU M2
        if '€/m²' in val_str or '€/m2' in val_str:
            pm2 = extraire_nombre(val_str, "prix")
            if pm2: prix_m2 = pm2

        # 4. SURFACE
        if 'm²' in val_str or 'm2' in val_str:
            if '€' not in val_str: # On évite la confusion avec le prix au m2
                s = extraire_nombre(val_str, "surface")
                if s and s < 1000: surface = s
        
        # 5. PIÈCES
        if 'pièce' in val_str:
            res = re.findall(r'\d+', val_str)
            if res: pieces = int(res[0])

        # 6. QUARTIER
        # Logique : SeLoger met souvent "Nom du quartier, Toulon (83...)"
        if 'toulon' in val_str and ',' in val_str:
            partie_quartier = val_str.split(',')[0].strip()
            # On ignore si c'est juste "Toulon"
            if partie_quartier.lower() != 'toulon':
                quartier = partie_quartier.title()

    return pd.Series([prix, surface, prix_m2, pieces, type_bien, quartier])

def clean_data():
    # Chemins
    chemin_script = os.path.dirname(os.path.abspath(__file__))
    racine_projet = os.path.dirname(chemin_script)
    dossier_csv = os.path.join(racine_projet, "CSV")
    dossier_data = os.path.join(racine_projet, "data")
    
    fichiers = ["seloger.csv", "seloger1.csv","seloger2.csv", "seloger3.csv","seloger.csv4","seloger5.csv","seloger6.csv","seloger7.csv","seloger8.csv","seloger9.csv"]
    li = []
    
    for f in fichiers:
        chemin = os.path.join(dossier_csv, f)
        if os.path.exists(chemin):
            print(f"✅ Lecture de : {chemin}")
            li.append(pd.read_csv(chemin))

    if not li:
        print("❌ Aucun fichier trouvé.")
        return

    df_raw = pd.concat(li, ignore_index=True)
    
    print("Analyse et extraction des colonnes en cours...")
    
    # Application de l'analyse sur chaque ligne
    new_cols = ['prix', 'surface', 'prix_m2', 'nb_pieces', 'type', 'quartier']
    df_raw[new_cols] = df_raw.apply(analyser_ligne, axis=1)

    # Nettoyage final : on garde les colonnes utiles et on vire les lignes sans prix
    df_final = df_raw[new_cols].copy()
    df_final['ville'] = "Toulon"
    
    # Suppression des lignes où le prix ou la surface manquent
    df_final = df_final.dropna(subset=['prix', 'surface'])
    
    # Suppression des doublons (si une annonce est sur seloger.csv et seloger1.csv)
    df_final = df_final.drop_duplicates()

    # Sauvegarde
    if not os.path.exists(dossier_data): os.makedirs(dossier_data)
    chemin_sortie = os.path.join(dossier_data, "annonces_propres.csv")
    df_final.to_csv(chemin_sortie, index=False)
    
    print("-" * 30)
    print(f"✨ NETTOYAGE TERMINÉ ✨")
    print(f"📊 Annonces extraites : {len(df_final)}")
    print(f"📍 Fichier créé : {chemin_sortie}")
    print("-" * 30)
    print("Aperçu des données :")
    print(df_final.head())

if __name__ == "__main__":
    clean_data()