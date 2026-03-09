import pandas as pd
import re
import os

def clean_val(text):
    if pd.isna(text) or str(text).strip() == "": return None
    # On enlève les espaces bizarres, les unités et les guillemets
    t = str(text).replace('\u202f', '').replace('\xa0', '').replace(' ', '')
    t = t.replace('€', '').replace('m²', '').replace('m2', '').replace('/m²', '')
    t = t.replace('pièces', '').replace('pièce', '').replace('chambres', '').replace('chambre', '')
    return t.strip().lower()

def extraire_num(texte):
    t = clean_val(texte)
    if not t: return None
    # Remplacement de la virgule par un point pour les surfaces (ex: 204,1 -> 204.1)
    t = t.replace(',', '.')
    res = re.findall(r"[-+]?\d*\.\d+|\d+", t)
    if res:
        return float(res[0])
    return None

def clean_data():
    # Gestion des chemins
    chemin_script = os.path.dirname(os.path.abspath(__file__))
    racine_projet = os.path.dirname(chemin_script)
    dossier_csv = os.path.join(racine_projet, "CSV")
    dossier_data = os.path.join(racine_projet, "data")
    
    # Lecture du fichier source
    chemin_input = os.path.join(dossier_csv, "seloger_document_base.csv")
    
    if not os.path.exists(chemin_input):
        print(f"❌ Fichier introuvable : {chemin_input}")
        return

    print(f"✅ Lecture de : {chemin_input}")
    df = pd.read_csv(chemin_input)

    # Création du nouveau DataFrame propre
    df_clean = pd.DataFrame()

    print("Nettoyage strict en cours (suppression des lignes sans DPE)...")

    # 1. PRIX
    df_clean['prix'] = df['prix_du_bien'].apply(extraire_num)
    
    # 2. SURFACE
    df_clean['surface'] = df['superficie'].apply(extraire_num)
    
    # 3. PRIX AU M2
    df_clean['prix_m2'] = df['prix_m2'].apply(extraire_num)
    
    # 4. NB PIECES
    df_clean['nb_pieces'] = df['nb_pièce'].apply(extraire_num)

    # 5. TYPE DE BIEN
    df_clean['type'] = df['type_de_bien'].str.contains('Maison|Villa', case=False, na=False).map({True: 'Maison', False: 'Appartement'})

    # 6. QUARTIER
    def clean_quartier(q):
        if pd.isna(q) or str(q).strip() == "": return None
        q_clean = str(q).split(',')[0].split('(')[0].strip()
        return q_clean.title() if q_clean.lower() != "toulon" else "Centre-Ville"

    df_clean['quartier'] = df['quartier'].apply(clean_quartier)
    
    # 7. ENERGIE (DPE) - On garde tel quel, sans valeur par défaut
    df_clean['energie'] = df['Cat_energie'].apply(lambda x: str(x).strip().upper() if pd.notna(x) and str(x).strip() != "" else None)

    # 8. VILLE
    df_clean['ville'] = "Toulon"

    # --- FILTRAGE STRICT ---
    # On ne garde que les lignes qui ont TOUTES ces informations
    colonnes_obligatoires = ['prix', 'surface', 'energie', 'quartier']
    df_clean = df_clean.dropna(subset=colonnes_obligatoires)
    
    # Suppression des doublons
    df_clean = df_clean.drop_duplicates()

    # Sauvegarde
    if not os.path.exists(dossier_data):
        os.makedirs(dossier_data)
    
    chemin_sortie = os.path.join(dossier_data, "annonces_propres.csv")
    df_clean.to_csv(chemin_sortie, index=False)
    
    print("-" * 30)
    print(f"✨ NETTOYAGE TERMINÉ ✨")
    print(f"📊 {len(df_clean)} annonces conservées (avec DPE et quartier).")
    print(f"📍 Fichier créé : {chemin_sortie}")
    print("-" * 30)
    print(df_clean.head())

if __name__ == "__main__":
    clean_data()