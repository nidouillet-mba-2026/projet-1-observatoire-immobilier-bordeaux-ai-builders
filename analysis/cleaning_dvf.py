import csv
import os

# Fichier CSV brut issu de data.gouv
INPUT_FILE = "donnees/raw/datagouv_83137_20242025.csv"

# Fichier CSV nettoyé généré par le script
OUTPUT_FILE = "donnees/processed/dvf_clean.csv"


# Conversion des valeurs

def to_float(value):
    """
    Conversion d'une valeur texte en nombre décimal.
    Retourne None si la valeur est vide ou invalide.
    """

    if value is None:
        return None

    value = value.strip()

    if value == "":
        return None

    # Remplacement des virgules par des points
    value = value.replace(",", ".")

    try:
        return float(value)
    except ValueError:
        return None


def to_int(value):
    """
    Conversion d'une valeur texte en entier.
    """

    if value is None:
        return None

    value = value.strip()

    if value == "":
        return None

    try:
        return int(float(value))
    except ValueError:
        return None


# Vérification des lignes vides


def is_empty_row(row):

    for value in row.values():
        if value is not None and value.strip() != "":
            return False

    return True

# Nettoyage d'une ligne de données


def clean_row(row):

    # Suppression des lignes totalement vides
    if is_empty_row(row):
        return None

    # Récupération des variables texte
    id_mutation = row.get("id_mutation", "").strip()
    date_mutation = row.get("date_mutation", "").strip()
    nature_mutation = row.get("nature_mutation", "").strip()
    code_postal = row.get("code_postal", "").strip()
    nom_commune = row.get("nom_commune", "").strip()
    type_local = row.get("type_local", "").strip()

    # Conversion des variables numériques
    prix = to_float(row.get("valeur_fonciere", ""))
    surface_bati = to_float(row.get("surface_reelle_bati", ""))
    nb_pieces = to_int(row.get("nombre_pieces_principales", ""))
    surface_terrain = to_float(row.get("surface_terrain", ""))
    longitude = to_float(row.get("longitude", ""))
    latitude = to_float(row.get("latitude", ""))

    # Suppression des lignes sans prix exploitable
    if prix is None or prix <= 0:
        return None

    # Suppression des lignes sans surface bâtie exploitable
    if surface_bati is None or surface_bati <= 0:
        return None

    # Création de la variable prix au mètre carré
    prix_au_m2 = prix / surface_bati

    # Construction de la ligne nettoyée avec les colonnes utiles
    cleaned = {
        "id_mutation": id_mutation,
        "date_mutation": date_mutation,
        "nature_mutation": nature_mutation.title(),
        "valeur_fonciere": round(prix, 2),
        "code_postal": code_postal,
        "nom_commune": nom_commune.title(),
        "type_local": type_local.title(),
        "surface_reelle_bati": round(surface_bati, 2),
        "nombre_pieces_principales": nb_pieces,
        "surface_terrain": round(surface_terrain, 2) if surface_terrain is not None else None,
        "longitude": longitude,
        "latitude": latitude,
        "prix_au_m2": round(prix_au_m2, 2)
    }

    return cleaned

# Chargement et nettoyage des données

def load_and_clean_data(input_file):

    cleaned_rows = []
    total_rows = 0
    empty_rows = 0
    invalid_price_rows = 0
    invalid_surface_rows = 0

    with open(input_file, "r", encoding="utf-8") as f:
        # Le fichier DVF utilise le séparateur ";"
        reader = csv.DictReader(f, delimiter=";")

        for row in reader:
            total_rows += 1

            if is_empty_row(row):
                empty_rows += 1
                continue

            prix = to_float(row.get("valeur_fonciere", ""))
            if prix is None or prix <= 0:
                invalid_price_rows += 1
                continue

            surface_bati = to_float(row.get("surface_reelle_bati", ""))
            if surface_bati is None or surface_bati <= 0:
                invalid_surface_rows += 1
                continue

            cleaned = clean_row(row)

            if cleaned is not None:
                cleaned_rows.append(cleaned)

    print("Nombre de lignes lues :", total_rows)
    print("Lignes vides supprimées :", empty_rows)
    print("Lignes supprimées car prix invalide :", invalid_price_rows)
    print("Lignes supprimées car surface invalide :", invalid_surface_rows)
    print("Nombre de lignes conservées :", len(cleaned_rows))

    return cleaned_rows


# Sauvegarde du fichier nettoyé


def save_clean_csv(rows, output_file):

    if not rows:
        print("Aucune donnée à sauvegarder.")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Tri des lignes pour faciliter la lecture du fichier final
    rows = sorted(rows, key=lambda row: (row["date_mutation"], row["valeur_fonciere"]))

    fieldnames = list(rows[0].keys())

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Fichier nettoyé enregistré :", output_file)
    print("Nombre de lignes exportées :", len(rows))


# Exécution du pipeline de nettoyage

def main():
    print("Nettoyage des données DVF en cours...")

    rows = load_and_clean_data(INPUT_FILE)

    save_clean_csv(rows, OUTPUT_FILE)

    print("Nettoyage terminé.")


if __name__ == "__main__":
    main()