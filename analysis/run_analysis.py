import csv
from stats import mean, median, variance, standard_deviation, covariance, correlation
from regression import least_squares_fit, predict, r_squared


# Fonctions utilitaires

def to_float(value):
    """
    Convertit une valeur texte en float.
    Retourne None si la valeur est vide ou invalide.
    """

    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    value = value.replace(",", ".")

    try:
        return float(value)
    except ValueError:
        return None


def load_numeric_column(file_path: str, column_name: str) -> list[float]:
    """
    Charge une colonne numerique depuis un CSV.
    Ignore les valeurs vides ou invalides.
    """

    values = []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            value = to_float(row.get(column_name))

            if value is not None:
                values.append(value)

    return values


def load_two_numeric_columns(file_path: str, x_column: str, y_column: str) -> tuple[list[float], list[float]]:
    """
    Charge deux colonnes numeriques d'un CSV en garantissant
    que chaque paire x / y provient de la meme ligne valide.
    """

    xs = []
    ys = []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            x_value = to_float(row.get(x_column))
            y_value = to_float(row.get(y_column))

            if x_value is not None and y_value is not None:
                xs.append(x_value)
                ys.append(y_value)

    return xs, ys


# Analyse statistique d'un dataset

def describe_column(file_path: str, label: str, column_name: str):
    """
    Calcule les statistiques descriptives d'une colonne.
    """

    values = load_numeric_column(file_path, column_name)

    print(f"\n--- {label} ({column_name}) ---")
    print("Nombre de valeurs :", len(values))
    print("Moyenne :", round(mean(values), 2))
    print("Mediane :", round(median(values), 2))
    print("Variance :", round(variance(values), 2))
    print("Ecart-type :", round(standard_deviation(values), 2))


def analyse_relation(file_path: str, label: str, x_column: str, y_column: str):
    """
    Calcule covariance, correlation et regression lineaire
    entre deux variables numeriques.
    """

    xs, ys = load_two_numeric_columns(file_path, x_column, y_column)

    print(f"\n=== Relation {label} : {x_column} / {y_column} ===")
    print("Nombre d'observations :", len(xs))
    print("Covariance :", round(covariance(xs, ys), 2))
    print("Correlation :", round(correlation(xs, ys), 4))

    alpha, beta = least_squares_fit(xs, ys)
    r2 = r_squared(xs, ys, alpha, beta)

    print("Modele lineaire : y = beta * x + alpha")
    print("alpha :", round(alpha, 2))
    print("beta :", round(beta, 2))
    print("R² :", round(r2, 4))

    # Exemple de prediction sur la premiere valeur de x
    example_x = xs[0]
    example_pred = predict(example_x, alpha, beta)

    print("Exemple prediction :")
    print(f"Pour x = {round(example_x, 2)}, y predit = {round(example_pred, 2)}")


# Analyse DVF

def analyse_dvf():
    """
    Analyse statistique du fichier DVF nettoye.
    """

    file_path = "donnees/processed/dvf_clean.csv"

    print("ANALYSE DU FICHIER DVF")

    describe_column(file_path, "DVF", "valeur_fonciere")
    describe_column(file_path, "DVF", "surface_reelle_bati")
    describe_column(file_path, "DVF", "prix_au_m2")

    analyse_relation(file_path, "DVF", "surface_reelle_bati", "valeur_fonciere")


# Analyse annonces

def analyse_annonces():
    """
    Analyse statistique du fichier d'annonces scrapees.
    Adapter les noms de colonnes si besoin.
    """

    file_path = "donnees/processed/annonces_propres.csv"

    print("ANALYSE DU FICHIER ANNONCES")

    describe_column(file_path, "ANNONCES", "prix")
    describe_column(file_path, "ANNONCES", "surface")
    describe_column(file_path, "ANNONCES", "prix_m2")

    analyse_relation(file_path, "ANNONCES", "surface", "prix")


# Lancement du script


def main():
    analyse_dvf()
    analyse_annonces()


if __name__ == "__main__":
    main()