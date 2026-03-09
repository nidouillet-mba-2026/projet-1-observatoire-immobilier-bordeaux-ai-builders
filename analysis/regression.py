"""
Regression lineaire simple from scratch.
Reference : Joel Grus, "Data Science From Scratch", chapitre 14.

IMPORTANT : N'importez pas sklearn, numpy ou scipy pour ces fonctions.
"""

from stats import mean


def predict(x: float, alpha: float, beta: float) -> float:
    """Prediction du modele lineaire."""
    return beta * x + alpha


def least_squares_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """
    Calcule les coefficients de regression lineaire.
    Retourne (alpha, beta)
    """

    mean_x = mean(xs)
    mean_y = mean(ys)

    numerator = 0
    denominator = 0

    for x, y in zip(xs, ys):
        numerator += (x - mean_x) * (y - mean_y)
        denominator += (x - mean_x) ** 2

    beta = numerator / denominator
    alpha = mean_y - beta * mean_x

    return alpha, beta


def sum_of_sqerrors(xs: list[float], ys: list[float], alpha: float, beta: float) -> float:
    """Somme des erreurs au carré."""

    total = 0

    for x, y in zip(xs, ys):
        predicted = predict(x, alpha, beta)
        total += (y - predicted) ** 2

    return total


def r_squared(xs: list[float], ys: list[float], alpha: float, beta: float) -> float:
    """Coefficient de determination R²."""

    mean_y = mean(ys)

    total_variance = 0
    for y in ys:
        total_variance += (y - mean_y) ** 2

    unexplained = sum_of_sqerrors(xs, ys, alpha, beta)

    return 1 - unexplained / total_variance