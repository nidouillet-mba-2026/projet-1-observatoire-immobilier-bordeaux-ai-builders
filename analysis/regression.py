"""
Regression lineaire simple from scratch.
Reference : Joel Grus, "Data Science From Scratch", chapitre 14.

IMPORTANT : N'importez pas de librairies ML externes pour ces fonctions.
"""

from analysis.stats import mean, variance, covariance


def predict(alpha: float, beta: float, x_i: float) -> float:
    """Predit y pour une valeur x : y = alpha + beta * x."""
    return alpha + beta * x_i


def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    """Calcule l'erreur de prediction pour un point."""
    return predict(alpha, beta, x_i) - y_i


def sum_of_sqerrors(alpha: float, beta: float, x: list, y: list) -> float:
    """Somme des erreurs au carre sur tous les points."""
    return sum(error(alpha, beta, x[i], y[i]) ** 2 for i in range(len(x)))


def least_squares_fit(x: list[float], y: list[float]) -> tuple[float, float]:
    """
    Trouve alpha et beta qui minimisent la somme des erreurs au carre.
    Retourne (alpha, beta) tels que y ≈ alpha + beta * x.
    """
    beta = covariance(x, y) / variance(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta


def r_squared(alpha: float, beta: float, x: list, y: list) -> float:
    """
    Coefficient de determination R².
    R² = 1 - (SS_res / SS_tot)
    1.0 = ajustement parfait, 0.0 = le modele n'explique rien.
    """
    mean_y = mean(y)
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    if ss_tot == 0:
        return 0
    ss_res = sum_of_sqerrors(alpha, beta, x, y)
    return 1 - ss_res / ss_tot
