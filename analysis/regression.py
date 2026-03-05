"""
Regression lineaire simple from scratch.
Reference : Joel Grus, "Data Science From Scratch", chapitre 14.

IMPORTANT : N'importez pas sklearn, numpy ou scipy pour ces fonctions.
"""

from analysis.stats import mean, variance, covariance


def predict(alpha: float, beta: float, x_i: float) -> float:
    """Predit y pour une valeur x : y = alpha + beta * x."""
    # VOTRE CODE ICI
    raise NotImplementedError("Implementez predict() - voir Grus ch.14")


def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    """Calcule l'erreur de prediction pour un point."""
    # VOTRE CODE ICI
    raise NotImplementedError("Implementez error() - voir Grus ch.14")


def sum_of_sqerrors(alpha: float, beta: float, x: list, y: list) -> float:
    """Somme des erreurs au carre sur tous les points."""
    # VOTRE CODE ICI
    raise NotImplementedError("Implementez sum_of_sqerrors() - voir Grus ch.14")


def least_squares_fit(x: list[float], y: list[float]) -> tuple[float, float]:
    """
    Trouve alpha et beta qui minimisent la somme des erreurs au carre.
    Retourne (alpha, beta) tels que y ≈ alpha + beta * x.
    """
    # VOTRE CODE ICI
    # Indices : beta = covariance(x, y) / variance(x)
    #           alpha = mean(y) - beta * mean(x)
    raise NotImplementedError("Implementez least_squares_fit() - voir Grus ch.14")


def r_squared(alpha: float, beta: float, x: list, y: list) -> float:
    """
    Coefficient de determination R².
    R² = 1 - (SS_res / SS_tot)
    1.0 = ajustement parfait, 0.0 = le modele n'explique rien.
    """
    # VOTRE CODE ICI
    raise NotImplementedError("Implementez r_squared() - voir Grus ch.14")
