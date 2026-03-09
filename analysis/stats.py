"""
Fonctions statistiques from scratch.
Reference : Joel Grus, "Data Science From Scratch", chapitre 5.

IMPORTANT : N'importez pas numpy, pandas ou statistics pour ces fonctions.
Implementez-les avec du Python pur (listes, boucles, math).
"""

import math


def mean(xs: list[float]) -> float:
    """Retourne la moyenne d'une liste de nombres."""
    return sum(xs) / len(xs)


def median(xs: list[float]) -> float:
    """Retourne la mediane d'une liste de nombres."""
    sorted_xs = sorted(xs)
    n = len(sorted_xs)
    mid = n // 2
    if n % 2 == 1:
        return sorted_xs[mid]
    return (sorted_xs[mid - 1] + sorted_xs[mid]) / 2


def variance(xs: list[float]) -> float:
    """Retourne la variance d'une liste de nombres (biaisee, n)."""
    n = len(xs)
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / n


def standard_deviation(xs: list[float]) -> float:
    """Retourne l'ecart-type d'une liste de nombres."""
    return math.sqrt(variance(xs))


def covariance(xs: list[float], ys: list[float]) -> float:
    """Retourne la covariance entre deux series (biaisee, n)."""
    n = len(xs)
    mx = mean(xs)
    my = mean(ys)
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / n


def correlation(xs: list[float], ys: list[float]) -> float:
    """
    Retourne le coefficient de correlation de Pearson entre deux series.
    Retourne 0 si l'une des series a un ecart-type nul.
    """
    std_x = standard_deviation(xs)
    std_y = standard_deviation(ys)
    if std_x == 0 or std_y == 0:
        return 0
    return covariance(xs, ys) / (std_x * std_y)
