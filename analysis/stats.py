"""
Fonctions statistiques from scratch.
Reference : Joel Grus, "Data Science From Scratch", chapitre 5.

IMPORTANT : N'importez pas numpy, pandas ou statistics pour ces fonctions.
Implementez-les avec du Python pur (listes, boucles, math).
"""

import math


def mean(xs: list[float]) -> float:
    """Retourne la moyenne d'une liste de nombres."""
    
    total = 0

    for x in xs:
        total += x

    return total / len(xs)


def median(xs: list[float]) -> float:
    """Retourne la mediane d'une liste de nombres."""

    sorted_xs = sorted(xs)
    n = len(sorted_xs)
    midpoint = n // 2

    if n % 2 == 1:
        return sorted_xs[midpoint]
    else:
        return (sorted_xs[midpoint - 1] + sorted_xs[midpoint]) / 2


def variance(xs: list[float]) -> float:
    """Retourne la variance d'une liste de nombres."""

    n = len(xs)
    m = mean(xs)

    total = 0

    for x in xs:
        total += (x - m) ** 2

    return total / (n - 1)


def standard_deviation(xs: list[float]) -> float:
    """Retourne l'ecart-type d'une liste de nombres."""

    return math.sqrt(variance(xs))


def covariance(xs: list[float], ys: list[float]) -> float:
    """Retourne la covariance entre deux series."""

    n = len(xs)

    mean_x = mean(xs)
    mean_y = mean(ys)

    total = 0

    for i in range(n):
        total += (xs[i] - mean_x) * (ys[i] - mean_y)

    return total / (n - 1)


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
