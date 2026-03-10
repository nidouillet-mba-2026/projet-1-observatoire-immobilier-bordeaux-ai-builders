"""
Regression lineaire simple from scratch.
Reference : Joel Grus, "Data Science From Scratch", chapitre 14.

IMPORTANT : N'importez pas sklearn, numpy ou scipy pour ces fonctions.
"""

from analysis.stats import mean, variance, covariance


def predict(alpha: float, beta: float, x_i: float) -> float:
    return alpha + beta * x_i


def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    return predict(alpha, beta, x_i) - y_i


def sum_of_sqerrors(alpha: float, beta: float, x: list, y: list) -> float:
    return sum(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))


def least_squares_fit(x: list[float], y: list[float]) -> tuple[float, float]:
    beta = covariance(x, y) / variance(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta


def r_squared(alpha: float, beta: float, x: list, y: list) -> float:
    ss_res = sum_of_sqerrors(alpha, beta, x, y)
    y_bar = mean(y)
    ss_tot = sum((y_i - y_bar) ** 2 for y_i in y)
    return 1 - ss_res / ss_tot