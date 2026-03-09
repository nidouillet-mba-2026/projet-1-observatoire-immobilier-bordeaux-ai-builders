"""Tests unitaires pour analysis/regression.py"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.regression import predict, error, least_squares_fit, r_squared, sum_of_sqerrors


def test_predict_basic():
    assert predict(1.0, 2.0, 3.0) == 7.0


def test_predict_zero_beta():
    assert predict(5.0, 0.0, 100.0) == 5.0


def test_error_positive():
    assert error(1.0, 2.0, 3.0, 5.0) == 2.0


def test_error_zero():
    assert error(1.0, 2.0, 3.0, 7.0) == 0.0


def test_least_squares_fit_y_equals_2x_plus_1():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [3.0, 5.0, 7.0, 9.0, 11.0]
    alpha, beta = least_squares_fit(x, y)
    assert abs(beta - 2.0) < 0.01
    assert abs(alpha - 1.0) < 0.01


def test_r_squared_perfect_fit():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [3.0, 5.0, 7.0, 9.0, 11.0]
    alpha, beta = least_squares_fit(x, y)
    assert abs(r_squared(alpha, beta, x, y) - 1.0) < 0.01


def test_r_squared_between_0_and_1():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 3.0, 5.0, 6.0]
    alpha, beta = least_squares_fit(x, y)
    r2 = r_squared(alpha, beta, x, y)
    assert 0.0 <= r2 <= 1.0


def test_sum_of_sqerrors_perfect_fit():
    x = [1.0, 2.0, 3.0]
    y = [3.0, 5.0, 7.0]
    alpha, beta = least_squares_fit(x, y)
    assert sum_of_sqerrors(alpha, beta, x, y) < 1e-10
