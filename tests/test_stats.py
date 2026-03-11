"""Tests unitaires pour analysis/stats.py"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.stats import mean, median, variance, standard_deviation, covariance, correlation


def test_mean_basic():
    assert mean([1, 2, 3]) == 2.0


def test_mean_two_elements():
    assert mean([10, 20]) == 15.0


def test_mean_single():
    assert mean([42.0]) == 42.0


def test_median_odd():
    assert median([1, 3, 5]) == 3.0


def test_median_even():
    assert median([1, 2, 3, 4]) == 2.5


def test_median_sorted_order():
    assert median([5, 1, 3]) == 3.0


def test_variance_grus():
    result = variance([2, 4, 4, 4, 5, 5, 7, 9])
    assert abs(result - 4.0) < 0.1


def test_standard_deviation():
    result = standard_deviation([2, 4, 4, 4, 5, 5, 7, 9])
    assert abs(result - 2.0) < 0.01


def test_correlation_identical():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert abs(correlation(xs, xs) - 1.0) < 0.01


def test_correlation_inverse():
    xs = [1, 2, 3, 4, 5]
    ys = [5, 4, 3, 2, 1]
    assert abs(correlation(xs, ys) - (-1.0)) < 0.01


def test_correlation_zero_std():
    xs = [1, 1, 1]
    ys = [2, 3, 4]
    assert correlation(xs, ys) == 0


def test_covariance_positive():
    xs = [1, 2, 3]
    ys = [1, 2, 3]
    assert covariance(xs, ys) > 0
