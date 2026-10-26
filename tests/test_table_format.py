"""Tests for the LaTeX formatting helpers used by the result tables."""

import math

from table_format import fmt, interval, latex_escape, signed


def test_should_escape_underscores_and_percent_signs():
    assert latex_escape("chain_D8 at 50%") == r"chain\_D8 at 50\%"


def test_should_render_missing_values_as_a_dash():
    assert fmt(math.nan, 2) == "--"


def test_should_round_to_the_requested_digits():
    assert fmt(312.156, 1) == "312.2"


def test_should_show_the_sign_of_a_delta():
    assert signed(-1.5, 1) == "-1.5"


def test_should_show_a_plus_sign_on_a_positive_delta():
    assert signed(2.0, 1) == "+2.0"


def test_should_render_a_point_with_its_bracket():
    assert interval(312.2, 234.1, 312.2, 1) == "312.2 [234.1, 312.2]"
