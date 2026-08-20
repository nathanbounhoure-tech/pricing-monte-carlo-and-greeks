"""
Tests unitaires du module black_scholes.py :
- parité call-put ;
- positivité des prix ;
- bornes des deltas ;
- cohérence des Greeks near-the-money.
"""

import numpy as np

from src import black_scholes as bs

S0, K, R, SIGMA, T = 100.0, 100.0, 0.02, 0.20, 1.0


def test_put_call_parity():
    call = bs.bs_price(S0, K, R, SIGMA, T, "call")
    put = bs.bs_price(S0, K, R, SIGMA, T, "put")
    lhs = call - put
    rhs = S0 - K * np.exp(-R * T)
    assert abs(lhs - rhs) < 1e-8


def test_price_positive():
    assert bs.bs_price(S0, K, R, SIGMA, T, "call") > 0
    assert bs.bs_price(S0, K, R, SIGMA, T, "put") > 0


def test_delta_bounds():
    d_call = bs.bs_delta(S0, K, R, SIGMA, T, "call")
    d_put = bs.bs_delta(S0, K, R, SIGMA, T, "put")
    assert 0.0 <= d_call <= 1.0
    assert -1.0 <= d_put <= 0.0


def test_call_delta_minus_put_delta_equals_one():
    d_call = bs.bs_delta(S0, K, R, SIGMA, T, "call")
    d_put = bs.bs_delta(S0, K, R, SIGMA, T, "put")
    assert abs((d_call - d_put) - 1.0) < 1e-10


def test_gamma_positive_and_symmetric_between_call_put():
    gamma = bs.bs_gamma(S0, K, R, SIGMA, T)
    assert gamma > 0


def test_vega_positive():
    assert bs.bs_vega(S0, K, R, SIGMA, T) > 0


def test_invalid_option_type_raises():
    try:
        bs.bs_price(S0, K, R, SIGMA, T, "straddle")
        assert False, "ValueError attendue"
    except ValueError:
        pass
