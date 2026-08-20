"""
Tests unitaires du module greeks.py :
- cohérence entre Greeks analytiques et bump-and-revalue sur les européennes
  (le bump-and-revalue est testé sur un produit dont on connaît la réponse
  analytique, ce qui permet de valider la méthode avant de l'appliquer aux
  exotiques, pour lesquelles aucune formule fermée n'existe).
"""

import numpy as np

from src import black_scholes as bs
from src.monte_carlo import simulate_terminal, mc_estimator
from src.payoffs import european_call_payoff
from src.greeks import bump_and_revalue_greeks, european_greeks_analytic

S0, K, R, SIGMA, T = 100.0, 100.0, 0.02, 0.20, 1.0
N_PATHS = 300_000


def _pricing_fn(S0_, sigma_):
    rng = np.random.default_rng(7)  # nombres aléatoires communs entre les appels
    ST = simulate_terminal(S0_, R, sigma_, T, N_PATHS, rng, antithetic=True)
    payoffs = np.exp(-R * T) * european_call_payoff(ST, K)
    return payoffs.mean()


def test_bump_delta_matches_analytic():
    res = bump_and_revalue_greeks(_pricing_fn, S0, SIGMA)
    analytic = bs.bs_delta(S0, K, R, SIGMA, T, "call")
    assert abs(res["delta"] - analytic) < 0.02


def test_bump_gamma_matches_analytic():
    res = bump_and_revalue_greeks(_pricing_fn, S0, SIGMA)
    analytic = bs.bs_gamma(S0, K, R, SIGMA, T)
    assert abs(res["gamma"] - analytic) < 0.01


def test_bump_vega_matches_analytic():
    res = bump_and_revalue_greeks(_pricing_fn, S0, SIGMA)
    analytic = bs.bs_vega(S0, K, R, SIGMA, T)
    assert abs(res["vega"] - analytic) < 2.0


def test_european_greeks_analytic_keys():
    greeks = european_greeks_analytic(S0, K, R, SIGMA, T, "call")
    assert set(greeks.keys()) == {"delta", "gamma", "vega", "theta"}
