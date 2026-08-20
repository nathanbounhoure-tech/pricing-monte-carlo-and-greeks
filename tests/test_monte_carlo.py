"""
Tests unitaires du moteur Monte Carlo :
- convergence de l'estimateur brut vers Black-Scholes ;
- convergence de l'estimateur avec variables antithétiques ;
- cohérence de l'intervalle de confiance.
"""

import numpy as np

from src import black_scholes as bs
from src.monte_carlo import simulate_terminal, simulate_paths, mc_estimator, mc_estimator_antithetic
from src.payoffs import european_call_payoff

S0, K, R, SIGMA, T = 100.0, 100.0, 0.02, 0.20, 1.0


def test_mc_converges_to_bs():
    rng = np.random.default_rng(123)
    ST = simulate_terminal(S0, R, SIGMA, T, 200_000, rng, antithetic=True)
    payoffs = np.exp(-R * T) * european_call_payoff(ST, K)
    res = mc_estimator(payoffs)
    ref = bs.bs_price(S0, K, R, SIGMA, T, "call")
    # tolérance à 5 erreurs-types : doit passer avec une probabilité écrasante
    assert abs(res["price"] - ref) < 5 * res["std_error"]


def test_confidence_interval_contains_price():
    rng = np.random.default_rng(1)
    ST = simulate_terminal(S0, R, SIGMA, T, 50_000, rng)
    payoffs = np.exp(-R * T) * european_call_payoff(ST, K)
    res = mc_estimator(payoffs)
    assert res["ci_low"] < res["price"] < res["ci_high"]
    assert res["ci_width"] > 0


def test_antithetic_reduces_std_error():
    # Le payoff d'un call est une fonction monotone de Z : la théorie garantit
    # Cov(f(Z), f(-Z)) <= 0, donc la variance de l'estimateur apparié doit être
    # strictement plus faible que celle de l'estimateur brut. IMPORTANT : il
    # faut utiliser mc_estimator_antithetic (moyennes de paires), pas
    # mc_estimator sur l'échantillon "poolé" brut, qui ignore la corrélation
    # négative entre paires et ne montre donc pas la réduction de variance.
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    ST_raw = simulate_terminal(S0, R, SIGMA, T, 50_000, rng1, antithetic=False)
    ST_anti = simulate_terminal(S0, R, SIGMA, T, 50_000, rng2, antithetic=True)
    se_raw = mc_estimator(np.exp(-R * T) * european_call_payoff(ST_raw, K))["std_error"]
    se_anti = mc_estimator_antithetic(np.exp(-R * T) * european_call_payoff(ST_anti, K))["std_error"]
    assert se_anti < se_raw


def test_naive_pooling_hides_antithetic_benefit():
    # Ce test documente explicitement le piège : le pooling brut ne capture
    # PAS la corrélation négative entre paires (Cov(X_i, X_i') n'apparaît pas
    # dans une variance calculée sur l'échantillon aplati), donc il ne faut
    # jamais l'utiliser pour évaluer l'apport des variables antithétiques.
    rng = np.random.default_rng(7)
    ST_anti = simulate_terminal(S0, R, SIGMA, T, 50_000, rng, antithetic=True)
    payoffs = np.exp(-R * T) * european_call_payoff(ST_anti, K)
    se_pooled = mc_estimator(payoffs)["std_error"]
    se_paired = mc_estimator_antithetic(payoffs)["std_error"]
    assert se_paired < se_pooled


def test_simulate_paths_shape_and_start_value():
    rng = np.random.default_rng(5)
    n_paths, n_steps = 100, 50
    paths = simulate_paths(S0, R, SIGMA, T, n_steps, n_paths, rng)
    assert paths.shape == (n_paths, n_steps + 1)
    assert np.allclose(paths[:, 0], S0)
    assert np.all(paths > 0)


def test_simulate_paths_terminal_matches_simulate_terminal_distribution():
    # Les deux schémas doivent produire des distributions terminales cohérentes
    # (même moyenne théorique de S_T sous la mesure risque-neutre).
    rng = np.random.default_rng(11)
    paths = simulate_paths(S0, R, SIGMA, T, 252, 100_000, rng, antithetic=True)
    ST = paths[:, -1]
    expected_mean = S0 * np.exp(R * T)
    assert abs(ST.mean() - expected_mean) / expected_mean < 0.01
