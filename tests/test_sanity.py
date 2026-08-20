"""
Tests de cohérence économique ("sanity checks") :
- le call asiatique arithmétique doit valoir moins qu'un call européen de même
  strike (la moyenne réduit la volatilité effective du payoff) ;
- le down-and-out call doit valoir moins ou autant qu'un call vanille de même
  strike (optionalité strictement dominée par le vanille) ;
- la variable de contrôle ne doit pas biaiser le prix (doit rester cohérente
  avec l'estimateur brut, à l'erreur statistique près).
"""

import numpy as np

from src import black_scholes as bs
from src.monte_carlo import simulate_paths, mc_estimator
from src.payoffs import european_call_payoff, asian_call_payoff, down_and_out_call_payoff
from src.variance_reduction import control_variate_estimate

S0, K, R, SIGMA, T = 100.0, 100.0, 0.02, 0.20, 1.0
N_STEPS = 252


def test_asian_cheaper_than_european():
    rng = np.random.default_rng(1)
    paths = simulate_paths(S0, R, SIGMA, T, N_STEPS, 40_000, rng, antithetic=True)
    fixing_idx = np.linspace(0, N_STEPS, 13, dtype=int)[1:]

    asian_price = np.exp(-R * T) * asian_call_payoff(paths[:, fixing_idx], K).mean()
    european_price = np.exp(-R * T) * european_call_payoff(paths[:, -1], K).mean()

    assert asian_price < european_price


def test_barrier_cheaper_or_equal_than_vanilla():
    rng = np.random.default_rng(2)
    paths = simulate_paths(S0, R, SIGMA, T, N_STEPS, 40_000, rng, antithetic=True)
    barrier = 80.0

    do_price = np.exp(-R * T) * down_and_out_call_payoff(paths, K, barrier).mean()
    european_price = np.exp(-R * T) * european_call_payoff(paths[:, -1], K).mean()

    assert do_price <= european_price


def test_lower_barrier_means_higher_price():
    # Une barrière plus basse (plus difficile à toucher) doit rapprocher le
    # down-and-out call du call vanille (moins de désactivations).
    rng1 = np.random.default_rng(3)
    rng2 = np.random.default_rng(3)
    paths1 = simulate_paths(S0, R, SIGMA, T, N_STEPS, 40_000, rng1, antithetic=True)
    paths2 = simulate_paths(S0, R, SIGMA, T, N_STEPS, 40_000, rng2, antithetic=True)

    price_barrier_80 = np.exp(-R * T) * down_and_out_call_payoff(paths1, K, 80.0).mean()
    price_barrier_60 = np.exp(-R * T) * down_and_out_call_payoff(paths2, K, 60.0).mean()

    assert price_barrier_60 >= price_barrier_80


def test_control_variate_is_unbiased_vs_raw_estimate():
    rng = np.random.default_rng(4)
    ST = S0 * np.exp((R - 0.5 * SIGMA ** 2) * T + SIGMA * np.sqrt(T) * rng.standard_normal(100_000))
    discounted_ST = np.exp(-R * T) * ST
    call_payoffs = np.exp(-R * T) * european_call_payoff(ST, K)

    raw = mc_estimator(call_payoffs)
    adjusted = control_variate_estimate(call_payoffs, discounted_ST, S0)

    # Le prix ajusté doit rester dans l'IC (large) de l'estimateur brut : pas de biais
    assert raw["ci_low"] - 3 * raw["std_error"] < adjusted["price"] < raw["ci_high"] + 3 * raw["std_error"]
    # et sa variance doit être strictement plus faible
    assert adjusted["std_error"] < raw["std_error"]


def test_payoffs_are_never_negative():
    rng = np.random.default_rng(6)
    paths = simulate_paths(S0, R, SIGMA, T, N_STEPS, 5_000, rng)
    assert (european_call_payoff(paths[:, -1], K) >= 0).all()
    assert (asian_call_payoff(paths[:, ::20], K) >= 0).all()
    assert (down_and_out_call_payoff(paths, K, 80.0) >= 0).all()
