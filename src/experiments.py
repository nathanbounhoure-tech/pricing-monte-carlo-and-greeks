"""
experiments.py

Deux familles d'expériences, volontairement séparées de la variable de contrôle
pour rester pédagogiques :

1. convergence_experiment : convergence de l'estimateur Monte Carlo "brut" (sans
   réduction de variance) vers le prix Black-Scholes lorsque N augmente, afin de
   visualiser la vitesse de convergence en O(1/sqrt(N)).
2. sensitivity_experiment : sensibilité du prix (formule fermée) à la volatilité,
   la maturité et le strike.
"""

import numpy as np
import pandas as pd

from . import black_scholes as bs
from .monte_carlo import simulate_terminal, mc_estimator
from .payoffs import european_call_payoff


def convergence_experiment(
    S0: float, K: float, r: float, sigma: float, T: float,
    n_values: list[int], rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    bs_ref = bs.bs_price(S0, K, r, sigma, T, "call")
    for n in n_values:
        ST = simulate_terminal(S0, r, sigma, T, n, rng, antithetic=False)
        payoffs = np.exp(-r * T) * european_call_payoff(ST, K)
        res = mc_estimator(payoffs)
        rows.append(
            {
                "n_paths": n,
                "price": res["price"],
                "std_error": res["std_error"],
                "ci_width": res["ci_width"],
                "bs_reference": bs_ref,
                "abs_error": abs(res["price"] - bs_ref),
            }
        )
    return pd.DataFrame(rows)


def sensitivity_experiment(
    S0: float, K: float, r: float, base_sigma: float, base_T: float,
    sigmas: list[float], maturities: list[float], strikes: list[float],
) -> pd.DataFrame:
    rows = []
    for sigma in sigmas:
        price = bs.bs_price(S0, K, r, sigma, base_T, "call")
        rows.append({"parameter": "sigma", "value": sigma, "price": price})
    for T in maturities:
        price = bs.bs_price(S0, K, r, base_sigma, T, "call")
        rows.append({"parameter": "T", "value": T, "price": price})
    for K_ in strikes:
        price = bs.bs_price(S0, K_, r, base_sigma, base_T, "call")
        rows.append({"parameter": "K", "value": K_, "price": price})
    return pd.DataFrame(rows)
