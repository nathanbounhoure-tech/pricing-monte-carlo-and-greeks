"""
black_scholes.py

Formule fermée de Black-Scholes pour options européennes (call / put) et Greeks
analytiques associés. Sert de référence pour valider le moteur Monte Carlo.

Convention : pas de dividendes, taux sans risque et volatilité constants.
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S0: float, K: float, r: float, sigma: float, T: float) -> tuple[float, float]:
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S0: float, K: float, r: float, sigma: float, T: float, option_type: str = "call") -> float:
    """Prix Black-Scholes fermé d'un call ou d'un put européen."""
    d1, d2 = _d1_d2(S0, K, r, sigma, T)
    if option_type == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    raise ValueError("option_type doit valoir 'call' ou 'put'")


def bs_delta(S0: float, K: float, r: float, sigma: float, T: float, option_type: str = "call") -> float:
    d1, _ = _d1_d2(S0, K, r, sigma, T)
    if option_type == "call":
        return norm.cdf(d1)
    if option_type == "put":
        return norm.cdf(d1) - 1.0
    raise ValueError("option_type doit valoir 'call' ou 'put'")


def bs_gamma(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    d1, _ = _d1_d2(S0, K, r, sigma, T)
    return norm.pdf(d1) / (S0 * sigma * np.sqrt(T))


def bs_vega(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Vega par 1.00 (soit 100%) de variation de sigma. Diviser par 100 pour une vega '1 vol point'."""
    d1, _ = _d1_d2(S0, K, r, sigma, T)
    return S0 * norm.pdf(d1) * np.sqrt(T)


def bs_theta(S0: float, K: float, r: float, sigma: float, T: float, option_type: str = "call") -> float:
    """Theta par an (diviser par 365 pour un theta journalier)."""
    d1, d2 = _d1_d2(S0, K, r, sigma, T)
    term1 = -S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option_type == "call":
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        return term1 + term2
    if option_type == "put":
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        return term1 + term2
    raise ValueError("option_type doit valoir 'call' ou 'put'")
