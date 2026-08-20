"""
greeks.py

Calcul des Greeks :
- formules analytiques Black-Scholes pour les options européennes ;
- différences finies (bump-and-revalue) avec nombres aléatoires communs (common
  random numbers) pour les options exotiques, afin de réduire le bruit de
  l'estimation par annulation partielle de la variance Monte Carlo entre les
  scénarios bumpés.
"""

from typing import Callable

from . import black_scholes as bs


def european_greeks_analytic(S0: float, K: float, r: float, sigma: float, T: float, option_type: str = "call") -> dict:
    result = {
        "delta": bs.bs_delta(S0, K, r, sigma, T, option_type),
        "gamma": bs.bs_gamma(S0, K, r, sigma, T),
        "vega": bs.bs_vega(S0, K, r, sigma, T),
        "theta": bs.bs_theta(S0, K, r, sigma, T, option_type),
    }
    return result


def bump_and_revalue_greeks(
    pricing_fn: Callable[[float, float], float],
    S0: float,
    sigma: float,
    h_S: float | None = None,
    h_sigma: float | None = None,
) -> dict:
    """
    pricing_fn(S0, sigma) -> prix Monte Carlo (float). L'implémentation de
    pricing_fn DOIT réinitialiser le générateur aléatoire avec la même seed à
    chaque appel, afin que les cinq évaluations (V0, S+h, S-h, sigma+h, sigma-h)
    partagent exactement les mêmes tirages (nombres aléatoires communs).

    h_S     : taille du bump en spot (par défaut 1% de S0).
    h_sigma : taille du bump en volatilité (par défaut 1% de sigma).
    """
    if h_S is None:
        h_S = 0.01 * S0
    if h_sigma is None:
        h_sigma = 0.01 * sigma

    V0 = pricing_fn(S0, sigma)
    V_up_S = pricing_fn(S0 + h_S, sigma)
    V_down_S = pricing_fn(S0 - h_S, sigma)
    V_up_sigma = pricing_fn(S0, sigma + h_sigma)
    V_down_sigma = pricing_fn(S0, sigma - h_sigma)

    delta = (V_up_S - V_down_S) / (2 * h_S)
    gamma = (V_up_S - 2 * V0 + V_down_S) / (h_S ** 2)
    vega = (V_up_sigma - V_down_sigma) / (2 * h_sigma)

    return {"price": V0, "delta": delta, "gamma": gamma, "vega": vega, "h_S": h_S, "h_sigma": h_sigma}
