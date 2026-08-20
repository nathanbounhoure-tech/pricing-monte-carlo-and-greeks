"""
payoffs.py

Fonctions de payoff pour les produits pricés dans le projet :
- call / put européens (dépendent uniquement de S_T) ;
- call asiatique arithmétique (dépend de la moyenne du sous-jacent sur des dates de fixing) ;
- down-and-out call (dépend du minimum de la trajectoire vs une barrière).

Toutes les fonctions sont vectorisées (NumPy) : elles opèrent sur un ensemble de
trajectoires simulées en une seule fois, sans boucle Python.
"""

import numpy as np


def european_call_payoff(ST: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(ST - K, 0.0)


def european_put_payoff(ST: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(K - ST, 0.0)


def asian_call_payoff(fixings: np.ndarray, K: float) -> np.ndarray:
    """
    fixings : tableau (n_paths, n_fixings) des valeurs du sous-jacent aux dates de fixing
              retenues pour le calcul de la moyenne arithmétique.
    """
    average = fixings.mean(axis=1)
    return np.maximum(average - K, 0.0)


def down_and_out_call_payoff(paths: np.ndarray, K: float, barrier: float) -> np.ndarray:
    """
    paths   : tableau (n_paths, n_steps + 1) incluant S0 en première colonne.
    barrier : niveau de la barrière désactivante (down-and-out).

    L'option est désactivée (payoff nul) si la trajectoire touche ou franchit
    la barrière à un instant quelconque de la surveillance discrète.
    """
    min_path = paths.min(axis=1)
    ST = paths[:, -1]
    alive = (min_path > barrier).astype(float)
    return np.maximum(ST - K, 0.0) * alive
