"""
monte_carlo.py

Moteur de simulation Monte Carlo sous le modèle Black-Scholes (mouvement brownien
géométrique), avec schéma de simulation exact (pas de biais de discrétisation, car
la solution de la SDE est connue en forme fermée à toute date).

Fournit également l'estimateur Monte Carlo standard (moyenne, erreur-type, IC 95%).
"""

import numpy as np


def simulate_terminal(
    S0: float, r: float, sigma: float, T: float, n_paths: int,
    rng: np.random.Generator, antithetic: bool = False,
) -> np.ndarray:
    """
    Simule uniquement S_T (suffisant pour les payoffs européens, qui ne dépendent
    que de la valeur terminale du sous-jacent).
    """
    if antithetic:
        half = n_paths // 2
        Z = rng.standard_normal(half)
        Z = np.concatenate([Z, -Z])
    else:
        Z = rng.standard_normal(n_paths)

    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    return ST


def simulate_paths(
    S0: float, r: float, sigma: float, T: float, n_steps: int, n_paths: int,
    rng: np.random.Generator, antithetic: bool = False,
) -> np.ndarray:
    """
    Simule des trajectoires complètes sur une grille régulière de n_steps pas
    (n_steps + 1 points, S0 inclus en première colonne), via le schéma exact du GBM.

    Retourne un tableau (n_paths, n_steps + 1).
    """
    dt = T / n_steps

    if antithetic:
        half = n_paths // 2
        Z = rng.standard_normal((half, n_steps))
        Z = np.concatenate([Z, -Z], axis=0)
    else:
        Z = rng.standard_normal((n_paths, n_steps))

    increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.concatenate([np.zeros((log_paths.shape[0], 1)), log_paths], axis=1)
    paths = S0 * np.exp(log_paths)
    return paths


def mc_estimator(discounted_payoffs: np.ndarray) -> dict:
    """
    Calcule l'estimateur Monte Carlo standard à partir d'un vecteur de payoffs
    déjà actualisés (échantillon i.i.d.) : prix, erreur-type, IC à 95%.

    À ne PAS utiliser directement sur un échantillon construit par variables
    antithétiques : le pooling brut ignore la corrélation négative entre
    paires (Z, -Z) et sous-estime donc la réduction de variance réellement
    obtenue. Utiliser mc_estimator_antithetic dans ce cas.
    """
    n = discounted_payoffs.shape[0]
    price = float(discounted_payoffs.mean())
    std_error = float(discounted_payoffs.std(ddof=1) / np.sqrt(n))
    ci_low = price - 1.96 * std_error
    ci_high = price + 1.96 * std_error
    return {
        "price": price,
        "std_error": std_error,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": ci_high - ci_low,
        "n_paths": n,
    }


def mc_estimator_antithetic(discounted_payoffs: np.ndarray) -> dict:
    """
    Estimateur Monte Carlo pour un échantillon construit par variables
    antithétiques, où discounted_payoffs a été produit par simulate_terminal /
    simulate_paths avec antithetic=True (donc organisé comme la concaténation
    [payoffs issus de Z, payoffs issus de -Z], les deux moitiés étant appariées
    élément par élément).

    On calcule la moyenne de chaque paire (Y_i = (X_i + X_i')/2), puis l'erreur-
    type sur ces n/2 moyennes de paires. Cela capture correctement la
    corrélation négative entre X_i et X_i' induite par l'antithétique, ce que
    le pooling brut ne fait pas.
    """
    n = discounted_payoffs.shape[0]
    half = n // 2
    pos, neg = discounted_payoffs[:half], discounted_payoffs[half:]
    pair_avg = 0.5 * (pos + neg)

    price = float(pair_avg.mean())
    std_error = float(pair_avg.std(ddof=1) / np.sqrt(half))
    ci_low = price - 1.96 * std_error
    ci_high = price + 1.96 * std_error
    return {
        "price": price,
        "std_error": std_error,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": ci_high - ci_low,
        "n_paths": n,
    }
