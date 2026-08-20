"""
variance_reduction.py

Techniques de réduction de variance :
- variables antithétiques (implémentées directement dans monte_carlo.simulate_*) ;
- variable de contrôle, implémentée ici.

Pour la variable de contrôle, on utilise l'estimateur classique :

    Y_i = X_i - c* (C_i - E[C])

où c* = Cov(X, C) / Var(C) est estimé empiriquement sur l'échantillon simulé,
X_i est le payoff actualisé du produit à pricer, C_i le payoff actualisé de la
variable de contrôle, et E[C] sa valeur théorique connue (fermée).
"""

import numpy as np

from .monte_carlo import mc_estimator, mc_estimator_antithetic


def control_variate_estimate(
    target_payoffs: np.ndarray, control_payoffs: np.ndarray, control_true_value: float,
    antithetic: bool = False,
) -> dict:
    """
    target_payoffs     : payoffs actualisés du produit à pricer.
    control_payoffs    : payoffs actualisés de la variable de contrôle, simulés
                          sur les MÊMES trajectoires que target_payoffs.
    control_true_value : valeur théorique exacte de la variable de contrôle.
    antithetic          : si True, target_payoffs et control_payoffs sont
                          supposés construits via variables antithétiques
                          (concaténation [Z, -Z] appariée) — l'erreur-type est
                          alors calculée sur les moyennes de paires plutôt que
                          sur l'échantillon brut, pour refléter correctement la
                          réduction de variance conjointe antithétique + contrôle.
    """
    cov_matrix = np.cov(target_payoffs, control_payoffs, ddof=1)
    cov_xc = cov_matrix[0, 1]
    var_c = cov_matrix[1, 1]
    c_star = cov_xc / var_c if var_c > 0 else 0.0

    adjusted = target_payoffs - c_star * (control_payoffs - control_true_value)

    result = mc_estimator_antithetic(adjusted) if antithetic else mc_estimator(adjusted)
    result["c_star"] = float(c_star)
    return result


def variance_reduction_pct(ci_width_base: float, ci_width_reduced: float) -> float:
    """Pourcentage de réduction de la largeur de l'intervalle de confiance à 95%."""
    return 100.0 * (1.0 - ci_width_reduced / ci_width_base)
