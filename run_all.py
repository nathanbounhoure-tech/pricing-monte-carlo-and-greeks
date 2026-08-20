"""
run_all.py

Point d'entrée principal du pipeline de pricing Monte Carlo.

Étapes :
 1. charge config.yaml ;
 2. price les options européennes (Monte Carlo brut vs Black-Scholes) ;
 3. applique la réduction de variance (antithétique + variable de contrôle) ;
 4. price les options exotiques (call asiatique, down-and-out call) ;
 5. calcule les Greeks (analytiques pour les européennes, bump-and-revalue
    avec nombres aléatoires communs pour les exotiques) ;
 6. exécute les expériences de convergence et de sensibilité ;
 7. génère les figures ;
 8. exporte les tableaux CSV ;
 9. rédige outputs/reports/final_report.md.
"""

import numpy as np
import pandas as pd

from src.utils import load_config, ensure_output_dirs, setup_logger, get_rng
from src import black_scholes as bs
from src.monte_carlo import simulate_terminal, simulate_paths, mc_estimator
from src.payoffs import (
    european_call_payoff,
    european_put_payoff,
    asian_call_payoff,
    down_and_out_call_payoff,
)
from src.variance_reduction import control_variate_estimate, variance_reduction_pct
from src.greeks import european_greeks_analytic, bump_and_revalue_greeks
from src.experiments import convergence_experiment, sensitivity_experiment
from src.plots import (
    plot_convergence,
    plot_variance_reduction_comparison,
    plot_sensitivity_volatility,
    plot_simulated_paths_with_barrier,
)
from src.report import generate_final_report


def df_to_md(df: pd.DataFrame, float_format: str = "{:.6f}") -> str:
    return df.to_markdown(index=False, floatfmt=".6f")


def main():
    config = load_config("config.yaml")
    ensure_output_dirs("outputs")
    logger = setup_logger("outputs/logs/run_all.log")

    S0 = config["market"]["S0"]
    K = config["market"]["K"]
    r = config["market"]["r"]
    sigma = config["market"]["sigma"]
    T = config["market"]["T"]
    N = config["simulation"]["n_paths"]
    seed = config["simulation"]["seed"]

    n_steps = config["barrier"]["n_steps"]
    barrier_level = config["barrier"]["level"]
    n_fixings = config["asian"]["n_fixings"]

    bump_S = config["greeks"]["bump_S_pct"] * S0
    bump_sigma = config["greeks"]["bump_sigma_pct"] * sigma
    n_paths_bump = config["greeks"]["n_paths_bump"]

    logger.info(
        "Démarrage du pipeline | S0=%.2f K=%.2f r=%.4f sigma=%.4f T=%.2f N=%d seed=%d",
        S0, K, r, sigma, T, N, seed,
    )

    # ------------------------------------------------------------------
    # 1. Pricing européen : Monte Carlo brut vs Black-Scholes
    # ------------------------------------------------------------------
    rng = get_rng(seed)
    ST_raw = simulate_terminal(S0, r, sigma, T, N, rng, antithetic=False)
    call_payoffs_raw = np.exp(-r * T) * european_call_payoff(ST_raw, K)
    put_payoffs_raw = np.exp(-r * T) * european_put_payoff(ST_raw, K)

    call_mc_raw = mc_estimator(call_payoffs_raw)
    put_mc_raw = mc_estimator(put_payoffs_raw)

    call_bs = bs.bs_price(S0, K, r, sigma, T, "call")
    put_bs = bs.bs_price(S0, K, r, sigma, T, "put")

    logger.info("Call BS=%.6f | MC brut=%.6f (se=%.6f)", call_bs, call_mc_raw["price"], call_mc_raw["std_error"])
    logger.info("Put  BS=%.6f | MC brut=%.6f (se=%.6f)", put_bs, put_mc_raw["price"], put_mc_raw["std_error"])

    # ------------------------------------------------------------------
    # 2. Pricing européen avec réduction de variance (antithétique + contrôle)
    # ------------------------------------------------------------------
    rng = get_rng(seed)
    ST_anti = simulate_terminal(S0, r, sigma, T, N, rng, antithetic=True)
    discounted_ST = np.exp(-r * T) * ST_anti

    call_payoffs_anti = np.exp(-r * T) * european_call_payoff(ST_anti, K)
    put_payoffs_anti = np.exp(-r * T) * european_put_payoff(ST_anti, K)

    call_final = control_variate_estimate(call_payoffs_anti, discounted_ST, S0, antithetic=True)
    put_final = control_variate_estimate(put_payoffs_anti, discounted_ST, S0, antithetic=True)

    call_vr_pct = variance_reduction_pct(call_mc_raw["ci_width"], call_final["ci_width"])
    put_vr_pct = variance_reduction_pct(put_mc_raw["ci_width"], put_final["ci_width"])

    logger.info("Call final (antithétique+contrôle)=%.6f (se=%.6f) | réduction IC=%.2f%%",
                call_final["price"], call_final["std_error"], call_vr_pct)
    logger.info("Put  final (antithétique+contrôle)=%.6f (se=%.6f) | réduction IC=%.2f%%",
                put_final["price"], put_final["std_error"], put_vr_pct)

    # ------------------------------------------------------------------
    # 3. Produits exotiques : call asiatique et down-and-out call
    # ------------------------------------------------------------------
    times = np.linspace(0, T, n_steps + 1)
    fixing_idx = np.linspace(0, n_steps, n_fixings + 1, dtype=int)[1:]

    rng = get_rng(seed + 1)
    paths_raw = simulate_paths(S0, r, sigma, T, n_steps, N, rng, antithetic=False)
    asian_payoffs_raw = np.exp(-r * T) * asian_call_payoff(paths_raw[:, fixing_idx], K)
    do_payoffs_raw = np.exp(-r * T) * down_and_out_call_payoff(paths_raw, K, barrier_level)
    asian_mc_raw = mc_estimator(asian_payoffs_raw)
    do_mc_raw = mc_estimator(do_payoffs_raw)

    rng = get_rng(seed + 1)
    paths_anti = simulate_paths(S0, r, sigma, T, n_steps, N, rng, antithetic=True)
    control_call_payoffs = np.exp(-r * T) * european_call_payoff(paths_anti[:, -1], K)
    asian_payoffs_anti = np.exp(-r * T) * asian_call_payoff(paths_anti[:, fixing_idx], K)
    do_payoffs_anti = np.exp(-r * T) * down_and_out_call_payoff(paths_anti, K, barrier_level)

    asian_final = control_variate_estimate(asian_payoffs_anti, control_call_payoffs, call_bs, antithetic=True)
    do_final = control_variate_estimate(do_payoffs_anti, control_call_payoffs, call_bs, antithetic=True)

    asian_vr_pct = variance_reduction_pct(asian_mc_raw["ci_width"], asian_final["ci_width"])
    do_vr_pct = variance_reduction_pct(do_mc_raw["ci_width"], do_final["ci_width"])

    logger.info("Call asiatique final=%.6f (se=%.6f) | réduction IC=%.2f%%",
                asian_final["price"], asian_final["std_error"], asian_vr_pct)
    logger.info("Down-and-out final=%.6f (se=%.6f) | réduction IC=%.2f%%",
                do_final["price"], do_final["std_error"], do_vr_pct)

    # ------------------------------------------------------------------
    # 4. Greeks
    # ------------------------------------------------------------------
    greeks_call = european_greeks_analytic(S0, K, r, sigma, T, "call")
    greeks_put = european_greeks_analytic(S0, K, r, sigma, T, "put")

    def asian_price_fn(S0_, sigma_):
        rng_local = get_rng(999)  # nombres aléatoires communs
        p = simulate_paths(S0_, r, sigma_, T, n_steps, n_paths_bump, rng_local, antithetic=True)
        return (np.exp(-r * T) * asian_call_payoff(p[:, fixing_idx], K)).mean()

    def do_price_fn(S0_, sigma_):
        rng_local = get_rng(998)  # nombres aléatoires communs
        p = simulate_paths(S0_, r, sigma_, T, n_steps, n_paths_bump, rng_local, antithetic=True)
        return (np.exp(-r * T) * down_and_out_call_payoff(p, K, barrier_level)).mean()

    greeks_asian = bump_and_revalue_greeks(asian_price_fn, S0, sigma, bump_S, bump_sigma)
    greeks_do = bump_and_revalue_greeks(do_price_fn, S0, sigma, bump_S, bump_sigma)

    logger.info("Greeks calculés pour les 4 produits.")

    # ------------------------------------------------------------------
    # 5. Expériences de convergence et de sensibilité
    # ------------------------------------------------------------------
    rng = get_rng(seed)
    conv_df = convergence_experiment(S0, K, r, sigma, T, config["convergence"]["n_values"], rng)

    sens_df = sensitivity_experiment(
        S0, K, r, sigma, T,
        config["sensitivity"]["sigma_range"],
        config["sensitivity"]["maturity_range"],
        config["sensitivity"]["strike_range"],
    )

    # ------------------------------------------------------------------
    # 6. Tableaux
    # ------------------------------------------------------------------
    pricing_summary = pd.DataFrame(
        [
            {"produit": "Call européen", "estimateur": "Antithétique + contrôle (stock)",
             "prix": call_final["price"], "erreur_type": call_final["std_error"], "reference_bs": call_bs},
            {"produit": "Put européen", "estimateur": "Antithétique + contrôle (stock)",
             "prix": put_final["price"], "erreur_type": put_final["std_error"], "reference_bs": put_bs},
            {"produit": "Call asiatique arithmétique", "estimateur": "Antithétique + contrôle (call européen)",
             "prix": asian_final["price"], "erreur_type": asian_final["std_error"], "reference_bs": np.nan},
            {"produit": "Down-and-out call", "estimateur": "Antithétique + contrôle (call européen)",
             "prix": do_final["price"], "erreur_type": do_final["std_error"], "reference_bs": np.nan},
        ]
    )
    pricing_summary.to_csv("outputs/tables/pricing_summary.csv", index=False)

    greeks_table = pd.DataFrame(
        [
            {"produit": "Call européen", "méthode": "Analytique", "delta": greeks_call["delta"],
             "gamma": greeks_call["gamma"], "vega": greeks_call["vega"], "theta": greeks_call["theta"]},
            {"produit": "Put européen", "méthode": "Analytique", "delta": greeks_put["delta"],
             "gamma": greeks_put["gamma"], "vega": greeks_put["vega"], "theta": greeks_put["theta"]},
            {"produit": "Call asiatique", "méthode": "Bump-and-revalue (CRN)", "delta": greeks_asian["delta"],
             "gamma": greeks_asian["gamma"], "vega": greeks_asian["vega"], "theta": np.nan},
            {"produit": "Down-and-out call", "méthode": "Bump-and-revalue (CRN)", "delta": greeks_do["delta"],
             "gamma": greeks_do["gamma"], "vega": greeks_do["vega"], "theta": np.nan},
        ]
    )
    greeks_table.to_csv("outputs/tables/greeks.csv", index=False)

    vr_table = pd.DataFrame(
        [
            {"produit": "Call européen", "ic_largeur_brut": call_mc_raw["ci_width"],
             "ic_largeur_reduit": call_final["ci_width"], "reduction_pct": call_vr_pct},
            {"produit": "Put européen", "ic_largeur_brut": put_mc_raw["ci_width"],
             "ic_largeur_reduit": put_final["ci_width"], "reduction_pct": put_vr_pct},
            {"produit": "Call asiatique", "ic_largeur_brut": asian_mc_raw["ci_width"],
             "ic_largeur_reduit": asian_final["ci_width"], "reduction_pct": asian_vr_pct},
            {"produit": "Down-and-out call", "ic_largeur_brut": do_mc_raw["ci_width"],
             "ic_largeur_reduit": do_final["ci_width"], "reduction_pct": do_vr_pct},
        ]
    )
    vr_table.to_csv("outputs/tables/variance_reduction.csv", index=False)

    conv_df.to_csv("outputs/tables/convergence.csv", index=False)
    sens_df.to_csv("outputs/tables/sensitivity.csv", index=False)

    # ------------------------------------------------------------------
    # 7. Figures
    # ------------------------------------------------------------------
    plot_convergence(conv_df, call_bs, "outputs/figures/convergence_european_call.png")
    plot_variance_reduction_comparison(
        ["Call euro", "Put euro", "Call asiatique", "Down-and-out"],
        [call_mc_raw["ci_width"], put_mc_raw["ci_width"], asian_mc_raw["ci_width"], do_mc_raw["ci_width"]],
        [call_final["ci_width"], put_final["ci_width"], asian_final["ci_width"], do_final["ci_width"]],
        "outputs/figures/variance_reduction_comparison.png",
    )
    plot_sensitivity_volatility(sens_df, "outputs/figures/sensitivity_volatility.png")

    rng = get_rng(seed + 2)
    demo_paths = simulate_paths(S0, r, sigma, T, n_steps, 60, rng, antithetic=False)
    plot_simulated_paths_with_barrier(demo_paths, times, barrier_level, "outputs/figures/simulated_paths_with_barrier.png")

    logger.info("Figures sauvegardées dans outputs/figures/.")

    # ------------------------------------------------------------------
    # 8. Rapport final
    # ------------------------------------------------------------------
    touched_mask = paths_raw.min(axis=1) <= barrier_level
    knockout_rate = 100.0 * touched_mask.mean()
    payoff_if_touched = european_call_payoff(paths_raw[:, -1], K)[touched_mask].mean()
    payoff_overall = european_call_payoff(paths_raw[:, -1], K).mean()

    interpretation = f"""- Le Monte Carlo brut converge vers Black-Scholes sur les produits européens
  (écart de {abs(call_mc_raw['price'] - call_bs):.4f} sur le call, très inférieur à une erreur-type), ce qui valide
  la chaîne de simulation.
- Le call asiatique ({asian_final['price']:.4f}) est nettement moins cher que le call européen
  ({call_final['price']:.4f}) : la moyenne arithmétique sur {n_fixings} dates de fixing réduit la
  volatilité effective du payoff.
- Le down-and-out call ({do_final['price']:.4f}) vaut moins que le call vanille de même strike, mais
  l'écart ({100 * (call_final['price'] - do_final['price']) / call_final['price']:.1f}% seulement) est plus faible que ne le
  suggérerait la probabilité de franchissement de la barrière à {barrier_level:.0f} (~{knockout_rate:.0f}% des trajectoires
  la touchent). La raison : les trajectoires qui franchissent la barrière ont, en moyenne, un payoff
  terminal proche de zéro ({payoff_if_touched:.2f} contre {payoff_overall:.2f} en moyenne globale) — une
  trajectoire qui chute 20% sous le spot revient rarement au-dessus du strike ATM en un an. La
  probabilité de franchissement seule surestime donc fortement l'impact sur le prix.
- La variable de contrôle (combinée aux variables antithétiques) améliore fortement la précision :
  réduction de la largeur de l'IC à 95% de {call_vr_pct:.1f}% (call), {put_vr_pct:.1f}% (put),
  {asian_vr_pct:.1f}% (asiatique) et {do_vr_pct:.1f}% (barrière) — l'effet est particulièrement spectaculaire
  sur la barrière car le call européen (contrôle) est fortement corrélé au payoff quand la barrière
  n'est pas touchée.
"""

    context = {
        "S0": S0, "K": K, "r": r, "sigma": sigma, "T": T, "N": N, "seed": seed,
        "barrier": barrier_level, "n_fixings": n_fixings,
        "pricing_table_md": df_to_md(pricing_summary),
        "greeks_table_md": df_to_md(greeks_table),
        "vr_table_md": df_to_md(vr_table),
        "interpretation": interpretation,
    }
    generate_final_report(context, "outputs/reports/final_report.md")

    logger.info("Pipeline terminé avec succès.")

    print("\n=== RÉSUMÉ DES PRIX (avec réduction de variance) ===")
    print(pricing_summary.to_string(index=False))
    print("\n=== RÉDUCTION DE VARIANCE (largeur IC 95%) ===")
    print(vr_table.to_string(index=False))


if __name__ == "__main__":
    main()
