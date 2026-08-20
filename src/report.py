"""
report.py

Génération du rapport final (Markdown) à partir des résultats calculés par
run_all.py. Le rapport est écrit dans outputs/reports/final_report.md.
"""


def generate_final_report(context: dict, path: str) -> None:
    md = f"""# Rapport final — Pricing Monte Carlo et Greeks

Rapport généré automatiquement par `run_all.py`.

## Calibration utilisée

| Paramètre | Valeur |
|---|---:|
| S0 | {context['S0']} |
| K | {context['K']} |
| r | {context['r'] * 100:.2f}% |
| sigma | {context['sigma'] * 100:.2f}% |
| T | {context['T']} an |
| N (pricing principal) | {context['N']:,} |
| Barrière (down-and-out) | {context['barrier']} |
| Fixings (call asiatique) | {context['n_fixings']} |
| Seed | {context['seed']} |

## Synthèse des prix

{context['pricing_table_md']}

## Greeks

{context['greeks_table_md']}

## Réduction de variance

{context['vr_table_md']}

## Interprétation

{context['interpretation']}

## Figures générées

- `figures/convergence_european_call.png`
- `figures/variance_reduction_comparison.png`
- `figures/sensitivity_volatility.png`
- `figures/simulated_paths_with_barrier.png`
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
