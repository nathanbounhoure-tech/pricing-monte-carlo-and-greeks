# Rapport final — Pricing Monte Carlo et Greeks

Rapport généré automatiquement par `run_all.py`.

## Calibration utilisée

| Paramètre | Valeur |
|---|---:|
| S0 | 100.0 |
| K | 100.0 |
| r | 2.00% |
| sigma | 20.00% |
| T | 1.0 an |
| N (pricing principal) | 50,000 |
| Barrière (down-and-out) | 80.0 |
| Fixings (call asiatique) | 12 |
| Seed | 42 |

## Synthèse des prix

| produit                     | estimateur                              |     prix |   erreur_type |   reference_bs |
|:----------------------------|:----------------------------------------|---------:|--------------:|---------------:|
| Call européen               | Antithétique + contrôle (stock)         | 8.912794 |      0.036823 |       8.916037 |
| Put européen                | Antithétique + contrôle (stock)         | 6.932662 |      0.036823 |       6.935905 |
| Call asiatique arithmétique | Antithétique + contrôle (call européen) | 5.378000 |      0.017433 |     nan        |
| Down-and-out call           | Antithétique + contrôle (call européen) | 8.840028 |      0.004554 |     nan        |

## Greeks

| produit           | méthode                |     delta |    gamma |      vega |      theta |
|:------------------|:-----------------------|----------:|---------:|----------:|-----------:|
| Call européen     | Analytique             |  0.579260 | 0.019552 | 39.104269 |  -4.890626 |
| Put européen      | Analytique             | -0.420740 | 0.019552 | 39.104269 |  -2.930228 |
| Call asiatique    | Bump-and-revalue (CRN) |  0.550529 | 0.032725 | 23.906327 | nan        |
| Down-and-out call | Bump-and-revalue (CRN) |  0.589556 | 0.013626 | 36.717774 | nan        |

## Réduction de variance

| produit           |   ic_largeur_brut |   ic_largeur_reduit |   reduction_pct |
|:------------------|------------------:|--------------------:|----------------:|
| Call européen     |          0.243144 |            0.144344 |       40.634278 |
| Put européen      |          0.170292 |            0.144344 |       15.237183 |
| Call asiatique    |          0.142607 |            0.068337 |       52.080039 |
| Down-and-out call |          0.243602 |            0.017850 |       92.672575 |

## Interprétation

- Le Monte Carlo brut converge vers Black-Scholes sur les produits européens
  (écart de 0.0102 sur le call, très inférieur à une erreur-type), ce qui valide
  la chaîne de simulation.
- Le call asiatique (5.3780) est nettement moins cher que le call européen
  (8.9128) : la moyenne arithmétique sur 12 dates de fixing réduit la
  volatilité effective du payoff.
- Le down-and-out call (8.8400) vaut moins que le call vanille de même strike, mais
  l'écart (0.8% seulement) est plus faible que ne le
  suggérerait la probabilité de franchissement de la barrière à 80 (~25% des trajectoires
  la touchent). La raison : les trajectoires qui franchissent la barrière ont, en moyenne, un payoff
  terminal proche de zéro (0.33 contre 9.15 en moyenne globale) — une
  trajectoire qui chute 20% sous le spot revient rarement au-dessus du strike ATM en un an. La
  probabilité de franchissement seule surestime donc fortement l'impact sur le prix.
- La variable de contrôle (combinée aux variables antithétiques) améliore fortement la précision :
  réduction de la largeur de l'IC à 95% de 40.6% (call), 15.2% (put),
  52.1% (asiatique) et 92.7% (barrière) — l'effet est particulièrement spectaculaire
  sur la barrière car le call européen (contrôle) est fortement corrélé au payoff quand la barrière
  n'est pas touchée.


## Figures générées

- `figures/convergence_european_call.png`
- `figures/variance_reduction_comparison.png`
- `figures/sensitivity_volatility.png`
- `figures/simulated_paths_with_barrier.png`
