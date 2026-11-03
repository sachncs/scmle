# Feasibility for five dataset sizes

Charter: min_effect 0.3, K_max 5, alpha_confirm 0.05, power 0.8, confirmation fraction 0.25. Bonferroni at alpha/K_max (conservative for BH).

| development rows | n_C (25%) | n required | MDE at n_C | achieved power | verdict |
|---:|---:|---:|---:|---:|---|
| 200 | 50 | 130 | 0.483 | 0.325 | downgrade |
| 500 | 125 | 130 | 0.306 | 0.782 | downgrade |
| 2000 | 500 | 130 | 0.153 | 1.000 | feasible |
| 10000 | 2500 | 130 | 0.068 | 1.000 | feasible |
| 100000 | 25000 | 130 | 0.022 | 1.000 | feasible |
