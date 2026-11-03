# Mutation survivors in `scmle/data/ledger/budget.py`

mutmut 3.x, scope `scmle.data.ledger.budget*`, tests
`tests/unit/test_error_budget.py` and
`tests/unit/test_error_budget_boundaries.py`: 310 mutants, 309 killed, 1
surviving. The survivor is equivalent and is deliberately left unkilled.

| Mutant | Change | Why it cannot change behaviour |
| --- | --- | --- |
| `x_split_budget__mutmut_25` | `sum(exact_weights.values(), Fraction(0))` becomes `sum(exact_weights.values(), )` | `weights` is checked non-empty above, so the sum has at least one `Fraction` term. `sum` then starts from the int `0` and `0 + Fraction(x)` is exactly `Fraction(x)`. The two totals are equal in value and type for every reachable input. The explicit `Fraction(0)` is only a typing aid. |

Before this change 44 mutants survived (comparison: `initial < 1` in
`ExplorationBudget` and `wealth < 1` in `create`; sign: `cost <= 0` in `spend`
and the `-` to `+` in the harmonic series; boundary: `n >= EXACT_HARMONIC_LIMIT`;
constants in the series; the rest were error-message and field-name
mutations). All are now killed by exact-boundary and exact-message tests.
