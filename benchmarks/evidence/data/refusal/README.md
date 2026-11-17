# Real tables added for claim C14

The refusal measurement (`benchmarks/evidence/refusal.py`) reads these
public OpenML tables from scikit-learn's local cache
(`sklearn.datasets.get_data_home()/openml`), not from this folder. They were
fetched once with `sklearn.datasets.fetch_openml(data_id=ID)` from
https://www.openml.org (no credentials); the runs then read them offline.

| OpenML id | name | version | licence | target | declared task |
| --- | --- | --- | --- | --- | --- |
| 3 | kr-vs-kp | 1 | Public | `class` | classification |
| 11 | balance-scale | 1 | Public | `class` | classification |
| 15 | breast-w | 1 | Public | `Class` | classification |
| 23 | cmc | 1 | Public | `Contraceptive_method_used` | classification |
| 50 | tic-tac-toe | 1 | Public | `Class` | classification |
| 53 | heart-statlog | 1 | Public | `class` | classification |
| 59 | ionosphere | 1 | Public | `class` | classification |
| 61 | iris | 1 | Public | `class` | classification |
| 188 | eucalyptus | 1 | Public | `Utility` | classification |
| 307 | vowel | 2 | Public | `Class` | classification |
| 333 | monks-problems-1 | 1 | Public | `class` | classification |
| 1462 | banknote-authentication | 1 | Public | `Class` | classification |
| 1480 | ilpd | 1 | Public | `Class` | classification |
| 1494 | qsar-biodeg | 1 | Public | `Class` | classification |
| 1049 | pc4 | 1 | Public | `c` | classification |
| 1063 | kc2 | 1 | Public | `problems` | classification |
| 1068 | pc1 | 1 | Public | `defects` | classification |
| 40982 | steel-plates-fault | 3 | Public | `target` | classification |
| 40983 | wilt | 2 | Public | `class` | classification |
| 40994 | climate-model-simulation-crashes | 4 | Public | `outcome` | classification |
| 1487 | ozone-level-8hr | 1 | Public | `Class` | classification |
| 197 | cpu_act | 1 | Public | `usr` | regression |
| 216 | elevators | 1 | Public | `Goal` | regression |
| 218 | house_8L | 1 | Public | `price` | regression |
| 507 | space_ga | 1 | Public | `ln(VOTES/POP)` | regression |
| 503 | wind | 1 | Public | `MAL` | regression |
| 196 | autoMpg | 1 | Public | `class` | regression |
| 230 | machine_cpu | 1 | Public | `class` | regression |
| 315 | us_crime | 1 | Public | `ViolentCrimesPerPop` | regression |
| 41021 | Moneyball | 2 | Public | `RS` | regression |
| 560 | bodyfat | 1 | Public | `class` | regression |
