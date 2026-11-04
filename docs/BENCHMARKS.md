# Benchmarks on more open datasets

## Reliability checks

Accuracy tables say how good a model is; they do not say whether the package
keeps its promises. `python -m benchmarks.reliability` runs nine checks on
synthetic data with known ground truth and exits non-zero if any fails. Results
are written to `benchmarks/results_reliability.json` with the measured value
next to the threshold.

| Check | Claim | Passes when |
| --- | --- | --- |
| `determinism` | Same data and seed give the same result. | same model, score difference <= 1e-9 |
| `replay` | `verify_replay` recognises an unchanged rerun. | level is `exact` |
| `no_false_skill` | Noise labels do not produce a convincing score. | mean ROC-AUC gap from 0.5 <= 0.08 over 5 seeds |
| `leak_detection` | A planted leaked column is found by the certified screen. | the only suspect is the planted column, it may block, and `leakage_policy="block"` stops the run |
| `bundle_roundtrip` | A saved signed bundle predicts identically. | difference is exactly 0 |
| `tamper_detection` | Wrong key or modified file is refused. | both raise `ArtifactError` |
| `interval_coverage` | A 90% interval covers fresh rows. | observed coverage >= 0.88 (nominal minus the 0.02 tolerance); the exact interval is recorded |
| `drift_detection` | A +3 sigma shift is flagged; clean data is not. | shifted != `none`, clean == `none` |
| `failure_isolation` | A crashing candidate does not sink the run. | run `COMPLETED`, crashing trial not `ok` |

The coverage and problem-type methods have their own measured gates in
`benchmarks/qualification/uncertainty_certify.py`, written to
`benchmarks/qualification/uncertainty_certificates.json` and checked by
`python -m benchmarks.qualification --check` (stale when the code changed).
Two gates missed their pre-registered floors and are listed as descriptive:
CQR conditional coverage in the highest-noise decile and the slice flag's
familywise false-flag rate. Gates that need outside data or tools have not
run and are listed under `not_measured` in the file.

These complement, and do not replace, `tests/leakage`, the mutation gate and
the operational rehearsal ([`OPERATIONS_REHEARSAL.md`](OPERATIONS_REHEARSAL.md)).
Reproduce the accuracy tables with the commands in
[`benchmarks/README.md`](../benchmarks/README.md).

Besides the nine core datasets in the README, the framework was run on 20 tables from the Hugging Face Hub and 21 more from OpenML (`python -m benchmarks.run_benchmarks --suite huggingface|extended`, 240 s budget, 2 workers). Every number is the score on the **locked test rows**; the reference is a plain scikit-learn `HistGradientBoosting` fitted on the same development rows. Lower is better for log loss and RMSE.

## Hugging Face Hub (inria-soda/tabular-benchmark and scikit-learn tables)

Ahead of the reference on **16 of 20**.

| dataset | rows | metric | selected | test | reference | vs reference |
| --- | ---: | --- | --- | ---: | ---: | --- |
| hf_compas | 4966 | roc_auc | logistic_regression | 0.7085 | 0.6936 | ahead |
| hf_credit_default | 13272 | roc_auc | hist_gradient_boosting | 0.777 | 0.7745 | ahead |
| hf_eye_movements | 7608 | roc_auc | blending | 0.7098 | 0.7111 | behind |
| hf_albert | 15000 | roc_auc | hist_gradient_boosting | 0.7162 | 0.7151 | ahead |
| hf_bioresponse | 3434 | roc_auc | hist_gradient_boosting | 0.8733 | 0.8738 | behind |
| hf_magic_telescope | 13376 | roc_auc | hist_gradient_boosting | 0.9291 | 0.9255 | ahead |
| hf_heloc | 10000 | roc_auc | hist_gradient_boosting | 0.7976 | 0.7843 | ahead |
| hf_pol | 10082 | roc_auc | hist_gradient_boosting | 0.9991 | 0.9992 | behind |
| hf_jannis | 15000 | roc_auc | hist_gradient_boosting | 0.8663 | 0.8615 | ahead |
| hf_adult_census | 15000 | roc_auc | hist_gradient_boosting | 0.9265 | 0.9257 | ahead |
| hf_brazilian_houses | 10692 | rmse | stacking | 0.04722 | 0.04786 | ahead |
| hf_mercedes | 4209 | rmse | stacking | 8.026 | 8.291 | ahead |
| hf_medical_charges | 15000 | rmse | stacking | 0.08551 | 0.08602 | ahead |
| hf_ailerons | 13750 | rmse | blending | 0.0001943 | 0.0002184 | ahead |
| hf_miami_housing | 13932 | rmse | stacking | 0.1547 | 0.1572 | ahead |
| hf_elevators | 16599 | rmse | blending | 0.002267 | 0.00232 | ahead |
| hf_superconduct | 12000 | rmse | stacking | 9.897 | 10.37 | ahead |
| hf_sulfur | 10081 | rmse | bagged_decision_tree | 0.02058 | 0.0239 | ahead |
| hf_auto_mpg | 398 | rmse | hist_gradient_boosting | 2.41 | 2.31 | behind |
| hf_fish_weight | 159 | rmse | bagged_decision_tree | 54.53 | 64.87 | ahead |

## OpenML

Ahead of the reference on **13 of 21**.

| dataset | rows | metric | selected | test | reference | vs reference |
| --- | ---: | --- | --- | ---: | ---: | --- |
| blood_transfusion | 748 | roc_auc | logistic_regression | 0.7834 | 0.7644 | ahead |
| pima_diabetes | 768 | roc_auc | logistic_regression | 0.8298 | 0.8202 | ahead |
| phoneme | 5404 | roc_auc | stacking | 0.9486 | 0.9395 | ahead |
| spambase | 4601 | roc_auc | hist_gradient_boosting | 0.9877 | 0.9877 | behind |
| bank_marketing | 15000 | roc_auc | hist_gradient_boosting | 0.9241 | 0.9163 | ahead |
| credit_card_fraud | 40000 | pr_auc | bagged_decision_tree | 0.829 | 0.6428 | ahead |
| telco_churn | 5000 | roc_auc | stacking | 0.9165 | 0.9297 | behind |
| electricity | 12000 | roc_auc | hist_gradient_boosting | 0.9401 | 0.9411 | tie |
| credit_approval | 690 | roc_auc | hist_gradient_boosting | 0.9159 | 0.9074 | ahead |
| horse_colic | 368 | roc_auc | bagged_decision_tree | 0.8566 | 0.8235 | ahead |
| vehicle | 846 | log_loss | stacking | 0.3798 | 0.6179 | ahead |
| segment | 2310 | log_loss | hist_gradient_boosting | 0.08911 | 0.08885 | behind |
| letter | 10000 | log_loss | hist_gradient_boosting | 0.1919 | 0.1883 | behind |
| covertype | 20000 | log_loss | bagged_logistic_regression | 0.6378 | 0.6264 | behind |
| bike_sharing | 17379 | rmse | stacking | 39.45 | 40.89 | ahead |
| diamonds | 15000 | rmse | stacking | 569.6 | 583.8 | ahead |
| cpu_act | 8192 | rmse | hist_gradient_boosting | 2.377 | 2.377 | behind |
| wine_quality | 6497 | rmse | bagged_decision_tree | 0.6083 | 0.6433 | ahead |
| kin8nm | 8192 | rmse | stacking | 0.1098 | 0.1259 | ahead |
| house_sales | 12000 | rmse | hist_gradient_boosting | 1.429e+05 | 1.406e+05 | behind |
| abalone | 4177 | rmse | stacking | 2.159 | 2.27 | ahead |

## Reading the results

* Ties at the third significant digit (ailerons, elevators, spambase, cpu_act, pol) are labelled by the unrounded numbers; treat them as ties.
* Chronological electricity data trailed the reference by about 0.05 ROC-AUC while the time column was only used to order rows; the reference, like most practitioners, used it as a feature. With `time_as_feature=True` the gap is 0.001. Models that key on absolute time must extrapolate, so it stays opt-in.
* The clearest weaknesses are covertype and letter (large multiclass sets where the 240 s budget leaves little room to tune), and house_sales, where the long-tail target transform won cross-validation but lost on the locked rows.
* Wide tables (bioresponse, 419 features) exposed that the post-selection diagnostics were not bounded by the time budget; they now share a fraction of it.
* Scores come from one split and one seed per dataset, so differences in the fourth digit are within noise.

## Repeated seeds with intervals

`--seeds N` repeats each dataset with N seeds (each seed changes the split and
the models) and reports the advantage over the plain HistGradientBoosting
reference with a 95% Student-t interval. A win needs the interval to exclude
zero; otherwise the verdict is `tie`. Three seeds, 150 s budget, the four
bundled datasets (`benchmarks/results_seeds.json`):

| dataset | metric | test (mean) | reference (mean) | advantage (95% CI) | verdict |
| --- | --- | --- | --- | --- | --- |
| breast_cancer | roc_auc | 0.9961 | 0.9916 | +0.0045 [-0.0045, +0.0135] | tie |
| wine | log_loss | 0.3677 | 0.0289 | -0.3388 [-1.7371, +1.0594] | tie |
| digits | log_loss | 0.0811 | 0.0902 | +0.0091 [-0.0184, +0.0365] | tie |
| diabetes_progression | rmse | 56.3792 | 63.5163 | +7.1370 [+5.4785, +8.7956] | ahead |

With three seeds only one result is distinguishable from noise (diabetes
progression, ahead). The single-seed "behind" label on wine earlier in this
document is not supported once the spread across seeds is counted: its
interval spans zero by a wide margin because one seed scored badly.

## New task types on real datasets

`benchmarks/real_tasks.py` runs every task type added after the first release
on public data (OpenML, R datasets, the Criteo uplift sample, MSLR-WEB10k,
covertype) with a 300 s budget per case, one seed, against a simple strong
baseline scored on the same locked rows. Results: `benchmarks/results_real_tasks.json`.

| dataset | task | metric | framework | baseline | verdict | selected |
| --- | --- | --- | --- | --- | --- | --- |
| AirPassengers (monthly, 144) | forecast | mean MAE over steps | 31.46 | 40.26 (seasonal naive) | ahead | multioutput_ridge |
| nottem (monthly temperature, 240) | forecast | mean MAE over steps | 1.828 | 2.446 (seasonal naive) | ahead | multi_random_forest |
| UKgas (quarterly, 108) | forecast | mean MAE over steps | 43.84 | 40.1 (seasonal naive) | behind | multioutput_ridge |
| rotterdam breast cancer (2982) | survival | c_index | 0.6929 | 0.6825 (untuned ridge Cox) | ahead | cox_ph |
| PBC liver (418) | survival | c_index | 0.844 | 0.8218 (untuned ridge Cox) | ahead | cox_ph |
| wine quality, red (1599) | ordinal | qwk | 0.5723 | 0.5671 (multiclass boosted classifier) | ahead | ordinal_random_forest |
| yeast (2417, 14 labels) | multilabel | macro F1 | 0.4175 | 0.4195 (one boosted classifier per label) | behind | multi_hist_gradient_boosting |
| energy efficiency (768, 2 loads) | multioutput | mean R2 | 0.978 | 0.9869 (one boosted regressor per target) | behind | multi_random_forest |
| NSW job training (randomized, 445) | uplift | qini | 0.02255 | -1.896e-05 (random ranking (mean of 50)) | ahead | uplift_s_learner |
| Lalonde NSW vs PSID (observational, 614) | uplift | qini | -4.107e-05 | -0.0002036 (random ranking (mean of 50)) | ahead | uplift_s_learner |
| Criteo uplift (randomized, sampled 420k) | uplift | qini | 0.003091 | -3.769e-05 (random ranking (mean of 50)) | ahead | uplift_s_learner |
| mslr-web10k (700 queries) | ranking | ndcg@10 | 0.5575 | 0.5583 (pointwise boosting) | behind | pointwise_ranker |
| sms-spam | text | roc_auc | 0.9889 | 0.9864 (TF-IDF terms) | ahead | logistic_regression |
| wine quality, white (4898) | ordinal | qwk | 0.5574 | 0.5231 (multiclass boosted classifier) | ahead | ordinal_hist_gradient_boosting |
| scene (2407, 6 labels) | multilabel | macro F1 | 0.8127 | 0.7949 (one boosted classifier per label) | ahead | multi_hist_gradient_boosting |
| covertype (581k rows, 50 MB budget) | streamed | accuracy on all locked rows | 0.8555 | 0.7794 (boosting on every development row) | ahead | catboost |

Reading it honestly: these are single-seed point estimates, not intervals, so
small gaps are ties (wine-red 0.572 vs 0.567; yeast 0.4175 vs 0.4195; mslr
0.5575 vs 0.5583). Clear wins: AirPassengers, nottem, wine-white, scene,
PBC, the randomized uplift sets and covertype under a 50 MB memory budget.
Clear losses: UKgas (the seasonal-naive forecast is better than the framework's
direct ridge on 108 points) and energy-efficiency (a per-target boosted
regressor is slightly better). On the observational Lalonde data the uplift
estimate rests on the no-unmeasured-confounding assumption and its Qini is
close to zero, so it should not be read as a causal result. Covertype's
baseline is a default boosted model on all development rows, which is weak, so
its margin overstates the framework's advantage.
