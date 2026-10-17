# Benchmarks on more open datasets

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
