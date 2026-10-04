# Project Report

## 1. Project identity

**Project name:** Electricity Consumption Intelligence

**Project title:** Consumer Profiling, Load Anomaly Detection, PCA, CBLOF, and Rooftop Solar Suitability

**Team:**

| Member | Student ID | Actual contribution |
| --- | --- | --- |
| Bony Thomas | 25280041 | Data validation, feature engineering, exploratory analysis, and profiling |
| Bhavin Kiritkumar Patel | To be completed | KNN/Decision Tree classification, tuning, evaluation, and anomaly analysis |
| Mayurkumar Bholabhai Bhalani  | To be completed | Solar scenarios, Streamlit dashboard, integration, and deployment documentation |

## 2. Abstract

This project develops an electricity-consumption intelligence workflow using consumer-level records from the Hansot subdivision. The workflow combines data analysis with course-aligned machine-learning methods: KNN classification, Decision Tree classification, K-Means clustering, PCA, and an in-project CBLOF-style anomaly detector. Monthly consumption and contract-load information are converted into behavioral features. A future-period proxy uses the first seven months for predictors and the final three months for the outcome, because the working dataset does not contain connection dates or post-connection histories. Hyperparameter tuning is used to compare KNN and Decision Tree performance. K-Means and PCA support consumer profiling, while CBLOF and a transparent load-ratio formula produce review priorities. Rooftop solar is presented as preliminary scenario analysis rather than certified technical feasibility.

## 3. What the project is about

Utilities and energy researchers need ways to summarize large billing datasets, identify consumers whose behavior deserves review, and estimate whether solar scenarios are worth investigating. The project addresses four connected questions:

1. What consumption patterns and consumer segments appear in the available monthly data?
2. Can early-period behavior classify a high-consumption outcome in a later period?
3. Which records are unusual relative to contract load or comparable behavioral peers?
4. What does a parameterized solar cost-benefit scenario look like under explicit assumptions?

The project does not use electricity consumption alone to claim socioeconomic development. It does not claim that an anomaly proves theft, unauthorized load, vacancy, or equipment failure.

## 4. Data assessment

The private working workbook contains 13,650 records and ten monthly consumption columns covering November 2025 through August 2026. The generated dataset summary records 2,306 missing monthly cells, 20 feeders, 50 villages, 562 consumers marked as solar, and 13,088 marked as non-solar. The source workbook is excluded from this public repository.

| Field or capability | Available in working data | Use or limitation |
| --- | --- | --- |
| Consumer label | Yes, anonymized in development outputs | Grouping and result display |
| Area, feeder, village | Yes | Filtering and profiling |
| Tariff | Yes | Classification feature and dashboard filter |
| Contract load in kW | Yes | Load-ratio benchmark and model feature |
| Monthly consumption | Yes | Features, target proxy, profiling, and solar input |
| Connection date | No | Genuine new-connection prediction is not validated |
| Billing start/end dates | No | Bimonthly allocation is not possible from this workbook |
| Contract demand in kVA | No | No kVA-to-kW conversion is performed |
| Measured peak demand | No | Physical load factor cannot be calculated |
| Property floor area | No | Consumption per square metre is not calculated |
| Rooftop area/shading | No | Solar output is preliminary only |
| Verified anomaly labels | No | Anomaly accuracy cannot be claimed |

Here we have considered only residential consumers for this specific project.The consumers of all other categories can be included and various other features such as last payment, arrears, current meter status, regulatory constraints etc can be added as a future part of this project

## 5. Data preparation and feature engineering

The loader reads the Excel workbook and coerces the monthly columns to numeric values. The first seven monthly columns form the predictor period and the final three form the future evaluation period. The following features are created:

- first-seven-month average, standard deviation, minimum, maximum, and total;
- coefficient of variation, with zero averages protected from division;
- consumption per contracted kW;
- future-period average and total for target construction only.

Missing monthly cells are not described as zero. The current feature calculations use pandas aggregation behavior that skips missing values; the final analysis should add a missingness indicator and a sensitivity comparison if missingness is material.

### Bimonthly billing decision

The requested bimonthly adjustment cannot be implemented from the supplied workbook because billing dates and billing-period metadata are absent. Alternate-month zeros must therefore not be silently redistributed. A future dataset with confirmed two-month billing periods should preserve the original reading, add an adjusted reading and adjustment flag, and allocate a confirmed two-month total equally or by documented day weights while preserving total energy.

## 6. Classification target and methodology

The data does not support a genuine new-connection model. Instead, the project uses a clearly labelled forecasting proxy:

- predictor period: the first seven months;
- outcome period: the final three months;
- threshold: the 75th percentile of first-seven-month average consumption;
- target: 1 when future three-month average consumption reaches or exceeds that threshold.

This target is defined from the project development data and should not be described as a validated new-connection probability. The split used in the saved development run is a stratified 75/25 held-out split with `random_state=42`. A chronological or repeated-consumer-aware evaluation should be added if future data supports it.

### Implemented algorithms

**KNN classification.** KNN uses standardized numeric features and one-hot encoded categorical features. It is suitable as an interpretable distance-based course baseline when feature scaling is explicit.

**Decision Tree classification.** The tree handles nonlinear relationships and provides a model that can be inspected through feature behavior. Class balancing and grid-search tuning are included.

**K-Means clustering.** Standardized behavioral/load features are clustered for consumer profiling. Candidate values of K from 2 through 6 are compared using silhouette score.

**PCA.** PCA reduces six standardized behavioral/load features to two components for visualization. It is a visualization and dimensionality-reduction method, not the classification target model.

**CBLOF-style anomaly detection.** The in-project implementation uses K-Means cluster structure, large/small cluster rules, and distance-based scores. The top 1% of scores are a configurable review set, not a validated anomaly detector.

## 7. Load-ratio anomaly baseline

The project uses the supplied assumptions:

```text
benchmark energy (kWh) = contract load (kW) x 8 hours/day x billing days x 0.8
consumption ratio = actual energy / benchmark energy
```

The factor 0.8 is an assumed average-to-contracted-load ratio for the reference scenario. It is not a measured conventional load factor, because measured peak demand and a complete reporting interval are unavailable. A kVA demand value must not be treated as kW without a stated power-factor assumption.

The dashboard labels low and high ratios as review priorities. It does not assert theft, unauthorized load, or a meter problem. Sensitivity analysis should compare alternative usage durations, such as 8, 12, and 24 hours, when the final dataset is available.

## 8. Rooftop solar analysis

The dashboard includes annual generation, on-site use, export, gross cost, subsidy, maintenance, annual savings, and simple payback fields. The supplied project assumptions are treated as scenario inputs rather than verified current prices or policy:

- 3 kW system cost: INR 80,000;
- above 3 kW rate: INR 50,000 per kW under an explicitly named scenario;
- subsidy: INR 78,000 up to 3 kW, subject to verification.

The final team must clarify whether the 3 kW cost is before or after subsidy, whether the above-3 kW rate applies to the entire system or only incremental capacity, whether subsidy is capped for larger systems, and what applies below 3 kW. Site-specific yield, daytime consumption share, export compensation, usable roof area, shading, losses, degradation, maintenance, lifetime, and discount rate must be verified or parameterized. Monthly consumption alone cannot establish technical feasibility.

## 9. Metrics used to evaluate the models

The saved classification files report accuracy, precision, recall, F1, and ROC-AUC. K-Means is evaluated with silhouette score. PCA is reported with explained-variance ratio. CBLOF is summarized by a configurable percentile review count because verified anomaly labels are not available.

For a stronger final evaluation, add PR-AUC and Brier score/calibration for probabilities, precision at a fixed review budget for verified anomaly labels, and chronological/repeated-consumer holdout checks. Do not enter metrics that were not produced by an executed experiment.

## 10. Results from the supplied development run

### Classification

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| KNN baseline | 0.8148 | 0.7794 | 0.7300 | 0.7539 | 0.9005 |
| KNN tuned | 0.8400 | 0.8166 | 0.7587 | 0.7866 | 0.9080 |
| Decision Tree baseline | 0.9021 | 0.8430 | 0.9193 | 0.8795 | 0.9647 |
| Decision Tree tuned | 0.9045 | 0.8587 | 0.9027 | 0.8801 | 0.9651 |

### Profiling and anomaly outputs

| Output | Development result |
| --- | ---: |
| Selected K | 2 |
| Best silhouette score | 0.9913 |
| PCA PC1 explained variance | 0.4997 |
| PCA PC2 explained variance | 0.1834 |
| CBLOF review records at 99th percentile | 137 |

## 11. How performance was improved

The project compares untuned baselines with grid-search-tuned KNN and Decision Tree models. KNN tuning searches over number of neighbors, distance weighting, and Minkowski distance parameter. Decision Tree tuning searches over maximum depth and minimum samples per leaf. Cross-validated F1 is used to select configurations within the declared search spaces.

| Model | Baseline F1 | Tuned F1 | Change |
| --- | ---: | ---: | ---: |
| KNN | 0.7539 | 0.7866 | +0.0327 |
| Decision Tree | 0.8795 | 0.8801 | +0.0006 |

The improvement is modest for the tree and stronger for KNN. This is a measured result from the saved development artifacts, not a claim that tuning always improves performance. Future iterations should test missingness features, calibration, time-aware validation, threshold selection, and error slices by tariff, feeder, village, and solar status.

## 12. Dashboard and implementation

`app.py` provides the following tabs:

- Overview: consumer count, average monthly consumption, solar count, feeder count, and monthly trend.
- Consumption: tariff-level summaries and consumer statistics.
- Load Anomalies: contract-load benchmark, ratio, and review label.
- PCA / Clusters: saved PCA visualization and clustering metrics.
- Solar: parameterized preliminary solar scenario.

The dashboard accepts only an authorized/anonymized Excel upload. The public repository includes no source workbook.

## 13. Limitations and future scope

The main limitations are missing billing-period metadata, connection dates, verified anomaly labels, measured peak demand, kVA demand, property area, rooftop information, tariff/export parameters, and site-specific solar yield. The current classification split is random and evaluates a project-defined proxy. Missing monthly cells are present and deserve explicit missingness sensitivity analysis.

Future scope includes confirmed bimonthly allocation, connection-date cohorts, post-connection outcomes, interval data, measured peak demand, verified field labels, probability calibration, privacy-preserving aggregation, and a technically reviewed solar feasibility module.

## 14. Requirement-to-file mapping

| Submission requirement | Repository evidence |
| --- | --- |
| Public code | `app.py`, `src/`, `requirements.txt` |
| Project abstract and description | This report and `README.md` |
| Approach and methodology | Sections 5 through 8 |
| Metrics | Section 9 and `results/model_metrics*.csv` |
| Experiments and tuning | Section 11 and `results/hyperparameter_experiments.csv` |
| Figures | `results/figures/` |
| Editable report | `docs/project_report.docx` |

## 15. Final publication checklist

- [ ] Replace team placeholders with real names, IDs, and actual contributions.
- [ ] Confirm data permission and anonymization before making the repository public.
- [ ] Inspect the Git diff for private workbooks and sensitive identifiers.
- [ ] Re-run training from the permitted workbook and record the command/date.
- [ ] Reconcile generated metrics with this report.
- [ ] Add course/syllabus confirmation for the two required ML algorithms.
- [ ] Add PR-AUC, calibration, and time-aware evaluation if required by the marking rubric.
