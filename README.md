# CUSTOMER PROFILING, ANOMALY DETECTION AND COST BENEFIT ANALYTICS FOR SOLAR ROOFTOP ADOPTION IN POWER DISCOMS

**Consumer Profiling, Load Anomaly Detection, PCA, CBLOF, and Rooftop Solar Suitability**

This repository contains a course-aligned machine-learning and data-analysis capstone for electricity-consumption records from the Hansot subdivision. It includes a standalone browser dashboard, reproducible training scripts, generated evaluation tables, figures, and an editable project report.

## Abstract

The project converts monthly electricity consumption and contract-load information into interpretable features for consumer classification, behavioral profiling, anomaly screening, and preliminary rooftop-solar scenarios. KNN and Decision Tree models classify a future-period high-consumption proxy. K-Means and PCA support consumer segmentation and visualization. A CBLOF-style clustering detector and an assumption-based load-ratio baseline produce review queues. Solar calculations remain preliminary because rooftop area, shading, daytime demand, tariff, export compensation, and site-specific yield are not present in the working dataset.


## Team

- Bony Thomas, 25280041
- Bhavin Kirtikumar Patel 25280040
- Mayurkumar Bholabhai Bhalani 25280045

## What has been built

- Data loading and feature engineering for ten monthly consumption columns.
- First-seven-month behavioral features: mean, standard deviation, minimum, maximum, total, coefficient of variation, and consumption per contracted kW.
- A clearly labelled future-period classification proxy: first seven months for predictors and final three months for the outcome.
- KNN and Decision Tree classification with preprocessing pipelines and grid-search tuning.
- K-Means clustering evaluated with silhouette score.
- PCA projection for consumer-profile visualization.
- In-project CBLOF-style clustering anomaly scoring.
- Transparent load-ratio screening using `contract load (kW) x 8 hours/day x billing days x 0.8`.
- Standalone HTML dashboard tabs for overview, consumption, load anomalies, saved model results, PCA/clusters, and preliminary solar scenarios.
- Sidebar filters for tariff, area, feeder, village, and solar status, plus CSV downloads for filtered consumer and anomaly review tables.
- Saved metrics, experiment records, anomaly outputs, and figures under `results/`.

## Working dataset summary

The private working workbook is intentionally excluded from this public repository. The checked-in result summary records:

| Item | Value |
| --- | ---: |
| Consumer records | 13,650 |
| Monthly columns | 10, November 2025 to August 2026 |
| Missing monthly cells | 2,306 |
| First-seven-month high-consumption threshold | 95.7143 kWh |
| Unique feeders | 20 |
| Unique villages | 50 |
| Consumers marked solar | 562 |
| Selected K | 2 |
| Selected K silhouette score | 0.9913 |
| CBLOF records flagged at the 99th percentile | 137 |

These values describe the supplied development run only. They must be regenerated if the input workbook changes.

## Classification results

The supplied result files use the same stratified 75/25 held-out split for baseline and tuned model comparisons. The target is the project-defined future-period proxy, not a validated new-connection outcome.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| KNN baseline | 0.8148 | 0.7794 | 0.7300 | 0.7539 | 0.9005 |
| KNN tuned | 0.8400 | 0.8166 | 0.7587 | 0.7866 | 0.9080 |
| Decision Tree baseline | 0.9021 | 0.8430 | 0.9193 | 0.8795 | 0.9647 |
| Decision Tree tuned | 0.9045 | 0.8587 | 0.9027 | 0.8801 | 0.9651 |


## Used Algorithms

| Topic | Implementation |
| --- | --- |
| Distance measures | Standardized numeric behavioral features and KNN |
| K-Means | Consumer profiling and CBLOF-style anomaly support |
| KNN classification | Future-period high-consumption proxy |
| Decision Trees | Future-period high-consumption proxy |
| Hyperparameter tuning | Grid search for KNN and Decision Tree |
| PCA | Two-dimensional consumer-profile visualization |
| Anomaly detection / CBLOF | In-project clustering-based anomaly score |



##Note: While running the consumer_consumption_project_final.py file , it is requested to upload Hansot_adjusted_consumption.xlsx from the repository before running the programme
