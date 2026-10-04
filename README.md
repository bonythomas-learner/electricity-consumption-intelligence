# Electricity Consumption Intelligence

**Consumer Profiling, Load Anomaly Detection, PCA, CBLOF, and Rooftop Solar Suitability**

This repository contains a course-aligned machine-learning and data-analysis capstone for electricity-consumption records from the Hansot subdivision. It includes a Streamlit dashboard, reproducible training scripts, generated evaluation tables, figures, and an editable project report.

## Abstract

The project converts monthly electricity consumption and contract-load information into interpretable features for consumer classification, behavioral profiling, anomaly screening, and preliminary rooftop-solar scenarios. KNN and Decision Tree models classify a future-period high-consumption proxy. K-Means and PCA support consumer segmentation and visualization. A CBLOF-style clustering detector and an assumption-based load-ratio baseline produce review queues. Solar calculations remain preliminary because rooftop area, shading, daytime demand, tariff, export compensation, and site-specific yield are not present in the working dataset.

The project does not claim to predict a genuinely new connection: connection dates and post-connection histories are unavailable. It also does not claim that an anomaly proves theft, unauthorized load, vacancy, or a meter fault.

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
- Streamlit dashboard tabs for overview, consumption, load anomalies, saved model results, PCA/clusters, and preliminary solar scenarios.
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

The table is evidence of the saved development run, not a guarantee of future performance. PR-AUC, calibration, Brier score, and a chronological/repeated-consumer evaluation should be added if required by the final marking rubric.

## Course alignment

| Course topic | Implementation |
| --- | --- |
| Distance measures | Standardized numeric behavioral features and KNN |
| K-Means | Consumer profiling and CBLOF-style anomaly support |
| KNN classification | Future-period high-consumption proxy |
| Decision Trees | Future-period high-consumption proxy |
| Hyperparameter tuning | Grid search for KNN and Decision Tree |
| PCA | Two-dimensional consumer-profile visualization |
| Anomaly detection / CBLOF | In-project clustering-based anomaly score |

## Run locally

This is a Streamlit application. Start it with `streamlit run app.py`, not with `python app.py`.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src\train_models.py --input data\Hansot_adjusted_consumption.xlsx --output results
streamlit run app.py
```

Place only an authorized and anonymized workbook at `data/Hansot_adjusted_consumption.xlsx` and the app will load it automatically. You can also upload the workbook from the dashboard sidebar. The real workbook is ignored by Git. The expected workbook columns are documented in [data/README.md](data/README.md) and [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md).

## Repository structure

```text
.
├── app.py                         # Streamlit dashboard
├── data/                          # Data instructions; private workbook excluded
├── docs/project_report.docx       # Editable report supplied with the project
├── notebooks/                     # Notebook notes and extension space
├── results/                       # Development metrics, outputs, and figures
├── src/data_utils.py              # Loading, features, and target construction
├── src/train_models.py            # KNN, tree, K-Means, PCA, CBLOF, and baseline
├── src/cblof.py                   # In-project CBLOF-style detector
├── src/solar.py                   # Parameterized solar scenarios
├── PROJECT_REPORT.md              # Public Markdown report and submission checklist
├── requirements.txt
└── .gitignore
```

## Data protection

Do not commit the private workbook, account numbers, meter numbers, addresses, phone numbers, or other personal/sensitive fields. Consumer identifiers visible in development outputs are placeholders/anonymized labels and should be reviewed before publication. The report must state the dataset permission and anonymization method used by the team.

## Limitations

- Billing dates and billing-period metadata are unavailable, so bimonthly allocation cannot be scientifically implemented from this workbook.
- Connection dates and post-connection history are unavailable, so the classifier is a future-period proxy rather than a new-connection forecast.
- Verified anomaly investigation labels are unavailable; CBLOF and load-ratio flags are review priorities only.
- Contract demand in kVA and measured peak demand are unavailable. kVA must not be silently treated as kW.
- Missing monthly cells are present and are handled through the current feature calculations; the missingness treatment should be reviewed for the final analysis.
- Rooftop area, shading, daytime load, tariff/export rules, and site-specific solar yield are unavailable.

## Report

The detailed submission outline, actual development metrics, limitations, future work, viva questions, and three-member contribution table are in [PROJECT_REPORT.md](PROJECT_REPORT.md). Replace contribution placeholders with the team's actual names and work before submission.
