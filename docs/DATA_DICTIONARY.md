# Data dictionary

The supplied workbook is private and is not committed. The training script currently expects these source columns:

| Column | Role | Notes |
| --- | --- | --- |
| `Consumer Name` | Identifier | Must be anonymized before publication |
| `AREA` | Geographic area | Not property floor area |
| `FEEDER_NAME` | Feeder | Area-level grouping |
| `VILLAGE_NAME` | Village/locality | Dashboard filter and feature |
| `TARRIF` | Tariff/category field | Preserve source spelling unless mapping is documented |
| `CONTRACT_LOAD` | Contracted load | Treat as kW only when the source confirms the unit |
| `SOLAR_CONSUMER` | Solar indicator | Source flag; not a technical feasibility label |
| Monthly columns | Consumption | November 2025 through August 2026 in the supplied run |

The current data does not provide billing start/end dates, billing-period metadata, connection dates, contracted demand in kVA, measured peak demand, property floor area, or verified anomaly labels. These fields should be added to a future schema only with a documented source and permission.
