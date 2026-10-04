# Data dictionary

The training script currently expects these source columns:

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
