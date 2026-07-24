# Medical Insurance Provider Fraud Detection
CMS Medicare Billing Anomaly Analysis

## 📌 Project Overview

This project analyzes 100,000 Medicare Part B billing records sourced from the CMS Open Data API to detect anomalies in provider billing patterns and flag potential fraudulent activity. The pipeline covers end-to-end data work — from raw API ingestion through cleaning, exploratory analysis, data-quality investigation, rule-based fraud flagging with peer-group normalization, and a non-circular machine learning classifier.

Data Source: CMS Medicare Physician & Other Practitioners Dataset

Rows Loaded: 100,000 (paginated, 5,000 per batch)

Columns: 28 (after cleaning and standardization)

Provider Specialties: 98 distinct types

States / Territories: 52 (military postal codes AA/AE/AP excluded from state-level rollups — see Data Quality Investigation below)

This notebook contains **two analytically distinct sections that are easy to conflate**:

1. **" Provider Types" (composite activity score)** — a pure billing-**volume** ranking (services, beneficiaries, charge, claim lines). It is explicitly **not** a fraud signal: a large, entirely legitimate high-volume provider scores identically to a fraudulent one. Treat it as descriptive EDA, not risk output.

<img width="993" height="492" alt="Screenshot 2026-07-24 at 4 34 25 PM" src="https://github.com/user-attachments/assets/6db3bc05-814d-4187-83b6-2faa53f5083d" />
   
2. **"Fraud Detection — Risk Flags" onward** — the actual anomaly-based logic (charge-to-payment ratio, services-per-beneficiary, peer-adjusted z-scores, Random Forest). This is where anything resembling "fraud risk" lives.


🗂️ Pipeline Structure
| # | Block | Description |
|---|---|---|
| 1 | Load CMS Dataset | Paginated API fetch, 20 batches × 5,000 rows |
| 2 | Data Profiling | Missing values, blank strings, duplicates |
| 3 | Clean Data | Snake_case renaming, type casting, null handling |
| 4 | Top 5 Providers | Composite **activity** score ranking (volume-based, not fraud) |
| 5 | Top 5 Provider Types | Composite activity score by specialty (volume-based, not fraud) |
| 6 | EDA — Provider Types | Top 10 by total services (bar chart) |
| 7 | EDA — States by Charge | Top 10 states by volume-weighted avg submitted charge, non-state codes excluded |
| 8 | EDA — Charge vs Payment Scatter | By place of service (facility vs office) |
| 9 | Fraud Detection — Risk Flags | Rule-based flagging (global ratio + utilization thresholds) **plus** specialty peer-relative z-score flags |
| 10 | Fraud Detection — Global vs Peer Comparison | How many providers flip risk status once compared to their own specialty instead of one global threshold |
| 11 | Fraud Viz — Charge/Payment Box Plot | Charge-payment ratio by place of service |
| 12 | Fraud Viz — SPB Top 15 | Services-per-beneficiary outliers, colored by peer-relative anomaly severity (continuous z-score gradient, not a binary global-threshold split) |
| 13 | Fraud Viz — States | Top 10 states by high-risk **rate** (not raw count, to avoid confounding by claim volume) |
| 14 | Data Quality Investigation | Explains the SPB outliers and the excluded military postal codes |
| 15 | Distribution — Services per Beneficiary | Histogram + box plot, percentile stats |
| 16 | Random Forest Classifier | Non-circular ML fraud model + evaluation |
| 17 | Fraud Viz — Fraud Scatter | 4-tier color-coded risk scatter plot |

🧹 Data Cleaning Summary
| Action | Detail |
|---|---|
| Column renaming | All 28 columns standardized to snake_case |
| Blank string handling | `""` → NaN → filled with "N/A" or "Unknown" |
| Type conversion | 7 billing columns cast from str → float64 / int64 |
| Duplicates removed | 0 (dataset was clean on exact-row duplicates; note "clean" here only means no full-row dupes and no blank/NaN cells post-fill — see Data Quality Investigation for cells that are technically valid but still worth scrutiny) |
| Final shape | 100,000 rows × 28 columns, zero nulls |

🔍 Data Quality Investigation
Before trusting downstream aggregates, two anomalies were traced back to their source rows rather than silently accepted:

**Extreme services-per-beneficiary (SPB) values.** The population max SPB is 7,934 (one HCPCS line item billed to a beneficiary count of just a few patients). Pulling the 204 rows with SPB > 500 shows they cluster heavily in Hematology-Oncology (63), Rheumatology (30), Medical Oncology (25), and Nurse Practitioner (23) — consistent with high-frequency drug-infusion administration codes billed against a small denominator, not an obvious data-entry error. This is kept in the data (no justification to drop it), but it means provider-level SPB aggregates that sum services/beneficiaries across many HCPCS rows before dividing can still be distorted if a provider's line items don't all share the same beneficiary population — a caveat carried into the "SPB Top 15" chart via peer-relative z-scores rather than a flat threshold.

**The "AE" state code.** "AE" (Armed Forces Europe, a military postal code, not a US state) briefly appeared in an earlier version of the "Top States by Avg Charge" ranking because a handful of low-volume providers there produced a high — but statistically meaningless — mean. `AA`/`AE`/`AP` are now excluded from all state-level rollups, and the states chart annotates each bar with its provider count (n=) so small-sample entries can't be mistaken for real geographic hotspots.

🚨 Fraud Detection Logic
Two engineered features feed both a **global threshold** and a **specialty peer-relative z-score**:

Flag 1 — `potential_fraud` (Charge Inflation)
```
charge_payment_ratio = avg_submitted_charge / avg_medicare_payment_amt
global threshold      = 3 × median(charge_payment_ratio) = 11.58
potential_fraud       = "High Risk" if ratio > 11.58 else "Normal"
```

Flag 2 — `utilization_flag` (Over-Billing per Patient)
```
services_per_beneficiary = total_services / total_beneficiaries
global threshold          = 99th percentile = 60.71
utilization_flag          = "High Utilization" if SPB > 60.71 else "Normal"
```

Flag 3 & 4 — Peer-relative versions (`potential_fraud_peer`, `utilization_flag_peer`)
A global threshold treats a Hematology-Oncology provider (naturally high-cost drug administration billing) the same as a Family Practice provider. Both `charge_payment_ratio` and `services_per_beneficiary` are additionally converted to a **z-score relative to the provider's own `provider_type` peer group** (falling back to the global mean/std when a specialty has fewer than 30 providers), and flagged at z > 2.

Global vs Peer Threshold — Row Counts
| Flag | Global Threshold | Peer Z-Score (>2) |
|---|---|---|
| High Charge Ratio | 8,084 (8.1%) | 1,919 (1.9%) |
| High Utilization | 1,000 (1.0%) | 1,433 (1.4%) |
| Both Triggered (global) | 159 (0.2%) | — |

Global vs Peer Overlap (High-Risk / charge-ratio flag)
| Category | Row Count | Interpretation |
|---|---|---|
| Global only | 6,601 | Likely false positives — mostly naturally high-cost specialties (e.g. oncology) where the raw ratio is unremarkable for that peer group |
| Peer only | 436 | Unusual relative to their own specialty even though the raw ratio isn't extreme globally — missed entirely by a global-only rule |
| Both | 1,483 | Anomalous under either method |

🤖 Machine Learning — Random Forest Classifier
**Circular-label problem, found and fixed.** The original model predicted `potential_fraud` — a label deterministically derived from `avg_submitted_charge / avg_medicare_payment_amt` — while also being fed both of those same raw columns as features. A tree ensemble can reconstruct the ratio from its own inputs, which is why the original run reported a suspicious ROC-AUC of 0.9998. **Both label-defining columns are now excluded from the feature set.**

Features (4 columns — excludes avg_submitted_charge & avg_medicare_payment_amt)
```
total_services
total_beneficiaries
avg_medicare_allowed_amt
avg_medicare_standardized_amt
```

Model Configuration
| Parameter | Value |
|---|---|
| Algorithm | Random Forest |
| Trees / Max Depth | 200 / 12 |
| Class Weight | Balanced |
| Train / Test Split | 80,000 / 20,000 (stratified) |
| Target Variable | `potential_fraud` (High Risk / Normal) |

Performance Metrics (non-circular feature set)
| Metric | Score |
|---|---|
| Accuracy | 74.78% |
| Precision | 17.47% |
| Recall | 56.90% |
| F1 Score | 26.73% |
| ROC-AUC | 0.7492 |

These are the honest numbers after removing label leakage — much lower than the original 99%+ figures, and that drop is the correct, expected outcome, not a regression. A ROC-AUC near 0.75 says the four remaining raw billing features carry real but modest signal about which rows get charge-ratio-flagged; it is not a strong standalone fraud detector.

Feature Importances
| Rank | Feature | Importance |
|---|---|---|
| 1 | avg_medicare_standardized_amt | 34% |
| 2 | avg_medicare_allowed_amt | 28% |
| 3 | total_services | 24% |
| 4 | total_beneficiaries | 14% |

🔑 Key Findings
1. **Charge inflation is systemic under the global rule, but partly specialty-driven.** The median provider charges 3.86× what Medicare pays, and the global rule flags 8.1% as "High Risk." Once compared against same-specialty peers instead, only 1.9% are flagged — most of the difference (6,601 rows) is naturally high-cost specialties like oncology, not necessarily anomalous behavior.
2. **Hematology-Oncology dominates the "both flags" list** even under the global rule (charge-to-payment ratios from ~100× to 675×, with Medicare paying as little as $0.05–$0.06 per claim) — consistent with administration-code inflation on drug infusion billing, and one of the few patterns that stays flagged under peer-adjustment too.
3. **Services-per-beneficiary outliers are extreme but need peer context to interpret.** Population median SPB is 1.07; the top of the list (Souza — Nurse Practitioner: 349.8, Eye Consultants of Atlanta — ASC: 298.8, Western Pacific Med-Corp — Opioid Treatment Program: 271.5) all clear the global P99 threshold by construction. Their peer z-scores tell a more differentiated story: an individual Nurse Practitioner or an Opioid Treatment Program at z≈22–32 is extreme even relative to peers doing similar work, while a Rheumatology entry at z≈3.4 is only mildly unusual for that specialty.
4. **Geographic hotspots shift once rate-normalized.** By raw high-risk count, TX (944) leads; by high-risk **rate** (high-risk rows ÷ all rows for that state), smaller states like WI (23%) and NH (22%) actually have a higher proportion of flagged billing than TX (13%) — raw counts were confounded by how many claims each state has overall.
5. **Facility billing is more inflated than office billing.** Facility median charge ratio: 4.97 vs Office median: 3.32 — this holds independent of the peer-adjustment work above.

⚠️ Limitations
| Limitation | Detail |
|---|---|
| Sample size | 100K rows from a dataset of millions — findings may not generalize fully |
| Label quality | Fraud labels are rule-derived proxies, not confirmed OIG/CMS enforcement labels |
| No temporal analysis | Single snapshot — no trend or spike detection over time |
| Peer benchmarking is partial | Specialty (`provider_type`) peer groups are used, but not combined with state or place-of-service cohorts; specialties with <30 providers fall back to the global baseline |
| No HCPCS unbundling/upcoding analysis | Procedure code-level review (bundling, upcoding) is not implemented |
| ML signal is modest, not leak-free-guaranteed | Removing the two label-defining columns fixed the direct circularity, but `total_services`/`total_beneficiaries` are still inputs to the SPB-based utilization flag, so some residual correlation with labeling logic may remain |


🛠️ Tech Stack
| Component | Library / Tool |
|---|---|
| Data Ingestion | requests, pandas |
| Data Cleaning | pandas, numpy |
| Visualization | matplotlib |
| Machine Learning | scikit-learn (RandomForestClassifier) |
