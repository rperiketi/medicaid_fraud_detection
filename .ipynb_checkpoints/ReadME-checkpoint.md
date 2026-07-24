🏥 Medical Insurance Provider Fraud Detection
CMS Medicare Part B Billing Anomaly Analysis

📌 Project Overview
This project analyzes 100,000 Medicare Part B billing records sourced from the CMS Open Data API to detect anomalies in provider billing patterns and flag potential fraudulent activity. The pipeline covers end-to-end data work — from raw API ingestion through cleaning, exploratory analysis, rule-based fraud flagging, statistical profiling, and machine learning classification.
Data Source: CMS Medicare Physician & Other Practitioners Dataset
Rows Loaded: 100,000 (paginated, 5,000 per batch)
Columns: 28 (after cleaning and standardization)
Provider Specialties: 98 distinct types
States / Territories: 52

🗂️ Pipeline Structure
#BlockDescription1Load CMS DatasetPaginated API fetch, 20 batches × 5,000 rows2Data ProfilingMissing values, blank strings, duplicates3Clean DataSnake_case renaming, type casting, null handling4EDA — Provider TypesTop 10 by total services (bar chart)5EDA — States by ChargeTop 10 states by avg submitted charge (bar chart)6EDA — Charge vs Payment ScatterBy place of service (facility vs office)7Top 5 ProvidersComposite activity score ranking8Top 5 Provider TypesComposite activity score by specialty9Fraud Detection — Risk FlagsRule-based flagging (ratio + utilization)10Distribution — Services per BeneficiaryHistogram + box plot, percentile stats11Fraud Viz — StatesTop 10 states by high-risk provider count12Fraud Viz — SPB Top 15Services per beneficiary outliers (red bars)13Fraud Viz — Fraud Scatter4-tier color-coded risk scatter plot14Fraud Viz — Box PlotCharge-payment ratio by place of service15Random Forest ClassifierML fraud detection model + evaluation16Final InsightsSummary findings and recommendations

🧹 Data Cleaning Summary
ActionDetailColumn renamingAll 28 columns standardized to snake_caseBlank string handling"" → NaN → filled with "N/A" or "Unknown"Type conversion7 billing columns cast from str → float64 / int64Duplicates removed0 (dataset was clean)Final shape100,000 rows × 28 columns, zero nulls

🚨 Fraud Detection Logic
Two independent flags are engineered from billing features:
Flag 1 — potential_fraud (Charge Inflation)
charge_payment_ratio = avg_submitted_charge / avg_medicare_payment_amt
threshold            = 3 × median(charge_payment_ratio) = 11.58
potential_fraud      = "High Risk" if ratio > 11.58 else "Normal"

Flag 2 — utilization_flag (Over-Billing per Patient)
services_per_beneficiary = total_services / total_beneficiaries
threshold                = 99th percentile = 60.71
utilization_flag         = True if services_per_beneficiary > 60.71

Combined Risk Summary
FlagCount% of DatasetHigh Charge Ratio8,0848.1%High Utilization1,0001.0%⚠️ Both Triggered1590.2%

🤖 Machine Learning — Random Forest Classifier
Features (6 raw billing columns — no derived ratios)

avg_submitted_charge
avg_medicare_payment_amt
avg_medicare_allowed_amt
avg_medicare_standardized_amt
total_services
total_beneficiaries

Model Configuration
ParameterValueAlgorithmRandom ForestTrees / Max Depth200 / 12Class WeightBalancedTrain / Test Split80,000 / 20,000 (stratified)Target Variablepotential_fraud (High Risk / Normal)
Performance Metrics
MetricScoreAccuracy99.31%Precision92.68%Recall99.38%F1 Score95.91%ROC-AUC0.9998
Confusion Matrix (20,000 test rows)
Predicted NormalPredicted High RiskActual Normal✅ 18,237 TN❌ 146 FPActual High Risk⚠️ 10 FN✅ 1,607 TP
Feature Importances
RankFeatureImportance🥇avg_submitted_charge65.4%2avg_medicare_payment_amt11.9%3avg_medicare_standardized_amt11.6%4avg_medicare_allowed_amt8.3%5total_services2.0%6total_beneficiaries0.8%

🔑 Key Findings
1. Charge Inflation Is Systemic
The median provider charges 3.86× what Medicare pays. The top 8.1% exceed 11.58×. This is a structural pattern across the dataset, not isolated incidents.
2. Hematology-Oncology Is the Highest-Risk Specialty
15 of the top 20 dual-flagged providers are oncologists. Charge-to-payment ratios range from 99× to 675×, with Medicare paying as little as 0.05–0.05–0.05–0.06 per claim. This pattern is consistent with administration code inflation on drug infusion billing.
3. Services-per-Beneficiary Outliers Are Extreme
Population median SPB is 1.07. Top outliers:

Souza, A. (Nurse Practitioner): 349.8 SPB
Eye Consultants of Atlanta (ASC): 298.8 SPB
Western Pacific Med-Corp (Opioid Treatment): 271.5 SPB

Values exceeding 100 SPB represent billing one patient more than 100 times — a critical anomaly signal.
4. Geographic Hotspots
TX → CA → FL → NY account for 35.7% of all high-risk providers. Florida's #3 position aligns with historical CMS enforcement data. TX leads with 944 high-risk providers.
5. Facility Billing Is More Inflated Than Office Billing

Facility median charge ratio: 4.97
Office median charge ratio: 3.32

Facility-based claims carry structurally higher charge inflation, independent of provider specialty.

⚠️ Limitations
LimitationDetailSample size100K rows from a dataset of millions — findings may not generalize fullyLabel qualityFraud labels are rule-derived proxies, not confirmed OIG/CMS enforcement labelsNo temporal analysisSingle snapshot — no trend or spike detection over timeNo peer benchmarkingNo specialty-specific baseline — some specialties (oncology, dialysis) naturally have higher visit frequencyNo HCPCS analysisUnbundling and upcoding require procedure code-level reviewCircular label riskcharge_payment_ratio was used to derive the target label and is excluded from the final model; however, model still benefits from correlated raw features

🔭 Recommended Next Steps

Cross-reference flagged NPIs against the OIG Exclusion List for confirmed bad actors
HCPCS code-level analysis — detect unbundling (billing separately for bundled procedures) and upcoding (billing higher-intensity codes)
Specialty peer benchmarking — flag providers >2σ above peers in the same specialty + state cohort to reduce false positives
Multi-year CMS data — load full historical dataset to detect temporal billing spikes
Network analysis — identify coordinated fraud rings via shared addresses, referring providers, or billing agents
Confirmed ground truth labels — integrate CMS/OIG enforcement actions as true fraud labels for a fully supervised model


🛠️ Tech Stack
ComponentLibrary / ToolData Ingestionrequests, pandasData Cleaningpandas, numpyVisualizationmatplotlibMachine Learningscikit-learn (RandomForestClassifier)