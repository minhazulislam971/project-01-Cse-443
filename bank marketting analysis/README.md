# Term Deposit Uptake Prediction with Logistic Regression

## Objective

Banks invest heavily in telemarketing to promote term deposits. This project applies **Logistic Regression** to historical campaign data so that the bank can estimate, in advance, which customers are most likely to open a term deposit.

## Dataset Details

The data comes from the **UCI Bank Marketing dataset** — **45,211 customer records** across **17 fields**.

| Field | Kind | What it captures |
|-------|------|------------------|
| age | Numeric | Customer's age in years |
| job | Categorical | Employment category |
| marital | Categorical | Marriage status |
| education | Categorical | Highest qualification |
| default | Binary | Credit default flag |
| balance | Numeric | Mean annual balance (EUR) |
| housing | Binary | Housing-loan flag |
| loan | Binary | Personal-loan flag |
| contact | Categorical | Communication channel used |
| day | Numeric | Calendar day of last call |
| month | Categorical | Calendar month of last call |
| duration | Numeric | Length of last call (sec) |
| campaign | Numeric | Contacts made in this campaign |
| pdays | Numeric | Days elapsed since prior campaign contact |
| previous | Numeric | Contacts made in earlier campaigns |
| poutcome | Categorical | Result of the prior campaign |
| **y** | **Binary** | **Opened a term deposit? yes / no** |

## Methodology

**Phase 1 — Exploration:**
Inspected data quality (no nulls, no duplicates). Plotted feature distributions — histograms for numerics, bar charts for categoricals. Examined how each attribute relates to the target via stacked proportion charts and box plots. Computed a correlation heatmap and isolated the strongest predictors.

**Phase 2 — Preprocessing:**
Converted binary and multi-level categorical fields into numeric codes (manual mapping for yes/no fields, `LabelEncoder` for the rest). Split the data 75 / 25 with stratification. Normalised features with `StandardScaler`.

**Phase 3 — Modelling:**
Fitted two Logistic Regression classifiers (solver: `liblinear`, C = 0.8):
- *Standard* — equal class weights.
- *Weighted* — `class_weight='balanced'` to compensate for the ~88 / 12 imbalance.

**Phase 4 — Evaluation:**
Compared both variants on Accuracy, Precision, Recall, F1, and ROC-AUC. Plotted confusion matrices and the ROC curve. Checked robustness via 5-fold stratified cross-validation.

## Findings

1. **Imbalance effect** — ~88 % of customers did not subscribe, so high accuracy alone does not mean the model is useful; recall and AUC matter more.
2. **Call duration dominates** — the length of the last marketing call is, by far, the strongest predictor of conversion.
3. **Prior success recurs** — a customer who subscribed in a previous campaign is very likely to subscribe again.
4. **Demographic sweet spots** — retirees, students, single individuals, and those with a tertiary education show above-average conversion.
5. **Channel matters** — cellular contact outperforms telephone and unknown.
6. **Weighted model trade-off** — it recovers substantially more true positives (higher recall) while giving up only a modest amount of precision, making it the better choice when missing a potential subscriber is costly.
7. **Consistent across folds** — cross-validation scores exhibit low variance, confirming the model is not over-fitting.

## Files

```
.
├── bank-data/
│   └── bank-full.csv               # Source data
├── term_deposit_analysis.py         # Full analysis script
├── requirements.txt                 # pip dependencies
├── README.md                        # Documentation
└── images/                          # Generated charts (after running)
```

## Execution

```bash
pip install -r requirements.txt
python term_deposit_analysis.py
```

Charts are written to the `images/` directory automatically.

## Libraries Used

Python 3.8+, pandas, numpy, matplotlib, seaborn, scikit-learn
