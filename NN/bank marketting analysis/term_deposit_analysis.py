"""
Term Deposit Prediction — Classification with Logistic Regression
=================================================================
Analyse customer data from a bank's direct marketing campaigns and
build a binary classifier to forecast term deposit uptake.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.figsize": (10, 6), "font.size": 11})
sns.set_style("ticks")

IMG_PATH = "images"
os.makedirs(IMG_PATH, exist_ok=True)

RS = 101
CLR_YES, CLR_NO = "#1abc9c", "#e74c3c"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 1 : READING THE DATA                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print("\n" + "~" * 65)
print("  PHASE 1 : Reading the data")
print("~" * 65)

raw = pd.read_csv("bank-data/bank-full.csv", sep=";")

total_rows, total_cols = raw.shape
print(f"  Records  = {total_rows}")
print(f"  Fields   = {total_cols}")

print("\n  --- Head of the dataset ---")
print(raw.head(7).to_string())

print("\n  --- Data types ---")
for col_name, dtype in raw.dtypes.items():
    print(f"    {col_name:15s}  {dtype}")

print("\n  --- Descriptive statistics (numeric) ---")
print(raw.describe().round(2).to_string())

print("\n  --- Descriptive statistics (categorical) ---")
print(raw.describe(include="object").to_string())

total_nulls = raw.isnull().sum().sum()
total_dups = raw.duplicated().sum()
print(f"\n  Null values  : {total_nulls}")
print(f"  Duplicates   : {total_dups}")

print("\n  --- Cardinality ---")
for col_name in raw.columns:
    print(f"    {col_name:15s}  {raw[col_name].nunique()} unique")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 2 : EXPLORATORY DATA ANALYSIS                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print("\n" + "~" * 65)
print("  PHASE 2 : Exploratory Data Analysis")
print("~" * 65)

num_cols = raw.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in raw.select_dtypes(include="object").columns if c != "y"]

print(f"  Numeric columns     = {num_cols}")
print(f"  Categorical columns = {cat_cols}")

# -- 2A : Target variable breakdown --
target_dist = raw["y"].value_counts()
ratio_no = (raw["y"] == "no").mean() * 100
ratio_yes = (raw["y"] == "yes").mean() * 100

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))

a1.barh(target_dist.index, target_dist.values,
        color=[CLR_NO, CLR_YES], edgecolor="black", height=0.45)
a1.set_title("Term Deposit — Class Counts", fontweight="bold", fontsize=13)
a1.set_xlabel("Customers")
for idx, val in enumerate(target_dist.values):
    a1.text(val + 250, idx, f"{val:,}", va="center", fontweight="bold")

a2.pie(target_dist.values, labels=["No", "Yes"], autopct="%1.1f%%",
       colors=[CLR_NO, CLR_YES], startangle=120,
       explode=(0.03, 0.03), textprops={"fontsize": 12})
a2.set_title("Term Deposit — Class Share", fontweight="bold", fontsize=13)

plt.tight_layout()
plt.savefig(f"{IMG_PATH}/target_overview.png", dpi=130, bbox_inches="tight")
plt.show()

print(f"\n  Class split => No: {ratio_no:.1f}%  |  Yes: {ratio_yes:.1f}%")

# -- 2B : Numeric histograms --
fig, axes = plt.subplots(3, 3, figsize=(17, 13))
flat = axes.flatten()
for i, col_name in enumerate(num_cols):
    ax = flat[i]
    ax.hist(raw[col_name], bins=38, color="#5dade2", edgecolor="white", alpha=0.8)
    mu = raw[col_name].mean()
    md = raw[col_name].median()
    ax.axvline(mu, color="#cb4335", ls="--", lw=1.5, label=f"Mean={mu:.1f}")
    ax.axvline(md, color="#28b463", ls="-.", lw=1.5, label=f"Median={md:.1f}")
    ax.set_title(col_name, fontweight="bold")
    ax.legend(fontsize=7)
for i in range(len(num_cols), 9):
    flat[i].axis("off")
fig.suptitle("Distributions of Numeric Attributes",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{IMG_PATH}/numeric_distributions.png", dpi=130, bbox_inches="tight")
plt.show()

# -- 2C : Categorical count charts --
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
flat = axes.flatten()
for i, col_name in enumerate(cat_cols):
    ax = flat[i]
    order = raw[col_name].value_counts().index
    sns.countplot(data=raw, x=col_name, order=order, ax=ax,
                  palette="Set3", edgecolor="black")
    ax.set_title(col_name, fontweight="bold")
    ax.tick_params(axis="x", rotation=42)
    ax.set_xlabel("")
for i in range(len(cat_cols), 9):
    flat[i].axis("off")
fig.suptitle("Distributions of Categorical Attributes",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{IMG_PATH}/categorical_distributions.png", dpi=130, bbox_inches="tight")
plt.show()

# -- 2D : Subscription rate per category --
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
flat = axes.flatten()
for i, col_name in enumerate(cat_cols):
    ax = flat[i]
    tbl = pd.crosstab(raw[col_name], raw["y"], normalize="index") * 100
    tbl.plot(kind="bar", stacked=True, ax=ax,
             color=[CLR_NO, CLR_YES], edgecolor="black")
    ax.set_title(f"% Subscription — {col_name}", fontweight="bold")
    ax.set_ylabel("Percent")
    ax.legend(title="y", labels=["No", "Yes"], fontsize=7)
    ax.tick_params(axis="x", rotation=42)
    ax.set_xlabel("")
for i in range(len(cat_cols), 9):
    flat[i].axis("off")
fig.suptitle("Subscription Rates by Category",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{IMG_PATH}/subscription_rates.png", dpi=130, bbox_inches="tight")
plt.show()

# -- 2E : Box plots (numeric vs outcome) --
fig, axes = plt.subplots(3, 3, figsize=(17, 13))
flat = axes.flatten()
for i, col_name in enumerate(num_cols):
    ax = flat[i]
    sns.boxplot(data=raw, x="y", y=col_name, ax=ax,
                palette=[CLR_NO, CLR_YES])
    ax.set_title(f"{col_name} vs Outcome", fontweight="bold")
    ax.set_xlabel("")
for i in range(len(num_cols), 9):
    flat[i].axis("off")
fig.suptitle("Numeric Attributes vs Subscription Outcome",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{IMG_PATH}/boxplots.png", dpi=130, bbox_inches="tight")
plt.show()

# -- 2F : Correlation matrix --
corr_df = raw.copy()
corr_df["deposit"] = (corr_df["y"] == "yes").astype(int)
corr_mat = corr_df[num_cols + ["deposit"]].corr()

plt.figure(figsize=(10, 8))
mask_upper = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, mask=mask_upper, annot=True, fmt=".2f",
            cmap="vlag", center=0, linewidths=0.6,
            cbar_kws={"shrink": 0.7})
plt.title("Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{IMG_PATH}/correlations.png", dpi=130, bbox_inches="tight")
plt.show()

print("\n  Correlation with target (deposit):")
dep_corr = corr_mat["deposit"].drop("deposit").sort_values(ascending=False)
for name, val in dep_corr.items():
    print(f"    {name:15s}  {val:+.4f}")

# -- 2G : Age by subscription --
fig, (a_no, a_yes) = plt.subplots(1, 2, figsize=(14, 5))
for lbl, clr, ax in [("no", CLR_NO, a_no), ("yes", CLR_YES, a_yes)]:
    ages = raw.loc[raw["y"] == lbl, "age"]
    ax.hist(ages, bins=25, color=clr, edgecolor="black", alpha=0.8)
    ax.axvline(ages.mean(), color="black", ls="--",
               label=f"Mean age = {ages.mean():.1f}")
    ax.set_title(f"{'Subscribed' if lbl == 'yes' else 'Did Not Subscribe'}",
                 fontweight="bold")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.legend()
plt.tight_layout()
plt.savefig(f"{IMG_PATH}/age_comparison.png", dpi=130, bbox_inches="tight")
plt.show()

# -- EDA summary --
print("\n" + "~" * 65)
print("  EDA SUMMARY")
print("~" * 65)
notes = [
    f"Imbalanced target: {ratio_no:.1f}% negative / {ratio_yes:.1f}% positive.",
    "Duration of the last contact is most correlated with conversion.",
    "Successful prior campaign result is a very strong positive signal.",
    "Customers with larger balances are slightly more likely to convert.",
    "Retired people and students subscribe at above-average rates.",
    "Singles and tertiary-educated customers also convert more often.",
    "Being contacted via cell phone beats other contact channels.",
    "Dataset has no nulls, though some fields contain 'unknown' values.",
    "More campaign contacts correlate slightly negatively with conversion.",
    "Number of prior contacts shows a mild positive association.",
]
for n, note in enumerate(notes, 1):
    print(f"  [{n:02d}] {note}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 3 : PREPROCESSING                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print("\n" + "~" * 65)
print("  PHASE 3 : Preprocessing")
print("~" * 65)

data = raw.copy()

data["y"] = data["y"].map({"yes": 1, "no": 0})
print("  Target binarised (yes=1, no=0)")

for bc in ["default", "housing", "loan"]:
    data[bc] = data[bc].map({"yes": 1, "no": 0})
print("  Binary columns converted: default, housing, loan")

encode_cols = ["job", "marital", "education", "contact", "month", "poutcome"]
le_store = {}
for ec in encode_cols:
    le = LabelEncoder()
    data[ec] = le.fit_transform(data[ec])
    le_store[ec] = le
    lbl_map = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"  {ec:12s} => {lbl_map}")

X_all = data.drop("y", axis=1)
y_all = data["y"]
print(f"\n  X shape = {X_all.shape}")
print(f"  y shape = {y_all.shape}")

X_tr, X_te, y_tr, y_te = train_test_split(
    X_all, y_all, test_size=0.25, random_state=RS, stratify=y_all
)
print(f"\n  Train : {len(X_tr)} ({len(X_tr)/len(X_all)*100:.0f}%)")
print(f"  Test  : {len(X_te)} ({len(X_te)/len(X_all)*100:.0f}%)")

sc = StandardScaler()
X_tr_n = pd.DataFrame(sc.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
X_te_n = pd.DataFrame(sc.transform(X_te), columns=X_te.columns, index=X_te.index)
print("  Standardisation applied (mean=0, std=1).")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 4 : LOGISTIC REGRESSION                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print("\n" + "~" * 65)
print("  PHASE 4 : Training Logistic Regression")
print("~" * 65)

model = LogisticRegression(solver="liblinear", C=0.8, max_iter=800, random_state=RS)
model.fit(X_tr_n, y_tr)

print(f"  Solver  = {model.solver}")
print(f"  C       = {model.C}")
print(f"  Iters   = {model.n_iter_[0]}")

coefs = pd.DataFrame({
    "col": X_all.columns,
    "coef": model.coef_[0],
}).assign(importance=lambda d: d["coef"].abs()).sort_values("importance", ascending=False)

print(f"\n  Intercept = {model.intercept_[0]:.4f}")
print("\n  Coefficients:")
for _, r in coefs.iterrows():
    marker = "+" if r["coef"] > 0 else "-"
    print(f"    {marker} {r['col']:15s} {r['coef']:+.4f}")

plt.figure(figsize=(10, 7))
c_clr = ["#1abc9c" if v > 0 else "#e74c3c" for v in coefs["coef"]]
plt.barh(coefs["col"], coefs["coef"], color=c_clr, edgecolor="black")
plt.axvline(0, color="black", lw=0.8)
plt.gca().invert_yaxis()
plt.xlabel("Coefficient")
plt.title("Logistic Regression — Feature Coefficients",
          fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig(f"{IMG_PATH}/coefficients.png", dpi=130, bbox_inches="tight")
plt.show()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 5 : EVALUATION                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print("\n" + "~" * 65)
print("  PHASE 5 : Model Evaluation")
print("~" * 65)

y_hat = model.predict(X_te_n)
y_prob = model.predict_proba(X_te_n)[:, 1]

m_acc  = accuracy_score(y_te, y_hat)
m_prec = precision_score(y_te, y_hat)
m_rec  = recall_score(y_te, y_hat)
m_f1   = f1_score(y_te, y_hat)
m_auc  = roc_auc_score(y_te, y_prob)

print("\n  [Standard Model]")
print(f"    Accuracy  = {m_acc:.4f}")
print(f"    Precision = {m_prec:.4f}")
print(f"    Recall    = {m_rec:.4f}")
print(f"    F1        = {m_f1:.4f}")
print(f"    AUC       = {m_auc:.4f}")

print("\n  Classification Report:")
print(classification_report(y_te, y_hat, target_names=["Not Subscribed", "Subscribed"]))

# Confusion matrix
cmat = confusion_matrix(y_te, y_hat)
fig, (ax_c, ax_p) = plt.subplots(1, 2, figsize=(13, 5))

sns.heatmap(cmat, annot=True, fmt="d", cmap="YlGnBu", ax=ax_c,
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"],
            linewidths=1.2, annot_kws={"size": 14})
ax_c.set_title("Confusion Matrix (Absolute)", fontweight="bold")
ax_c.set_xlabel("Predicted")
ax_c.set_ylabel("True")

cmat_pct = cmat.astype(float) / cmat.sum(axis=1, keepdims=True) * 100
sns.heatmap(cmat_pct, annot=True, fmt=".1f", cmap="RdYlGn", ax=ax_p,
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"],
            linewidths=1.2, annot_kws={"size": 14})
ax_p.set_title("Confusion Matrix (Row-wise %)", fontweight="bold")
ax_p.set_xlabel("Predicted")
ax_p.set_ylabel("True")

plt.tight_layout()
plt.savefig(f"{IMG_PATH}/confusion.png", dpi=130, bbox_inches="tight")
plt.show()

tn, fp, fn, tp = cmat.ravel()
print(f"  TN = {tn} | FP = {fp} | FN = {fn} | TP = {tp}")
print(f"  Specificity = {tn / (tn + fp):.4f}")
print(f"  Sensitivity = {tp / (tp + fn):.4f}")

# ROC
fpr_arr, tpr_arr, thr_arr = roc_curve(y_te, y_prob)
opt_i = np.argmax(tpr_arr - fpr_arr)
opt_thr = thr_arr[opt_i]

plt.figure(figsize=(8, 6))
plt.plot(fpr_arr, tpr_arr, color="#8e44ad", lw=2.5,
         label=f"Model AUC = {m_auc:.4f}")
plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random (0.5)")
plt.fill_between(fpr_arr, tpr_arr, alpha=0.10, color="#8e44ad")
plt.scatter(fpr_arr[opt_i], tpr_arr[opt_i], color="red", s=100, zorder=5,
            label=f"Best cutoff = {opt_thr:.3f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve", fontweight="bold", fontsize=13)
plt.legend(loc="lower right")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f"{IMG_PATH}/roc.png", dpi=130, bbox_inches="tight")
plt.show()

print(f"\n  Optimal cutoff = {opt_thr:.4f}")
print(f"  TPR @ cutoff   = {tpr_arr[opt_i]:.4f}")
print(f"  FPR @ cutoff   = {fpr_arr[opt_i]:.4f}")

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
sc_acc = cross_val_score(model, X_tr_n, y_tr, cv=skf, scoring="accuracy")
sc_f1  = cross_val_score(model, X_tr_n, y_tr, cv=skf, scoring="f1")
sc_auc = cross_val_score(model, X_tr_n, y_tr, cv=skf, scoring="roc_auc")

print("\n  5-fold Stratified CV:")
print(f"    Accuracy = {sc_acc.mean():.4f} (std {sc_acc.std():.4f})")
print(f"    F1       = {sc_f1.mean():.4f} (std {sc_f1.std():.4f})")
print(f"    AUC      = {sc_auc.mean():.4f} (std {sc_auc.std():.4f})")

fig, ax_trio = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (cv_vals, label, clr) in zip(ax_trio, [
    (sc_acc, "Accuracy", "#2e86c1"),
    (sc_f1,  "F1",       "#ca6f1e"),
    (sc_auc, "AUC",      "#1e8449"),
]):
    ax.bar(range(1, 6), cv_vals, color=clr, edgecolor="black", alpha=0.85)
    ax.axhline(cv_vals.mean(), color="red", ls="--", lw=1.6,
               label=f"Mean={cv_vals.mean():.4f}")
    ax.set_title(f"CV — {label}", fontweight="bold")
    ax.set_xlabel("Fold #")
    ax.set_ylabel(label)
    ax.set_ylim(cv_vals.min() - 0.02, cv_vals.max() + 0.02)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{IMG_PATH}/cv_scores.png", dpi=130, bbox_inches="tight")
plt.show()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 6 : WEIGHTED MODEL (class imbalance)                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print("\n" + "~" * 65)
print("  PHASE 6 : Weighted Logistic Regression")
print("~" * 65)

model_w = LogisticRegression(
    solver="liblinear", C=0.8, max_iter=800,
    class_weight="balanced", random_state=RS,
)
model_w.fit(X_tr_n, y_tr)

y_hat_w = model_w.predict(X_te_n)
y_prob_w = model_w.predict_proba(X_te_n)[:, 1]

w_acc  = accuracy_score(y_te, y_hat_w)
w_prec = precision_score(y_te, y_hat_w)
w_rec  = recall_score(y_te, y_hat_w)
w_f1   = f1_score(y_te, y_hat_w)
w_auc  = roc_auc_score(y_te, y_prob_w)

print("\n  [Weighted Model]")
print(f"    Accuracy  = {w_acc:.4f}")
print(f"    Precision = {w_prec:.4f}")
print(f"    Recall    = {w_rec:.4f}")
print(f"    F1        = {w_f1:.4f}")
print(f"    AUC       = {w_auc:.4f}")

print("\n  Classification Report:")
print(classification_report(y_te, y_hat_w, target_names=["Not Subscribed", "Subscribed"]))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 7 : HEAD-TO-HEAD COMPARISON                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print("\n" + "~" * 65)
print("  PHASE 7 : Head-to-Head Comparison")
print("~" * 65)

compare = pd.DataFrame({
    "Metric":   ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    "Standard": [m_acc, m_prec, m_rec, m_f1, m_auc],
    "Weighted":  [w_acc, w_prec, w_rec, w_f1, w_auc],
})
compare["Gap"] = compare["Weighted"] - compare["Standard"]
print(compare.to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 5))
pos = np.arange(len(compare))
bw = 0.30
g1 = ax.bar(pos - bw / 2, compare["Standard"], bw,
            label="Standard", color="#5499c7", edgecolor="black")
g2 = ax.bar(pos + bw / 2, compare["Weighted"], bw,
            label="Weighted", color="#f5b041", edgecolor="black")
ax.set_xticks(pos)
ax.set_xticklabels(compare["Metric"])
ax.set_ylim(0, 1.10)
ax.set_ylabel("Score")
ax.set_title("Standard vs Weighted Model", fontweight="bold", fontsize=13)
ax.legend()
for grp in (g1, g2):
    for b in grp:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"{b.get_height():.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{IMG_PATH}/head_to_head.png", dpi=130, bbox_inches="tight")
plt.show()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 8 : WRAP-UP                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print("\n" + "~" * 65)
print("  PHASE 8 : Wrap-up")
print("~" * 65)

print(f"""
  Data snapshot
    Records        : {total_rows}
    Attributes     : 16 inputs + 1 target
    Class balance  : {ratio_no:.1f}% No / {ratio_yes:.1f}% Yes

  Standard model
    Acc={m_acc:.4f}  Prec={m_prec:.4f}  Rec={m_rec:.4f}  F1={m_f1:.4f}  AUC={m_auc:.4f}

  Weighted model
    Acc={w_acc:.4f}  Prec={w_prec:.4f}  Rec={w_rec:.4f}  F1={w_f1:.4f}  AUC={w_auc:.4f}

  Observations
    - The target is heavily skewed toward non-subscribers.
    - Duration of the most recent call dominates predictions.
    - A previously successful campaign outcome is the second best signal.
    - The weighted model recovers many more true subscribers at
      a modest drop in precision — a worthwhile trade-off for the bank.
    - Five-fold CV confirms the model generalises well.
    - Retirees, students, singles, and the tertiary-educated are
      the most receptive customer segments.

  Practical advice
    - Adopt the weighted model for campaign targeting.
    - Extend call conversations when engagement is high.
    - Re-contact customers from past successful campaigns first.
    - Concentrate on retiree and student demographics.
    - Use cellular as the preferred contact channel.
""")

print("~" * 65)
print(f"  Finished. Visualisations stored in ./{IMG_PATH}/")
print("~" * 65)
