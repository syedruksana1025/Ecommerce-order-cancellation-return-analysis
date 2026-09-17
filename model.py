"""
model.py  –  Data loading, cleaning, EDA, feature engineering, and
             ML model training for E-Commerce Order Cancellation & Return Analysis.
             Saves report images to  report_images/  and the trained model to
             model/ecommerce_order_model.pkl.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
IMG_DIR     = os.path.join(BASE_DIR, "report_images")
MODEL_DIR   = os.path.join(BASE_DIR, "model")

os.makedirs(IMG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = ["#3b82d4", "#7c5cd8", "#e74c3c", "#2ecc71", "#f39c12",
           "#1abc9c", "#e67e22", "#9b59b6", "#34495e", "#e91e63"]
sns.set_theme(style="whitegrid", palette=PALETTE)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "clean_final_data.csv"), parse_dates=["OrderDate", "SignupDate"])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df["Age"]       = df["Age"].fillna(df["Age"].median())
    df["Discount"]  = df["Discount"].fillna(0)
    df["Quantity"]  = df["Quantity"].fillna(1)

    # Normalise Status labels
    df["Status"] = df["Status"].str.strip().str.title()

    # Derived date features
    df["OrderYear"]  = df["OrderDate"].dt.year
    df["OrderMonth"] = df["OrderDate"].dt.month
    df["OrderDOW"]   = df["OrderDate"].dt.dayofweek   # 0=Mon

    # Customer tenure (days from signup to order)
    df["Tenure"] = (df["OrderDate"] - df["SignupDate"]).dt.days.clip(lower=0)

    # Revenue per row
    df["Revenue"] = df["Sales"]

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EDA  –  VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def eda_visualisations(df: pd.DataFrame):
    status_counts = df["Status"].value_counts()

    # 1. Order Status Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(status_counts.index, status_counts.values,
                  color=PALETTE[:len(status_counts)])
    ax.set_title("Order Status Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Status"); ax.set_ylabel("Number of Orders")
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 80,
                f"{int(b.get_height()):,}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "01_order_status_distribution.png"), dpi=150)
    plt.close()

    # 2. Cancellation Rate by Category
    cat_stats = df.groupby("Category")["Status"].apply(
        lambda s: (s == "Cancelled").sum() / len(s) * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    cat_stats.plot(kind="bar", color=PALETTE[:len(cat_stats)], ax=ax)
    ax.set_title("Cancellation Rate by Product Category (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Category"); ax.set_ylabel("Cancellation Rate (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "02_cancellation_by_category.png"), dpi=150)
    plt.close()

    # 3. Return Rate by Category
    ret_stats = df.groupby("Category")["Status"].apply(
        lambda s: (s == "Returned").sum() / len(s) * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ret_stats.plot(kind="bar", color=PALETTE[1:len(ret_stats)+1], ax=ax)
    ax.set_title("Return Rate by Product Category (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Category"); ax.set_ylabel("Return Rate (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "03_return_by_category.png"), dpi=150)
    plt.close()

    # 4. Monthly Order Trends
    monthly = df.groupby(["OrderYear", "OrderMonth"])["OrderID"].count().reset_index()
    monthly["Period"] = monthly["OrderYear"].astype(str) + "-" + monthly["OrderMonth"].astype(str).str.zfill(2)
    monthly = monthly.sort_values(["OrderYear", "OrderMonth"])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly["Period"], monthly["OrderID"], marker="o", color=PALETTE[0], linewidth=2)
    ax.set_title("Monthly Order Volume", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Number of Orders")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "04_monthly_order_trends.png"), dpi=150)
    plt.close()

    # 5. Payment Method vs Status
    pm_status = df.groupby(["PaymentMethod", "Status"]).size().unstack(fill_value=0)
    pm_status_pct = pm_status.div(pm_status.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    pm_status_pct.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
    ax.set_title("Order Status by Payment Method (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Payment Method"); ax.set_ylabel("Percentage (%)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "05_payment_method_vs_status.png"), dpi=150)
    plt.close()

    # 6. Customer Segment vs Cancellation Rate
    seg_cancel = df.groupby("CustomerSegment")["Status"].apply(
        lambda s: (s == "Cancelled").sum() / len(s) * 100)
    fig, ax = plt.subplots(figsize=(7, 5))
    seg_cancel.plot(kind="bar", color=PALETTE[2:6], ax=ax)
    ax.set_title("Cancellation Rate by Customer Segment (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Segment"); ax.set_ylabel("Cancellation Rate (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "06_segment_cancellation_rate.png"), dpi=150)
    plt.close()

    # 7. Discount vs Order Status (boxplot)
    fig, ax = plt.subplots(figsize=(8, 5))
    order = df["Status"].value_counts().index.tolist()
    sns.boxplot(data=df, x="Status", y="Discount", order=order, palette=PALETTE, ax=ax)
    ax.set_title("Discount Distribution by Order Status", fontsize=14, fontweight="bold")
    ax.set_xlabel("Status"); ax.set_ylabel("Discount (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "07_discount_vs_status.png"), dpi=150)
    plt.close()

    # 8. Age Distribution by Status
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, st in enumerate(df["Status"].unique()):
        subset = df[df["Status"] == st]["Age"].dropna()
        ax.hist(subset, bins=25, alpha=0.6, label=st, color=PALETTE[i])
    ax.set_title("Age Distribution by Order Status", fontsize=14, fontweight="bold")
    ax.set_xlabel("Customer Age"); ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "08_age_distribution_by_status.png"), dpi=150)
    plt.close()

    # 9. Top 10 Cities by Cancellation Volume
    city_cancel = df[df["Status"] == "Cancelled"]["City"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(9, 5))
    city_cancel.plot(kind="barh", color=PALETTE[0], ax=ax)
    ax.set_title("Top 10 Cities by Cancellation Volume", fontsize=14, fontweight="bold")
    ax.set_xlabel("Cancellations"); ax.set_ylabel("City")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "09_top_cities_cancellation.png"), dpi=150)
    plt.close()

    # 10. Correlation Heatmap (numeric features)
    num_cols = ["Age", "Quantity", "Discount", "UnitPrice", "Sales", "Tenure"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap – Numeric Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "10_correlation_heatmap.png"), dpi=150)
    plt.close()

    print(f"  [OK] 10 EDA images saved to {IMG_DIR}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING  &  MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def encode_and_split(df: pd.DataFrame):
    df = df.copy()
    # Binary target: 1 = Cancelled or Returned, 0 = Completed / other
    df["Target"] = df["Status"].apply(lambda s: 1 if s in ("Cancelled", "Returned") else 0)

    feature_cols = [
        "Age", "Quantity", "Discount", "UnitPrice", "Sales", "Tenure",
        "OrderMonth", "OrderDOW",
        "PaymentMethod", "Category", "CustomerSegment", "City"
    ]
    df_feat = df[feature_cols + ["Target"]].copy()

    # Label encode categoricals
    cat_cols = ["PaymentMethod", "Category", "CustomerSegment", "City"]
    le_map = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_feat[col] = le.fit_transform(df_feat[col].astype(str))
        le_map[col] = le

    X = df_feat.drop("Target", axis=1)
    y = df_feat["Target"]

    # SMOTE to balance classes
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.20, random_state=42, stratify=y_res
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train, y_test, X.columns.tolist(), scaler, le_map


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, random_state=42),
        "XGBoost":             XGBClassifier(n_estimators=200, eval_metric="logloss",
                                             random_state=42, verbosity=0),
    }

    results = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        roc     = roc_auc_score(y_test, y_proba)
        report  = classification_report(y_test, y_pred, output_dict=True)
        results[name] = {"model": clf, "roc_auc": roc, "report": report,
                         "y_pred": y_pred, "y_proba": y_proba}
        print(f"  {name:25s} | ROC-AUC: {roc:.4f} | F1(macro): {report['macro avg']['f1-score']:.4f}")

    # ── Pick best model ────────────────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best      = results[best_name]
    print(f"\n  *** Best model: {best_name}  (ROC-AUC = {best['roc_auc']:.4f})")

    # ── Confusion matrix ───────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, best["y_pred"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Not Cancelled/Returned", "Cancelled/Returned"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix – {best_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "11_confusion_matrix.png"), dpi=150)
    plt.close()

    # ── ROC curves ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title("ROC Curves – All Models", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "12_roc_curves.png"), dpi=150)
    plt.close()

    # ── Feature importance (best tree model) ─────────────────────────────────
    clf_best = best["model"]
    if hasattr(clf_best, "feature_importances_"):
        fi = pd.Series(clf_best.feature_importances_, index=feature_names).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        fi.plot(kind="bar", color=PALETTE[0], ax=ax)
        ax.set_title(f"Feature Importances – {best_name}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Feature"); ax.set_ylabel("Importance")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, "13_feature_importance.png"), dpi=150)
        plt.close()

    print(f"  [OK] Model visualisations saved to {IMG_DIR}")
    return best_name, best["model"], results


def save_model(model, scaler, le_map, feature_names):
    bundle = {
        "model":    model,
        "scaler":   scaler,
        "le_map":   le_map,
        "features": feature_names,
    }
    path = os.path.join(MODEL_DIR, "ecommerce_order_model.pkl")
    joblib.dump(bundle, path)
    print(f"  [OK] Model bundle saved -> {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SUMMARY STATISTICS  (used by generate_report.py)
# ═══════════════════════════════════════════════════════════════════════════════

def get_summary_stats(df: pd.DataFrame) -> dict:
    total    = len(df)
    status_c = df["Status"].value_counts()
    canc_n   = int(status_c.get("Cancelled", 0))
    ret_n    = int(status_c.get("Returned",  0))
    comp_n   = int(status_c.get("Completed", 0))

    stats = {
        "total_orders":       total,
        "completed_orders":   comp_n,
        "cancelled_orders":   canc_n,
        "returned_orders":    ret_n,
        "cancellation_rate":  round(canc_n / total * 100, 2),
        "return_rate":        round(ret_n  / total * 100, 2),
        "unique_customers":   int(df["CustomerID"].nunique()),
        "unique_products":    int(df["ProductID"].nunique()),
        "avg_order_value":    round(df["Sales"].mean(), 2),
        "total_revenue":      round(df[df["Status"] == "Completed"]["Sales"].sum(), 2),
        "top_cancel_cat":     df[df["Status"] == "Cancelled"]["Category"].value_counts().idxmax(),
        "top_return_cat":     df[df["Status"] == "Returned"]["Category"].value_counts().idxmax(),
        "top_cancel_city":    df[df["Status"] == "Cancelled"]["City"].value_counts().idxmax(),
        "top_cancel_pm":      df[df["Status"] == "Cancelled"]["PaymentMethod"].value_counts().idxmax(),
    }
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n=== E-Commerce Order Cancellation & Return Analysis ===\n")

    print(">> Loading data ...")
    df_raw = load_data()
    print(f"  Rows loaded: {len(df_raw):,}")

    print(">> Cleaning data ...")
    df = clean_data(df_raw)

    print(">> Running EDA visualisations ...")
    eda_visualisations(df)

    stats = get_summary_stats(df)
    print("\n-- Summary --")
    for k, v in stats.items():
        print(f"  {k:25s}: {v}")

    print("\n>> Engineering features & training models ...")
    X_tr, X_te, y_tr, y_te, feat_names, scaler, le_map = encode_and_split(df)
    best_name, best_model, all_results = train_and_evaluate(X_tr, X_te, y_tr, y_te, feat_names)

    print("\n>> Saving model ...")
    save_model(best_model, scaler, le_map, feat_names)

    print("\nmodel.py complete.\n")
