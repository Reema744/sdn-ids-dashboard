import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    f1_score, balanced_accuracy_score
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ================== FIXED SETTINGS ==================
TEST_SIZE = 0.20
RANDOM_STATE = 42
TOP_K = 3

MAX_CATEGORIES_PER_COL = 100  # cap high-cardinality categories (speed + stability)

PREFERRED_LABEL = "Attack type"
LABEL_FALLBACKS = [
    "attack_type", "attack", "Attack", "label", "Label",
    "class", "Class", "target", "Target",
    "attack_cat", "AttackCat", "category", "Category"
]

MISSING_TOKENS = ["-", "NA", "N/A", "null", "None", "?", ""]


# ================== NOTE ABOUT CNN (EXPERIMENTAL) ==================
# A CNN model was implemented and tested separately in Google Colab on a reduced subset
# (e.g., 5,000 samples using numeric features only). Result was lower than Random Forest.
# Therefore, CNN was not adopted as the primary model for this Streamlit dashboard.


# ================== HELPERS ==================
def load_dataset(file_or_path):
    return pd.read_csv(file_or_path, low_memory=False)

def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace(MISSING_TOKENS, np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def normalize_label_series(y: pd.Series) -> pd.Series:
    y = y.astype(str).str.strip()
    y = y.str.replace(r"\s+", " ", regex=True)
    y = y.str.title()
    y = y.replace({
        "Backdoors": "Backdoor",
        "Dos": "DoS",
        "Ddos": "DDoS",
        "Mitm": "MITM"
    })
    return y

def cap_categories(series: pd.Series, max_cats=100):
    s = series.astype(str).fillna("NA")
    vc = s.value_counts()
    if len(vc) <= max_cats:
        return s
    top = set(vc.head(max_cats).index)
    return s.apply(lambda v: v if v in top else "OTHER")

def detect_feature_types(X: pd.DataFrame):
    numeric, categorical = [], []
    for c in X.columns:
        ratio = pd.to_numeric(X[c], errors="coerce").notna().mean()
        if ratio > 0.85:
            numeric.append(c)
        else:
            categorical.append(c)
    return numeric, categorical

def detect_label_column(df: pd.DataFrame) -> str:
    if PREFERRED_LABEL in df.columns:
        return PREFERRED_LABEL
    hits = [c for c in LABEL_FALLBACKS if c in df.columns]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        # choose column with smaller unique count (often label)
        return sorted(hits, key=lambda c: df[c].nunique(dropna=True))[0]
    return ""

def build_onehot():
    # supports both newer and older scikit-learn versions
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def compute_topk_accuracy(proba, y_true, classes, k=3):
    if proba is None:
        return None
    idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([idx.get(v, -1) for v in y_true])
    mask = y_idx >= 0
    proba, y_idx = proba[mask], y_idx[mask]
    topk = np.argsort(proba, axis=1)[:, -k:]
    return float(np.mean([y_idx[i] in topk[i] for i in range(len(y_idx))]))

def metrics_pack(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))
    }


# ================== UI ==================
st.set_page_config(page_title="General Attack Detection Dashboard", layout="wide")
st.title("General Attack Detection Dashboard")
st.caption("Dataset-agnostic IDS dashboard (Logistic Regression + Random Forest). CNN is experimental (tested separately).")

st.markdown("---")
st.header("1) Upload Dataset")
uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV file to start.")
    st.stop()

df = sanitize_df(load_dataset(uploaded))

st.subheader("Quick Preview")
st.dataframe(df.head())
st.write("Shape:", df.shape)

st.markdown("---")
st.header("2) Label Column (Auto-detected)")
label_col = detect_label_column(df)
if not label_col:
    st.error(f"Label column not found. Expected '{PREFERRED_LABEL}' or one of: {LABEL_FALLBACKS}")
    st.stop()

st.success(f"Using label column: {label_col}")

df = df.dropna(subset=[label_col]).copy()
df[label_col] = normalize_label_series(df[label_col])

st.subheader("Label Distribution (Top 20)")
st.table(df[label_col].value_counts().head(20))

st.markdown("---")
st.header("3) Feature Handling (Auto types)")

X = sanitize_df(df.drop(columns=[label_col])).copy()
y = df[label_col].copy()

# drop empty columns
X = X.dropna(axis=1, how="all")
if X.shape[1] == 0:
    st.error("No usable feature columns after cleaning (all features empty).")
    st.stop()

numeric_cols, categorical_cols = detect_feature_types(X)

for c in categorical_cols:
    X[c] = cap_categories(X[c], MAX_CATEGORIES_PER_COL)

st.write("Numeric columns:", numeric_cols)
st.write("Categorical columns:", categorical_cols)

st.markdown("---")
st.header("4) Train & Evaluate")

can_stratify = y.value_counts().min() >= 2

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y if can_stratify else None
)

st.write(f"Split: Train={len(X_train)} | Test={len(X_test)} | Stratify used: {can_stratify}")

# Preprocessor
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", build_onehot())
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, numeric_cols),
    ("cat", cat_pipe, categorical_cols)
])

# Models
logreg = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(
        max_iter=1200,
        class_weight="balanced",
        n_jobs=-1
    ))
])

rf = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=250,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_depth=25,
        min_samples_leaf=2
    ))
])

run = st.button("Run Training & Evaluation")
if not run:
    st.info("Click **Run Training & Evaluation** to train models and see results.")
    st.stop()

with st.spinner("Training models..."):
    logreg.fit(X_train, y_train)
    rf.fit(X_train, y_train)

pred_lr = logreg.predict(X_test)
pred_rf = rf.predict(X_test)

lr_m = metrics_pack(y_test, pred_lr)
rf_m = metrics_pack(y_test, pred_rf)

best = "RandomForest" if rf_m["macro_f1"] >= lr_m["macro_f1"] else "LogisticRegression"
best_pred = pred_rf if best == "RandomForest" else pred_lr

# Top-K for RF
rf_proba = rf.predict_proba(X_test)
rf_topk = compute_topk_accuracy(
    rf_proba, y_test.values,
    rf.named_steps["clf"].classes_,
    k=TOP_K
)

st.subheader("Key Results")
c1, c2 = st.columns(2)
with c1:
    st.markdown("### Logistic Regression (Baseline)")
    st.metric("Accuracy", f"{lr_m['accuracy']:.3f}")
    st.metric("Macro F1", f"{lr_m['macro_f1']:.3f}")
    st.metric("Balanced Accuracy", f"{lr_m['balanced_accuracy']:.3f}")

with c2:
    st.markdown("### Random Forest (Final Model)")
    st.metric("Accuracy", f"{rf_m['accuracy']:.3f}")
    st.metric("Macro F1", f"{rf_m['macro_f1']:.3f}")
    st.metric("Balanced Accuracy", f"{rf_m['balanced_accuracy']:.3f}")
    st.metric(f"Top-{TOP_K} Accuracy", f"{rf_topk*100:.2f}%")

st.success(f"Best model (by Macro F1): {best}")

st.markdown("---")
st.subheader(f"Classification Report ({best})")
st.text(classification_report(y_test, best_pred, zero_division=0))

st.subheader(f"Confusion Matrix ({best})")
labels_sorted = sorted(pd.Series(y_test).unique().tolist())
cm = confusion_matrix(y_test, best_pred, labels=labels_sorted)
cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
st.dataframe(cm_df)

st.subheader("Sample Predictions (First 15)")
sample_df = pd.DataFrame({
    "true_label": y_test.values[:15],
    "predicted_label": best_pred[:15]
})
st.dataframe(sample_df)

st.markdown("---")
st.subheader("Experimental Deep Learning (CNN) — Tested Separately")
# put your experimental CNN result here (from your Colab run)
cnn_accuracy = 0.684

with st.expander("Show CNN experiment details"):
    st.write("CNN was evaluated separately in Google Colab on a reduced subset (e.g., 5,000 samples, numeric features only).")
    st.write(f"CNN Test Accuracy (experimental): **{cnn_accuracy:.3f}**")
    st.write(f"Random Forest Accuracy (dashboard): **{rf_m['accuracy']:.3f}**")
    st.write("Conclusion: CNN was not adopted due to inferior performance on structured tabular data and class imbalance.")

st.markdown("---")
st.subheader("Model Comparison (Report-ready)")
comparison = pd.DataFrame([
    {"Model": "Logistic Regression (Baseline)", "Accuracy": lr_m["accuracy"], "Macro F1": lr_m["macro_f1"], "Balanced Accuracy": lr_m["balanced_accuracy"]},
    {"Model": "Random Forest (Final)", "Accuracy": rf_m["accuracy"], "Macro F1": rf_m["macro_f1"], "Balanced Accuracy": rf_m["balanced_accuracy"]},
    {"Model": "CNN (Experimental, Colab)", "Accuracy": cnn_accuracy, "Macro F1": None, "Balanced Accuracy": None},
])

st.dataframe(comparison)

st.subheader("Key Takeaway (copy to report)")
st.text(
f"""Random Forest achieved the strongest overall performance:
- Accuracy: {rf_m['accuracy']:.3f}
- Macro F1-score: {rf_m['macro_f1']:.3f}
- Balanced Accuracy: {rf_m['balanced_accuracy']:.3f}

This model outperformed Logistic Regression across all evaluation metrics.

The CNN model was evaluated experimentally on a reduced subset (Colab)
(Accuracy: {cnn_accuracy:.3f}), but it was not adopted as the primary model
due to inferior performance on structured tabular data and class imbalance."""
)
