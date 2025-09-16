# patient_monitoring1.py
import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

st.set_page_config(page_title="Breast Cancer Detection", layout="wide")
st.title("Breast Cancer Detection")

# -------- Fixed defaults (no UI for these) --------
FIXED_FEATURES = ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean"]
TEST_SIZE = 0.20
RANDOM_STATE = 42

# ----------------- Helpers -----------------
def read_tabular(file) -> pd.DataFrame:
    name = getattr(file, "name", "")
    ext = os.path.splitext(name)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(file)
    return pd.read_csv(file)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def pick_target_column(df: pd.DataFrame) -> str:
    # Try common names first
    candidates = ["target","diagnosis","label","class","outcome","is_cancer","malignant"]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    # Fallbacks: any binary-like column, else let user pick
    binary_like = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
    if binary_like:
        return st.selectbox("Pick target column", options=binary_like, index=0)
    st.warning("No obvious target column found. Please choose one.")
    return st.selectbox("Pick target column", options=list(df.columns), index=0)

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots()
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center")
    st.pyplot(fig)

# ----------------- Sidebar: upload only -----------------
with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx","xls"])

if not uploaded:
    st.info("Upload a dataset to continue.")
    st.stop()

# ----------------- Load & preview -----------------
try:
    df_raw = read_tabular(uploaded)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

df = normalize_columns(df_raw)

st.subheader("Dataset Preview")
st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
st.dataframe(df.head(), use_container_width=True)

with st.expander("Detected columns"):
    st.write(list(df.columns))

# ----------------- Target -----------------
target_col = pick_target_column(df)
if target_col is None:
    st.error("A target column is required.")
    st.stop()

# Map text labels to 0/1 if needed
df_work = df.copy()
if df_work[target_col].dtype == object:
    uniq = df_work[target_col].dropna().unique().tolist()
    if len(uniq) == 2:
        df_work[target_col] = df_work[target_col].map({uniq[0]:0, uniq[1]:1})

# Must be at least 2 classes
if df_work[target_col].nunique() < 2:
    st.error(f"Target '{target_col}' has fewer than 2 classes; cannot train.")
    st.stop()

# ----------------- Feature selection -----------------
# 1) Try the fixed five if present
available_features = [c for c in FIXED_FEATURES if c in df_work.columns]

# 2) If none present, auto-pick ALL numeric columns except target
if len(available_features) == 0:
    numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    available_features = numeric_cols

# 3) As a safety, require at least 2 features
if len(available_features) < 2:
    st.error("Not enough numeric feature columns found to train a model. "
             "Please ensure your sheet has numeric predictors (besides the target).")
    st.stop()

st.markdown("### Features Used")
st.success(available_features)

# Build modeling frame (drop rows with NA in used columns)
model_cols = available_features + [target_col]
df_model = df_work.dropna(subset=model_cols).copy()

X = df_model[available_features]
y = df_model[target_col]

# ----------------- Train & evaluate -----------------
st.markdown("## Model Performance")
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y if y.nunique() == 2 else None
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{acc*100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    class_names = [str(v) for v in sorted(y.unique())]
    plot_confusion_matrix(cm, class_names)

    st.text("Classification Report")
    st.code(classification_report(y_test, y_pred, zero_division=0))
except Exception as e:
    st.error(f"Training/evaluation failed: {e}")
    st.stop()

# ----------------- Inference -----------------
st.markdown("## Predict for a New Patient")
st.caption("Defaults are column medians; adjust as needed.")
with st.form("inference"):
    new_values = {}
    for feat in available_features:
        col = df_model[feat]
        default = float(np.nanmedian(col.values)) if pd.api.types.is_numeric_dtype(col) else 0.0
        new_values[feat] = st.number_input(feat, value=default)
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        new_df = pd.DataFrame([new_values], columns=available_features)
        pred = int(clf.predict(new_df)[0])
        proba = getattr(clf, "predict_proba", None)
        probs = clf.predict_proba(new_df)[0] if proba else None

        label_text = "Malignant" if pred == 1 else "Benign"
        (st.error if pred == 1 else st.success)(f"Prediction: **{label_text}**")

        if probs is not None and len(probs) == 2:
            st.write(f"Benign probability: {probs[0]:.3f} | Malignant probability: {probs[1]:.3f}")
    except Exception as e:
        st.error(f"Inference failed: {e}")
