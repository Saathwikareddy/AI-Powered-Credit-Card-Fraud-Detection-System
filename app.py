import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI-Powered Credit Card Fraud Detection System",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================
st.title("💳 AI-Powered Credit Card Fraud Detection System")

st.markdown("""
### Algorithms Used
- Logistic Regression → Fraud / Non-Fraud Detection
- Random Forest → Fraud Prediction
- Isolation Forest → Anomaly Detection
- PCA Analysis → Feature Analysis
""")

# =========================================================
# LOAD DATASET
# =========================================================
@st.cache_data
def load_data():

    df = pd.read_csv("creditcard.csv")

    return df

df = load_data()

# =========================================================
# DATA PREVIEW
# =========================================================
st.subheader("📌 Dataset Preview")

st.dataframe(df.head())

# =========================================================
# DATASET SHAPE
# =========================================================
st.subheader("📊 Dataset Shape")

st.write(df.shape)

# =========================================================
# MISSING VALUES
# =========================================================
st.subheader("🔍 Missing Values")

st.write(df.isnull().sum())

# =========================================================
# HANDLE MISSING VALUES
# =========================================================
df = df.dropna(subset=['Class'])

df['Class'] = df['Class'].astype(int)

# =========================================================
# FRAUD DISTRIBUTION
# =========================================================
st.subheader("💰 Fraud vs Non-Fraud Transactions")

fraud_count = df['Class'].value_counts()

st.write(fraud_count)

fig1, ax1 = plt.subplots(figsize=(6,4))

sns.countplot(
    x='Class',
    data=df,
    ax=ax1
)

ax1.set_title("Fraud vs Non-Fraud Transactions")

st.pyplot(fig1)

# =========================================================
# FEATURE SCALING
# =========================================================
scaler = StandardScaler()

df['Amount'] = scaler.fit_transform(
    df[['Amount']]
)

# =========================================================
# FEATURES & TARGET
# =========================================================
X = df.drop('Class', axis=1)

y = df['Class']

# =========================================================
# TRAIN TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# SIDEBAR MODEL SELECTION
# =========================================================
st.sidebar.title("⚙️ Model Selection")

model_name = st.sidebar.selectbox(
    "Choose Algorithm",
    [
        "Logistic Regression",
        "Random Forest",
        "Isolation Forest"
    ]
)

# =========================================================
# LOGISTIC REGRESSION
# =========================================================
if model_name == "Logistic Regression":

    st.subheader("📈 Logistic Regression Results")

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

# =========================================================
# RANDOM FOREST
# =========================================================
elif model_name == "Random Forest":

    st.subheader("🌲 Random Forest Results")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

# =========================================================
# ISOLATION FOREST
# =========================================================
else:

    st.subheader("🛡️ Isolation Forest Results")

    model = IsolationForest(
        n_estimators=100,
        contamination=0.001,
        random_state=42
    )

    model.fit(X_train)

    y_pred = model.predict(X_test)

    # Convert predictions
    # -1 = Fraud
    # 1 = Normal
    y_pred = np.where(
        y_pred == -1,
        1,
        0
    )

# =========================================================
# MODEL EVALUATION
# =========================================================
accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

# =========================================================
# DISPLAY METRICS
# =========================================================
st.subheader("✅ Model Performance")

st.write(f"Accuracy : {accuracy:.4f}")

st.write(f"Precision : {precision:.4f}")

st.write(f"Recall : {recall:.4f}")

st.write(f"F1 Score : {f1:.4f}")

# =========================================================
# CLASSIFICATION REPORT
# =========================================================
st.subheader("📄 Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)

# =========================================================
# CONFUSION MATRIX
# =========================================================
st.subheader("📌 Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)

fig2, ax2 = plt.subplots(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=ax2
)

ax2.set_xlabel("Predicted")

ax2.set_ylabel("Actual")

st.pyplot(fig2)

# =========================================================
# PCA ANALYSIS
# =========================================================
st.subheader("📉 PCA Feature Analysis")

pca_columns = [
    col for col in df.columns
    if col.startswith('V')
]

st.write("PCA Columns:")

st.write(pca_columns)

fig3, ax3 = plt.subplots(figsize=(12,8))

sns.heatmap(
    df[pca_columns].corr(),
    cmap='coolwarm',
    ax=ax3
)

ax3.set_title("PCA Correlation Heatmap")

st.pyplot(fig3)

# =========================================================
# AMOUNT VS FRAUD
# =========================================================
st.subheader("💵 Transaction Amount vs Fraud")

fig4, ax4 = plt.subplots(figsize=(8,6))

sns.boxplot(
    x='Class',
    y='Amount',
    data=df,
    ax=ax4
)

ax4.set_title("Transaction Amount vs Fraud")

st.pyplot(fig4)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.markdown("""
Developed using:
- Streamlit
- Scikit-Learn
- Pandas
- Seaborn
- Matplotlib
""")