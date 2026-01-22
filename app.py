import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC


# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Smart Loan Approval System",
    page_icon="🏦",
    layout="wide"
)

# ---------------------- CUSTOM CSS ----------------------
st.markdown("""
<style>
/* Main background */
.main {
    background-color: #0b1220;
}

/* Title */
.big-title {
    font-size: 40px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0px;
}
.sub-title {
    font-size: 16px;
    color: #cbd5e1;
    margin-top: 5px;
}

/* Section headers */
.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #ffffff;
    margin-top: 10px;
    margin-bottom: 10px;
}

/* Card */
.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 18px;
}

/* Highlight result boxes */
.approved {
    background: rgba(34,197,94,0.15);
    border: 1px solid rgba(34,197,94,0.6);
    padding: 16px;
    border-radius: 16px;
    font-size: 22px;
    font-weight: 800;
    color: #22c55e;
    text-align: center;
}

.rejected {
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.6);
    padding: 16px;
    border-radius: 16px;
    font-size: 22px;
    font-weight: 800;
    color: #ef4444;
    text-align: center;
}

/* badge */
.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 700;
    color: #0b1220;
    background: #facc15;
}
.small {
    color: #e2e8f0;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------- DATA LOAD ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
    return df


df = load_data()


# ---------------------- FEATURES ----------------------
features = ["ApplicantIncome", "LoanAmount", "Credit_History", "Self_Employed", "Property_Area"]
X = df[features]
y = df["Loan_Status"].map({"N": 0, "Y": 1})


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_cols = ["ApplicantIncome", "LoanAmount", "Credit_History"]
cat_cols = ["Self_Employed", "Property_Area"]

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])


def build_model(kernel_choice):
    if kernel_choice == "Linear SVM":
        clf = SVC(kernel="linear", C=1, probability=True, random_state=42)
    elif kernel_choice == "Polynomial SVM":
        clf = SVC(kernel="poly", degree=3, C=1, gamma="scale", probability=True, random_state=42)
    else:
        clf = SVC(kernel="rbf", C=1, gamma="scale", probability=True, random_state=42)

    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", clf)
    ])

    model.fit(X_train, y_train)
    return model


# ---------------------- HEADER ----------------------
st.markdown('<div class="big-title">Smart Loan Approval System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">This system uses Support Vector Machines (SVM) to predict loan approval in real-time.</div>', unsafe_allow_html=True)
st.write("")


# ---------------------- LAYOUT ----------------------
left, right = st.columns([1.1, 1])

# ---------------------- INPUT PANEL ----------------------
with left:
    st.markdown('<div class="section-title">Applicant Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    applicant_income = st.number_input("Applicant Income", min_value=0.0, value=5000.0, step=100.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, value=120.0, step=10.0)

    credit_history = st.selectbox("Credit History", ["Yes", "No"])
    credit_value = 1.0 if credit_history == "Yes" else 0.0

    employment_status = st.selectbox("Employment Status", ["No", "Yes"])  # Self_Employed
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section-title">Model Selection</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    kernel_choice = st.radio(
        "Choose SVM Kernel",
        ["Linear SVM", "Polynomial SVM", "RBF SVM"],
        horizontal=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    predict_btn = st.button("Check Loan Eligibility", use_container_width=True)


# ---------------------- OUTPUT PANEL ----------------------
with right:
    st.markdown('<div class="section-title">Prediction Output</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<span class="badge">Real-time Decision</span>', unsafe_allow_html=True)
    st.write("")

    if predict_btn:
        model = build_model(kernel_choice)

        user_input = pd.DataFrame([{
            "ApplicantIncome": applicant_income,
            "LoanAmount": loan_amount,
            "Credit_History": credit_value,
            "Self_Employed": employment_status,
            "Property_Area": property_area
        }])

        pred = model.predict(user_input)[0]
        conf = model.predict_proba(user_input)[0][1]  # probability of approval

        if pred == 1:
            st.markdown('<div class="approved">Loan Approved</div>', unsafe_allow_html=True)
            decision_text = "likely"
        else:
            st.markdown('<div class="rejected">Loan Rejected</div>', unsafe_allow_html=True)
            decision_text = "unlikely"

        st.write("")
        st.markdown(f"**Kernel Used:** `{kernel_choice}`")
        st.markdown("**Model Confidence (Approval Probability):**")
        st.progress(float(conf))
        st.markdown(f"<p class='small'>{conf*100:.2f}% confidence of approval</p>", unsafe_allow_html=True)

        st.write("")
        st.markdown("**Business Explanation:**")
        st.info(
            f"Based on credit history and income pattern, the applicant is {decision_text} to repay the loan."
        )
    else:
        st.warning("Enter applicant details and click **Check Loan Eligibility** to get prediction.")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------- FOOTER ----------------------
st.write("")
st.caption("Developed using Streamlit + Support Vector Machine (SVM)")
