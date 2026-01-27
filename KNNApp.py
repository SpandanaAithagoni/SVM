import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk Prediction", page_icon="🏦", layout="wide")

st.markdown("""
<style>
body { background-color: #f4f7fb; }
h1 { color: #3a506b; }
.subtle { color: #6c757d; }
.stButton button {
    background-color: #5bc0be;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🏦 Credit Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtle'>KNN-based loan approval model</p>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("credit_risk_dataset.csv")
    df['person_emp_length'].fillna(df['person_emp_length'].mean(), inplace=True)
    df['loan_int_rate'].fillna(df['loan_int_rate'].mean(), inplace=True)
    df['loan_grade'] = df['loan_grade'].map({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7})
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y':1,'N':0})
    df = pd.get_dummies(df, columns=['person_home_ownership','loan_intent'], drop_first=True)
    return df

df = load_data()

X = df.drop('loan_status', axis=1)
y = df['loan_status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

k = st.slider("Select K (number of neighbors)", 3, 15, 7, step=2)

knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
knn.fit(X_train, y_train)

st.subheader("Applicant Details")

c1, c2, c3 = st.columns(3)

with c1:
    age = st.number_input("Age", 18, 70)
    income = st.number_input("Annual Income", 10000, 10000000)

with c2:
    loan_amt = st.number_input("Loan Amount", 1000, 1000000)
    interest = st.number_input("Interest Rate (%)", 1.0, 30.0)

with c3:
    grade = st.selectbox("Loan Grade", list("ABCDEFG"))
    default = st.selectbox("Previous Default", ["No", "Yes"])

input_df = pd.DataFrame([X.mean()], columns=X.columns)

input_df['person_age'] = age
input_df['person_income'] = income
input_df['loan_amnt'] = loan_amt
input_df['loan_int_rate'] = interest
input_df['loan_grade'] = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}[grade]
input_df['cb_person_default_on_file'] = 1 if default == "Yes" else 0

input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    pred = knn.predict(input_scaled)[0]
    prob = knn.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.success(f"Loan Approved ✅  (Probability: {prob:.2f})")
    else:
        st.error(f"Loan Rejected ❌  (Approval Probability: {prob:.2f})")

st.subheader("Model Performance")
st.metric("Accuracy", f"{accuracy_score(y_test, knn.predict(X_test)):.2f}")

fig, ax = plt.subplots()
sns.heatmap(
    confusion_matrix(y_test, knn.predict(X_test)),
    annot=True,
    fmt="d",
    cmap="Pastel2",
    ax=ax
)
st.pyplot(fig)
