import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="Social Network Ads Prediction",
    layout="centered"
)

st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: #e5e7eb;
}
h1, h2, h3 {
    color: #e5e7eb;
}
.stSelectbox label, .stNumberInput label {
    color: #cbd5f5;
}
.stButton > button {
    background-color: #3b82f6;
    color: white;
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
}
.stButton > button:hover {
    background-color: #2563eb;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Social Network Ads Prediction")
st.caption("Stacking Classifier • Clean • Professional")

data = pd.read_csv("Social_Network_Ads.csv")

X = data[['Gender', 'Age', 'EstimatedSalary']]
y = data['Purchased']

X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})

st.subheader("Enter User Details")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
salary = st.number_input("Estimated Salary", min_value=1000, max_value=200000, value=50000)

if st.button("Submit"):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model = StackingClassifier(
        estimators=[
            ('lr', LogisticRegression()),
            ('knn', KNeighborsClassifier()),
            ('svc', SVC(probability=True))
        ],
        final_estimator=LogisticRegression(),
        cv=5
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    st.subheader("Model Performance")
    st.write(f"Accuracy : {acc:.2%}")
    st.write(f"True Positives (TP) : {tp}")
    st.write(f"True Negatives (TN) : {tn}")
    st.write(f"False Positives (FP) : {fp}")
    st.write(f"False Negatives (FN) : {fn}")

    g_val = 1 if gender == "Male" else 0
    input_df = pd.DataFrame([[g_val, age, salary]], columns=X.columns)
    input_scaled = sc.transform(input_df)

    result = model.predict(input_scaled)[0]

    st.subheader("Final Prediction")
    if result == 1:
        st.success("Customer is likely to PURCHASE the product.")
    else:
        st.warning("Customer is NOT likely to purchase the product.")
