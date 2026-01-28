import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

st.markdown("""
<style>
body {
    background-color: #f6f8fb;
}
h1, h2, h3 {
    color: #1f3c88;
}
.note {
    color: #1f6f8b;
    font-size: 15px;
    font-weight: 500;
    background-color: #eaf4f8;
    padding: 8px 12px;
    border-left: 4px solid #1f6f8b;
}

</style>
""", unsafe_allow_html=True)

st.title("🟢 Customer Segmentation Dashboard")
st.write(
    "This system uses **K-Means Clustering** to group customers based on their "
    "purchasing behavior and similarities."
)

@st.cache_data
def load_data():
    return pd.read_csv("Wholesale customers data.csv")

df = load_data()
numeric_df = df.select_dtypes(include=["int64", "float64"])

st.markdown("### 🔧 Clustering Controls")

c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 2])

with c1:
    feature_1 = st.selectbox("Feature 1", numeric_df.columns)

with c2:
    feature_2 = st.selectbox("Feature 2", numeric_df.columns, index=1)

with c3:
    k = st.slider("Clusters (K)", 2, 10, 4)

with c4:
    random_state = st.number_input("Random State", min_value=0, value=42, step=1)

with c5:
    run_button = st.button("🟦 Run Clustering")

if feature_1 == feature_2:
    st.warning("Please select two different features.")
    st.stop()

if run_button:

    st.markdown("## 📊 Cluster Visualization")

    X = numeric_df[[feature_1, feature_2]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=random_state)
    clusters = kmeans.fit_predict(X_scaled)

    fig, ax = plt.subplots()
    ax.scatter(X[feature_1], X[feature_2], c=clusters)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    ax.scatter(centers[:, 0], centers[:, 1], marker='X', s=200)
    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_title("Customer Clusters")

    st.pyplot(fig)

    st.markdown("## 📋 Cluster Summary")

    summary = (
        X.assign(Cluster=clusters)
        .groupby("Cluster")
        .agg(
            Count=("Cluster", "count"),
            Avg_Feature_1=(feature_1, "mean"),
            Avg_Feature_2=(feature_2, "mean")
        )
        .reset_index()
    )

    st.dataframe(summary, use_container_width=True)

    st.markdown("## 💼 Business Interpretation")

    for i in summary["Cluster"]:
        avg1 = summary.loc[summary["Cluster"] == i, "Avg_Feature_1"].values[0]
        avg2 = summary.loc[summary["Cluster"] == i, "Avg_Feature_2"].values[0]

        if avg1 > X[feature_1].mean() and avg2 > X[feature_2].mean():
            desc = "High-spending customers across selected categories"
            icon = "🟢"
        elif avg1 < X[feature_1].mean() and avg2 < X[feature_2].mean():
            desc = "Budget-conscious customers with low spending"
            icon = "🟡"
        else:
            desc = "Moderate spenders with selective purchasing behavior"
            icon = "🔵"

        st.write(f"{icon} **Cluster {i}:** {desc}")

    st.markdown("## 📌 Insight")

    st.markdown(
        "<p class='note'>"
        "Customers in the same cluster exhibit similar purchasing behaviour "
        "and can be targeted with similar business strategies."
        "</p>",
        unsafe_allow_html=True
    )

else:
    st.info("Select features and click **Run Clustering** to begin.")
