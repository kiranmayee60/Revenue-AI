import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Revenue AI",
    page_icon="💰",
    layout="wide"
)

# ---------------- CUSTOM STYLE ----------------
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.title {
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    font-size: 18px;
    color: #666;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #f5f7fa;
    text-align: center;
}

.prediction {
    padding: 25px;
    border-radius: 15px;
    background-color: #eef6ff;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    '<div class="title">💰 Revenue AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Revenue Analysis & Prediction Dashboard</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------- FILE UPLOAD ----------------
st.subheader("📂 Upload Revenue Data")

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is None:

    st.info(
        "👆 Upload a CSV file containing a column named 'Revenue' to begin."
    )

else:

    # ---------------- READ DATA ----------------
    df = pd.read_csv(uploaded_file)

    st.success("✅ Dataset uploaded successfully!")

    # ---------------- CHECK REVENUE COLUMN ----------------
    revenue_column = None

    for column in df.columns:

        if column.strip().lower() == "revenue":
            revenue_column = column
            break

    if revenue_column is None:

        st.error(
            "❌ Revenue column not found. Please make sure your CSV contains a column named 'Revenue'."
        )

        st.stop()

    # ---------------- CLEAN DATA ----------------
    df[revenue_column] = pd.to_numeric(
        df[revenue_column],
        errors="coerce"
    )

    df = df.dropna(subset=[revenue_column])

    revenue = df[revenue_column]

    # ---------------- METRICS ----------------
    total_revenue = revenue.sum()
    average_revenue = revenue.mean()
    highest_revenue = revenue.max()

    # ---------------- MODEL ----------------
    if len(revenue) >= 2:

        X = pd.DataFrame({
            "Period": range(1, len(revenue) + 1)
        })

        y = revenue.values

        model = LinearRegression()

        model.fit(X, y)

        next_period = len(revenue) + 1

        predicted_revenue = model.predict(
            [[next_period]]
        )[0]

        growth_percentage = (
            (predicted_revenue - average_revenue)
            / average_revenue
        ) * 100

    else:

        predicted_revenue = 0
        growth_percentage = 0

    # ---------------- DASHBOARD CARDS ----------------
    st.subheader("📊 Revenue Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "💰 Total Revenue",
            f"₹{total_revenue:,.0f}"
        )

    with col2:

        st.metric(
            "📊 Average Revenue",
            f"₹{average_revenue:,.0f}"
        )

    with col3:

        st.metric(
            "🚀 Highest Revenue",
            f"₹{highest_revenue:,.0f}"
        )

    with col4:

        if len(revenue) >= 2:

            st.metric(
                "🔮 Predicted Revenue",
                f"₹{predicted_revenue:,.0f}"
            )

        else:

            st.metric(
                "🔮 Predicted Revenue",
                "N/A"
            )

    st.divider()

    # ---------------- DATA PREVIEW ----------------
    with st.expander("📋 View Uploaded Data"):

        st.dataframe(
            df,
            use_container_width=True
        )

    # ---------------- REVENUE TREND ----------------
    st.subheader("📈 Historical Revenue Trend")

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        range(1, len(revenue) + 1),
        revenue,
        marker="o"
    )

    ax.set_xlabel("Period")
    ax.set_ylabel("Revenue")

    ax.set_title(
        "Historical Revenue Performance"
    )

    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    # ---------------- AI PREDICTION ----------------
    if len(revenue) >= 2:

        st.divider()

        st.subheader("🤖 AI Revenue Prediction")

        prediction_col1, prediction_col2 = st.columns(2)

        with prediction_col1:

            st.markdown(
                '<div class="prediction">',
                unsafe_allow_html=True
            )

            st.write("### 🔮 Next Period Prediction")

            st.write(
                f"## ₹{predicted_revenue:,.0f}"
            )

            st.write(
                "Predicted using Linear Regression"
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        with prediction_col2:

            st.markdown(
                '<div class="prediction">',
                unsafe_allow_html=True
            )

            st.write("### 📈 Expected Growth")

            if growth_percentage >= 0:

                st.write(
                    f"## +{growth_percentage:.2f}%"
                )

                st.write(
                    "Revenue is expected to increase."
                )

            else:

                st.write(
                    f"## {growth_percentage:.2f}%"
                )

                st.write(
                    "Revenue may decrease."
                )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        # ---------------- ACTUAL VS PREDICTED ----------------
        st.subheader(
            "📊 Actual vs AI Predicted Revenue"
        )

        future_periods = list(
            range(1, len(revenue) + 2)
        )

        predictions = model.predict(
            pd.DataFrame({
                "Period": future_periods
            })
        )

        fig2, ax2 = plt.subplots(
            figsize=(10, 4)
        )

        ax2.plot(
            range(1, len(revenue) + 1),
            revenue,
            marker="o",
            label="Actual Revenue"
        )

        ax2.plot(
            future_periods,
            predictions,
            marker="o",
            linestyle="--",
            label="AI Prediction"
        )

        ax2.set_xlabel("Period")
        ax2.set_ylabel("Revenue")

        ax2.set_title(
            "Actual Revenue vs AI Prediction"
        )

        ax2.legend()

        ax2.grid(True, alpha=0.3)

        st.pyplot(fig2)

        # ---------------- BUSINESS INSIGHTS ----------------
        st.divider()

        st.subheader("💡 AI Business Insights")

        if predicted_revenue > average_revenue:

            st.success(
                "📈 Positive Revenue Trend: "
                "The model predicts revenue growth. "
                "Consider increasing marketing activities "
                "and maintaining sufficient inventory."
            )

        elif predicted_revenue < average_revenue:

            st.warning(
                "📉 Revenue Risk Detected: "
                "The model predicts weaker revenue. "
                "Consider improving pricing, marketing "
                "and customer retention."
            )

        else:

            st.info(
                "➡️ Stable Revenue: "
                "Revenue is expected to remain relatively stable."
            )

        # ---------------- DATASET INFORMATION ----------------
        st.divider()

        st.subheader("📌 Dataset Information")

        info_col1, info_col2, info_col3 = st.columns(3)

        with info_col1:

            st.metric(
                "Total Records",
                len(df)
            )

        with info_col2:

            st.metric(
                "Revenue Records",
                len(revenue)
            )

        with info_col3:

            st.metric(
                "Average Growth Indicator",
                f"{growth_percentage:.2f}%"
            )