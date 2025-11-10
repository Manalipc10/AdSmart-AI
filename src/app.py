# src/app.py
import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import roc_auc_score
from segmentation import segment_customers
from askmydata import ask_data


# ---------------- CONFIG ----------------
MODEL_PATH = "models/marketing_response_model.joblib"
DATA_PATH = "../dataset/marketing_campaign.csv"
THRESHOLD = 0.175

st.set_page_config(page_title="AdSmart AI Dashboard", layout="wide")
st.title("AdSmart AI – Predictive Marketing Intelligence")

st.info("""
AdSmart AI helps marketing teams understand customer behavior and optimize campaigns 
using machine learning and AI-driven reasoning.  
It transforms data into actionable marketing personas and strategy insights for decision-makers.
""")


# ---------------- LOAD DATA ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(" Trained model not found. Please train it using `train.py` first.")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, sep=';')


model = load_model()
df_raw = load_data()


# ---------------- FEATURE ENGINEERING ----------------
def feature_engineering(df):
    df = df.copy()
    df["Income"] = df["Income"].fillna(df["Income"].median())
    df["Age"] = 2024 - df["Year_Birth"]
    spend_cols = ["MntWines", "MntFruits", "MntMeatProducts",
                  "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    df["TotalSpent"] = df[spend_cols].sum(axis=1)
    purchase_cols = ["NumWebPurchases", "NumCatalogPurchases",
                     "NumStorePurchases", "NumDealsPurchases"]
    df["TotalPurchases"] = df[purchase_cols].sum(axis=1)
    df["FamilySize"] = df["Kidhome"] + df["Teenhome"] + 1
    drop_cols = ["ID", "Dt_Customer", "Z_CostContact", "Z_Revenue"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return df


df = feature_engineering(df_raw)


# ---------------- NAVIGATION ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Marketing Overview", "Customer Intelligence", "Ask My Data"))


# =====================================================
# TAB 1 — MARKETING OVERVIEW
# =====================================================
if page == "Marketing Overview":
    st.subheader("Campaign and Revenue Overview")

    total_customers = len(df)
    total_spent = df["TotalSpent"].sum()
    avg_spent = df["TotalSpent"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Total Revenue", f"${total_spent:,.0f}")
    col3.metric("Avg Spend / Customer", f"${avg_spent:,.0f}")

    st.markdown("""
    **Customer Summary:**  
    - Most customers fall between 40–65 years old.  
    - Income and spend are positively correlated, with a few high-value outliers.  
    - Moderate-income families contribute most to revenue.
    """)

    st.subheader("Model Predictions")
    proba = model.predict_proba(df)[:, 1]
    preds = (proba >= THRESHOLD).astype(int)
    df["Predicted_Probability"], df["Predicted_Response"] = proba, preds

    responders = df["Predicted_Response"].sum()
    response_rate = responders / len(df) * 100
    high_prob = (df["Predicted_Probability"] > 0.7).mean() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Responders", f"{responders:,}")
    c2.metric("Response Rate", f"{response_rate:.1f}%")
    c3.metric("High-Confidence Responders (>70%)", f"{high_prob:.1f}%")

    if "Response" in df_raw.columns:
        auc = roc_auc_score(df_raw["Response"], proba)
        st.metric("Model AUC", f"{auc:.3f}")


# =====================================================
# TAB 2 — CUSTOMER INTELLIGENCE
# =====================================================
elif page == "Customer Intelligence":
    st.subheader("Customer Personas Based on Income & Spend")

    df_seg, seg_summary = segment_customers(df, n_clusters=4)
    cols = st.columns(2)

    for i, (seg_id, row) in enumerate(seg_summary.iterrows()):
        income, spend, family = row["Income"], row["TotalSpent"], row["FamilySize"]

        # Simplified personas based on spend/income
        if income < 40000:
            persona, tone = "Cost-Sensitive Buyers", "Motivated by discounts and deals."
        elif income < 60000:
            persona, tone = "Value-Driven Families", "Appreciate balance of affordability and convenience."
        else:
            persona, tone = "Affluent Enthusiasts", "Seek premium experiences and exclusivity."

        with cols[i % 2]:
            st.markdown(
                f"""
                <div style="background-color:#f8f9fa;border:1px solid #d6d6d6;
                border-radius:12px;padding:18px;margin-bottom:15px;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
                    <h4>Segment {seg_id}: {persona}</h4>
                    <p><b>Average Income:</b> ${income:,.0f}<br>
                       <b>Average Spend:</b> ${spend:,.0f}<br>
                       <b>Family Size:</b> {family:.1f}</p>
                    <p><b>Behavioral Insight:</b> {tone}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# =====================================================
# TAB 3 — ASK MY DATA
# =====================================================
elif page == "Ask My Data":
    st.subheader("Ask My Data (Conversational Insights)")
    st.caption("Ask natural questions about your marketing dataset. Powered by Groq Llama3.")

    query = st.text_area("Ask a question (e.g., 'Which age group spends the most?')")

    if st.button("Generate Insight"):
        with st.spinner("Thinking..."):
            answer = ask_data(query, df)
        st.success("Insight generated successfully.")
        st.markdown(answer)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("Developed with Streamlit, Scikit-learn, and Groq API for AdSmart AI.")
