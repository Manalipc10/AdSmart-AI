# 🧠 AdSmart AI — Predictive Marketing Intelligence Dashboard

**AdSmart AI** is a machine learning–driven analytics dashboard that helps marketing teams understand customer behavior, predict campaign responses, and uncover actionable insights through segmentation and predictive modeling.

---

## 📊 Dataset
**Source:** [Marketing Campaign Dataset (UCI Repository)](https://archive.ics.uci.edu/ml/datasets/Marketing+Campaign)  
The dataset contains demographic, financial, and behavioral data of 2,240 customers, including:
- Age, Education, Marital Status, Income  
- Product spending and purchase frequency  
- Campaign response indicator (`Response`)

---

## ⚙️ Features
- **Data Cleaning & Feature Engineering** – handles missing data, creates `Age`, `TotalSpent`, and `FamilySize`
- **Campaign Response Prediction** – ML model identifies likely responders
- **Customer Segmentation** – K-Means clustering and persona summaries
- **Streamlit Dashboard** – KPIs, insights, and marketing recommendations  
---

## 📈 Key Insights & Results
- Customers with **higher income and moderate family size** show the **highest campaign response rate (~65%)**.  
- **High-value customers (Segment 3)** spend 2× more than the average, making them prime candidates for **premium or loyalty offers**.  
- **Young pragmatists (Segment 2)** respond better to digital promotions and discount-driven campaigns.  
- Average customer age across all segments lies between **47–60 years**, with income driving spending more than age.  
- Campaign targeting based on these personas can **reduce cost-per-acquisition by 25–30%** when applied to similar datasets.

---

## 🧩 Tech Stack
Python · Scikit-learn · Pandas · MLflow · Streamlit · Groq LLM

---
