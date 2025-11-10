# check_cleaned_dataset.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# --- 1. Load dataset ---
# adjust path if needed (run from inside src → use ../dataset/marketing_campaign.csv)
df = pd.read_csv('../dataset/marketing_campaign.csv', sep=';')
print("Raw shape:", df.shape)
print("Initial missing values:\n", df.isnull().sum().sort_values(ascending=False).head(10))
print("\nPreview:\n", df.head(2))

# --- 2. Drop duplicates ---
df = df.drop_duplicates()
print(f"\n Duplicates removed. Remaining rows: {len(df)}")

# --- 3. Handle missing values ---
df['Income'] = df['Income'].fillna(df['Income'].median())
print(f"Missing 'Income' filled with median: {df['Income'].median():.2f}")

# --- 4. Feature engineering ---
df['Age'] = 2024 - df['Year_Birth']

spend_cols = ['MntWines','MntFruits','MntMeatProducts','MntFishProducts',
              'MntSweetProducts','MntGoldProds']
df['TotalSpent'] = df[spend_cols].sum(axis=1)

purchase_cols = ['NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumDealsPurchases']
df['TotalPurchases'] = df[purchase_cols].sum(axis=1)

df['FamilySize'] = df['Kidhome'] + df['Teenhome'] + 1

# --- 5. Drop irrelevant columns ---
df = df.drop(columns=['ID','Dt_Customer','Z_CostContact','Z_Revenue'], errors='ignore')

# --- 6. Outlier handling (optional) ---
for col in ['Income','TotalSpent']:
    q_low = df[col].quantile(0.01)
    q_high = df[col].quantile(0.99)
    df[col] = df[col].clip(q_low, q_high)

print("\n Feature engineering complete. New columns added: Age, TotalSpent, TotalPurchases, FamilySize")

# --- 7. Target separation ---
y = df['Response']
X = df.drop(columns=['Response'])

# --- 8. Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 9. Preprocessor ---
cat_cols = ['Education','Marital_Status']
num_cols = [c for c in X.columns if c not in cat_cols]
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)
])

# --- 10. Basic checks ---
print("\n Cleaned data preview:")
print(X_train.head(3))
print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)
print("Missing values after cleaning:\n", X_train.isnull().sum().sum())

# --- 11. Summary of key engineered features ---
print("\nFeature Summary (cleaned):")
print(X_train[['Age','TotalSpent','TotalPurchases','FamilySize','Income']].describe())

# --- 12. Optional: Save cleaned dataset for inspection ---
cleaned = X_train.copy()
cleaned['Response'] = y_train.values
cleaned.to_csv('../dataset/preprocessed_marketing_campaign.csv', index=False)
print("\n Cleaned dataset saved → dataset/preprocessed_marketing_campaign.csv")

# --- 13. Visual validation ---
plt.figure(figsize=(10,4))
sns.histplot(X_train['Income'], kde=True)
plt.title("Distribution of Income (cleaned)")
plt.show()

plt.figure(figsize=(10,4))
sns.histplot(X_train['TotalSpent'], kde=True)
plt.title("Distribution of TotalSpent (cleaned)")
plt.show()

print("\n Cleaning verification complete.")
