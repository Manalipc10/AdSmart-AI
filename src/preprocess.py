# src/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def load_and_prepare(path='../dataset/marketing_campaign.csv'):
    df = pd.read_csv(path, sep=';')

    # --- Cleaning & feature engineering ---
    df = df.drop_duplicates()
    df['Income'] = df['Income'].fillna(df['Income'].median())
    df['Age'] = 2024 - df['Year_Birth']
    df['TotalSpent'] = df[['MntWines','MntFruits','MntMeatProducts',
                           'MntFishProducts','MntSweetProducts','MntGoldProds']].sum(axis=1)
    df['TotalPurchases'] = df[['NumWebPurchases','NumCatalogPurchases',
                               'NumStorePurchases','NumDealsPurchases']].sum(axis=1)
    df['FamilySize'] = df['Kidhome'] + df['Teenhome'] + 1
    df = df.drop(columns=['ID','Dt_Customer','Z_CostContact','Z_Revenue'])

    # --- Split ---
    y = df['Response']
    X = df.drop(columns=['Response'])

    cat_features = ['Education','Marital_Status']
    num_features = [col for col in X.columns if col not in cat_features]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, stratify=y)
    return preprocessor, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocessor, X_train, X_test, y_train, y_test = load_and_prepare()
    print(" Data preprocessing successful!")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    print(X_train.head())
