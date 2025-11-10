# src/predict.py
import os
import pandas as pd
import joblib

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate engineered features if missing."""
    # Fill missing Income
    if 'Income' in df.columns:
        df['Income'] = df['Income'].fillna(df['Income'].median())

    # Age
    if 'Age' not in df.columns and 'Year_Birth' in df.columns:
        df['Age'] = 2024 - df['Year_Birth']

    # TotalSpent
    spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts',
                  'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    if not all(col in df.columns for col in ['TotalSpent']):
        df['TotalSpent'] = df[[c for c in spend_cols if c in df.columns]].sum(axis=1)

    # TotalPurchases
    purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases',
                     'NumStorePurchases', 'NumDealsPurchases']
    if not all(col in df.columns for col in ['TotalPurchases']):
        df['TotalPurchases'] = df[[c for c in purchase_cols if c in df.columns]].sum(axis=1)

    # FamilySize
    if not all(col in df.columns for col in ['FamilySize']):
        df['FamilySize'] = df[['Kidhome', 'Teenhome']].sum(axis=1) + 1

    return df


def predict(input_csv_path="../dataset/marketing_campaign.csv",
            model_path="models/marketing_response_model.joblib",
            threshold=0.175):
    """
    Load the trained model and predict campaign responses on new data.
    Automatically handles missing engineered features.
    """

    # --- Load model ---
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")

    print("Loading trained model...")
    model = joblib.load(model_path)

    # --- Load input data ---
    print(f"Loading data from: {input_csv_path}")
    df = pd.read_csv(input_csv_path, sep=';')

    # Drop target column if present
    if 'Response' in df.columns:
        df = df.drop(columns=['Response'])

    # --- Apply feature engineering if needed ---
    df = feature_engineering(df)

    # Drop unused columns
    drop_cols = ['ID', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # --- Ensure all expected columns exist ---
    missing_cols = [col for col in model.feature_names_in_ if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}. Filling with 0.")
        for col in missing_cols:
            df[col] = 0

    # --- Predict ---
    print("Generating predictions...")
    proba = model.predict_proba(df)[:, 1]
    preds = (proba >= threshold).astype(int)

    # --- Combine and save ---
    results = df.copy()
    results['Predicted_Probability'] = proba
    results['Predicted_Response'] = preds

    output_path = "../dataset/predictions.csv"
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")

    print("\nSample predictions:")
    print(results[['Predicted_Probability', 'Predicted_Response']].head())

    return results


if __name__ == "__main__":
    predict()
