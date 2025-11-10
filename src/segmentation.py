# src/segmentation.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def prepare_segmentation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that all required engineered features for segmentation exist.
    If they don't, compute them using standard feature engineering logic.
    """

    df = df.copy()

    # --- Basic feature engineering ---
    if 'Age' not in df.columns and 'Year_Birth' in df.columns:
        df['Age'] = 2024 - df['Year_Birth']

    spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts',
                  'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    if not all(col in df.columns for col in ['TotalSpent']):
        df['TotalSpent'] = df[[c for c in spend_cols if c in df.columns]].sum(axis=1)

    purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases',
                     'NumStorePurchases', 'NumDealsPurchases']
    if not all(col in df.columns for col in ['TotalPurchases']):
        df['TotalPurchases'] = df[[c for c in purchase_cols if c in df.columns]].sum(axis=1)

    if 'FamilySize' not in df.columns and {'Kidhome', 'Teenhome'}.issubset(df.columns):
        df['FamilySize'] = df['Kidhome'] + df['Teenhome'] + 1

    # Fill Income if missing
    if 'Income' in df.columns:
        df['Income'] = df['Income'].fillna(df['Income'].median())

    return df


def segment_customers(df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42):
    """
    Cluster customers into behavioral segments based on key features.
    Returns:
        df_seg: DataFrame with 'Segment' column
        segment_summary: mean stats per segment
    """

    df_seg = prepare_segmentation_features(df)

    # Features to use for segmentation
    features = ['Age', 'Income', 'TotalSpent', 'TotalPurchases', 'FamilySize']

    # Check for missing required columns
    missing = [c for c in features if c not in df_seg.columns]
    if missing:
        raise ValueError(f"Missing required columns for segmentation: {missing}")

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_seg[features])

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df_seg['Segment'] = kmeans.fit_predict(X_scaled)

    # Segment summary statistics
    segment_summary = df_seg.groupby('Segment')[features].mean().round(2)

    return df_seg, segment_summary


if __name__ == "__main__":
    # Example standalone run (for testing)
    df = pd.read_csv("../dataset/marketing_campaign.csv", sep=';')

    # Run segmentation pipeline
    df_seg, summary = segment_customers(df, n_clusters=4)

    print("Segmentation complete. Summary:")
    print(summary)
