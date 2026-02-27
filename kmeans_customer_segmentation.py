"""
K-Means Customer Segmentation on Credit Card Dataset

- Loads Credit_Card_Dataset.csv
- Selects key financial risk features
- Scales data
- Runs Elbow Method to help choose K
- Trains K-Means model
- Attaches cluster labels to original data
- Creates seaborn visualisations
- Outputs cluster profile + silhouette score

Author: Kiko + team
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sns.set(style="whitegrid")


# -----------------------------
# Configuration
# -----------------------------

# Path to your dataset (relative to this script)
DATA_PATH = "Credit_Card_Dataset.csv"

# Features used for clustering (update if your column names differ)
FEATURE_COLS = [
    "Debt_To_Income_Ratio",
    "Credit_Utilization_Ratio",
    "Fraud_Transactions",
    "Defaulted",
]

# Final number of clusters (K) – update after checking the elbow plot
FINAL_K = 3


# -----------------------------
# Helper Functions
# -----------------------------

def load_dataset(path: str) -> pd.DataFrame:
    """Load the credit card dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. "
            f"Make sure the CSV is in the same folder as this script."
        )
    df = pd.read_csv(path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df


def preprocess_features(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """
    Select and scale features for clustering.
    Returns the scaled feature matrix and the fitted scaler.
    """
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"The following columns are missing from the dataset: {missing_cols}"
        )

    X = df[feature_cols].copy()

    # Drop rows with missing values in selected features (simple handling)
    before_rows = X.shape[0]
    X = X.dropna()
    after_rows = X.shape[0]
    if before_rows != after_rows:
        print(f"Dropped {before_rows - after_rows} rows with NaN in feature columns.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Features scaled. Shape:", X_scaled.shape)
    return X_scaled, scaler, X.index  # return original index to align later


def plot_elbow_method(X_scaled: np.ndarray, k_min: int = 1, k_max: int = 10) -> None:
    """Plot the Elbow Method to help choose the optimal K."""
    wcss = []

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    sns.lineplot(x=range(k_min, k_max + 1), y=wcss, marker="o")
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
    plt.xticks(range(k_min, k_max + 1))
    plt.tight_layout()
    plt.show()


def run_kmeans(X_scaled: np.ndarray, k: int) -> np.ndarray:
    """Fit K-Means and return cluster labels."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    print(f"K-Means fitted with K = {k}")
    return labels


def plot_clusters(df: pd.DataFrame, feature_x: str, feature_y: str) -> None:
    """
    Plot a scatter plot of two features coloured by cluster label.
    Assumes df has a 'Cluster' column.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x=feature_x,
        y=feature_y,
        hue="Cluster",
        palette="viridis",
        s=40,
        alpha=0.8,
    )
    plt.title(f"Customer Segments: {feature_x} vs {feature_y}")
    plt.tight_layout()
    plt.show()


def plot_cluster_counts(df: pd.DataFrame) -> None:
    """Plot the number of customers in each cluster."""
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Cluster", palette="viridis")
    plt.title("Number of Customers in Each Cluster")
    plt.tight_layout()
    plt.show()


def compute_cluster_profile(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Return a DataFrame summarising average values per cluster."""
    profile = df.groupby("Cluster")[feature_cols].mean()
    print("\nCluster Profile (feature means by cluster):")
    print(profile)
    return profile


# -----------------------------
# Main Script
# -----------------------------

def main():
    # 1. Load data
    df = load_dataset(DATA_PATH)

    # 2. Preprocess + scale features
    X_scaled, scaler, valid_idx = preprocess_features(df, FEATURE_COLS)

    # Align original df to rows that were used after dropping NaNs
    df_used = df.loc[valid_idx].copy()

    # 3. Elbow method (inspect the plot and adjust FINAL_K accordingly)
    print("Plotting Elbow Method – use this to choose a good K.")
    plot_elbow_method(X_scaled, k_min=1, k_max=10)

    # 4. Run final K-Means with chosen K
    labels = run_kmeans(X_scaled, FINAL_K)

    # Attach cluster labels
    df_used["Cluster"] = labels

    # 5. Compute silhouette score
    sil_score = silhouette_score(X_scaled, labels)
    print(f"\nSilhouette Score for K = {FINAL_K}: {sil_score:.3f}")

    # 6. Cluster profile table
    profile = compute_cluster_profile(df_used, FEATURE_COLS)

    # 7. Visualisations
    # Scatter plot of two key features
    if "Debt_To_Income_Ratio" in df_used.columns and "Credit_Utilization_Ratio" in df_used.columns:
        plot_clusters(df_used, "Debt_To_Income_Ratio", "Credit_Utilization_Ratio")

    # Count of customers per cluster
    plot_cluster_counts(df_used)

    # 8. (Optional) Save outputs for reporting / GitHub
    output_clusters_path = "credit_card_clusters.csv"
    output_profile_path = "cluster_profile.csv"

    df_used.to_csv(output_clusters_path, index=False)
    profile.to_csv(output_profile_path)

    print(f"\nSaved clustered dataset to: {output_clusters_path}")
    print(f"Saved cluster profile to: {output_profile_path}")


if __name__ == "__main__":
    main()