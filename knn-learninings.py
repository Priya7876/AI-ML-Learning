# ============================================================================
# IMPORTING LIBRARIES
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

np.random.seed(42)

# ============================================================================
# STEP 1: CREATE SAMPLE DATA
# ============================================================================

print("="*70)
print("K-MEANS CLUSTERING - CUSTOMER SEGMENTATION PROJECT")
print("="*70)

n_customers = 300

budget_shoppers = pd.DataFrame({
    'Annual_Income': np.random.normal(30000, 5000, 100),
    'Spending_Score': np.random.normal(30, 10, 100),
    'Age': np.random.normal(40, 10, 100)
})

average_customers = pd.DataFrame({
    'Annual_Income': np.random.normal(60000, 8000, 100),
    'Spending_Score': np.random.normal(50, 10, 100),
    'Age': np.random.normal(35, 12, 100)
})

premium_shoppers = pd.DataFrame({
    'Annual_Income': np.random.normal(100000, 10000, 100),
    'Spending_Score': np.random.normal(80, 8, 100),
    'Age': np.random.normal(32, 8, 100)
})

customers = pd.concat([budget_shoppers, average_customers, premium_shoppers], ignore_index=True)

customers['Annual_Income'] = customers['Annual_Income'].clip(15000, 150000)
customers['Spending_Score'] = customers['Spending_Score'].clip(1, 100)
customers['Age'] = customers['Age'].clip(18, 70).astype(int)

print("\nCustomer Dataset (First 10 rows):")
print(customers.head(10))

print(f"\nDataset Shape: {customers.shape}")
print(f"   - Number of customers: {customers.shape[0]}")
print(f"   - Number of features: {customers.shape[1]}")

print("\nStatistical Summary:")
print(customers.describe())

# ============================================================================
# STEP 2: VISUALIZE RAW DATA
# ============================================================================

print("\n" + "="*70)
print("VISUALIZING DATA BEFORE CLUSTERING")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(
    customers['Annual_Income'],
    customers['Spending_Score'],
    alpha=0.6,
    s=50,
    c='gray',
    edgecolors='black',
    linewidth=0.5
)
axes[0].set_xlabel('Annual Income ($)')
axes[0].set_ylabel('Spending Score (1-100)')
axes[0].set_title('Income vs Spending (Before Clustering)')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(
    customers['Age'],
    customers['Spending_Score'],
    alpha=0.6,
    s=50,
    c='gray',
    edgecolors='black',
    linewidth=0.5
)
axes[1].set_xlabel('Age (years)')
axes[1].set_ylabel('Spending Score (1-100)')
axes[1].set_title('Age vs Spending (Before Clustering)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('before_clustering.png', dpi=150)
plt.show()

print("Visualization saved as 'before_clustering.png'")

# ============================================================================
# STEP 3: STANDARDIZATION
# ============================================================================

print("\n" + "="*70)
print("STEP 3: STANDARDIZING FEATURES")
print("="*70)

print("\nBEFORE Standardization:")
print(f"   Annual Income: ${customers['Annual_Income'].min():.0f} - ${customers['Annual_Income'].max():.0f}")
print(f"   Spending Score: {customers['Spending_Score'].min():.0f} - {customers['Spending_Score'].max():.0f}")
print(f"   Age: {customers['Age'].min()} - {customers['Age'].max()}")

scaler = StandardScaler()
customers_scaled = scaler.fit_transform(customers)

customers_scaled_df = pd.DataFrame(customers_scaled, columns=customers.columns)

print("\nAFTER Standardization:")
print(customers_scaled_df.describe())

# ============================================================================
# STEP 4: FIND OPTIMAL K
# ============================================================================

print("\n" + "="*70)
print("STEP 4: FINDING OPTIMAL K")
print("="*70)

inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(customers_scaled)

    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(customers_scaled, kmeans.labels_))

print("\nInertia values:")
for k, inertia in zip(K_range, inertias):
    print(f"K={k}: {inertia:.2f}")

print("\nSilhouette Scores:")
for k, score in zip(K_range, silhouette_scores):
    print(f"K={k}: {score:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_title('Elbow Method')
axes[0].set_xlabel('K')
axes[0].set_ylabel('Inertia')
axes[0].axvline(x=3, color='red', linestyle='--')

axes[1].plot(K_range, silhouette_scores, 'go-')
axes[1].set_title('Silhouette Score')
axes[1].set_xlabel('K')
axes[1].set_ylabel('Score')

plt.tight_layout()
plt.savefig('elbow_method.png', dpi=150)
plt.show()

print("Optimal K = 3")

# ============================================================================
# STEP 5: APPLY K-MEANS
# ============================================================================

optimal_k = 3
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)

cluster_labels = kmeans_final.fit_predict(customers_scaled)
customers['Cluster'] = cluster_labels

print("K-Means completed")

# ============================================================================
# STEP 6: ANALYZE CLUSTERS
# ============================================================================

print("\n" + "="*70)
print("STEP 6: ANALYZING CLUSTERS")
print("="*70)

cluster_names = {}

for cluster_id in range(optimal_k):
    cluster_data = customers[customers['Cluster'] == cluster_id]

    avg_income = cluster_data['Annual_Income'].mean()
    avg_spending = cluster_data['Spending_Score'].mean()

    if avg_income > 80000 and avg_spending > 65:
        cluster_names[cluster_id] = "Premium Customers"
    elif avg_income < 45000 and avg_spending < 40:
        cluster_names[cluster_id] = "Budget Shoppers"
    else:
        cluster_names[cluster_id] = "Regular Customers"

customers['Segment'] = customers['Cluster'].map(cluster_names)

print("Cluster labeling done")

# ============================================================================
# STEP 7: VISUALIZE CLUSTERS
# ============================================================================

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for cluster_id in range(optimal_k):
    cluster_data = customers[customers['Cluster'] == cluster_id]
    axes[0].scatter(
        cluster_data['Annual_Income'],
        cluster_data['Spending_Score'],
        c=colors[cluster_id],
        label=cluster_names[cluster_id],
        alpha=0.6
    )

centers = scaler.inverse_transform(kmeans_final.cluster_centers_)

axes[0].scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')
axes[0].set_title('Income vs Spending')
axes[0].legend()

for cluster_id in range(optimal_k):
    cluster_data = customers[customers['Cluster'] == cluster_id]
    axes[1].scatter(
        cluster_data['Age'],
        cluster_data['Spending_Score'],
        c=colors[cluster_id],
        label=cluster_names[cluster_id],
        alpha=0.6
    )

axes[1].scatter(centers[:, 2], centers[:, 1], c='red', marker='X', s=200, label='Centroids')
axes[1].set_title('Age vs Spending')
axes[1].legend()

plt.tight_layout()
plt.savefig('customer_clusters.png', dpi=150)
plt.show()

print("Clustering visualization complete")
print("PROJECT COMPLETE")