# =============================================================================
# INSTALLATION GUIDE:
# Run these commands in your terminal/command prompt before running this code:
# 
# pip install pandas numpy matplotlib seaborn scikit-learn
# 
# OR create a requirements.txt file with the above libraries and run:
# pip install -r requirements.txt
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PROBLEM 2: UNSUPERVISED LEARNING ON CREDIT CARD DATA
# CC GENERAL.csv Analysis - Clustering and Anomaly Detection
# =============================================================================

def main():
    print("=" * 70)
    print("UNSUPERVISED LEARNING ANALYSIS - CREDIT CARD DATA")
    print("=" * 70)
    
    # Set style for better visualizations
    plt.style.use('default')
    sns.set_palette("husl")
    
    # =============================================================================
    # 1. DATA LOADING AND PREPROCESSING
    # =============================================================================
    
    print("\n1. LOADING AND PREPROCESSING DATA")
    print("-" * 40)
    
    try:
        # Load the dataset
        df = pd.read_csv('CC GENERAL.csv')
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError:
        print("ERROR: CC GENERAL.csv file not found!")
        print("Please make sure the file is in the same directory as this script.")
        return
    
    # Data preprocessing
    def preprocess_data(data):
        df_clean = data.copy()
        
        # Drop CUST_ID column
        if 'CUST_ID' in df_clean.columns:
            df_clean = df_clean.drop('CUST_ID', axis=1)
        
        # Handle missing values
        print(f"Missing values before cleaning: {df_clean.isnull().sum().sum()}")
        for col in df_clean.columns:
            if df_clean[col].dtype in ['float64', 'int64']:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        df_clean = df_clean.dropna()
        print(f"Missing values after cleaning: {df_clean.isnull().sum().sum()}")
        print(f"Final data shape: {df_clean.shape}")
        
        return df_clean
    
    df_clean = preprocess_data(df)
    
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_clean)
    df_scaled = pd.DataFrame(df_scaled, columns=df_clean.columns)
    
    # =============================================================================
    # 2. CLUSTERING ANALYSIS
    # =============================================================================
    
    print("\n2. CLUSTERING ANALYSIS")
    print("-" * 40)
    
    # 2.1 Determine optimal number of clusters
    print("2.1 Finding optimal number of clusters...")
    
    def find_optimal_clusters(data, max_k=15):
        wcss = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)
            silhouette_avg = silhouette_score(data, kmeans.labels_)
            silhouette_scores.append(silhouette_avg)
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(k_range, wcss, 'bo-', markersize=8, linewidth=2)
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('WCSS')
        ax1.set_title('Elbow Method')
        ax1.grid(True)
        
        ax2.plot(k_range, silhouette_scores, 'ro-', markersize=8, linewidth=2)
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('optimal_clusters.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {optimal_k}")
        return optimal_k
    
    optimal_k = find_optimal_clusters(df_scaled)
    
    # 2.2 K-means Clustering
    print("\n2.2 Performing K-means clustering...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(df_scaled)
    df_clean_kmeans = df_clean.copy()
    df_clean_kmeans['KMeans_Cluster'] = kmeans_labels
    kmeans_silhouette = silhouette_score(df_scaled, kmeans_labels)
    print(f"K-means Silhouette Score: {kmeans_silhouette:.4f}")
    
    # 2.3 Gaussian Mixture Model Clustering
    print("\n2.3 Performing GMM clustering...")
    gmm = GaussianMixture(n_components=optimal_k, random_state=42)
    gmm_labels = gmm.fit_predict(df_scaled)
    df_clean_gmm = df_clean.copy()
    df_clean_gmm['GMM_Cluster'] = gmm_labels
    gmm_silhouette = silhouette_score(df_scaled, gmm_labels)
    print(f"GMM Silhouette Score: {gmm_silhouette:.4f}")
    
    # 2.4 Cluster Visualization
    print("\n2.4 Visualizing clusters...")
    
    def visualize_clusters(data, labels, method_name):
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(data)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, 
                             cmap='viridis', alpha=0.7, s=50)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title(f'{method_name} Clustering - PCA Visualization')
        plt.colorbar(scatter)
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{method_name.lower()}_clusters.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return pca_result
    
    pca_kmeans = visualize_clusters(df_scaled, kmeans_labels, 'K-means')
    pca_gmm = visualize_clusters(df_scaled, gmm_labels, 'GMM')
    
    # 2.5 Cluster Stability Analysis
    print("\n2.5 Analyzing cluster stability...")
    
    def cluster_stability_analysis(data, n_clusters, n_iterations=10, sample_ratio=0.95):
        ari_scores = []
        sample_size = int(len(data) * sample_ratio)
        
        for i in range(n_iterations):
            indices = np.random.choice(len(data), size=sample_size, replace=False)
            sample_data = data[indices]
            
            kmeans_sample = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            sample_labels = kmeans_sample.fit_predict(sample_data)
            original_sample_labels = kmeans_labels[indices]
            
            ari = adjusted_rand_score(original_sample_labels, sample_labels)
            ari_scores.append(ari)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(ari_scores) + 1), ari_scores, 'bo-', markersize=8)
        plt.axhline(y=np.mean(ari_scores), color='r', linestyle='--', 
                   label=f'Mean ARI: {np.mean(ari_scores):.4f}')
        plt.xlabel('Iteration')
        plt.ylabel('Adjusted Rand Index')
        plt.title('Cluster Stability Analysis')
        plt.legend()
        plt.grid(True)
        plt.savefig('cluster_stability.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return ari_scores
    
    stability_scores = cluster_stability_analysis(df_scaled.values, optimal_k)
    print(f"Mean Adjusted Rand Index: {np.mean(stability_scores):.4f}")
    
    # 2.6 Cluster Profiling and Labeling
    print("\n2.6 Profiling and labeling clusters...")
    
    def profile_clusters(data, cluster_labels, method_name):
        df_profiling = pd.DataFrame(data, columns=df_clean.columns)
        df_profiling['Cluster'] = cluster_labels
        
        cluster_sizes = df_profiling['Cluster'].value_counts().sort_index()
        cluster_means = df_profiling.groupby('Cluster').mean()
        global_means = df_profiling.mean()
        
        print(f"\n{method_name} CLUSTER PROFILES:")
        print("=" * 50)
        
        for cluster_id in range(len(cluster_means)):
            print(f"\n--- Cluster {cluster_id} (Size: {cluster_sizes[cluster_id]}) ---")
            
            deviations = (cluster_means.loc[cluster_id] - global_means).abs()
            top_features = deviations.nlargest(3).index.tolist()
            
            print("Key characteristics:")
            for feature in top_features:
                cluster_val = cluster_means.loc[cluster_id, feature]
                global_val = global_means[feature]
                deviation_pct = ((cluster_val - global_val) / global_val) * 100
                direction = "above" if deviation_pct > 0 else "below"
                print(f"  - {feature}: {deviation_pct:+.1f}% {direction} average")
            
            # Simple labeling logic
            balance_ratio = cluster_means.loc[cluster_id, 'BALANCE'] / global_means['BALANCE']
            purchases_ratio = cluster_means.loc[cluster_id, 'PURCHASES'] / global_means['PURCHASES']
            
            if balance_ratio > 1.5 and purchases_ratio > 1.5:
                label = "High-Value Active Users"
            elif balance_ratio > 1.5 and purchases_ratio < 0.7:
                label = "High Balance, Low Spenders"
            elif cluster_means.loc[cluster_id, 'CASH_ADVANCE'] / global_means['CASH_ADVANCE'] > 1.5:
                label = "Heavy Cash Advance Users"
            elif purchases_ratio < 0.5:
                label = "Inactive/Light Users"
            else:
                label = "Average Users"
                
            print(f"Suggested label: {label}")
        
        return cluster_means
    
    print("\nK-MEANS CLUSTERING RESULTS:")
    kmeans_means = profile_clusters(df_scaled, kmeans_labels, "K-means")
    
    print("\nGMM CLUSTERING RESULTS:")
    gmm_means = profile_clusters(df_scaled, gmm_labels, "GMM")
    
    # =============================================================================
    # 3. ANOMALY DETECTION
    # =============================================================================
    
    print("\n3. ANOMALY DETECTION")
    print("-" * 40)
    
    # 3.1 Perform anomaly detection
    print("3.1 Detecting anomalies using multiple methods...")
    
    def perform_anomaly_detection(data):
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.02, random_state=42)
        iso_labels = iso_forest.fit_predict(data)
        iso_scores = iso_forest.decision_function(data)
        
        # Local Outlier Factor
        lof = LocalOutlierFactor(contamination=0.02, n_neighbors=20)
        lof_labels = lof.fit_predict(data)
        lof_scores = lof.negative_outlier_factor_
        
        iso_anomalies = iso_labels == -1
        lof_anomalies = lof_labels == -1
        consensus_anomalies = iso_anomalies & lof_anomalies
        
        print(f"Isolation Forest anomalies: {iso_anomalies.sum()}")
        print(f"Local Outlier Factor anomalies: {lof_anomalies.sum()}")
        print(f"Consensus anomalies: {consensus_anomalies.sum()}")
        
        return {
            'iso_forest': {'labels': iso_labels, 'scores': iso_scores, 'anomalies': iso_anomalies},
            'lof': {'labels': lof_labels, 'scores': lof_scores, 'anomalies': lof_anomalies},
            'consensus': consensus_anomalies
        }
    
    anomaly_results = perform_anomaly_detection(df_scaled)
    
    # 3.2 Analyze and visualize anomalies
    print("\n3.2 Analyzing detected anomalies...")
    
    def analyze_anomalies(data, anomaly_results, original_data):
        consensus_anomalies = anomaly_results['consensus']
        anomaly_indices = np.where(consensus_anomalies)[0]
        
        df_anomalies = original_data.iloc[anomaly_indices].copy()
        df_normal = original_data.iloc[~consensus_anomalies].copy()
        
        print("\nCOMPARISON - Anomalies vs Normal:")
        print("=" * 40)
        
        key_features = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS']
        
        comparison_df = pd.DataFrame({
            'Normal_Mean': df_normal[key_features].mean(),
            'Anomaly_Mean': df_anomalies[key_features].mean(),
            'Difference_Pct': ((df_anomalies[key_features].mean() - df_normal[key_features].mean()) / 
                             df_normal[key_features].mean()) * 100
        })
        
        print(comparison_df.round(2))
        
        # Visualize anomalies
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(data)
        
        plt.figure(figsize=(12, 8))
        normal_mask = ~consensus_anomalies
        plt.scatter(pca_result[normal_mask, 0], pca_result[normal_mask, 1], 
                   c='blue', alpha=0.6, s=50, label='Normal')
        plt.scatter(pca_result[anomaly_indices, 0], pca_result[anomaly_indices, 1], 
                   c='red', alpha=0.8, s=100, label='Anomalies', marker='X')
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('Anomaly Detection Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('anomaly_detection.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df_anomalies
    
    anomalies_df = analyze_anomalies(df_scaled, anomaly_results, df_clean)
    
    # 3.3 Show top anomalies
    print("\n3.3 Top 3 most anomalous instances:")
    print("=" * 40)
    
    top_anomaly_indices = np.argsort(anomaly_results['iso_forest']['scores'])[:3]
    
    for i, idx in enumerate(top_anomaly_indices):
        print(f"\nTop Anomaly #{i+1}:")
        anomaly_data = df_clean.iloc[idx]
        
        extreme_features = anomaly_data.nlargest(3)
        for feature, value in extreme_features.items():
            global_mean = df_clean[feature].mean()
            deviation = ((value - global_mean) / global_mean) * 100
            print(f"  {feature}: {value:.2f} ({deviation:+.1f}% from mean)")
    
    # =============================================================================
    # 4. RESULTS AND SAVING FILES
    # =============================================================================
    
    print("\n4. SAVING RESULTS")
    print("-" * 40)
    
    # Save clustered data
    df_clean_kmeans.to_csv('cc_data_kmeans_clusters.csv', index=False)
    df_clean_gmm.to_csv('cc_data_gmm_clusters.csv', index=False)
    
    # Save anomaly results
    anomaly_summary = pd.DataFrame({
        'IsolationForest_Anomaly': anomaly_results['iso_forest']['anomalies'],
        'LOF_Anomaly': anomaly_results['lof']['anomalies'],
        'Consensus_Anomaly': anomaly_results['consensus']
    })
    anomaly_summary.to_csv('anomaly_results.csv', index=False)
    
    print("Saved files:")
    print("- cc_data_kmeans_clusters.csv")
    print("- cc_data_gmm_clusters.csv") 
    print("- anomaly_results.csv")
    print("- optimal_clusters.png")
    print("- k-means_clusters.png")
    print("- gmm_clusters.png")
    print("- cluster_stability.png")
    print("- anomaly_detection.png")
    
    # =============================================================================
    # 5. FINAL SUMMARY
    # =============================================================================
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print(f"\nCLUSTERING PERFORMANCE:")
    print(f"• Optimal clusters: {optimal_k}")
    print(f"• K-means Silhouette: {kmeans_silhouette:.4f}")
    print(f"• GMM Silhouette: {gmm_silhouette:.4f}")
    print(f"• Cluster stability (ARI): {np.mean(stability_scores):.4f}")
    
    print(f"\nANOMALY DETECTION:")
    consensus_count = anomaly_results['consensus'].sum()
    print(f"• Consensus anomalies: {consensus_count} ({consensus_count/len(df_scaled)*100:.1f}%)")
    
    print(f"\nKEY INSIGHTS:")
    print("• Clusters represent distinct customer segments (High-Value, Inactive, etc.)")
    print("• Anomalies show extreme spending/balance patterns worth investigating")
    print("• Both clustering methods show reasonable performance and stability")
    
    print(f"\nDATASET INFO:")
    print(f"• Original size: {df.shape}")
    print(f"• Cleaned size: {df_clean.shape}")
    print(f"• Features: {len(df_clean.columns)}")
    
    print("\nANALYSIS COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()