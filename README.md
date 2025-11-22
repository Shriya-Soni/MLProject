# Unsupervised Learning Analysis - Credit Card Data

##  Project Overview
This project performs comprehensive unsupervised learning analysis on credit card customer data, including clustering and anomaly detection to identify customer segments and unusual patterns.

##  Key Objectives
- Perform customer segmentation using clustering algorithms
- Identify optimal number of clusters
- Detect anomalous transactions and customer behaviors
- Provide actionable insights for business strategy

##  Dataset Information
- **Original Dataset**: 8,950 customers × 18 features
- **Cleaned Dataset**: 8,950 customers × 17 features (after preprocessing)
- **Missing Values Handled**: 314 values imputed using median strategy

##  Methodology

### Data Preprocessing
- Removed customer identifier (`CUST_ID`)
- Handled missing values using median imputation
- Standardized features for clustering algorithms

### Clustering Analysis
- **Optimal Clusters**: 3 (determined using Elbow Method and Silhouette Analysis)
- **Algorithms Used**: K-means and Gaussian Mixture Models (GMM)
- **Validation**: Cluster stability testing with Adjusted Rand Index

### Anomaly Detection
- **Techniques**: Isolation Forest and Local Outlier Factor (LOF)
- **Consensus Approach**: Combined results from both methods
- **Contamination Rate**: 2% (0.02)

##  Results

### Clustering Performance
| Metric | K-means | GMM |
|--------|---------|-----|
| **Silhouette Score** | 0.2510 | 0.0978 |
| **Cluster Stability (ARI)** | 0.9792 | 0.9792 |

### K-means Cluster Profiles

#### Cluster 0 - High-Value Active Users (1,275 customers)
- **Characteristics**: Extremely high purchase transactions
- **Behavior**: Frequent and high-value purchasers
- **Business Implication**: Premium customer segment with high revenue potential

#### Cluster 1 - Inactive/Light Users (6,114 customers)
- **Characteristics**: Low balance, low credit limit, infrequent cash advances
- **Behavior**: Minimal card usage and transactions
- **Business Implication**: Opportunity for engagement and activation campaigns

#### Cluster 2 - High Balance, Low Spenders (1,561 customers)
- **Characteristics**: High cash advance frequency and amounts
- **Behavior**: Prefer cash advances over purchases
- **Business Implication**: Potential risk segment requiring monitoring

### GMM Cluster Profiles
- **Cluster 0**: High-Value Active Users (2,606 customers)
- **Cluster 1**: High Balance, Low Spenders (2,934 customers)  
- **Cluster 2**: Average Users (3,410 customers)

### Anomaly Detection Results

#### Detection Summary
- **Isolation Forest**: 179 anomalies detected
- **Local Outlier Factor**: 179 anomalies detected  
- **Consensus Anomalies**: 30 instances (0.3% of dataset)

#### Anomaly Characteristics vs Normal Behavior
| Feature | Normal Mean | Anomaly Mean | Difference |
|---------|-------------|--------------|------------|
| **BALANCE** | $1,545.83 | $7,108.68 | +359.86% |
| **PURCHASES** | $968.92 | $11,197.03 | +1055.62% |
| **CASH_ADVANCE** | $955.40 | $7,957.68 | +732.92% |
| **CREDIT_LIMIT** | $4,465.32 | $13,106.67 | +193.52% |
| **PAYMENTS** | $1,682.21 | $16,877.84 | +903.31% |

#### Top 3 Most Anomalous Instances

**Anomaly #1**
- Payments: $23,018.58 (+1,228.1% above mean)
- Purchases: $22,009.92 (+2,094.0% above mean) 
- Balance: $19,043.14 (+1,117.2% above mean)

**Anomaly #2**
- Purchases: $41,050.40 (+3,991.9% above mean)
- One-off Purchases: $40,624.06 (+6,757.1% above mean)
- Payments: $36,066.75 (+1,981.0% above mean)

**Anomaly #3**
- Credit Limit: $20,000.00 (+345.0% above mean)
- Balance: $13,673.08 (+774.0% above mean)
- Payments: $11,717.31 (+576.1% above mean)

##  Key Insights

### Clustering Insights
1. **Clear Customer Segmentation**: Three distinct customer groups identified
2. **K-means Superiority**: Better performance than GMM for this dataset (Silhouette: 0.2510 vs 0.0978)
3. **High Stability**: Excellent cluster consistency (ARI: 0.9792)

### Anomaly Insights
1. **Extreme Spending Patterns**: Anomalies show purchase amounts 10x+ above average
2. **High-Value Transactions**: Unusual payment and balance behaviors
3. **Potential Fraud Indicators**: Extreme deviations from normal customer behavior

### Business Implications
- **Cluster 0**: Target for premium services and loyalty programs
- **Cluster 1**: Focus on activation and engagement strategies  
- **Cluster 2**: Monitor for cash advance risks and offer purchase incentives
- **Anomalies**: Investigate for potential fraud or data quality issues

## Files Generated
- `cc_data_kmeans_clusters.csv` - Data with K-means cluster assignments
- `cc_data_gmm_clusters.csv` - Data with GMM cluster assignments
- `anomaly_results.csv` - Anomaly detection results
- `optimal_clusters.png` - Elbow method and silhouette analysis plots
- `k-means_clusters.png` - K-means clustering visualization
- `gmm_clusters.png` - GMM clustering visualization  
- `cluster_stability.png` - Cluster stability analysis
- `anomaly_detection.png` - Anomaly visualization

##  Conclusion
The analysis successfully identified three meaningful customer segments with distinct behavioral patterns and detected significant anomalies showing extreme financial behaviors. The K-means algorithm demonstrated superior performance for this dataset, and the high cluster stability ensures reliable results for business decision-making.

**Recommendations**:
1. Develop targeted marketing strategies for each cluster
2. Investigate anomalous instances for potential fraud
3. Use insights for risk management and customer relationship optimization

---
*Analysis completed using Python with scikit-learn, pandas, and matplotlib*
