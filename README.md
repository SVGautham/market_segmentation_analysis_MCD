# market_segmentation_analysis_MCD
This repository contains a Python-based implementation of a customer segmentation analysis for McDonald's, which was originally written in R. The analysis includes data preprocessing, dimensionality reduction using PCA, clustering, and logistic regression to explore and segment customer preferences based on survey responses.

# Overview
The project is designed to achieve the following:

Data Preprocessing:

Load and inspect the McDonald's customer survey dataset.
Convert categorical responses (Yes/No) to binary format (0/1).
Compute and display the column means for the binary data.
Dimensionality Reduction:

Perform Principal Component Analysis (PCA) to reduce the dataset's dimensionality and identify key components.
Visualize the PCA projection and analyze the explained variance ratios.
Clustering:

Implement K-Means clustering to identify distinct customer segments.
Use the Elbow Method to determine the optimal number of clusters.
Visualize the distribution of clusters and the results of hierarchical clustering.
Logistic Regression:

Fit a logistic regression model to understand the factors influencing customer satisfaction (using 'Like' as the target variable).
Analyze the regression results to determine significant predictors.
Predict cluster memberships and visualize the results.
Visualization:

Generate various plots including the PCA projection, cluster distributions, hierarchical clustering dendrogram, and the relationship between customer age and satisfaction.
# Key Libraries Used
pandas: For data manipulation and preprocessing.

numpy: For numerical operations.

scikit-learn: For PCA, K-Means clustering, and preprocessing.

scipy: For hierarchical clustering and dendrogram visualization.

matplotlib: For data visualization.

statsmodels: For logistic regression modeling.
