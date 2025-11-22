# Rainfall Prediction Using Machine Learning

This project is a machine learning-based rainfall prediction analysis built from historical temperature, month, year, and rainfall data. The entire project workflow includes data exploration, preprocessing, feature engineering, model training, tuning, and evaluation using various classification algorithms.

## Project Description Summary

The goal is to classify rainfall intensity into classes based on weather features, primarily temperature, month, and year. The dataset was explored visually and statistically to understand its distribution, correlation, and trends over time. Outliers and skewness in rainfall data were treated with capping and log transformation. The target variable was converted into a binary class based on median rainfall.

Data was preprocessed with feature scaling, and class imbalance was addressed using SMOTE oversampling. Several machine learning classifiers—Random Forest, SVM, Logistic Regression, KNN, Gradient Boosting, and Neural Network—were trained and hyperparameter-tuned using grid search. An ensemble voting classifier combined some base models for improved robustness.

Models were evaluated using accuracy, precision, recall, F1 score, ROC AUC, confusion matrices, ROC curves, and detailed classification reports. The top-performing models based on F1 score were SVM, KNN, and Neural Network.

## Key Code Elements & Workflows

### 1. Exploratory Data Analysis (EDA) Mapping

- **Initial Data Inspection**  
  Functions like `.shape`, `.info()`, and `.describe()` were used to check dataset size, data types, and summary statistics, helping understand overall data structure and quality.

- **Univariate Distribution and Outlier Analysis**  
  Histograms and boxplots were created using `sns.histplot` and `sns.boxplot` to visualize distributions and identify outliers in temperature and rainfall.

- **Bivariate Relationship with Regression**  
  Scatter plots with regression lines (using `sns.regplot` or `plt.scatter`) analyzed the linear relationship between temperature and rainfall.

- **Multivariate and Seasonal Pattern Detection**  
  Pairplots colored by month (`sns.pairplot(hue='Month')`) helped detect seasonal variations and multivariate relationships.

- **Time Series Trend Analysis**  
  Line plots (`plt.plot` or `sns.lineplot`) of average monthly rainfall over years visualized long-term trends and seasonality.

- **Feature Correlation Quantification**  
  A heatmap (`sns.heatmap`) of the correlation matrix identified the strength and direction of relationships between features.

### 2. Preprocessing

- Verified no missing values were present.
- Detected and capped extreme rainfall outliers using Interquartile Range (IQR).
- Applied log transformation to correct skewness in rainfall data.
- Created a binary classification target based on median rainfall.

### 3. Feature Scaling and Train-Test Split

- Data was split into training and testing sets.
- Features were scaled using `StandardScaler`.
- SMOTE was applied on training data to address class imbalance synthetically.

### 4. Model Training and Optimization

- Six classifiers were used: Logistic Regression, SVM, Random Forest, Gradient Boosting, KNN, and Neural Network (MLP).
- Hyperparameters were optimized with `GridSearchCV` using cross-validation to ensure model generalization.
- Models were trained on SMOTE-balanced data to improve recall and F1 scores on the minority class.

### 5. Ensemble Model

- A `VotingClassifier` ensemble combined Random Forest, Gradient Boosting, and Logistic Regression with soft voting for improved prediction stability and accuracy.

### 6. Model Evaluation

- Metrics computed: accuracy, precision, recall, F1 score, and ROC AUC.
- Visualizations included ROC curves and confusion matrix heatmaps.
- Detailed classification reports showed per-class performance metrics.

### 7. Final Comparison

- Models ranked by F1 score to balance precision and recall in imbalanced data.
- SVM, KNN, and Neural Network were identified as best performers.

### 8. Visualization Outputs

- Distribution histograms and boxplots for temperature and rainfall.
- Regression plots of temperature vs rainfall.
- Pairplots colored by months.
- Monthly rainfall trends over years.
- Correlation heatmaps.
- ROC curve comparisons.
- Confusion matrix heatmaps.

---

This description captures the project's core methodologies, workflows, and outcomes. It uses common Python data science libraries including pandas, NumPy, matplotlib, seaborn, scikit-learn, and imblearn for SMOTE. The project showcases comprehensive predictive modeling for environmental data, transforming raw weather records into robust rainfall classification insights.
