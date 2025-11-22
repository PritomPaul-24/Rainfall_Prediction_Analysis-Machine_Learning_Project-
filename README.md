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
<img width="954" height="703" alt="image" src="https://github.com/user-attachments/assets/e1b57931-4be5-40c6-92f7-28689fc7166b" />

- **Bivariate Relationship with Regression**  
  Scatter plots with regression lines (using `sns.regplot` or `plt.scatter`) analyzed the linear relationship between temperature and rainfall.
<img width="1051" height="672" alt="image" src="https://github.com/user-attachments/assets/d7c6157d-8127-48fb-99cd-639a5653ab31" />

- **Multivariate and Seasonal Pattern Detection**  
  Pairplots colored by month (`sns.pairplot(hue='Month')`) helped detect seasonal variations and multivariate relationships.
<img width="853" height="806" alt="image" src="https://github.com/user-attachments/assets/65e81c7d-b8a7-4570-91d4-891a2f298d46" />

- **Time Series Trend Analysis**  
  Line plots (`plt.plot` or `sns.lineplot`) of average monthly rainfall over years visualized long-term trends and seasonality.
<img width="940" height="705" alt="image" src="https://github.com/user-attachments/assets/91a5493e-7fb6-4c3a-b2b3-74f1f7cbdcee" />

- **Feature Correlation Quantification**  
  A heatmap (`sns.heatmap`) of the correlation matrix identified the strength and direction of relationships between features.
<img width="938" height="793" alt="image" src="https://github.com/user-attachments/assets/3c690e81-44ac-418e-96b8-e5203ca4b69a" />

### 2. Preprocessing

- Verified no missing values were present.
- Detected and capped extreme rainfall outliers using Interquartile Range (IQR).
- Applied log transformation to correct skewness in rainfall data.
- Created a binary classification target based on median rainfall.

### 3. Feature Scaling and Train-Test Split

- Data was split into training and testing sets.
- Features were scaled using `StandardScaler`.
- SMOTE was applied on training data to address class imbalance synthetically.

## 4. Model Training and Optimization

The core of the project involved training and optimizing a diverse set of classifiers to learn the complex relationship between the weather features and the target rainfall class. This process ensured achieving the highest possible predictive accuracy.

### 4.1. Defined Multiple Classifiers

We selected six robust and varied machine learning algorithms to thoroughly explore the solution space for our binary classification problem. The models included:  
- Linear/Kernel Methods: Logistic Regression and Support Vector Machine (SVM)  
- Ensemble Methods: Random Forest and Gradient Boosting  
- Distance-Based Method: K-Nearest Neighbors (KNN)  
- Deep Learning: Neural Network (Multi-layer Perceptron)  

### 4.2. Hyperparameter Tuning using GridSearchCV

To ensure each model performed optimally, we utilized GridSearchCV with cross-validation on the training data.  
- **Process:** GridSearchCV systematically explores a defined grid of hyperparameter combinations for each algorithm. Cross-validation (e.g., k-fold) validates the model on different data partitions to robustly estimate performance and identify the best parameters.  
- **Goal:** Minimize the risk of sub-optimal parameters and significantly improve the model's ability to generalize to unseen data.

### 4.3. Trained Optimized Models on SMOTE-Balanced Data

The final, best-performing hyperparameter sets were used to train the models on the preprocessed data, balanced using SMOTE (Synthetic Minority Oversampling Technique).  
- **SMOTE's Role:** By synthetically generating samples of the minority class, SMOTE mitigates class imbalance common in real-world datasets.  
- **Impact on Performance:** Training on SMOTE-balanced data prevents bias toward the majority class, boosting Recall and F1 Score on the positive (high rainfall) class.

### Visualization Attachment Points and Descriptions

- **Visualization: ROC Curves Comparison**  
  - *Placement:* Typically shown immediately after the training summary to illustrate the discriminatory power of the trained models.  
  - *Description:* Plots True Positive Rate vs. False Positive Rate for all six classifiers and the ensemble model. Area Under the Curve (AUC) values (e.g., AUC = 0.99) quantify overall model quality, showing successful high-performance training.
<img width="645" height="520" alt="image" src="https://github.com/user-attachments/assets/af0d7a13-0b52-42f0-861b-c45c3feb8ac9" />

- **Visualization: Confusion Matrices Heatmaps**  
  - *Placement:* Follows ROC curves for granular detail of prediction errors.  
  - *Description:* Six heatmaps (one per model) display counts of True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN) from the test set, illustrating how well each model predicts both high and low rainfall classes (e.g., low FN count in the SVM matrix).
<img width="814" height="545" alt="image" src="https://github.com/user-attachments/assets/dd2c8992-0f7d-4536-8786-e511a551d6d3" />


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
