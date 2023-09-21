import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV



data = pd.read_csv('C:\\Users\\BANTA\\Desktop\\ML projects\\dataSets\\winequality-red_decision_tree.csv')

# Explore the data:
print(data.head().T)
print(data.shape)  # Check the dimensions of the dataset
print(data.columns)  # Check the column names
print(data.info())  # Get information about the data types
print(data.describe())  # Summary statistics of the dataset

print(data.isnull().sum())  # Check for missing values
data = data.fillna(data.mean())

# Explore the distribution of the target variable
print(data['quality'].value_counts())
# Assuming your DataFrame is called 'df' and the label column is 'quality'
data['quality'] = data['quality'].apply(lambda x: 1 if x > 6 else 0)
print(data)

# Create histograms for each feature
data.hist(figsize=(12, 10))
plt.show()

# Create a correlation matrix
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Perform exploratory data analysis (EDA)
# Analyze the distribution of the target variable

# Analyze the relationships between features and the target variable
sns.boxplot(x='quality', y='alcohol', data=data)
plt.show()

# Splitting into training set & test set using scikit learn's function
X = data.drop('quality', axis=1)  # Features
y = data['quality']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create a RobustScaler object
scaler = RobustScaler()
# Fit the scaler on the training data
scaler.fit(X_train)
#By fitting the RobustScaler on the training data, it calculates the median and interquartile range (IQR) for each feature. Then, it scales the features using the formula (X - median) / IQR, where X is the original value of the feature. This scaling method is more robust to outliers compared to standard scaling techniques like the MinMaxScaler or StandardScaler
# Transform the training and testing data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Training model on training set
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

# finding the feature importance
feature_importances = rf_classifier.feature_importances_
# Create a DataFrame to display feature importances
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values('Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Print feature importances in descending order
print(importance_df)


y_pred = rf_classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision
precision = precision_score(y_test, y_pred)
# Calculate F1-score
f1 = f1_score(y_test, y_pred)



# Calculate recall
recall = recall_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
#print(classification_report(y_test, y_pred))


#define a parameter grid with different values for the hyperparameters n_estimators, max_depth, min_samples_split, and min_samples_leaf
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the random forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

#The GridSearchCV object performs a grid search over these parameter values using 5-fold cross-validation (cv=5) and evaluates the models based on accuracy (scoring='accuracy').
# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')

#After fitting the GridSearchCV object on the training data, the best parameters is obtain using best_params_ and the best score using best_score_
# Fit the GridSearchCV object on the training data
grid_search.fit(X_train, y_train)

y_pred = grid_search.predict(y_test)
accuracy = accuracy_score(y_test, y_pred)



# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best accuracy Score:", best_score)






