import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
# Load the dataset
df = pd.read_csv('C:\\Users\\BANTA\\Desktop\\ML projects\\dataSets\\winequality-red_decision_tree.csv')
# Display the first 10 rows of the dataset
print(df.head(10))
# Get statistical summary of the dataset
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Explore the distribution of the target variable
print(df['quality'].value_counts())
# Assuming your DataFrame is called 'df' and the label column is 'quality'
df['binary_label'] = df['quality'].apply(lambda x: 1 if x > 6 else 0)
print(df)


# Visualize the correlation between features
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Selected features for histogram plots
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar']
'''
# Create histograms for each feature
for feature in features:
    plt.figure()
    plt.hist(df[feature], bins=20, edgecolor='black')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {feature}')
    plt.show()

'''

# Convert int64 columns to int32
int_cols = [col for col in df.columns if df[col].dtype == "int64"]
df[int_cols] = df[int_cols].astype(np.int32)
# Convert float64 columns to float32
int_cols = [col for col in df.columns if df[col].dtype == "float64"]
df[int_cols] = df[int_cols].astype(np.float32)
# Handle missing values (if any)
# Drop rows with missing values
df = df.dropna()
# Separate features and target variable
X = df.drop('quality', axis=1)
y = df['quality']

# Encode categorical variables (if any)
# If you don't have any categorical variables, you can skip this step

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Randomize the order of the data
X_shuffled, y_shuffled = shuffle(X_scaled, y, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create an instance of the Random Forest classifier
dt_classifier = DecisionTreeClassifier()
# Train the classifier on the training data
dt_classifier.fit(X_train, y_train)


# Make predictions on the testing data
y_pred = dt_classifier.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')


# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)



# Define the hyperparameter grid
param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Create an instance of the Decision Tree classifier
dt_classifier = DecisionTreeClassifier()

# Perform grid search with cross-validation
grid_search = GridSearchCV(dt_classifier, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model on the testing data
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the best hyperparameters and evaluation metrics
print("Best Hyperparameters:", best_params)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)