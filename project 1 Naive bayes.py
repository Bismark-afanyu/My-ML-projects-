# importing the Libraries needed for my model 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from sklearn.utils import shuffle

# This code is reading a CSV file called 'adult_naive_bayes.csv' and storing it in a pandas DataFrame
# called 'data'  and look for some general insite about the data
data = pd.read_csv('C:\\Users\\BANTA\\Desktop\\ML projects\\dataSets\\adult_naive_bayes.csv')
print (data.head(10).T)
print(data.info())
print(data.describe())

#checking and handelling missing values 
# This code is checking for missing values in the dataset and handling them.
missing_values = data.isnull().sum()
print (missing_values)
data = data.replace("?", np.nan)
print (data)
missing_values_percentage = (data.isnull().sum() / len(data)) * 100
data = data.fillna(data.mode().iloc[0])
print(data.head(10))

print(missing_values_percentage)

# Visualization 
#visualizing the data to look for trends and outliers in the data
# Create subplots for multiple histograms
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Plot histogram for 'age'
# This code is creating a histogram of the 'age' column in the dataset.
axes[0].hist(data['age'], bins=20, edgecolor='black')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Age')

# Plot histogram for 'education.num'
axes[1].hist(data['education.num'], bins=20, edgecolor='black')
axes[1].set_xlabel('Education Number')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Education Number')

# Plot histogram for 'hours.per.week'
axes[2].hist(data['hours.per.week'], bins=20, edgecolor='black',)
axes[2].set_xlabel('Hours per Week')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Distribution of Hours per Week')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

# Create subplots for multiple bar plots
fig, axes = plt.subplots(1, 3, figsize=(30, 4))

# Plot bar plot for 'workclass'
sns.countplot(data=data, x='workclass', ax=axes[0])
plt.xticks(rotation = 45)
axes[0].set_xlabel('Workclass')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Workclass')

# Plot bar plot for 'education'
sns.countplot(data=data, x='education', ax=axes[1])
plt.xticks(rotation = 45)
axes[1].set_xlabel('Education')
axes[1].set_ylabel('Count')
axes[1].set_title('Distribution of Education')

# Plot bar plot for 'marital.status'
sns.countplot(data=data, x='marital.status', ax=axes[2])
axes[2].set_xlabel('Marital Status')
axes[2].set_ylabel('Count')
axes[2].set_title('Distribution of Marital Status')

# Rotate x-axis labels if needed
plt.xticks(rotation=45)

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

# Create subplots for multiple box plots
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# Plot box plot for 'age' grouped by 'income'
sns.boxplot(data=data, x='income', y='age', ax=axes[0])
axes[0].set_xlabel('Income')
axes[0].set_ylabel('Age')
axes[0].set_title('Distribution of Age by Income')
# Plot box plot for 'education.num' grouped by 'income'
sns.boxplot(data=data, x='income', y='education.num', ax=axes[1])
axes[1].set_xlabel('Income')
axes[1].set_ylabel('Education Number')
axes[1].set_title('Distribution of Education Number by Income')
# Plot box plot for 'hours.per.week' grouped by 'income'
sns.boxplot(data=data, x='income', y='hours.per.week', ax=axes[2])
axes[2].set_xlabel('Income')
axes[2].set_ylabel('Hours per Week')
axes[2].set_title('Distribution of Hours per Week by Income')
# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

# This code is grouping the data by the 'income' column and calculating the rates for each feature in
# the dataset. It creates a dictionary called 'feature_rates' to store the rates for each feature.
# and calculate the rates by dividing the count of each category within each income level by the total count for that income level. it also show how your income is affected if you belong to a perticular category in the data set.
# Group the data by income and calculate the rates for each feature
feature_rates = {}
for feature in data.columns[:-1]:  # Exclude the last column (income)
    rates = data.groupby([feature, 'income']).size() / data.groupby('income').size()
    feature_rates[feature] = rates

# Print the rates for each feature
for feature, rates in feature_rates.items():
    print(f"Rates for feature '{feature}':")
    print(rates)
    print()


# This code snippet is performing label encoding on the string columns in the dataset. Label encoding
# is a process of converting categorical variables into numerical values. It assigns a unique
# numerical value to each category in a column.
#one hot encoding is not used becuse it will create a lot of columns with a lot of zero's that may lead to high dimentionalty.
# Select the string columns to be encoded
string_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country', 'income']

# Apply label encoding to each string column
for column in string_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])

# Print the updated dataset
print(data)

# The code snippet is computing the correlation matrix between the numerical features in the dataset.
# It selects the columns specified in the `numerical_features` list and calculates the correlation
# coefficients between them.
numerical_features = ["age","workclass","fnlwgt","education","education.num","marital.status","occupation","relationship","race","sex","capital.gain","capital.loss","hours.per.week","native.country","income" ]

# Compute the correlation matrix
correlation_matrix = data[numerical_features].corr()

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)

# Set the title
plt.title('Correlation Heatmap of Numerical Features')

# Show the plot
plt.show()



# Split the data into features (X) and label (y)
X = data.drop('income', axis=1)
y = data['income']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Get the feature importance scores
importance_scores = model.feature_importances_

# Create a dataframe to display the feature importance scores
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance_scores})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Print the feature importance scores
print(feature_importance)

# Plot the feature importance scores
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['Feature'], feature_importance['Importance'])
plt.xticks(rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()



# Convert int64 columns to int32
# The code snippet is converting the columns with data type "int64" in the DataFrame 'data' to "int32"
# data type. This is done to reduce the memory usage of the DataFrame. The 'int32' data type uses 32
# bits of memory to store each integer value, while the 'int64' data type uses 64 bits. By converting
# the columns to 'int32', the memory usage of the DataFrame is reduced, which can be beneficial when
# working with large datasets.
int_cols = [col for col in data.columns if data[col].dtype == "int64"]
data[int_cols] = data[int_cols].astype(np.int32)
int_cols = [col for col in data.columns if data[col].dtype == "float64"]
data[int_cols] = data[int_cols].astype(np.float32)


# Split the data into features (X) and label (y)
X = data.drop('income', axis=1)
y = data['income']
# Scale the features using StandardScaler
# The code snippet `scaler = StandardScaler()` creates an instance of the `StandardScaler` class,
# which is used to standardize the features in the dataset.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Randomize the order of the data
# The code `X_shuffled, y_shuffled = shuffle(X_scaled, y, random_state=42)` is shuffling the order of
# the data points in the dataset. It takes the scaled features `X_scaled` and the corresponding labels
# `y` and shuffles them randomly. The `random_state=42` parameter ensures that the shuffling is done
# in a reproducible manner, meaning that the same shuffling order will be obtained every time the code
# is run with the same random state value. This is useful for ensuring consistent results when working
# with randomized algorithms or when comparing different models.
X_shuffled, y_shuffled = shuffle(X_scaled, y, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the Naive Bayes classifier
# This code snippet is training a Naive Bayes classifier using the GaussianNB algorithm.
model = GaussianNB()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)

# This code snippet is making predictions on the training set using the trained Naive Bayes classifier
# model. It then calculates and prints the accuracy of the model on the training set. Similarly, it
# calculates and prints the accuracy of the model on the test set. The accuracy score is a common
# evaluation metric for classification models and represents the percentage of correctly predicted
# labels out of the total number of samples.
# Make predictions on the training set
y_train_pred = model.predict(X_train)
# Calculate and print the accuracy on the training set
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", model.score(X_train, y_train))
# Calculate and print the accuracy on the test set
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)

# The below code is printing the confusion matrix and classification report for a machine learning
# model's predictions. The confusion matrix shows the number of true positives, true negatives, false
# positives, and false negatives. The classification report provides precision, recall, F1-score, and
# support for each class in the prediction.
# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# The below code is calculating additional evaluation metrics for a classification model. It
# calculates the accuracy, precision, recall, F1 score, and AUC-ROC score. These metrics are commonly
# used to evaluate the performance of a classification model.
#additional evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred)


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("AUC-ROC:", auc_roc)


# Model Evalution and tuning
# This code is performing model evaluation and tuning using grid search.
# Define the parameter grid for grid search
# The above code is creating a dictionary called `param_grid` with a single key-value pair. The key is
# `'var_smoothing'` and the value is a list of floating-point numbers `[1e-9, 1e-8, 1e-7, 1e-6,
# 1e-5]`. This dictionary is commonly used as a parameter grid for hyperparameter tuning in machine
# learning algorithms.
param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}
# Create the Naive Bayes classifier
model = GaussianNB()
# Pedtorm grid search to find the best hyperparameters
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
# Get the best hyperparameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
# Make predictions on the test set using the best model
y_test_pred = best_model.predict(X_test)
# Calculate and print the accuracy on the test set
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy after model tuning:", test_accuracy)

# The above code is creating and training a Naive Bayes classifier using the GaussianNB() function
# from the scikit-learn library. It then saves the trained model using the joblib.dump() function, so
# that it can be used for future predictions.
#model deployment
# Create the final Naive Bayes classifier
model1 = GaussianNB()
# Train the final model on the entire dataset
model1.fit(X_scaled, y)
# Save the final model for future use
import joblib
joblib.dump(model1, 'final_model.joblib')

