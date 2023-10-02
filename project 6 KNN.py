import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mstats
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns


data = pd.read_csv('C:\\Users\\BANTA\\Desktop\\ML projects\\dataSets\\customers - customers.csv')  
# Explore the data:
print(data.head().T)
print(data.shape)  # Check the dimensions of the dataset
print(data.columns)  # Check the column names
print(data.info())  # Get information about the data types
print(data.describe())  # Summary statistics of the dataset

print(data.isnull().sum())
print("yes")
original_count = data.shape[0]
drop_dup_count = data.drop_duplicates().shape[0]
duplicate_percentage = (original_count - drop_dup_count) / original_count * 100
duplicates = data[data.duplicated()] 
print(duplicate_percentage)
print(duplicates)

# Visualize the relationships using scatter plots
features = ['tenure', 'age', 'income', 'ed', 'employ', 'reside']
target_variable = 'custcat'


# for feature in features:
#     plt.figure(figsize=(8, 6))
#     #Violin plots provide a combination of box plots and kernel density plots, allowing you to see the distribution of each feature for each class in the target variable.
#     sns.violinplot(data=data, x=target_variable, y=feature)
#     plt.xlabel(target_variable)
#     plt.ylabel(feature)
#     plt.title(f'{target_variable} vs. {feature}')
#     plt.show()

# Count the occurrences of each category in the target column
target_variable = 'custcat'
category_counts = data[target_variable].value_counts()

# Visualize the category counts using a bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.xlabel(target_variable)
plt.ylabel('Count')
plt.title('Category Counts in Target Variable')
plt.show()

# Apply Winsorization to the numerical features
#Instead of removing outliers,i replace them with the nearest non-outlier values. This technique helps to minimize the impact of outliers while retaining the overall distribution of the data
numeric_features = ['tenure', 'age', 'income', 'ed', 'employ', 'reside']

for feature in numeric_features:
    winsorized_data = mstats.winsorize(data[feature], limits=[0.05, 0.05])  
    data[feature] = winsorized_data

# Check the updated dataset
# print(data.head())


# Separate the features and target variable
X = data.drop('custcat', axis=1)  # Features
y = data['custcat']  # Target variable

# Feature Importance using Random Forest
rf = RandomForestClassifier()
rf.fit(X, y)
feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_importance)

# X is the feature matrix, y is the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


