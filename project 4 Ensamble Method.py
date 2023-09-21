import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_squared_error, accuracy_score, precision_score, recall_score

#Load and preprocess the data
data = pd.read_csv('C:\\Users\\BANTA\\Desktop\\ML projects\\dataSets\\house_price_test.csv')  
# Explore the data:
print(data.head().T)
print(data.shape)  # Check the dimensions of the dataset
print(data.columns)  # Check the column names
print(data.info())  # Get information about the data types
print(data.describe())  # Summary statistics of the dataset

print(data.isnull().sum())  # Check for missing values
# Plot the categories in the target variable
# plt.figure(figsize=(8, 6))
# sns.countplot(data=data, x='SaleCondition')
# plt.xlabel('Sale Condition')
# plt.ylabel('Count')
# plt.title('Distribution of Sale Condition')
# plt.xticks(rotation=45)
# plt.show()

# # Visualize the distribution of numerical features
# numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
# data[numerical_columns].hist(bins=20, figsize=(10, 8))
# plt.tight_layout()
# plt.show()

# # Visualize the distribution of categorical features
# categorical_columns = data.select_dtypes(include=['object']).columns
# for column in categorical_columns:
#     plt.figure(figsize=(8, 6))
#     sns.countplot(data=data, x=column)
#     plt.xticks(rotation=45)
#     plt.show()

# # Visualize the relationship between numerical features and the target column
# for column in numerical_columns:
#     plt.figure(figsize=(8, 6))
#     sns.scatterplot(data=data, x=column, y='SaleCondition')
#     plt.show()

# # Create a heatmap of the correlation matrix
# correlation_matrix = data[numerical_columns].corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.show()



# Handle missing values in categorical features
categorical_columns = data.select_dtypes(include=['object']).columns
categorical_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])

# Handle missing values in numerical features
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
numerical_imputer = SimpleImputer(strategy='mean')
data[numerical_columns] = numerical_imputer.fit_transform(data[numerical_columns])

# # Handle categorical variables

# Encode categorical features using label encoding
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# data[categorical_columns] = pd.get_dummies(data[categorical_columns])

 # Handle numerical variables
numerical_scaler = StandardScaler()
data[numerical_columns] = numerical_scaler.fit_transform(data[numerical_columns])

# # Check the cleaned data
print(data.head())

# Randomize the data
randomized_data = data.sample(frac=1, random_state=42)

data = randomized_data

# Check the randomized data
print(randomized_data.head())

# Split the data into features (X) and target variable (y)
X = data.drop('SaleCondition', axis=1)  # Replace 'target_column' with the actual name of your target column
y = data['SaleCondition']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a RobustScaler object
scaler = RobustScaler()
# Fit the scaler on the training data
scaler.fit(X_train)
#By fitting the RobustScaler on the training data, it calculates the median and interquartile range (IQR) for each feature. Then, it scales the features using the formula (X - median) / IQR, where X is the original value of the feature. This scaling method is more robust to outliers compared to standard scaling techniques like the MinMaxScaler or StandardScaler
# Transform the training and testing data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a list of classification models
models = [
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier()
]

# Train and evaluate each model
average = []
for model in models:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    average.append(accuracy)
    print(f"{model.__class__.__name__} Accuracy: {accuracy}\n")
print(f"Average Accuracy: {(sum(average))/3}\n")  

# Create an ensemble of classifiers
#The VotingClassifier combines the predictions of multiple classifiers using majority voting.
#The BaggingClassifier applies bootstrap sampling to train multiple classifiers on different subsets of the training data. 
#The AdaBoostClassifier trains multiple classifiers sequentially, with each subsequent classifier focusing on the samples that were misclassified by the previous classifiers
ensemble = [
    ('voting', VotingClassifier(estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('svc', SVC()),
        ('nb', GaussianNB()),
        ('knn', KNeighborsClassifier())
    ])),
    ('bagging', BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10)),
    ('adaboost', AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=50))
]

# Train and evaluate each ensemble model
#We iterate over each ensemble model, train it on the training set using fit(), make predictions on the test set using predict(), and calculate the accuracy using accuracy_score(). Finally, we print the accuracy for each ensemble model.
for name, model in ensemble:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy}\n")



