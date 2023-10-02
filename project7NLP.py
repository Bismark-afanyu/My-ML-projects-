import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.svm import SVC
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import class_weight
from nltk.tokenize import word_tokenize
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('C:\\Users\\BANTA\\Desktop\\ML projects\\dataSets\\reviews.csv', )

# Explore the data:
print(data.head().T)
print(data.isnull().sum())  # Check for missing values

print(data.shape)  # Check the dimensions of the dataset
print(data.columns)  # Check the column names
print(data.info())  # Get information about the data types
print(data.describe())
# split and Convert ratings to integer
data['Review Rating'] = data['Review Rating'].str.split('/').str[0].astype(int)
print(data.head(10))

#preprocessing function
def preprocess_text(text):

    # Regex allows defining patterns to match unwanted characters through character classes.
    # The pattern '[^\w\s]' matches any character that is not a word character (\w) or whitespace (\s).
    #Word characters include [a-zA-Z0-9_] and whitespace includes spaces, tabs etc.
    #The re.sub() method replaces all matches of the pattern with an empty string, thereby removing them
    
    # Clean the text
    text = re.sub(r'[^a-zA-Z]',' ',text)

    # Lowercasing
    #Convert all text to lowercase to ensure consistency and avoid treating the same word differently based on its case.
    text = text.lower()
    
    # Tokenization
    #Split the text into individual words or tokens. This step helps in creating a vocabulary of words that can be used as features.
    tokens = word_tokenize(text)
    
    # Removing punctuation
    #Remove any punctuation marks from the text, as they usually do not carry much meaning in spam detection
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # Removing stop words
    #Remove common words that do not contribute much to the classification task, such as "the," "is," "and," etc.
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    #Reduce words to their base or root form to handle variations of the same word. Stemming and lemmatization techniques help in reducing the dimensionality of the feature space
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

text_data = data["Review_body"]

sentiment = np.where(data['Review Rating'] < 5, -1,np.where(data['Review Rating'] == 5, 0, 1))

# Count samples in each class  
target_labels = sentiment

# Plot the histogram
plt.hist(target_labels)
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.title('Distribution of Categories')
plt.show()


# Apply preprocess function to selected columns
data['Review title'] = data['Review title'].apply(preprocess_text)  
data['Review_body'] = data['Review_body'].apply(preprocess_text)

print(data.head(10))
print(data.info())  # Get information about the data types

# Apply preprocessing to each text data
preprocessed_texts = [preprocess_text(text) for text in text_data]
#print(preprocessed_texts)


# Initialize the TF-IDF vectorizer
#Term Frequency-Inverse Document Frequency (TF-IDF): TF-IDF is a numerical statistic that reflects the importance of a word in a document relative to the entire corpus. It combines term frequency (TF) and inverse document frequency (IDF) to assign weights to words
vectorizer = TfidfVectorizer()

# Fiting and transforming the preprocessed texts
#This step converts the preprocessed texts into a feature matrix, where each row represents a document and each column represents a feature (word)
features = vectorizer.fit_transform(preprocessed_texts)
target_labels = sentiment
#feature matrix and labels
feature = features
labels = target_labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Print the sizes of the training and testing sets
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Apply oversampling to the training data
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Apply undersampling to the training data
undersampler = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

# Adjust class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

models = [
  LogisticRegression(),
  DecisionTreeClassifier(), 
  RandomForestClassifier(),
  MultinomialNB(),
  SVC(kernel='linear', class_weight='balanced')
]

for model in models:
  model.fit(X_train_resampled, y_train_resampled)
  y_pred = model.predict(X_test)
  y_train_pred = model.predict(X_train_resampled)

  
  print(f'Model: {model}')
  print(f'Training Accuracy: {accuracy_score(y_train_resampled, y_train_pred)}')
  print(f'Test Accuracy: {accuracy_score(y_test, y_pred)}\n')


# Create the final Naive Bayes classifier
# Apply oversampling to the entire data
oversampler = RandomOverSampler(random_state=42)
X_entire_resampled, y_entire_resampled = oversampler.fit_resample(features, labels)

# Apply undersampling to the entire data
undersampler = RandomUnderSampler(random_state=42)
X_entire_resampled, y_entire_resampled = undersampler.fit_resample(features, labels)

# Adjust class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
model1 = MultinomialNB()
# Train the final model on the entire dataset
model1.fit(X_entire_resampled, y_entire_resampled)
# Save the final model for future use
import joblib
joblib.dump(model1, 'sentiment_model.joblib')


# params = {
#   'LogisticRegression': {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']},
#   'DecisionTreeClassifier': {'max_depth': [5, 8, 15], 'min_samples_split': [2, 5, 10]}, 
#   'RandomForestClassifier': {'n_estimators': [100, 500, 1000], 'max_features': ['auto', 'sqrt', 'log2']},
#   'MultinomialNB': {'alpha': [1e-3, 1e-2, 1e-1]},
#   'SVC': {'C': [1, 2, 5], 'kernel': ['linear', 'rbf']}  
# }

# model_names = [
#   "LogisticRegression",
#   "DecisionTreeClassifier", 
#   "RandomForestClassifier",
#   "MultinomialNB",
#   "SVC"
# ]
# for model_name in model_names:
#   clf = GridSearchCV(eval(model_name +"()"), param_grid=params[model_name], cv=5)
#   clf.fit(X_train_resampled, y_train_resampled)
#   print(model)

#   print(f'Best params for {model}: {clf.best_params_}')
#   print(f'Best score for {model}: {clf.best_score_}\n')
