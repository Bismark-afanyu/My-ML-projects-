import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.svm import SVC
from nltk.stem import PorterStemmer
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


data = pd.read_csv('C:\\Users\\BANTA\\Desktop\\ML projects\\dataSets\\spam.csv', encoding= "latin_1")
# Extract the target labels from the CSV file
target_labels = data['v1']

# Plot the histogram
plt.hist(target_labels)
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.title('Distribution of Categories')
plt.show()

# Explore the data:
print(data.head(15))
print(data.isnull().sum())  # Check for missing values

print(data.shape)  # Check the dimensions of the dataset
print(data.columns)  # Check the column names
print(data.info())  # Get information about the data types
print(data.describe())
original_count = data.shape[0]
drop_dup_count = data.drop_duplicates().shape[0]
duplicate_percentage = (original_count - drop_dup_count) / original_count * 100
duplicates = data[data.duplicated()] 
print(duplicate_percentage)
print(duplicates)
print(duplicates.shape)
print (data['v1'])
print (data['v2'])


# Extracting text data and target labels from the CSV file
text_data = data['v2']
target_labels = data['v1']

label_encoder = LabelEncoder()
data['v1'] = label_encoder.fit_transform(data['v1'])
target_labels = (data['v1'])


# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')

#preprocessing function
def preprocess_text(text):
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

# Example 
#text = "Hello! This is an example sentence for preprocessing."
# Apply preprocessing to each text data
preprocessed_texts = [preprocess_text(text) for text in text_data]
#print(preprocessed_texts)


# Example preprocessed texts
# preprocessed_texts = [
#     "this is a sample text",
#     "another example text",
#     "spam spam spam"
# ]

# Initialize the TF-IDF vectorizer
#Term Frequency-Inverse Document Frequency (TF-IDF): TF-IDF is a numerical statistic that reflects the importance of a word in a document relative to the entire corpus. It combines term frequency (TF) and inverse document frequency (IDF) to assign weights to words
vectorizer = TfidfVectorizer()

# Fiting and transforming the preprocessed texts
#This step converts the preprocessed texts into a feature matrix, where each row represents a document and each column represents a feature (word)
features = vectorizer.fit_transform(preprocessed_texts)

# Get the feature names (words)
#feature_names = vectorizer.get_feature_names_out()

# Print the feature matrix
#print(features.toarray())
print ('yes')

# Print the feature names
#print(feature_names)


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

# Create an SVM model object with adjusted class weights
svm_model = SVC(kernel='linear', class_weight='balanced')

# Fit the SVM model on the resampled training data
svm_model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the testing set
y_pred = svm_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# #Model tuining
# # Define the parameter grid that specifies the different values to be tried for the hyperparameters
# param_grid = {
#     'C': [0.1, 1, 10],
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     'gamma': ['scale', 'auto']
# }
# # Create an SVM model
# svm_model = SVC()

# # Perform grid search with cross-validation
# grid_search = GridSearchCV(svm_model, param_grid, cv=5)
# grid_search.fit(X_train, y_train)

# # Get the best parameters and best score
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_

# # Print the best parameters and best score
# print("Best Parameters:", best_params)
# print("Best Score:", best_score)

# test case
def predict_spam(sentence):
    # Preprocess the input sentence
    preprocessed_sentence = preprocess_text(sentence)

    # Convert the preprocessed sentence into numerical features
    feature_vector = vectorizer.transform([preprocessed_sentence])

    # Predict the label using the trained SVM model
    predicted_label = svm_model.predict(feature_vector)
    #print(predicted_label)

    # Return the predicted label
    if predicted_label == 0:
        return "Not Spam"
    else:
        return "Spam"

sentence1 = "Customer Support Chat Job: $25/hr."
print(predict_spam(sentence1))  

sentence2 = "Congratulations! You've won a free car"
print(predict_spam(sentence2))  






