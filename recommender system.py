import re
import string
import difflib
from nltk import PorterStemmer, word_tokenize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gensim
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder



data = pd.read_csv('C:\\Users\\BANTA\\Desktop\\ML projects\\dataSets\\IMDB.csv')  
# # Explore the data:
# print(data.head().T)
# print(data.shape)  # Check the dimensions of the dataset
# print(data.columns)  # Check the column names
# print(data.info())  # Get information about the data types
# print(data.describe())  # Summary statistics of the dataset

# print(data.isnull().sum())
#data cleaning 

def clean_imdb_dataset(data):
    # Drop irrelevant columns
    data = data.drop(['index', 'tconst', 'endYear', 'ordering','types', 'attributes'], axis=1)
    # Drop rows with missing values
    data = data.dropna()
    # Convert startYear to datetime format and set it as the index
    data['startYear'] = pd.to_datetime(data['startYear'], format='%Y')
    #data.set_index('startYear', inplace=True)
    # Normalize numerical features
    data['averageRating'] = (data['averageRating'] - data['averageRating'].mean()) / data['averageRating'].std()
    data['numVotes'] = (data['numVotes'] - data['numVotes'].mean()) / data['numVotes'].std()
    return data


# Clean the dataset using the function
cleaned_data = clean_imdb_dataset(data)

# Save the cleaned dataset to a new CSV file
new_data = cleaned_data
# Explore the data:
print(new_data.head().T)
print(new_data.shape)  # Check the dimensions of the dataset
print(new_data.columns)  # Check the column names
print(new_data.info())  # Get information about the data types
print(new_data.describe())  # Summary statistics of the dataset

print(new_data.isnull().sum())

# EDA 

# import matplotlib.pyplot as plt

# Histogram of averageRating
plt.hist(new_data['averageRating'], bins=10)
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Average Ratings')
plt.show()

# Scatter plot of averageRating vs. numVotes
plt.scatter(new_data['averageRating'], new_data['numVotes'])
plt.xlabel('Average Rating')
plt.ylabel('Number of Votes')
plt.title('Average Rating vs. Number of Votes')
plt.show()

# Bar chart of genres
genre_counts = new_data['genres'].value_counts().head(10)
plt.bar(genre_counts.index, genre_counts.values)
plt.xlabel('Genres')
plt.ylabel('Count')
plt.title('Top 10 Genres')
plt.xticks(rotation=45)
plt.show()



#Time Series Analysis, i ploted the average rating and number of votes over the years to identify any trends or patterns.
# Group the data by year and calculate the average rating and total number of votes for each year
yearly_data = new_data.groupby(new_data.index).agg({'averageRating': 'mean', 'numVotes': 'sum'})
# Plot the average rating over the years
plt.plot(yearly_data.index, yearly_data['averageRating'])
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.title('Average Rating Over the Years')
plt.show()

# Plot the total number of votes over the years
plt.plot(yearly_data.index, yearly_data['numVotes'])
plt.xlabel('Year')
plt.ylabel('Total Number of Votes')
plt.title('Total Number of Votes Over the Years')
plt.show()


#I perform Genre Analysis, by visualizing the distribution of genres in the dataset and analyze the average rating or number of votes for each genre. 
# Split the genres column into multiple genres
new_data['genres'] = new_data['genres'].str.split(',')
# Create a list of all unique genres
all_genres = set()
for genres in new_data['genres']:
    all_genres.update(genres)

# Count the frequency of each genre
genre_counts = {}
for genre in all_genres:
    genre_counts[genre] = new_data[new_data['genres'].apply(lambda x: genre in x)].shape[0]

# Plot the genre distribution
plt.bar(genre_counts.keys(), genre_counts.values())
plt.xlabel('Genres')
plt.ylabel('Count')
plt.title('Genre Distribution')
plt.xticks(rotation=45)
plt.show()

# Calculate the average rating for each genre
genre_avg_rating = {}
for genre in all_genres:
    genre_avg_rating[genre] = new_data[new_data['genres'].apply(lambda x: genre in x)]['averageRating'].mean()

# Plot the average rating for each genre
plt.bar(genre_avg_rating.keys(), genre_avg_rating.values())
plt.xlabel('Genres')
plt.ylabel('Average Rating')
plt.title('Average Rating by Genre')
plt.xticks(rotation=45)
plt.show()



 #analyzing movie ratings
# Plot the distribution of movie ratings
# Plot the distribution of movie ratings
plt.hist(new_data['averageRating'], bins=10)
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Movie Ratings')
plt.show()

# Identify the top-rated movies based on average rating
top_movies = new_data.nlargest(10, 'averageRating')
plt.bar(top_movies['primaryTitle'], top_movies['averageRating'])
plt.xlabel('Movie Title')
plt.ylabel('Average Rating')
plt.title('Top 10 Highest Rated Movies')
plt.xticks(rotation=45)
plt.show()

print(new_data.head().T)


    # Temporal Feature
new_data['month'] = pd.to_datetime(new_data['startYear']).dt.month
new_data['season'] = pd.to_datetime(new_data['startYear']).dt.quarter
new_data['day_of_week'] = pd.to_datetime(new_data['startYear']).dt.dayofweek

new_data = new_data.drop('startYear', axis= 1)


joined_features = new_data['titleType']+' '+new_data['genres']+' '+new_data['language']+' '+new_data['primaryTitle']+' '+new_data['originalTitle']+' '+new_data['Description']+' '+new_data['region']

vectorizer = TfidfVectorizer()


features = vectorizer.fit_transform(joined_features)

# getting the similarity score using cosine similarities

similarity = cosine_similarity(features)
 # Getting the movie name from the user 

movie_name = input("Enter your favourite movie name:")

#creating a list with all the movie names that are present in the dataset 
list_of_all_title = data['title'].tolist()
#print(list_of_all_title)

# finding the close match for the movie entered by the user
find_close_match = difflib.get_close_matches(movie_name, list_of_all_title)
print(find_close_match)

close_match = find_close_match[0]

# finding the index of the movie with title 
index_of_the_movie = data[data.title == close_match]['index'].values[0]

#print (index_of_the_movie)

# getting a list of similar movies
similarity_score = list(enumerate(similarity[index_of_the_movie])) 
#print (similarity_score)
len(similarity_score)

#sorting the movies based on their similarity score
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse= True)
#print(sorted_similar_movies)

# print the names of the similar movies based on the index
print('Movies suggested for you : \n')

i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = data[data.index == index]['title'].values[0]
    if(i<25):
        print('\t' + str(i)+'. '+str((title_from_index)))
        i += 1





