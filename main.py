import numpy as np 
import pandas as pd 
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_and_process_data(filepath):
    # Load data
    movies_data = pd.read_csv('movies.csv')
    
    # Fill NA values
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')
    
    # Combine features
    movies_data['combined_features'] = movies_data[selected_features].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    
    return movies_data

def compute_feature_vectors(data):
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(data['combined_features'])
    return feature_vectors

def get_movie_recommendations(movie_name, data, vectors, mood=None):
    similarity = cosine_similarity(vectors)
    list_of_all_titles = data['title'].tolist()
    
    # Get closest match for movie name
    close_match = difflib.get_close_matches(movie_name, list_of_all_titles)[0]
    index_of_the_movie = data[data.title == close_match].index[0]
    
    # Get similarity scores and sort them
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)[1:30]  # excluding the input movie

    mood_to_genres = {
    'happy': ['Comedy', 'Family', 'Animation'],
    'sad': ['Drama', 'Romance'],
    'adventurous': ['Adventure', 'Action', 'Fantasy'],
    'scared': ['Horror', 'Mystery', 'Thriller'],
    'thoughtful': ['Documentary', 'History', 'Biography']
    }

    # If mood is specified, prioritize movies of that mood
    if mood and mood in mood_to_genres:
        genres_for_mood = mood_to_genres[mood]
        sorted_similar_movies = [movie for movie in sorted_similar_movies if any(genre in data.iloc[movie[0]]['genres'] for genre in genres_for_mood)]

    return [data.iloc[i]['title'] for i, _ in sorted_similar_movies]


if __name__ == "__main__":
    filepath = '/content/movies.csv'
    movies_data = load_and_process_data(filepath)
    feature_vectors = compute_feature_vectors(movies_data)
    
    movie_name = input('Enter your favourite movie name: ')
    user_mood = input('How are you feeling today? (happy, sad, adventurous, scared, thoughtful, enter if none): ')

    recommended_movies = get_movie_recommendations(movie_name, movies_data, feature_vectors, user_mood)
    
    print('Movies suggested for you:\n')
    for idx, movie in enumerate(recommended_movies, 1):
        print(f"{idx}. {movie}")