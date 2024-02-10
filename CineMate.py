import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import requests
import ast
import nltk
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('wordnet')

# Load the dataset
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge datasets
movies = movies.merge(credits, left_on="id", right_on="movie_id")

# Select relevant columns
movies = movies[["genres", "movie_id", "keywords", "overview", "title_x", "cast", "crew"]]

# Rename columns
movies.rename(columns={"title_x": "title"}, inplace=True)

# Drop rows with missing values
movies.dropna(inplace=True)

# Extract relevant information from columns
movies["genres"] = movies["genres"].apply(lambda x: [i["name"] for i in ast.literal_eval(x)])
movies["keywords"] = movies["keywords"].apply(lambda x: [i["name"] for i in ast.literal_eval(x)])
movies["cast"] = movies["cast"].apply(lambda x: [i["name"] for i in ast.literal_eval(x)])
movies["crew"] = movies["crew"].apply(lambda x: [i["name"] for i in ast.literal_eval(x)])
movies["tags"] = movies["overview"] + movies["genres"] + movies["cast"] + movies["crew"] + movies["keywords"]

# Select relevant columns
new_data = movies[["movie_id", "title", "tags"]]

# Combine text columns into a single feature
new_data["tags"] = new_data["tags"].apply(lambda x: " ".join(x))

# Stemming function
ps = PorterStemmer()
def stemming(text):
    return " ".join([ps.stem(word) for word in text.split()])

# Apply stemming to the 'tags' column
new_data["tags"] = new_data["tags"].apply(stemming)

# Load preprocessed data and similarity matrix
movies = pickle.load(open('movie_list.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Function to fetch movie poster
def fetch_poster(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=2736a08daef7a534d3cf2d8c371e0427&language=en-US'
    data = requests.get(url).json()
    poster_path = data['poster_path']
    full_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
    return full_path

# Function to recommend movies
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), key=lambda x: x[1], reverse=True)
    recommended_movie_names = []
    recommended_movie_posters = []

    for i in distances[1:11]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_names.append(movies.iloc[i[0]].title)
        recommended_movie_posters.append(fetch_poster(movie_id))
    return recommended_movie_names, recommended_movie_posters

# Streamlit app
st.header("CineMate.ai ")
movie_list = movies['title'].values
selected_movie = st.selectbox("Type or select a movie from the list", movie_list)

if st.button("Show Recommendations"):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = st.columns(10)
    with c1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with c2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])
    with c3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with c4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with c5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])
    with c6:
        st.text(recommended_movie_names[5])
        st.image(recommended_movie_posters[5])
    with c7:
        st.text(recommended_movie_names[6])
        st.image(recommended_movie_posters[6])
    with c8:
        st.text(recommended_movie_names[7])
        st.image(recommended_movie_posters[7])
    with c9:
        st.text(recommended_movie_names[8])
        st.image(recommended_movie_posters[8])
    with c10:
        st.text(recommended_movie_names[9])
        st.image(recommended_movie_posters[9])
