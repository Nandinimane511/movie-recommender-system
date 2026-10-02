"""
CineMatch AI - Model Preprocessing & Training Pipeline
Author: Nandini Mane
Description: Merges TMDB datasets, performs NLP feature extraction (Porter Stemming, CountVectorizer),
             and builds a high-dimensional Cosine Similarity matrix for content-based movie recommendation.
"""

import os
import ast
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

print("Loading datasets...")
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

print(f"Movies shape: {movies.shape}, Credits shape: {credits.shape}")

# Merge on title
df = movies.merge(credits, on='title')
print(f"Merged shape: {df.shape}")

# Drop rows with null essential columns
df.dropna(subset=['overview', 'genres', 'keywords', 'cast', 'crew'], inplace=True)
df['vote_average'] = df['vote_average'].fillna(0.0)
df['release_date'] = df['release_date'].fillna("Unknown")
df['runtime'] = df['runtime'].fillna(0).astype(int)

# Parsing helpers
def get_list(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj)]
    except Exception:
        return []

def get_top3_cast(obj):
    try:
        cast = ast.literal_eval(obj)
        return [i['name'] for i in cast[:3]]
    except Exception:
        return []

def get_director(obj):
    try:
        for i in ast.literal_eval(obj):
            if i.get('job') == 'Director':
                return i.get('name', 'Unknown')
        return 'Unknown'
    except Exception:
        return 'Unknown'

print("Extracting metadata...")
df['genres_list'] = df['genres'].apply(get_list)
df['keywords_list'] = df['keywords'].apply(get_list)
df['cast_list'] = df['cast'].apply(get_top3_cast)
df['director'] = df['crew'].apply(get_director)

# Format for NLP tags (remove spaces for unique entities)
genres_clean = df['genres_list'].apply(lambda x: [i.replace(" ", "") for i in x])
keywords_clean = df['keywords_list'].apply(lambda x: [i.replace(" ", "") for i in x])
cast_clean = df['cast_list'].apply(lambda x: [i.replace(" ", "") for i in x])
director_clean = df['director'].apply(lambda x: [x.replace(" ", "")] if x != 'Unknown' else [])
overview_tokens = df['overview'].apply(lambda x: str(x).split())

# Combined tags
df['tags'] = overview_tokens + genres_clean + keywords_clean + cast_clean + director_clean
df['tags_str'] = df['tags'].apply(lambda x: " ".join(x).lower())

# Stemming
ps = PorterStemmer()
def stem(text):
    return " ".join([ps.stem(w) for w in text.split()])

print("Stemming tags...")
df['tags_stemmed'] = df['tags_str'].apply(stem)

# Vectorization & Cosine Similarity
print("Computing Cosine Similarity...")
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags_stemmed']).toarray()
similarity = cosine_similarity(vectors)

# Clean DataFrame to save with rich metadata
rich_movies = df[[
    'movie_id', 'title', 'overview', 'genres_list', 'cast_list', 
    'director', 'vote_average', 'release_date', 'runtime', 'tags_stemmed'
]].reset_index(drop=True)

os.makedirs("model", exist_ok=True)
print("Saving model artifacts...")
pickle.dump(rich_movies, open('model/movie_list.pkl', 'wb'))
pickle.dump(similarity, open('model/similarity.pkl', 'wb'))

print("Enhanced Model Saved Successfully! (Total movies:", len(rich_movies), ")")
