# recommender.py - Movie Recommendation Logic

import pandas as pd
import numpy as np
import re
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

class MovieRecommender:
    def __init__(self, data_path='data/netflix_titles.csv'):
        """Initialize the movie recommender system"""
        self.data_path = data_path
        self.df = None
        self.cosine_sim_combined = None
        self.indices = None
        self.tfidf_vectorizer_combined = None
        
        # Load and prepare data
        self.load_and_prepare_data()
        self.build_recommendation_model()
    
    def load_and_prepare_data(self):
        """Load and preprocess the Netflix dataset"""
        print("Loading dataset...")
        
        # Load the dataset
        try:
            self.df = pd.read_csv(self.data_path, encoding='latin1')
        except UnicodeDecodeError:
            print("Error decoding with latin1, trying utf-8")
            self.df = pd.read_csv(self.data_path, encoding='utf-8')
        except FileNotFoundError:
            print(f"Dataset not found at {self.data_path}")
            print("Please ensure netflix_titles.csv is in the data/ directory")
            return
        
        print(f"Dataset loaded with shape: {self.df.shape}")
        
        # Drop columns with all missing values
        self.df.dropna(axis=1, how='all', inplace=True)
        
        # Fill missing values in specified columns with 'Unknown'
        cols_to_fill = ['director', 'cast', 'country', 'rating']
        self.df[cols_to_fill] = self.df[cols_to_fill].fillna('Unknown')
        
        # Extract genres from the 'listed_in' column
        self.df['genres'] = self.df['listed_in'].apply(lambda x: [i.strip() for i in x.split(',')])
        
        # Clean description text
        self.df['cleaned_description'] = self.df['description'].astype(str).apply(self.clean_text)
        
        # Prepare features for modeling
        self.df['processed_cast'] = self.df['cast'].apply(
            lambda x: ' '.join(x.split(',')[:3]).replace('Unknown', '').strip() if x != 'Unknown' else ''
        )
        self.df['processed_director'] = self.df['director'].apply(
            lambda x: x.replace('Unknown', '').strip() if x != 'Unknown' else ''
        )
        self.df['processed_genres'] = self.df['genres'].apply(lambda x: ' '.join(x).strip())
        self.df['processed_country'] = self.df['country'].apply(
            lambda x: x if x != 'Unknown' else ''
        )
        
        # Create combined 'soup' feature
        self.df['soup'] = (
            self.df['processed_genres'] + ' ' +
            self.df['processed_director'] + ' ' +
            self.df['processed_cast'] + ' ' +
            self.df['cleaned_description']
        )
        
        print("Data preprocessing completed.")
    
    def clean_text(self, text):
        """Clean text by converting to lowercase and removing punctuation"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def build_recommendation_model(self):
        """Build the TF-IDF model and cosine similarity matrix"""
        print("Building recommendation model...")
        
        if self.df is None:
            print("No data available to build model")
            return
        
        # Initialize the TF-IDF Vectorizer
        self.tfidf_vectorizer_combined = TfidfVectorizer(stop_words='english', max_features=10000)
        
        # Fit and transform the 'soup' column
        tfidf_matrix_combined = self.tfidf_vectorizer_combined.fit_transform(self.df['soup'])
        
        # Calculate the cosine similarity matrix
        self.cosine_sim_combined = cosine_similarity(tfidf_matrix_combined, tfidf_matrix_combined)
        
        # Create a mapping from title to index
        self.indices = pd.Series(self.df.index, index=self.df['title']).drop_duplicates()
        
        print("Recommendation model built successfully.")
    
    def get_all_genres(self):
        """Get all unique genres from the dataset"""
        if self.df is None:
            return []
        
        all_genres = set()
        for genres_list in self.df['genres']:
            all_genres.update(genres_list)
        
        return sorted(list(all_genres))
    
    def get_all_countries(self):
        """Get all unique countries from the dataset, splitting multi-country cells."""
        if self.df is None:
            return []

        all_countries = []
        for entry in self.df['country'].dropna():
            if entry != 'Unknown':
                all_countries.extend([country.strip() for country in entry.split(',') if country.strip()])

        return sorted(set(all_countries))

    
    def get_filtered_movies(self, selected_genres=None, selected_countries=None):
        """Get movies filtered by genres and countries"""
        if self.df is None:
            return []
        
        filtered_df = self.df.copy()
        
        # Filter by genres
        if selected_genres:
            filtered_df = filtered_df[
                filtered_df['genres'].apply(
                    lambda gs: any(g in gs for g in selected_genres)
                )
            ]
        
        # Filter by countries
        if selected_countries:
            filtered_df = filtered_df[
                filtered_df['processed_country'].isin(selected_countries)
            ]
        
        # Return sorted list of movie titles
        return sorted(filtered_df['title'].unique().tolist())
    
    def get_recommendations(self, title, selected_genres=None, selected_countries=None, num_recommendations=10):
        """Get movie recommendations based on similarity"""
        if self.df is None or self.cosine_sim_combined is None:
            return None
        
        # Filter dataset based on selections
        filtered_df = self.df.copy()
        
        if selected_genres:
            filtered_df = filtered_df[
                filtered_df['genres'].apply(
                    lambda gs: any(g in gs for g in selected_genres)
                )
            ]
        
        if selected_countries:
            filtered_df = filtered_df[
                filtered_df['processed_country'].isin(selected_countries)
            ]
        
        # Create indices for filtered dataset
        filtered_indices = pd.Series(filtered_df.index, index=filtered_df['title']).drop_duplicates()
        
        # Check if title exists
        if title not in filtered_indices:
            return None
        
        # Get the index of the movie
        idx = filtered_indices[title]
        
        # Get pairwise similarity scores
        sim_scores = list(enumerate(self.cosine_sim_combined[idx]))
        
        # Filter similarity scores to only include movies in filtered dataset
        filtered_sim_scores = [
            (i, score) for i, score in sim_scores 
            if i in filtered_df.index.tolist() and i != idx
        ]
        
        # Sort by similarity scores
        filtered_sim_scores = sorted(filtered_sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        filtered_sim_scores = filtered_sim_scores[:num_recommendations]
        
        # Get movie indices and scores
        movie_indices = [i[0] for i in filtered_sim_scores]
        scores = [i[1] for i in filtered_sim_scores]
        
        # Create recommendation DataFrame
        recommendations = pd.DataFrame({
            'Title': self.df['title'].iloc[movie_indices].values,
            'Similarity Score': [f"{score:.3f}" for score in scores],
            'Genres': self.df['processed_genres'].iloc[movie_indices].values,
            'Director': self.df['processed_director'].iloc[movie_indices].values,
            'Cast': self.df['processed_cast'].iloc[movie_indices].values,
            'Country': self.df['processed_country'].iloc[movie_indices].values,
            'Description': self.df['description'].iloc[movie_indices].values,
            'Rating': self.df['rating'].iloc[movie_indices].values,
            'Type': self.df['type'].iloc[movie_indices].values
        })
        
        return recommendations
    
    def get_random_movie(self, selected_genres=None, selected_countries=None):
        """Get a random movie from the filtered dataset"""
        if self.df is None:
            return None
        
        movies = self.get_filtered_movies(selected_genres, selected_countries)
        
        if movies:
            return random.choice(movies)
        else:
            return None