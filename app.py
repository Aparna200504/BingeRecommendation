# app.py - Main Flask Application for Binge Recommendation

from flask import Flask, render_template, request, jsonify
import pandas as pd
import random
from recommender import MovieRecommender

app = Flask(__name__)

# Initialize the recommendation system
print("Loading movie recommendation system...")
recommender = MovieRecommender()
print("Recommendation system ready!")

@app.route('/')
def index():
    """Homepage with filters and search functionality"""
    # Get unique genres and countries for filter dropdowns
    genres = recommender.get_all_genres()
    countries = recommender.get_all_countries()
    
    return render_template('index.html', 
                         genres=genres, 
                         countries=countries,
                         movies=None,
                         recommendations=None)

@app.route('/get_filtered_movies', methods=['POST'])
def get_filtered_movies():
    """Get movies based on selected filters"""
    data = request.json
    selected_genres = data.get('genres', [])
    selected_countries = data.get('countries', [])
    
    movies = recommender.get_filtered_movies(selected_genres, selected_countries)
    return jsonify(movies)

@app.route('/recommend', methods=['POST'])
def recommend():
    """Get recommendations for a selected movie"""
    data = request.json
    movie_title = data.get('movie_title')
    selected_genres = data.get('genres', [])
    selected_countries = data.get('countries', [])
    
    if not movie_title:
        return jsonify({'error': 'No movie selected'})
    
    recommendations = recommender.get_recommendations(
        movie_title, selected_genres, selected_countries
    )
    
    if recommendations is None:
        return jsonify({'error': 'Movie not found'})
    
    return jsonify({
        'recommendations': recommendations.to_dict('records'),
        'selected_movie': movie_title
    })

@app.route('/random_movie', methods=['POST'])
def random_movie():
    """Get a random movie suggestion"""
    data = request.json
    selected_genres = data.get('genres', [])
    selected_countries = data.get('countries', [])
    
    random_movie = recommender.get_random_movie(selected_genres, selected_countries)
    
    if random_movie:
        recommendations = recommender.get_recommendations(
            random_movie, selected_genres, selected_countries
        )
        return jsonify({
            'random_movie': random_movie,
            'recommendations': recommendations.to_dict('records') if recommendations is not None else []
        })
    else:
        return jsonify({'error': 'No movies found with selected filters'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)