# src/content_based.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

class ContentBasedRecommender:
    def __init__(self, movies_df, content_matrix=None, tfidf_vectorizer=None):
        self.movies_df = movies_df
        self.content_matrix = content_matrix
        self.tfidf_vectorizer = tfidf_vectorizer
        self.similarity_matrix = None
        
    def compute_content_similarity(self):
        """Compute content similarity matrix using cosine similarity"""
        print("Computing content similarity matrix...")
        
        if self.content_matrix is None:
            raise ValueError("Content matrix not available. Run preprocessing first.")
            
        # Use linear_kernel for TF-IDF matrices (more efficient than cosine_similarity)
        self.similarity_matrix = linear_kernel(self.content_matrix, self.content_matrix)
        
        print(f"Content similarity matrix shape: {self.similarity_matrix.shape}")
        
    def get_movie_recommendations(self, movie_title, n_recommendations=10):
        """Get recommendations based on movie content similarity"""
        if self.similarity_matrix is None:
            self.compute_content_similarity()
            
        # Find movie index
        try:
            movie_idx = self.movies_df[self.movies_df['title'].str.lower() == movie_title.lower()].index[0]
        except IndexError:
            print(f"Movie '{movie_title}' not found in the database")
            return []
            
        # Get similarity scores for the movie
        sim_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        
        # Sort movies by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar movies (excluding the movie itself)
        sim_scores = sim_scores[1:n_recommendations+1]
        
        # Get movie indices and create recommendations list
        movie_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
        recommendations = []
        for idx, score in zip(movie_indices, similarity_scores):
            movie_info = self.movies_df.iloc[idx]
            recommendations.append({
                'movie_id': movie_info['id'],
                'title': movie_info['title'],
                'similarity_score': score,
                'genres': movie_info.get('genres', ''),
                'overview': movie_info.get('overview', ''),
                'release_date': movie_info.get('release_date', ''),
                'vote_average': movie_info.get('vote_average', 0)
            })
            
        return recommendations
        
    def get_recommendations_by_features(self, genres=None, keywords=None, cast=None, n_recommendations=10):
        """Get recommendations based on specific features"""
        if self.tfidf_vectorizer is None:
            print("TF-IDF vectorizer not available")
            return []
            
        # Create query string from features
        query_features = []
        if genres:
            query_features.append(genres)
        if keywords:
            query_features.append(keywords)
        if cast:
            query_features.append(cast)
            
        query_string = ' '.join(query_features).lower()
        
        if not query_string:
            print("No features provided for recommendation")
            return []
            
        # Transform query using the fitted vectorizer
        query_vector = self.tfidf_vectorizer.transform([query_string])
        
        # Compute similarity with all movies
        similarities = linear_kernel(query_vector, self.content_matrix).flatten()
        
        # Get top recommendations
        top_indices = similarities.argsort()[::-1][:n_recommendations]
        
        recommendations = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include movies with some similarity
                movie_info = self.movies_df.iloc[idx]
                recommendations.append({
                    'movie_id': movie_info['id'],
                    'title': movie_info['title'],
                    'similarity_score': similarities[idx],
                    'genres': movie_info.get('genres', ''),
                    'overview': movie_info.get('overview', ''),
                    'release_date': movie_info.get('release_date', ''),
                    'vote_average': movie_info.get('vote_average', 0)
                })
                
        return recommendations
        
    def find_similar_movies_by_id(self, movie_id, n_recommendations=10):
        """Get similar movies by movie ID"""
        try:
            movie_idx = self.movies_df[self.movies_df['id'] == movie_id].index[0]
            movie_title = self.movies_df.iloc[movie_idx]['title']
            return self.get_movie_recommendations(movie_title, n_recommendations)
        except IndexError:
            print(f"Movie with ID {movie_id} not found")
            return []
            
    def get_movie_info(self, movie_title):
        """Get detailed information about a specific movie"""
        try:
            movie = self.movies_df[self.movies_df['title'].str.lower() == movie_title.lower()].iloc[0]
            return {
                'movie_id': movie['id'],
                'title': movie['title'],
                'overview': movie.get('overview', ''),
                'genres': movie.get('genres', ''),
                'release_date': movie.get('release_date', ''),
                'vote_average': movie.get('vote_average', 0),
                'vote_count': movie.get('vote_count', 0),
                'runtime': movie.get('runtime', 0),
                'budget': movie.get('budget', 0),
                'revenue': movie.get('revenue', 0),
                'cast': movie.get('cast', ''),
                'crew': movie.get('crew', ''),
                'keywords': movie.get('keywords', '')
            }
        except IndexError:
            print(f"Movie '{movie_title}' not found")
            return None
            
    def search_movies(self, query, n_results=10):
        """Search for movies by title"""
        query = query.lower()
        matches = self.movies_df[self.movies_df['title'].str.lower().str.contains(query, na=False)]
        
        results = []
        for _, movie in matches.head(n_results).iterrows():
            results.append({
                'movie_id': movie['id'],
                'title': movie['title'],
                'overview': movie.get('overview', ''),
                'genres': movie.get('genres', ''),
                'release_date': movie.get('release_date', ''),
                'vote_average': movie.get('vote_average', 0)
            })
            
        return results
        
    def save_model(self, filepath):
        """Save the content-based model"""
        model_data = {
            'similarity_matrix': self.similarity_matrix,
            'movies_df': self.movies_df,
            'content_matrix': self.content_matrix,
            'tfidf_vectorizer': self.tfidf_vectorizer
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Content-based model saved to {filepath}")
        
    @classmethod
    def load_model(cls, filepath):
        """Load a trained content-based model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        recommender = cls(
            movies_df=model_data['movies_df'],
            content_matrix=model_data['content_matrix'],
            tfidf_vectorizer=model_data['tfidf_vectorizer']
        )
        recommender.similarity_matrix = model_data['similarity_matrix']
        
        print(f"Content-based model loaded from {filepath}")
        return recommender

if __name__ == "__main__":
    print("This is the content-based filtering module. Import and use in your main application.")