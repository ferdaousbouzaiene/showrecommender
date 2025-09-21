# src/hybrid_engine.py - FIXED VERSION
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
import os
import sys

# Fix import issues by using absolute imports
try:
    # Try relative imports first (when run as package)
    from .collaborative_filtering import CollaborativeFilteringRecommender
    from .content_based import ContentBasedRecommender  
    from .gpt_recommender import GPTRecommender
except ImportError:
    # Fall back to absolute imports (when run directly)
    from collaborative_filtering import CollaborativeFilteringRecommender
    from content_based import ContentBasedRecommender
    from gpt_recommender import GPTRecommender

class HybridRecommendationEngine:
    def __init__(
        self,
        movies_df: pd.DataFrame,
        user_item_matrix: pd.DataFrame = None,
        content_matrix = None,
        tfidf_vectorizer = None,
        openai_api_key: str = None
    ):
        self.movies_df = movies_df
        
        # Initialize recommenders
        self.collaborative_recommender = None
        self.content_recommender = None
        self.gpt_recommender = None
        
        # Initialize collaborative filtering if data is available
        if user_item_matrix is not None:
            self.collaborative_recommender = CollaborativeFilteringRecommender(user_item_matrix)
            
        # Initialize content-based filtering
        if content_matrix is not None and tfidf_vectorizer is not None:
            self.content_recommender = ContentBasedRecommender(
                movies_df, content_matrix, tfidf_vectorizer
            )
            
        # Initialize GPT recommender if API key is available
        if openai_api_key:
            self.gpt_recommender = GPTRecommender(openai_api_key, movies_df)
    
    def train_models(self):
        """Train all available recommendation models"""
        print("Training hybrid recommendation engine...")
        
        if self.collaborative_recommender:
            print("Training collaborative filtering model...")
            self.collaborative_recommender.train_model()
            
        if self.content_recommender:
            print("Training content-based model...")
            self.content_recommender.compute_content_similarity()
            
        print("Model training completed!")
    
    def get_hybrid_recommendations(
        self,
        movie_title: str = None,
        user_id: int = None,
        user_preferences: str = None,
        n_recommendations: int = 10,
        weights: Dict[str, float] = None
    ) -> List[Dict]:
        """
        Generate hybrid recommendations combining multiple approaches
        
        Args:
            movie_title: Title of movie for content-based recommendations
            user_id: User ID for collaborative filtering
            user_preferences: Natural language preferences for GPT recommendations
            n_recommendations: Number of recommendations to return
            weights: Weights for different recommendation methods
        """
        
        if weights is None:
            weights = {
                'collaborative': 0.4,
                'content': 0.4,
                'gpt': 0.2
            }
        
        all_recommendations = {}
        
        # Content-based recommendations
        if movie_title and self.content_recommender:
            print(f"Getting content-based recommendations for '{movie_title}'...")
            content_recs = self.content_recommender.get_movie_recommendations(
                movie_title, n_recommendations * 2
            )
            
            for rec in content_recs:
                movie_id = rec['movie_id']
                if movie_id not in all_recommendations:
                    all_recommendations[movie_id] = {
                        'movie_info': rec,
                        'scores': {},
                        'sources': []
                    }
                    
                all_recommendations[movie_id]['scores']['content'] = rec['similarity_score']
                all_recommendations[movie_id]['sources'].append('content-based')
        
        # Collaborative filtering recommendations
        if user_id and self.collaborative_recommender:
            print(f"Getting collaborative recommendations for user {user_id}...")
            collab_recs = self.collaborative_recommender.get_user_recommendations(
                user_id, n_recommendations * 2
            )
            
            for movie_id, rating in collab_recs:
                if movie_id not in all_recommendations:
                    # Get movie info from database
                    movie_info = self._get_movie_info_by_id(movie_id)
                    if movie_info:
                        all_recommendations[movie_id] = {
                            'movie_info': movie_info,
                            'scores': {},
                            'sources': []
                        }
                
                if movie_id in all_recommendations:
                    all_recommendations[movie_id]['scores']['collaborative'] = rating / 5.0  # Normalize to 0-1
                    all_recommendations[movie_id]['sources'].append('collaborative')
        
        # GPT-based recommendations
        if user_preferences and self.gpt_recommender:
            print("Getting GPT-based recommendations...")
            gpt_result = self.gpt_recommender.generate_recommendations_by_preference(
                user_preferences, n_recommendations
            )
            
            if gpt_result['success']:
                for rec in gpt_result['recommendations']:
                    movie_id = rec.get('movie_id')
                    if movie_id and movie_id not in all_recommendations:
                        all_recommendations[movie_id] = {
                            'movie_info': rec,
                            'scores': {},
                            'sources': []
                        }
                    
                    if movie_id and movie_id in all_recommendations:
                        confidence = rec.get('gpt_confidence', 0.5)
                        all_recommendations[movie_id]['scores']['gpt'] = confidence
                        all_recommendations[movie_id]['sources'].append('gpt')
                        all_recommendations[movie_id]['gpt_reason'] = rec.get('gpt_reason', '')
        
        # Calculate hybrid scores
        hybrid_recommendations = []
        
        for movie_id, data in all_recommendations.items():
            scores = data['scores']
            
            # Calculate weighted hybrid score
            hybrid_score = 0.0
            total_weight = 0.0
            
            for method, weight in weights.items():
                if method in scores:
                    hybrid_score += scores[method] * weight
                    total_weight += weight
            
            if total_weight > 0:
                hybrid_score /= total_weight
                
                # Boost score if multiple methods agree
                method_count = len(scores)
                if method_count > 1:
                    hybrid_score *= (1 + 0.1 * (method_count - 1))  # 10% bonus per additional method
            
            recommendation = data['movie_info'].copy()
            recommendation.update({
                'hybrid_score': hybrid_score,
                'individual_scores': scores,
                'recommendation_sources': data['sources'],
                'method_count': len(scores)
            })
            
            # Add GPT reasoning if available
            if 'gpt_reason' in data:
                recommendation['gpt_reason'] = data['gpt_reason']
            
            hybrid_recommendations.append(recommendation)
        
        # Sort by hybrid score and return top N
        hybrid_recommendations.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return hybrid_recommendations[:n_recommendations]
    
    def get_movie_recommendations_by_title(self, movie_title: str, n_recommendations: int = 10) -> List[Dict]:
        """Get recommendations based on a single movie title"""
        if not self.content_recommender:
            print("Content-based recommender not available")
            return []
            
        recommendations = self.content_recommender.get_movie_recommendations(movie_title, n_recommendations)
        
        # Enhance with GPT explanations if available
        if self.gpt_recommender and recommendations:
            try:
                explanation = self.gpt_recommender.generate_similar_movie_explanation(
                    movie_title, [rec['title'] for rec in recommendations[:5]]
                )
                for rec in recommendations:
                    rec['gpt_explanation'] = explanation
            except Exception as e:
                print(f"Could not generate GPT explanations: {e}")
        
        return recommendations
    
    def get_recommendations_by_preferences(self, preferences: str, n_recommendations: int = 10) -> Dict:
        """Get recommendations based on natural language preferences"""
        if not self.gpt_recommender:
            print("GPT recommender not available")
            return {'recommendations': [], 'explanation': 'GPT recommender not configured'}
        
        return self.gpt_recommender.generate_recommendations_by_preference(preferences, n_recommendations)
    
    def search_movies(self, query: str, n_results: int = 10) -> List[Dict]:
        """Search for movies by title"""
        if self.content_recommender:
            return self.content_recommender.search_movies(query, n_results)
        else:
            # Fallback search using pandas
            query_lower = query.lower()
            matches = self.movies_df[
                self.movies_df['title'].str.lower().str.contains(query_lower, na=False)
            ]
            
            results = []
            for _, movie in matches.head(n_results).iterrows():
                results.append({
                    'movie_id': movie.get('id'),
                    'title': movie['title'],
                    'overview': movie.get('overview', ''),
                    'genres': movie.get('genres', ''),
                    'release_date': movie.get('release_date', ''),
                    'vote_average': movie.get('vote_average', 0)
                })
            
            return results
    
    def _get_movie_info_by_id(self, movie_id: int) -> Optional[Dict]:
        """Get movie information by ID"""
        try:
            movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
            return {
                'movie_id': movie['id'],
                'title': movie['title'],
                'overview': movie.get('overview', ''),
                'genres': movie.get('genres', ''),
                'release_date': movie.get('release_date', ''),
                'vote_average': movie.get('vote_average', 0),
                'vote_count': movie.get('vote_count', 0),
                'runtime': movie.get('runtime', 0)
            }
        except (IndexError, KeyError):
            return None
    
    def get_trending_movies(self, n_movies: int = 20) -> List[Dict]:
        """Get trending movies based on popularity and rating"""
        # Sort by a combination of vote_average and vote_count
        trending = self.movies_df.copy()
        trending = trending[trending['vote_count'] >= 100]  # Filter out movies with too few votes
        
        # Create a weighted score combining rating and popularity
        trending['trending_score'] = (
            trending['vote_average'] * 0.7 + 
            np.log1p(trending['vote_count']) * 0.3
        )
        
        trending = trending.nlargest(n_movies, 'trending_score')
        
        results = []
        for _, movie in trending.iterrows():
            results.append({
                'movie_id': movie.get('id'),
                'title': movie['title'],
                'overview': movie.get('overview', ''),
                'genres': movie.get('genres', ''),
                'release_date': movie.get('release_date', ''),
                'vote_average': movie.get('vote_average', 0),
                'vote_count': movie.get('vote_count', 0),
                'trending_score': movie['trending_score']
            })
        
        return results
    
    def get_random_recommendations(self, n_movies: int = 10) -> List[Dict]:
        """Get random movie recommendations"""
        random_movies = self.movies_df.sample(n=min(n_movies, len(self.movies_df)))
        
        results = []
        for _, movie in random_movies.iterrows():
            results.append({
                'movie_id': movie.get('id'),
                'title': movie['title'],
                'overview': movie.get('overview', ''),
                'genres': movie.get('genres', ''),
                'release_date': movie.get('release_date', ''),
                'vote_average': movie.get('vote_average', 0)
            })
        
        return results

if __name__ == "__main__":
    print("This is the hybrid recommendation engine. Import and use in your main application.")