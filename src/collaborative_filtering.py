# src/collaborative_filtering.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import pickle

class CollaborativeFilteringRecommender:
    def __init__(self, user_item_matrix, n_components=50):
        self.user_item_matrix = user_item_matrix
        self.n_components = n_components
        self.user_similarity = None
        self.item_similarity = None
        self.svd_model = None
        self.user_factors = None
        self.item_factors = None
        
    def compute_user_similarity(self):
        """Compute user-user similarity matrix using cosine similarity"""
        print("Computing user similarity matrix...")
        
        # Transpose to get users as columns for similarity computation
        user_matrix = self.user_item_matrix.values
        
        # Compute cosine similarity between users
        self.user_similarity = cosine_similarity(user_matrix)
        
        print(f"User similarity matrix shape: {self.user_similarity.shape}")
        
    def compute_item_similarity(self):
        """Compute item-item similarity matrix using cosine similarity"""
        print("Computing item similarity matrix...")
        
        # Items are columns, so transpose for item-item similarity
        item_matrix = self.user_item_matrix.values.T
        
        # Compute cosine similarity between items
        self.item_similarity = cosine_similarity(item_matrix)
        
        print(f"Item similarity matrix shape: {self.item_similarity.shape}")
        
    def fit_svd(self):
        """Fit SVD model for matrix factorization"""
        print(f"Fitting SVD model with {self.n_components} components...")
        
        self.svd_model = TruncatedSVD(n_components=self.n_components, random_state=42)
        
        # Fit SVD on user-item matrix
        self.user_factors = self.svd_model.fit_transform(self.user_item_matrix.values)
        self.item_factors = self.svd_model.components_.T
        
        print("SVD model fitted successfully!")
        
    def predict_user_rating(self, user_idx, item_idx, method='svd'):
        """Predict rating for a user-item pair"""
        if method == 'svd' and self.user_factors is not None:
            # SVD-based prediction
            user_vector = self.user_factors[user_idx]
            item_vector = self.item_factors[item_idx]
            prediction = np.dot(user_vector, item_vector)
            return np.clip(prediction, 0.5, 5.0)  # Clip to valid rating range
            
        elif method == 'user_based' and self.user_similarity is not None:
            # User-based collaborative filtering
            similar_users = self.user_similarity[user_idx]
            user_ratings = self.user_item_matrix.iloc[:, item_idx].values
            
            # Weighted average of similar users' ratings
            numerator = np.sum(similar_users * user_ratings)
            denominator = np.sum(np.abs(similar_users))
            
            if denominator > 0:
                prediction = numerator / denominator
                return np.clip(prediction, 0.5, 5.0)
                
        return 3.0  # Default rating if prediction fails
        
    def get_user_recommendations(self, user_id, n_recommendations=10, method='svd'):
        """Get recommendations for a specific user"""
        if user_id not in self.user_item_matrix.index:
            print(f"User {user_id} not found in the dataset")
            return []
            
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_ratings = self.user_item_matrix.iloc[user_idx]
        
        # Get items the user hasn't rated
        unrated_items = user_ratings[user_ratings == 0].index
        
        # Predict ratings for unrated items
        predictions = []
        for item_id in unrated_items:
            item_idx = self.user_item_matrix.columns.get_loc(item_id)
            predicted_rating = self.predict_user_rating(user_idx, item_idx, method)
            predictions.append((item_id, predicted_rating))
            
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
        
    def get_item_recommendations(self, movie_id, n_recommendations=10):
        """Get similar items based on item-item similarity"""
        if movie_id not in self.user_item_matrix.columns:
            print(f"Movie {movie_id} not found in the dataset")
            return []
            
        if self.item_similarity is None:
            self.compute_item_similarity()
            
        item_idx = self.user_item_matrix.columns.get_loc(movie_id)
        item_similarities = self.item_similarity[item_idx]
        
        # Get most similar items (excluding the item itself)
        similar_items_idx = np.argsort(item_similarities)[::-1][1:n_recommendations+1]
        
        recommendations = []
        for idx in similar_items_idx:
            similar_movie_id = self.user_item_matrix.columns[idx]
            similarity_score = item_similarities[idx]
            recommendations.append((similar_movie_id, similarity_score))
            
        return recommendations
        
    def train_model(self, use_svd=True, compute_similarities=True):
        """Train the collaborative filtering model"""
        print("Training collaborative filtering model...")
        
        if use_svd:
            self.fit_svd()
            
        if compute_similarities:
            self.compute_user_similarity()
            self.compute_item_similarity()
            
        print("Model training completed!")
        
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'user_similarity': self.user_similarity,
            'item_similarity': self.item_similarity,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'svd_components': self.svd_model.components_ if self.svd_model else None,
            'user_item_matrix_index': self.user_item_matrix.index,
            'user_item_matrix_columns': self.user_item_matrix.columns
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Model saved to {filepath}")
        
    @classmethod
    def load_model(cls, filepath, user_item_matrix):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        recommender = cls(user_item_matrix)
        recommender.user_similarity = model_data['user_similarity']
        recommender.item_similarity = model_data['item_similarity']
        recommender.user_factors = model_data['user_factors']
        recommender.item_factors = model_data['item_factors']
        
        print(f"Model loaded from {filepath}")
        return recommender

if __name__ == "__main__":
    # Example usage
    print("This is the collaborative filtering module. Import and use in your main application.")