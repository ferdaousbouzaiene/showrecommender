# src/data_preprocessing.py - MINIMAL WORKING VERSION
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, movies_path, ratings_path):
        self.movies_path = movies_path
        self.ratings_path = ratings_path
        self.movies_df = None
        self.ratings_df = None
        self.user_item_matrix = None
        self.tfidf_vectorizer = None
        self.content_matrix = None
        
    def load_data(self):
        """Load and clean data simply"""
        print("Loading data...")
        
        # Load movies
        self.movies_df = pd.read_csv(self.movies_path, low_memory=False)
        self.ratings_df = pd.read_csv(self.ratings_path, low_memory=False)
        
        print(f"Movies columns: {list(self.movies_df.columns)}")
        print(f"Ratings columns: {list(self.ratings_df.columns)}")
        print(f"Movies shape: {self.movies_df.shape}")
        print(f"Ratings shape: {self.ratings_df.shape}")
        
        # Simple cleaning - no complex renaming
        # Just ensure we have the basic required columns
        required_movie_cols = ['id', 'title', 'overview', 'genres']
        required_rating_cols = ['userId', 'movieId', 'rating']
        
        # Check movies
        for col in required_movie_cols:
            if col not in self.movies_df.columns:
                if col == 'overview' and 'original_title' in self.movies_df.columns:
                    self.movies_df['overview'] = self.movies_df['original_title'] + ' - Movie'
                elif col == 'genres':
                    self.movies_df['genres'] = 'Unknown'
                elif col == 'overview':
                    self.movies_df['overview'] = 'No overview available'
        
        # Check ratings - they should already be correct
        for col in required_rating_cols:
            if col not in self.ratings_df.columns:
                print(f"ERROR: Missing {col} in ratings")
                
        # Clean data types
        self.movies_df['id'] = pd.to_numeric(self.movies_df['id'], errors='coerce')
        self.movies_df = self.movies_df.dropna(subset=['id', 'title'])
        
        self.ratings_df['userId'] = pd.to_numeric(self.ratings_df['userId'], errors='coerce')
        self.ratings_df['movieId'] = pd.to_numeric(self.ratings_df['movieId'], errors='coerce')
        self.ratings_df['rating'] = pd.to_numeric(self.ratings_df['rating'], errors='coerce')
        self.ratings_df = self.ratings_df.dropna(subset=['userId', 'movieId', 'rating'])
        
        # Add missing columns with defaults
        for col in ['keywords', 'cast', 'crew', 'vote_average', 'vote_count', 'release_date']:
            if col not in self.movies_df.columns:
                if col in ['vote_average', 'vote_count']:
                    self.movies_df[col] = 0
                else:
                    self.movies_df[col] = ''
        
        print(f"After cleaning: {len(self.movies_df)} movies, {len(self.ratings_df)} ratings")
        
    def create_user_item_matrix(self):
        """Create user-item matrix"""
        print("Creating user-item matrix...")
        
        # Check for common movies
        movie_ids = set(self.movies_df['id'])
        rating_movie_ids = set(self.ratings_df['movieId'])
        common_movies = movie_ids.intersection(rating_movie_ids)
        
        print(f"Common movies: {len(common_movies)}")
        
        if len(common_movies) > 10:
            # Filter to common movies
            filtered_ratings = self.ratings_df[self.ratings_df['movieId'].isin(common_movies)]
            
            # Limit size for memory
            if len(filtered_ratings) > 50000:
                filtered_ratings = filtered_ratings.sample(n=50000)
                
            try:
                self.user_item_matrix = filtered_ratings.pivot_table(
                    index='userId', columns='movieId', values='rating', fill_value=0
                )
                print(f"User-item matrix shape: {self.user_item_matrix.shape}")
            except Exception as e:
                print(f"Matrix creation failed: {e}")
                self.user_item_matrix = pd.DataFrame()
        else:
            print("Not enough common movies for collaborative filtering")
            self.user_item_matrix = pd.DataFrame()
        
    def process_content_features(self):
        """Process content features"""
        print("Processing content features...")
        
        # Create simple combined features
        self.movies_df['combined_features'] = (
            self.movies_df['title'].astype(str) + ' ' +
            self.movies_df['overview'].astype(str) + ' ' +
            self.movies_df['genres'].astype(str)
        ).str.lower().str.replace(r'[^a-zA-Z0-9\s]', ' ', regex=True)
        
        # Remove very short descriptions
        mask = self.movies_df['combined_features'].str.len() > 10
        self.movies_df = self.movies_df[mask]
        
        try:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=2000,
                min_df=2,
                stop_words='english'
            )
            self.content_matrix = self.tfidf_vectorizer.fit_transform(self.movies_df['combined_features'])
            print(f"Content matrix shape: {self.content_matrix.shape}")
        except Exception as e:
            print(f"TF-IDF failed: {e}")
            # Create dummy matrix
            from scipy.sparse import csr_matrix
            self.content_matrix = csr_matrix((len(self.movies_df), 1000))
            self.tfidf_vectorizer = None
        
    def save_processed_data(self, output_dir='data/processed'):
        """Save processed data"""
        os.makedirs(output_dir, exist_ok=True)
        print("Saving processed data...")
        
        with open(f'{output_dir}/user_item_matrix.pkl', 'wb') as f:
            pickle.dump(self.user_item_matrix, f)
            
        with open(f'{output_dir}/content_matrix.pkl', 'wb') as f:
            pickle.dump(self.content_matrix, f)
            
        with open(f'{output_dir}/tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
            
        self.movies_df.to_csv(f'{output_dir}/processed_movies.csv', index=False)
        self.ratings_df.to_csv(f'{output_dir}/processed_ratings.csv', index=False)
        
        print("✅ Data saved successfully!")
        
    def run_preprocessing(self):
        """Run preprocessing"""
        try:
            self.load_data()
            self.create_user_item_matrix()
            self.process_content_features()
            self.save_processed_data()
            
            return {
                'movies_df': self.movies_df,
                'ratings_df': self.ratings_df,
                'user_item_matrix': self.user_item_matrix,
                'content_matrix': self.content_matrix,
                'tfidf_vectorizer': self.tfidf_vectorizer
            }
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
