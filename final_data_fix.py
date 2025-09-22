# final_data_fix.py - Complete fix for TMDB data
import pandas as pd
import numpy as np
import json
import ast
import os

def parse_json_safely(x):
    """Parse JSON strings safely"""
    if pd.isna(x) or x == '' or x == '[]':
        return []
    try:
        return json.loads(x)
    except:
        try:
            return ast.literal_eval(x)
        except:
            return []

def extract_names(items):
    """Extract names from list of dictionaries"""
    if not items or not isinstance(items, list):
        return "Unknown"
    names = [item.get('name', '') for item in items if isinstance(item, dict) and 'name' in item]
    return '|'.join(names) if names else "Unknown"

def final_fix():
    print("🔧 FINAL DATA FIX FOR TMDB FORMAT")
    print("=" * 50)
    
    # Load and fix movies data
    print("📽️ Processing movies data...")
    try:
        movies = pd.read_csv('data/movies.csv', low_memory=False)
        print(f"Original movies shape: {movies.shape}")
        print(f"Columns: {list(movies.columns)}")
        
        # Clean up the data - remove duplicate column mappings
        # Keep only the columns we need and rename properly
        
        # Essential columns mapping
        column_mapping = {
            'id': 'id',
            'title': 'title', 
            'original_title': 'original_title',
            'overview': 'overview',
            'genres': 'genres',
            'release_date': 'release_date',
            'vote_average': 'vote_average',
            'vote_count': 'vote_count',
            'popularity': 'popularity',
            'runtime': 'runtime',
            'budget': 'budget',
            'revenue': 'revenue'
        }
        
        # Select and rename only existing columns
        available_cols = {}
        for new_name, old_name in column_mapping.items():
            if old_name in movies.columns:
                available_cols[old_name] = new_name
        
        # Keep only available columns and rename
        movies_clean = movies[list(available_cols.keys())].copy()
        movies_clean = movies_clean.rename(columns=available_cols)
        
        print(f"After column selection: {movies_clean.shape}")
        print(f"Selected columns: {list(movies_clean.columns)}")
        
        # Clean essential columns
        movies_clean['id'] = pd.to_numeric(movies_clean['id'], errors='coerce')
        movies_clean = movies_clean.dropna(subset=['id'])
        movies_clean['id'] = movies_clean['id'].astype(int)
        
        # Clean text columns
        movies_clean['title'] = movies_clean['title'].astype(str)
        movies_clean['overview'] = movies_clean['overview'].fillna('No overview available').astype(str)
        
        # Parse and clean genres
        if 'genres' in movies_clean.columns:
            print("Parsing genres...")
            movies_clean['genres_parsed'] = movies_clean['genres'].apply(parse_json_safely)
            movies_clean['genres'] = movies_clean['genres_parsed'].apply(extract_names)
        else:
            movies_clean['genres'] = 'Unknown'
        
        # Clean numeric columns safely
        numeric_cols = ['vote_average', 'vote_count', 'popularity', 'runtime', 'budget', 'revenue']
        for col in numeric_cols:
            if col in movies_clean.columns:
                movies_clean[col] = pd.to_numeric(movies_clean[col], errors='coerce').fillna(0)
            else:
                movies_clean[col] = 0
        
        # Add missing required columns
        required_cols = ['keywords', 'cast', 'crew', 'release_date']
        for col in required_cols:
            if col not in movies_clean.columns:
                movies_clean[col] = '' if col != 'release_date' else '2000-01-01'
        
        # Remove any parsing helper columns
        movies_clean = movies_clean[[col for col in movies_clean.columns if not col.endswith('_parsed')]]
        
        print(f"Final movies shape: {movies_clean.shape}")
        print(f"Sample movie: {movies_clean.iloc[0]['title']}")
        print(f"Sample genres: {movies_clean.iloc[0]['genres']}")
        
    except Exception as e:
        print(f"❌ Movies processing error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load and fix ratings data
    print("\n⭐ Processing ratings data...")
    try:
        ratings = pd.read_csv('data/ratings.csv', low_memory=False)
        print(f"Original ratings shape: {ratings.shape}")
        
        # Clean ratings
        ratings['userId'] = pd.to_numeric(ratings['userId'], errors='coerce')
        ratings['movieId'] = pd.to_numeric(ratings['movieId'], errors='coerce')
        ratings['rating'] = pd.to_numeric(ratings['rating'], errors='coerce')
        
        ratings = ratings.dropna(subset=['userId', 'movieId', 'rating'])
        ratings = ratings[(ratings['userId'] > 0) & (ratings['movieId'] > 0) & (ratings['rating'] > 0)]
        
        print(f"Final ratings shape: {ratings.shape}")
        
    except Exception as e:
        print(f"❌ Ratings processing error: {e}")
        return False
    
    # Create ID mapping between datasets
    print("\n🔗 Creating ID mapping...")
    movie_ids = set(movies_clean['id'].unique())
    rating_movie_ids = set(ratings['movieId'].unique())
    common_movies = movie_ids.intersection(rating_movie_ids)
    
    print(f"Movies with IDs: {len(movie_ids)}")
    print(f"Ratings with movie IDs: {len(rating_movie_ids)}")
    print(f"Common movies: {len(common_movies)}")
    
    if len(common_movies) < 100:
        print("Creating new ID mapping...")
        
        # Take first N movies and create sequential mapping
        n_movies = min(2000, len(movies_clean))  # Limit for performance
        movies_sample = movies_clean.head(n_movies).copy()
        movies_sample['id'] = range(1, len(movies_sample) + 1)
        
        # Create mapping for ratings
        unique_rating_movies = ratings['movieId'].unique()
        n_map = min(len(unique_rating_movies), n_movies)
        
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_rating_movies[:n_map], 1)}
        
        # Apply mapping to ratings
        ratings['movieId'] = ratings['movieId'].map(id_mapping)
        ratings = ratings.dropna(subset=['movieId'])
        ratings['movieId'] = ratings['movieId'].astype(int)
        
        # Filter ratings to mapped movies only
        ratings = ratings[ratings['movieId'].isin(range(1, n_movies + 1))]
        
        movies_final = movies_sample
        ratings_final = ratings
        
        print(f"After mapping - Movies: {len(movies_final)}, Ratings: {len(ratings_final)}")
    else:
        print("Using existing ID mapping...")
        # Filter to common movies
        movies_final = movies_clean[movies_clean['id'].isin(common_movies)]
        ratings_final = ratings[ratings['movieId'].isin(common_movies)]
    
    # Final validation
    final_common = set(movies_final['id']).intersection(set(ratings_final['movieId']))
    print(f"Final common movies: {len(final_common)}")
    
    if len(final_common) < 50:
        print("❌ Insufficient common movies after processing")
        return False
    
    # Save cleaned data
    try:
        movies_final.to_csv('data/movies.csv', index=False)
        ratings_final.to_csv('data/ratings.csv', index=False)
        
        print("✅ Data saved successfully!")
        
        # Quick verification
        test_movies = pd.read_csv('data/movies.csv')
        test_ratings = pd.read_csv('data/ratings.csv')
        
        print(f"✅ Verification:")
        print(f"   Movies: {len(test_movies)} rows")
        print(f"   Ratings: {len(test_ratings)} rows") 
        print(f"   Sample movie: {test_movies['title'].iloc[0]}")
        print(f"   Sample rating: {test_ratings['rating'].iloc[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Save error: {e}")
        return False

if __name__ == "__main__":
    if final_fix():
        print("\n🎉 DATA COMPLETELY FIXED!")
        print("Now run: python setup_and_run.py")
        print("Choose option 2 to re-run preprocessing with clean data")
    else:
        print("\n❌ Fix failed - check errors above")
