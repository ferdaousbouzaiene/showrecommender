# simple_data_fix.py - Simple fix for the CSV issues
import pandas as pd
import numpy as np

def simple_fix():
    """Simple fix for the data format issues"""
    
    print("🔧 SIMPLE DATA FIX")
    print("=" * 30)
    
    # Fix movies data
    print("📽️ Fixing movies...")
    
    # Your movies data has headers as first row data - let's skip that
    try:
        # Read raw data
        movies_raw = pd.read_csv('data/movies.csv', header=None, nrows=10)
        print("First few raw rows:")
        print(movies_raw.head(3))
        
        # Skip the problematic first row and read properly
        movies = pd.read_csv('data/movies.csv', skiprows=1)
        
        # If that doesn't work, manually set headers
        if movies.empty or 'adult,belongs_to_collection' in str(movies.columns[0]):
            print("Using manual headers...")
            movies = pd.read_csv('data/movies.csv', header=None, skiprows=1)
            movies.columns = ['id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + \
                           [f'genre_{i}' for i in range(19)] + \
                           ['overview', 'genres', 'keywords', 'cast', 'crew', 'vote_average', 'vote_count']
        
        print(f"Movies shape: {movies.shape}")
        
        # Clean up the ID column - it might have mixed data
        movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
        movies = movies.dropna(subset=['id'])
        movies['id'] = movies['id'].astype(int)
        
        # Clean up other columns
        movies['title'] = movies['title'].astype(str)
        movies['overview'] = movies['overview'].fillna('No overview available')
        movies['genres'] = movies['genres'].fillna('Unknown')
        
        # Convert genre columns to a single genre string
        genre_cols = [f'genre_{i}' for i in range(19)]
        genre_names = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 
                      'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'Unknown']
        
        def get_genres(row):
            genres = []
            for i, col in enumerate(genre_cols):
                if col in movies.columns and i < len(genre_names) and row[col] == 1:
                    genres.append(genre_names[i])
            return '|'.join(genres) if genres else 'Unknown'
        
        movies['genres'] = movies.apply(get_genres, axis=1)
        
        print(f"Cleaned movies: {len(movies)}")
        
    except Exception as e:
        print(f"Movies error: {e}")
        return False
    
    # Fix ratings data
    print("\n⭐ Fixing ratings...")
    
    try:
        # Check first few rows
        ratings_raw = pd.read_csv('data/ratings.csv', header=None, nrows=5)
        print("First few rating rows:")
        print(ratings_raw.head())
        
        # Skip the header row that's in the data
        if str(ratings_raw.iloc[0, 0]) == 'userId,movieId,rating,timestamp':
            print("Skipping header row in data...")
            ratings = pd.read_csv('data/ratings.csv', skiprows=1)
        else:
            ratings = pd.read_csv('data/ratings.csv')
        
        # If columns are wrong, set manually
        if 'userId,movieId,rating,timestamp' in str(ratings.columns[0]):
            ratings = pd.read_csv('data/ratings.csv', header=None, skiprows=1)
            ratings.columns = ['userId', 'movieId', 'rating', 'timestamp']
        
        print(f"Ratings shape: {ratings.shape}")
        
        # Clean ratings data
        ratings['userId'] = pd.to_numeric(ratings['userId'], errors='coerce')
        ratings['movieId'] = pd.to_numeric(ratings['movieId'], errors='coerce') 
        ratings['rating'] = pd.to_numeric(ratings['rating'], errors='coerce')
        
        # Remove invalid rows
        ratings = ratings.dropna(subset=['userId', 'movieId', 'rating'])
        ratings = ratings[(ratings['userId'] > 0) & (ratings['movieId'] > 0) & (ratings['rating'] > 0)]
        
        print(f"Cleaned ratings: {len(ratings)}")
        
    except Exception as e:
        print(f"Ratings error: {e}")
        return False
    
    # Save cleaned data
    try:
        movies.to_csv('data/movies.csv', index=False)
        ratings.to_csv('data/ratings.csv', index=False)
        print("\n✅ Saved cleaned data files")
        
        # Quick test
        test_movies = pd.read_csv('data/movies.csv')
        test_ratings = pd.read_csv('data/ratings.csv')
        
        print(f"✅ Test - Movies: {len(test_movies)}, Ratings: {len(test_ratings)}")
        print(f"Sample movie: {test_movies['title'].iloc[0]}")
        print(f"Sample rating: {test_ratings['rating'].iloc[0]}")
        
        return True
        
    except Exception as e:
        print(f"Save error: {e}")
        return False

if __name__ == "__main__":
    if simple_fix():
        print("\n🎉 DATA FIXED! Now run: python setup_and_run.py")
    else:
        print("\n❌ Fix failed")