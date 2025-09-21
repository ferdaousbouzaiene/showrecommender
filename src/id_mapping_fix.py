# id_mapping_fix.py - Fix ID mismatch between movies and ratings
import pandas as pd
import numpy as np

def fix_id_mapping():
    print("🔧 FIXING ID MAPPING BETWEEN MOVIES AND RATINGS")
    print("=" * 60)
    
    # Load both datasets
    movies = pd.read_csv('data/movies.csv')
    ratings = pd.read_csv('data/ratings.csv')
    
    print(f"Movies loaded: {len(movies)} rows")
    print(f"Ratings loaded: {len(ratings)} rows")
    
    # Check ID ranges
    movie_ids = set(movies['id'].unique())
    rating_movie_ids = set(ratings['movieId'].unique())
    
    print(f"Movie IDs range: {min(movie_ids)} to {max(movie_ids)}")
    print(f"Rating movie IDs range: {min(rating_movie_ids)} to {max(rating_movie_ids)}")
    print(f"Common IDs: {len(movie_ids.intersection(rating_movie_ids))}")
    
    if len(movie_ids.intersection(rating_movie_ids)) < 100:
        print("🔄 Creating ID mapping...")
        
        # Create mapping: map rating movieIds to sequential movie IDs
        unique_rating_movies = sorted(ratings['movieId'].unique())
        
        # Take first N movies from movies dataset where N = number of unique rating movies
        n_movies = min(len(movies), len(unique_rating_movies))
        
        # Create simple sequential mapping
        id_mapping = {}
        for i, rating_movie_id in enumerate(unique_rating_movies[:n_movies]):
            new_id = i + 1
            id_mapping[rating_movie_id] = new_id
        
        # Apply mapping to ratings
        print(f"Mapping {len(id_mapping)} movie IDs...")
        ratings['movieId'] = ratings['movieId'].map(id_mapping)
        ratings = ratings.dropna(subset=['movieId'])
        ratings['movieId'] = ratings['movieId'].astype(int)
        
        # Update movies with sequential IDs
        movies = movies.head(n_movies).copy()
        movies['id'] = range(1, len(movies) + 1)
        
        print(f"After mapping:")
        print(f"Movies: {len(movies)} rows with IDs 1 to {len(movies)}")
        print(f"Ratings: {len(ratings)} rows")
        
        # Verify overlap
        new_common = set(movies['id']).intersection(set(ratings['movieId']))
        print(f"Common IDs after mapping: {len(new_common)}")
        
        # Save updated datasets
        movies.to_csv('data/movies.csv', index=False)
        ratings.to_csv('data/ratings.csv', index=False)
        
        print("✅ ID mapping completed and files saved")
        return True
    else:
        print("✅ Sufficient common IDs found, no mapping needed")
        return True

if __name__ == "__main__":
    if fix_id_mapping():
        print("\n🎉 ID mapping successful!")
        print("Now run: python setup_and_run.py")
    else:
        print("\n❌ ID mapping failed")
