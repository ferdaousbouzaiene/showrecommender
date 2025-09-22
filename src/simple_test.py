# simple_test.py - Simple test of your data
import pandas as pd

print("🧪 SIMPLE DATA TEST")
print("=" * 30)

# Test data loading
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

print(f"Movies: {len(movies)} rows")
print(f"Ratings: {len(ratings)} rows")

# Check common IDs
movie_ids = set(movies['id'])
rating_ids = set(ratings['movieId'])
common = movie_ids.intersection(rating_ids)

print(f"Common movie IDs: {len(common)}")
print(f"Movie ID range: {min(movie_ids)} - {max(movie_ids)}")
print(f"Rating movie ID range: {min(rating_ids)} - {max(rating_ids)}")

if len(common) > 10:
    print("✅ Sufficient overlap - should work!")
else:
    print("❌ Need ID mapping")

# Test content features
print(f"Sample movie: {movies['title'].iloc[0]}")
print(f"Sample genres: {movies['genres'].iloc[0]}")
print(f"Sample overview: {movies['overview'].iloc[0][:100]}...")
