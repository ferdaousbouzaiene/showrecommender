import pandas as pd
import json
import ast
import numpy as np

def parse_json_column(x):
    """Parse JSON-like strings safely"""
    if pd.isna(x) or x == '' or x == '[]':
        return []
    try:
        # Try parsing as JSON first
        return json.loads(x)
    except:
        try:
            # Try literal_eval for Python-like strings
            return ast.literal_eval(x)
        except:
            return []

def extract_names_from_list(items):
    """Extract 'name' fields from list of dictionaries"""
    if not items or not isinstance(items, list):
        return "Unknown"
    names = [item.get('name', '') for item in items if isinstance(item, dict) and 'name' in item]
    return '|'.join(names) if names else "Unknown"

# Fix the CSV parsing issue first
print("🔧 Fixing movies data with JSON columns...")

# Read the file correctly - your header seems malformed
try:
    # Skip problematic header and use manual column names for TMDB format
    movies = pd.read_csv('data/movies.csv', low_memory=False)
    
    # If the header is still problematic, read without header
    if 'adult,belongs_to_collection' in str(movies.columns[0]):
        movies = pd.read_csv('data/movies.csv', header=None, skiprows=1, low_memory=False)
        # TMDB standard columns
        movies.columns = [
            'adult', 'belongs_to_collection', 'budget', 'genres', 'homepage', 'id', 
            'imdb_id', 'original_language', 'original_title', 'overview', 'popularity',
            'poster_path', 'production_companies', 'production_countries', 
            'release_date', 'revenue', 'runtime', 'spoken_languages', 'status', 
            'tagline', 'title', 'video', 'vote_average', 'vote_count'
        ]
    
    print(f"Loaded movies shape: {movies.shape}")
    
    # Clean and convert basic columns
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
    movies = movies.dropna(subset=['id', 'title'])
    movies['id'] = movies['id'].astype(int)
    
    # Parse JSON columns
    print("Parsing JSON columns...")
    
    # Parse genres
    if 'genres' in movies.columns:
        movies['genres_parsed'] = movies['genres'].apply(parse_json_column)
        movies['genres'] = movies['genres_parsed'].apply(extract_names_from_list)
    else:
        movies['genres'] = 'Unknown'
    
    # Parse spoken_languages 
    if 'spoken_languages' in movies.columns:
        movies['languages_parsed'] = movies['spoken_languages'].apply(parse_json_column)
        movies['languages'] = movies['languages_parsed'].apply(extract_names_from_list)
    else:
        movies['languages'] = 'English'
    
    # Parse production_companies
    if 'production_companies' in movies.columns:
        movies['companies_parsed'] = movies['production_companies'].apply(parse_json_column)
        movies['production_companies'] = movies['companies_parsed'].apply(extract_names_from_list)
    
    # Parse production_countries
    if 'production_countries' in movies.columns:
        movies['countries_parsed'] = movies['production_countries'].apply(parse_json_column)
        movies['production_countries'] = movies['countries_parsed'].apply(extract_names_from_list)
    
    # Clean other required columns
    movies['overview'] = movies['overview'].fillna('No overview available')
    movies['keywords'] = movies.get('keywords', '')
    movies['cast'] = movies.get('cast', '')
    movies['crew'] = movies.get('crew', '')
    movies['vote_average'] = pd.to_numeric(movies['vote_average'], errors='coerce').fillna(0)
    movies['vote_count'] = pd.to_numeric(movies['vote_count'], errors='coerce').fillna(0)
    movies['release_date'] = movies['release_date'].fillna('')
    
    # Drop parsing helper columns
    movies = movies.drop([col for col in movies.columns if col.endswith('_parsed')], axis=1)
    
    print(f"Cleaned movies: {len(movies)} rows")
    print(f"Sample genres: {movies['genres'].iloc[0]}")
    
    # Save cleaned movies
    movies.to_csv('data/movies.csv', index=False)
    print("✅ Movies data fixed and saved")
    
except Exception as e:
    print(f"Error processing movies: {e}")
    import traceback
    traceback.print_exc()

# Fix ratings data  
print("\n⭐ Fixing ratings data...")
try:
    ratings = pd.read_csv('data/ratings.csv', low_memory=False)
    
    # Check if header is in data
    if 'userId,movieId,rating,timestamp' in str(ratings.iloc[0, 0]):
        ratings = pd.read_csv('data/ratings.csv', header=None, skiprows=1, low_memory=False)
        ratings.columns = ['userId', 'movieId', 'rating', 'timestamp']
    
    # Clean ratings
    ratings['userId'] = pd.to_numeric(ratings['userId'], errors='coerce')
    ratings['movieId'] = pd.to_numeric(ratings['movieId'], errors='coerce')
    ratings['rating'] = pd.to_numeric(ratings['rating'], errors='coerce')
    
    ratings = ratings.dropna(subset=['userId', 'movieId', 'rating'])
    ratings = ratings[(ratings['userId'] > 0) & (ratings['movieId'] > 0) & (ratings['rating'] > 0)]
    
    print(f"Cleaned ratings: {len(ratings)} rows")
    
    # Save cleaned ratings
    ratings.to_csv('data/ratings.csv', index=False)
    print("✅ Ratings data fixed and saved")
    
except Exception as e:
    print(f"Error processing ratings: {e}")

print("\n🎉 Data preprocessing complete! Now run: python setup_and_run.py")
